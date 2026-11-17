"""Frozen PushT collection. Runtime commands are intended for the server only."""
from __future__ import annotations

import argparse
import copy
import json
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from research_common import (
    Case, append_jsonl, derived_seed, file_hash, finish_run, new_run,
    read_cases, runtime_metadata, stable_case_id, sync_device, write_cases, write_json,
)

DEFAULTS = {
    'env_name': 'swm/PushT-v1', 'dataset': 'pusht_expert_train', 'cache_dir': None,
    'policy': 'quentinll/lewm-pusht', 'device': 'cuda', 'img_size': 224,
    'horizon': 5, 'receding_horizon': 5, 'action_block': 5, 'eval_budget': 50,
    'goal_offset': 25, 'warm_start': True, 'num_samples': 100, 'iterations': 10,
    'topk': 10, 'planning_seed': 42, 'environment_seed': 123,
    'diagnostics': False, 'save_snapshots': False, 'keep_best': 2,
    'keep_random': 2, 'max_snapshot_plans': 1,
}


def _load_cfg(args):
    import yaml

    cfg = DEFAULTS.copy()
    if args.config:
        loaded = yaml.safe_load(Path(args.config).read_text()) or {}
        if not isinstance(loaded, dict) or set(loaded) - set(cfg):
            raise ValueError('Unknown config keys or non-mapping config')
        cfg.update(loaded)
    # Parser suppresses absent options: only explicit CLI arguments override YAML.
    cfg.update({k: v for k, v in vars(args).items() if k in cfg})
    if cfg['save_snapshots']:
        cfg['diagnostics'] = True
    if cfg['env_name'] != 'swm/PushT-v1' or cfg['action_block'] != 5 or cfg['img_size'] != 224:
        raise ValueError('This protocol supports PushT at 224px with action_block=5 only')
    for key in ('horizon', 'receding_horizon', 'eval_budget', 'goal_offset', 'num_samples', 'iterations',
                'topk', 'max_snapshot_plans'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('keep_best', 'keep_random', 'planning_seed', 'environment_seed'):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('warm_start', 'diagnostics', 'save_snapshots'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if not 1 <= cfg['receding_horizon'] <= cfg['horizon']:
        raise ValueError('Require 1 <= K <= H')
    if cfg['horizon'] * cfg['action_block'] > cfg['eval_budget']:
        raise ValueError('H * action_block must not exceed eval_budget')
    if not 2 <= cfg['topk'] <= cfg['num_samples']:
        raise ValueError('Require 2 <= topk <= num_samples')
    if min(cfg['keep_best'], cfg['keep_random']) < 0 or cfg['max_snapshot_plans'] < 1:
        raise ValueError('Invalid snapshot sampling limits')
    if str(cfg['device']).split(':')[0] not in ('cpu', 'cuda', 'mps'):
        raise ValueError('device must be cpu, cuda[:index] or mps')
    return cfg


def generate_cases(args):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(args.dataset, cache_dir=args.cache_dir, keys_to_cache=[])
    if args.goal_offset <= 0 or args.num_cases <= 0:
        raise ValueError('goal-offset and num-cases must be positive')
    with h5py.File(dataset.h5_path, 'r') as handle:
        lengths = handle['ep_len'][:]
    eligible = np.flatnonzero(lengths > args.goal_offset)
    if args.num_cases > len(eligible):
        raise ValueError('Use at most one case per eligible episode for independent case clusters')
    rng = np.random.default_rng(args.case_seed)
    cases = []
    for ep in rng.choice(eligible, args.num_cases, replace=False):
        start = int(rng.integers(0, int(lengths[ep]) - args.goal_offset))
        ep = int(ep)  # dataset.load_chunk uses ep_len row indices, not arbitrary episode labels.
        cases.append(Case(stable_case_id(ep, start, args.goal_offset), ep, start, args.goal_offset))
    write_cases(args.output, cases, metadata={
        'case_seed': args.case_seed, 'dataset': args.dataset,
        'dataset_size': dataset.h5_path.stat().st_size,
        'dataset_sha256': file_hash(dataset.h5_path),
        'sampling': 'one start per distinct episode; development cases from supplied dataset',
    })
    if dataset.h5_file is not None:
        dataset.h5_file.close()
    print(f'Wrote {len(cases)} cases to {args.output}')


def expand_info(info, count):
    import numpy as np
    import torch

    out = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            # Do not mutate caller-owned tensors. The model mutates the new dict.
            out[key] = value.unsqueeze(1).expand(value.shape[0], count, *value.shape[1:])
        elif isinstance(value, np.ndarray):
            out[key] = np.repeat(value[:, None], count, axis=1)
        else:
            out[key] = value
    return out


class LoggingCEMSolver:
    """Single-env reference CEM: final elite mean, unbiased std, zero tail padding.

    All backends use CPU noise. This is an explicit research baseline, not a
    claim of bitwise equivalence to the upstream CUDA RNG. Diagnostics use a
    separate NumPy RNG and are disabled entirely for formal timing runs.
    """

    def __init__(self, model, *, num_samples, n_steps, topk, device, seed,
                 diagnostics=False, snapshot_dir=None, keep_best=2, keep_random=2,
                 max_snapshot_plans=1, run_id='unassigned', case_id='unassigned'):
        import numpy as np
        import torch

        self.model = model
        self.num_samples, self.n_steps, self.topk = int(num_samples), int(n_steps), int(topk)
        if not 2 <= self.topk <= self.num_samples or self.n_steps < 1:
            raise ValueError('Invalid CEM configuration')
        self.device = torch.device(device)
        self.seed = int(seed)
        self.torch_gen = torch.Generator(device='cpu').manual_seed(self.seed)
        self.diagnostic_rng = np.random.default_rng(derived_seed(self.seed, 'diagnostics'))
        self.diagnostics = bool(diagnostics or snapshot_dir)
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.keep_best, self.keep_random = keep_best, keep_random
        self.max_snapshot_plans = max_snapshot_plans
        self.run_id, self.case_id = run_id, case_id
        self.plan_id, self.environment_step = 0, 0
        self.iteration_rows, self.plan_rows = [], []
        self.context_provider = None
        self._n_envs = 1

    def configure(self, *, action_space, n_envs, config):
        import numpy as np

        if n_envs != 1 or not 1 <= config.receding_horizon <= config.horizon:
            raise ValueError('Research collector supports one environment and 1 <= K <= H')
        self._action_space, self._n_envs, self._config = action_space, n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self):
        return self._n_envs

    @property
    def action_dim(self):
        return self._action_dim * self._config.action_block

    @property
    def horizon(self):
        return self._config.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None):
        import torch

        with torch.inference_mode():
            return self._solve(info_dict, init_action)

    def _solve(self, info_dict, init_action):
        import numpy as np
        import torch

        total_start = time.perf_counter()
        plan_id = self.plan_id
        self.plan_id += 1
        key = {'run_id': self.run_id, 'case_id': self.case_id, 'plan_id': plan_id,
               'environment_step': self.environment_step, 'planning_seed': self.seed}
        capture = self.snapshot_dir is not None and plan_id < self.max_snapshot_plans
        arrays, context = {}, {}
        if capture:
            if self.context_provider is None:
                raise RuntimeError('Snapshot collection requires a replay context provider')
            context, arrays = self.context_provider()
            arrays['planner_rng_state'] = self.torch_gen.get_state().cpu().numpy()
            arrays['init_action'] = (init_action.detach().cpu().numpy() if init_action is not None
                                     else np.zeros((1, 0, self.action_dim), dtype=np.float32))
        info = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in info_dict.items()}
        if capture:
            for name in ('pixels', 'goal', 'action'):
                if name not in info or not torch.is_tensor(info[name]):
                    raise ValueError(f'Missing model input {name}')
                arrays['info_' + name] = info[name].detach().cpu().numpy()
        sync_device(self.device)
        segment_start = time.perf_counter()
        mean = torch.zeros((1, self.horizon, self.action_dim), device=self.device)
        if init_action is not None:
            if init_action.ndim != 3 or init_action.shape[0] != 1 or init_action.shape[2] != self.action_dim:
                raise ValueError('Invalid warm-start shape')
            length = init_action.shape[1]
            if length > self.horizon:
                raise ValueError('Warm-start exceeds horizon')
            mean[:, :length] = init_action.to(device=self.device, dtype=mean.dtype)
        std = torch.ones_like(mean)
        search_seconds = 0.0
        extra_score_seconds = 0.0
        saved_actions, saved_scores, candidate_meta = [], [], []
        return_score = None
        for iteration in range(self.n_steps):
            noise = torch.randn((1, self.num_samples, self.horizon, self.action_dim),
                                generator=self.torch_gen, device='cpu').to(self.device)
            candidates = noise * std[:, None] + mean[:, None]
            candidates[:, 0] = mean
            costs = self.model.get_cost(expand_info(info, self.num_samples), candidates)
            if costs.shape != (1, self.num_samples) or not bool(torch.isfinite(costs).all()):
                raise ValueError('Invalid/nonfinite candidate costs')
            values, indices = torch.topk(costs, self.topk, dim=1, largest=False)
            elites = candidates[torch.arange(1, device=self.device)[:, None], indices]
            mean, std = elites.mean(1), elites.std(1)
            if self.diagnostics:
                sync_device(self.device)
                search_seconds += time.perf_counter() - segment_start
                extra_start = time.perf_counter()
                mean_cost = self.model.get_cost(expand_info(info, 1), mean[:, None])[0, 0]
                sync_device(self.device)
                extra_score_seconds += time.perf_counter() - extra_start
                if not bool(torch.isfinite(mean_cost)):
                    raise ValueError('Nonfinite mean plan cost')
                return_score = float(mean_cost.cpu())
                self.iteration_rows.append({
                    **key, 'iteration': iteration, 'candidate_count': self.num_samples,
                    'horizon': self.horizon, 'topk': self.topk,
                    'iteration_min_cost': float(costs.min().cpu()),
                    'elite_mean_cost': float(values.mean().cpu()),
                    'updated_mean_cost': return_score,
                    'distribution_std_mean': float(std.mean().cpu()),
                })
                if capture:
                    ranked = torch.argsort(costs[0]).cpu().numpy()
                    best = ranked[:min(self.keep_best, self.num_samples)]
                    pool = np.setdiff1d(np.arange(self.num_samples), best)
                    random = self.diagnostic_rng.choice(pool, min(self.keep_random, len(pool)), replace=False)
                    for idx in [*best, *random]:
                        idx = int(idx)
                        saved_actions.append(candidates[0, idx].cpu().numpy().copy())
                        saved_scores.append(float(costs[0, idx].cpu()))
                        candidate_meta.append({'iteration': iteration, 'candidate_id': f'sample_{idx}',
                                               'kind': 'best' if idx in best else 'random', 'selected': False})
                    saved_actions.append(mean[0].cpu().numpy().copy())
                    saved_scores.append(return_score)
                    candidate_meta.append({'iteration': iteration, 'candidate_id': 'mean',
                                           'kind': 'updated_mean', 'selected': iteration == self.n_steps - 1})
                sync_device(self.device)
                segment_start = time.perf_counter()
        actions = mean.detach().cpu()
        final_std = std.detach().cpu()
        elite_mean = values.mean(1).cpu().tolist()
        sync_device(self.device)
        search_seconds += time.perf_counter() - segment_start
        if capture:
            folder = self.snapshot_dir / self.case_id / f'plan_{plan_id:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            arrays['actions'] = np.stack(saved_actions)
            arrays['J_pred'] = np.asarray(saved_scores, dtype=np.float64)
            arrays['returned_actions'] = actions.numpy()
            with (folder / 'arrays.npz').open('xb') as handle:
                np.savez_compressed(handle, **arrays)
            write_json(folder / 'snapshot.json', {
                'schema': 'lewm-snapshot-v2', **key, **context,
                'horizon': self.horizon, 'action_block': self._config.action_block,
                'num_samples': self.num_samples, 'iterations': self.n_steps,
                'topk': self.topk, 'candidates': candidate_meta,
                'arrays_sha256': file_hash(folder / 'arrays.npz'),
            })
        wall = time.perf_counter() - total_start
        if not self.diagnostics:
            search_seconds = wall
        self.plan_rows.append({
            **key, 'solver_seconds': search_seconds,
            'solver_wall_seconds': wall, 'diagnostic_and_capture_seconds': max(0.0, wall - search_seconds),
            'extra_score_seconds': extra_score_seconds,
            'returned_mean_cost': return_score, 'diagnostics': self.diagnostics,
            'candidate_predictions': self.num_samples * self.n_steps * self.horizon,
            'extra_prediction_steps': self.n_steps * self.horizon if self.diagnostics else 0,
        })
        return {'actions': actions, 'mean': [actions], 'var': [final_std], 'costs': elite_mean}


class InstrumentedPolicy:
    def __init__(self, policy, solver, world, recipe, cfg):
        self.policy, self.solver, self.world, self.recipe, self.cfg = policy, solver, world, recipe, cfg
        self.environment_step = 0
        self.prefix_actions = []
        self.raw_pixels = None
        self.solver.context_provider = self.snapshot_context

    def set_env(self, env):
        self.policy.set_env(env)

    def get_action(self, info):
        import numpy as np

        if self.environment_step == 0:
            self.recipe.update(initial_state=np.array(info['state'][0, -1], copy=True),
                               goal_state=np.array(info['goal_state'][0, -1], copy=True))
        self.solver.environment_step = self.environment_step
        if self.solver.snapshot_dir is not None and self.solver.plan_id < self.solver.max_snapshot_plans:
            self.raw_pixels = np.array(info['pixels'][0, -1], copy=True)
        action = self.policy.get_action(info)
        self.prefix_actions.append(np.array(action[0], copy=True))
        self.environment_step += 1
        return action

    def snapshot_context(self):
        import numpy as np
        from research_runtime import physics_vector, render_frame

        rendered = render_frame(self.world, self.cfg['img_size'])
        return ({'recipe': copy.deepcopy(self.recipe)}, {
            'prefix_actions': np.asarray(self.prefix_actions).reshape(-1, 2),
            'physics': physics_vector(self.world), 'rendered_pixels': rendered,
            'observation_pixels': self.raw_pixels.copy(),
        })


def run_experiment(args):
    import numpy as np
    import stable_worldmodel as swm
    from research_runtime import (
        fit_processors, image_transform, install_reset_recorder,
        load_model, make_world, source_fingerprints, execution_identity,
    )

    cfg = _load_cfg(args)
    cases, case_meta = read_cases(args.cases)
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise ValueError('limit-cases must be positive')
        cases = cases[:args.limit_cases]
    if not cases or any(c.goal_offset != cfg['goal_offset'] for c in cases):
        raise ValueError('Empty cases or manifest goal_offset disagrees with configuration')
    if len({c.episode for c in cases}) != len(cases):
        raise ValueError('Use one case per episode to keep case-level inference meaningful')
    if case_meta.get('dataset') != cfg['dataset']:
        raise ValueError('Manifest dataset disagrees with configuration')
    output = new_run(args.output)
    dataset = None
    try:
        dataset = swm.data.HDF5Dataset(cfg['dataset'], cache_dir=cfg['cache_dir'], keys_to_cache=[])
        dataset_sha256 = file_hash(dataset.h5_path)
        if case_meta.get('dataset_sha256') and dataset_sha256 != case_meta['dataset_sha256']:
            raise ValueError('Dataset content changed since manifest generation')
        for case in cases:
            if case.episode >= len(dataset.lengths) or case.start_step + case.goal_offset >= dataset.lengths[case.episode]:
                raise ValueError(f'Case outside dataset: {case}')
        model, model_identity = load_model(cfg['policy'], cfg['cache_dir'], cfg['device'])
        process, processor_state = fit_processors(dataset)
        write_json(output / 'processors.json', processor_state)
        run_id = uuid.uuid4().hex
        write_json(output / 'run_metadata.json', {
            **runtime_metadata(Path(__file__).parent), 'schema': 'lewm-run-v2', 'run_id': run_id,
            'config': cfg, 'cases_sha256': file_hash(args.cases), 'cases': [asdict(c) for c in cases],
            'case_metadata': case_meta, 'model': model_identity,
            'processors_sha256': file_hash(output / 'processors.json'),
            'sources': source_fingerprints(),
            'research_sources': {p.name: file_hash(p) for p in Path(__file__).parent.glob('research_*.py')},
            'dataset_size': dataset.h5_path.stat().st_size,
            'dataset_sha256': dataset_sha256,
            'execution': execution_identity(cfg['device']),
            'sampling': 'CPU torch RNG for all devices',
            'protocol': 'upstream evaluate_from_dataset; history_len=1; fixed full episode budget',
        })
        print(json.dumps({'config': cfg, 'model': model_identity}, indent=2))
        config = swm.PlanConfig(horizon=cfg['horizon'], receding_horizon=cfg['receding_horizon'],
                                action_block=cfg['action_block'], history_len=1, warm_start=cfg['warm_start'])
        for case in cases:
            solver = LoggingCEMSolver(
                model, num_samples=cfg['num_samples'], n_steps=cfg['iterations'], topk=cfg['topk'],
                device=cfg['device'], seed=derived_seed(cfg['planning_seed'], case.case_id),
                diagnostics=cfg['diagnostics'], snapshot_dir=output / 'snapshots' if cfg['save_snapshots'] else None,
                keep_best=cfg['keep_best'], keep_random=cfg['keep_random'],
                max_snapshot_plans=cfg['max_snapshot_plans'], run_id=run_id, case_id=case.case_id,
            )
            world = make_world(cfg)
            try:
                recipe = install_reset_recorder(world, derived_seed(cfg['environment_seed'], case.case_id))
                policy = InstrumentedPolicy(swm.policy.WorldModelPolicy(
                    solver=solver, config=config, process=process,
                    transform={'pixels': image_transform(cfg['img_size']), 'goal': image_transform(cfg['img_size'])}),
                    solver, world, recipe, cfg)
                world.set_policy(policy)
                sync_device(cfg['device'])
                started = time.perf_counter()
                metrics = world.evaluate_from_dataset(
                    dataset, [case.episode], [case.start_step], case.goal_offset, cfg['eval_budget'],
                    callables=[{'method': '_set_state', 'args': {'state': {'value': 'state'}}},
                               {'method': '_set_goal_state', 'args': {'goal_state': {'value': 'goal_state'}}}],
                    save_video=False,
                )
                sync_device(cfg['device'])
                elapsed = time.perf_counter() - started
                # Log writes are after episode timing. NPZ capture costs remain explicitly diagnostic.
                for row in solver.iteration_rows:
                    append_jsonl(output / 'cem_iterations.jsonl', row)
                for row in solver.plan_rows:
                    append_jsonl(output / 'plans.jsonl', row)
                case_dir = output / 'cases' / case.case_id
                case_dir.mkdir(parents=True, exist_ok=False)
                write_json(case_dir / 'recipe.json', recipe)
                np.save(case_dir / 'executed_actions.npy', np.asarray(policy.prefix_actions))
                append_jsonl(output / 'episodes.jsonl', {
                    'run_id': run_id, **asdict(case), 'success': bool(metrics['episode_successes'][0]),
                    'environment_steps': policy.environment_step, 'planning_calls': solver.plan_id,
                    'episode_seconds': elapsed, 'solver_seconds': sum(r['solver_seconds'] for r in solver.plan_rows),
                    'diagnostic_and_capture_seconds': sum(r['diagnostic_and_capture_seconds'] for r in solver.plan_rows),
                    'candidate_predictions': sum(r['candidate_predictions'] for r in solver.plan_rows),
                    'extra_prediction_steps': sum(r['extra_prediction_steps'] for r in solver.plan_rows),
                    'planning_seed': cfg['planning_seed'], 'effective_planning_seed': solver.seed,
                    'environment_seed': recipe['reset_seed'], 'num_samples': cfg['num_samples'],
                    'iterations': cfg['iterations'], 'diagnostics': cfg['diagnostics'],
                })
                print('Completed case', case.case_id)
            finally:
                world.close()
        finish_run(output, 'complete', completed_cases=len(cases))
    except BaseException as exc:
        finish_run(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if dataset is not None and dataset.h5_file is not None:
            dataset.h5_file.close()


def preflight(args):
    from research_common import source_identity
    from research_runtime import load_model

    model, identity = load_model(args.policy, args.cache_dir, args.device)
    print(json.dumps({'model': identity, 'source': source_identity(model), 'device': args.device,
                      'frozen': all(not p.requires_grad for p in model.parameters()),
                      'parameters': sum(p.numel() for p in model.parameters())}, indent=2))


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate-cases')
    g.add_argument('--dataset', default='pusht_expert_train')
    g.add_argument('--cache-dir')
    g.add_argument('--output', required=True)
    g.add_argument('--num-cases', type=int, default=20)
    g.add_argument('--goal-offset', type=int, default=25)
    g.add_argument('--case-seed', type=int, default=123)
    r = sub.add_parser('run', argument_default=argparse.SUPPRESS)
    r.add_argument('--cases', required=True)
    r.add_argument('--output', required=True)
    r.add_argument('--config', default=None)
    r.add_argument('--limit-cases', type=int, default=None)
    for name, default in DEFAULTS.items():
        option = '--' + name.replace('_', '-')
        if isinstance(default, bool):
            r.add_argument(option, action=argparse.BooleanOptionalAction)
        else:
            r.add_argument(option, type=int if isinstance(default, int) else str)
    f = sub.add_parser('preflight')
    f.add_argument('--policy', default='quentinll/lewm-pusht')
    f.add_argument('--cache-dir')
    f.add_argument('--device', default='cuda')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    {'generate-cases': generate_cases, 'run': run_experiment, 'preflight': preflight}[args.command](args)
