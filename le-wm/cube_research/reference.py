"""Read-only reference Cube protocol: original World, policy, solver and model."""

import inspect
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import stable_worldmodel as swm
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from research_common import append_jsonl, file_hash, write_json
from research_device import model_tensor
from research_runtime import execution_identity
from .simulator import Harness


ROOT = Path(__file__).resolve().parents[1]
EVAL = yaml.safe_load((ROOT / 'config/eval/cube.yaml').read_text())
CEM = yaml.safe_load((ROOT / 'config/eval/solver/cem.yaml').read_text())
PROTOCOLS = {'paper-300x10-top30': 10, 'public-300x30-top30': CEM['n_steps']}
SEED, OFFSET, BUDGET = EVAL['seed'], EVAL['eval']['goal_offset_steps'], EVAL['eval']['eval_budget']
PLAN = swm.PlanConfig(**EVAL['plan_config'])


def open_dataset(path, cache_dir):
    # An absolute stem bypasses name lookup/download; the separate cache root
    # keeps the read-only source volume untouched even for arbitrary local paths.
    path = Path(path).resolve(strict=True)
    if path.suffix != '.h5':
        raise ValueError('Expected an explicit local .h5 file')
    return swm.data.HDF5Dataset(str(path.with_suffix('')), frameskip=1,
                                keys_to_cache=['action'], cache_dir=cache_dir)


def sample_rows(ep, step, num_eval=50, seed=SEED, offset=OFFSET):
    # Vectorized equivalent of eval.py's per-episode max(step)+1. Match its
    # row-uniform sampling exactly, including exclusion of the last valid row.
    _, inv = np.unique(ep, return_inverse=True)
    end = np.full(int(inv.max()) + 1, -1, dtype=np.int64)
    np.maximum.at(end, inv, step)
    valid = np.flatnonzero(step <= end[inv] - offset)
    chosen = np.random.default_rng(seed).choice(len(valid) - 1, num_eval, replace=False)
    return np.sort(valid[chosen]), valid


def fixed_manifest(data):
    col = 'episode_idx' if 'episode_idx' in data.column_names else 'ep_idx'
    episode, step = data.get_col_data(col), data.get_col_data('step_idx')
    rows, valid = sample_rows(episode, step, EVAL['eval']['num_eval'])
    # Dataset.load_chunk addresses episodes by positional index.
    if not (np.array_equal(episode[rows], np.searchsorted(data.offsets, rows, side='right') - 1)
            and np.array_equal(data.offsets[episode[rows]] + step[rows], rows)
            and np.all(episode[rows + OFFSET] == episode[rows])
            and np.all(step[rows + OFFSET] == step[rows] + OFFSET)):
        raise ValueError('Selected rows are not contiguous indexed episodes')
    p = data.h5_path
    return dict(schema='cube-reference50-v1', dataset=str(p), dataset_bytes=p.stat().st_size,
                seed=SEED, goal_offset=OFFSET, frameskip=data.frameskip,
                sampling='row-uniform without replacement, exclude last eligible row, sort raw indices',
                valid_rows=len(valid), excluded_last_valid_row=int(valid[-1]),
                cases=[dict(case_id=f'row-{i}', row=int(i), episode=int(episode[i]),
                            start_step=int(step[i]), goal_row=int(i + OFFSET), goal_offset=OFFSET)
                       for i in rows])


def fit_actions(data):
    raw = data.get_col_data('action')
    if raw.ndim != 2 or raw.shape[-1] != 5:
        raise ValueError(f'Reference expects per-step 5D actions, got {raw.shape}')
    valid = raw[~np.isnan(raw).any(axis=1)]
    return StandardScaler().fit(valid)


def action_blocks(raw, scaler):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-2:] != (OFFSET, 5):
        raise ValueError(f'Expected (...,25,5) raw temporal commands, got {raw.shape}')
    return scaler.transform(raw.reshape(-1, 5)).reshape(*raw.shape[:-2], PLAN.horizon, 25)


def command_rows(blocks, scaler):
    return scaler.inverse_transform(np.asarray(blocks).reshape(-1, 5)).reshape(-1, OFFSET, 5)


def policy_inputs(info):
    # The recorded action isn't a past-action context: LeWM overwrites it
    # with candidate actions before encode(). Its key is required by goal.pop.
    rgb = {k: info[k] for k in ('pixels', 'goal')}
    rgb['action'] = np.zeros((*rgb['pixels'].shape[:2], 5), dtype=np.float32)
    return rgb


class RGBPolicy(swm.policy.WorldModelPolicy):
    def get_action(self, info_dict, **kwargs):
        return super().get_action(policy_inputs(info_dict), **kwargs)


def make_policy(solver, scaler):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as T

    # Same pipeline as eval.img_transform, without its import-time EGL override.
    tx = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                    T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(EVAL['eval']['img_size'])])
    return RGBPolicy(solver, PLAN, process={'action': scaler}, transform={'pixels': tx, 'goal': tx})


def expanded_inputs(info, samples, device):
    return {k: model_tensor(v, device).unsqueeze(1).expand(v.shape[0], samples, *v.shape[1:])
            for k, v in info.items()}


class TracedCost:
    """Observe candidate costs without changing the optimizer or its RNG."""

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.captures = []

    def get_cost(self, info, actions):
        costs = self.model.get_cost({k: model_tensor(v, self.device) for k, v in info.items()},
                                    model_tensor(actions, self.device))
        value, index = costs.min(dim=1)
        selected = actions[torch.arange(len(index), device=actions.device), index]
        value, selected = value.cpu().numpy(), selected.cpu().numpy()
        self.captures.append(dict(cost=float(value.item()), normalized=selected[0]))
        return costs


class TracedSolver(swm.solver.CEMSolver):
    def __init__(self, model, scaler, device, protocol):
        super().__init__(TracedCost(model, device), batch_size=CEM['batch_size'],
                         num_samples=CEM['num_samples'], var_scale=CEM['var_scale'],
                         n_steps=PROTOCOLS[protocol], topk=CEM['topk'], device=device, seed=SEED)
        self.scaler = scaler
        self.plans = []

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        # Keep exact reference batch_size=1 ordering; the logger observes
        # n_steps calls per vector slot and the returned mean is scored anew.
        captures = []
        self.model.captures = captures
        out = super().solve(info_dict, init_action)
        mean = out['actions']
        means_cost = []
        for i in range(len(mean)):
            model_info = expanded_inputs({k: v[i:i+1] for k, v in info_dict.items()}, 1, self.device)
            means_cost.append(float(self.model.model.get_cost(
                model_info, model_tensor(mean[i:i+1, None], self.device))[0, 0].cpu()))
        self.plans.append(dict(mean=mean.numpy(), variance_field_is_std=out['var'][0].numpy(),
                               raw_commands=command_rows(mean.numpy(), self.scaler),
                               final_mean_cost=means_cost, final_elite_average=out['costs'],
                               sampled=[dict(best=min(chunk, key=lambda r: r['cost']),
                                             last=chunk[-1], iteration_min=[r['cost'] for r in chunk])
                                        for i in range(len(mean))
                                        for chunk in [captures[i*self.n_steps:(i+1)*self.n_steps]]],
                               warm_start_shape=None if init_action is None else list(init_action.shape)))
        return out


class Baseline:
    def __init__(self, name):
        self.name = name
        self.rng = np.random.default_rng(SEED)

    def set_env(self, env):
        self.env = env

    def get_action(self, info_dict):
        shape = self.env.action_space.shape
        if self.name == 'zero':
            return np.zeros(shape, dtype=np.float32)
        return self.rng.uniform(-1, 1, shape).astype(np.float32)


class LoggedPolicy:
    def __init__(self, policy):
        self.policy = policy

    def set_env(self, env):
        self.policy.set_env(env)

    def get_action(self, info):
        self.last_action = self.policy.get_action({k: info[k] for k in ('pixels', 'goal')})
        return self.last_action


class AuditedWorld(swm.World):
    def __init__(self, cases, output, solver=None):
        world = dict(EVAL['world'], num_envs=len(cases), max_episode_steps=2 * BUDGET)
        super().__init__(**world, image_shape=(224, 224), verbose=0)
        self.cases, self.output, self.solver = cases, output, solver
        self.rows = [[] for _ in cases]
        self.steps = 0
        self.evaluators = [Harness(e.unwrapped, {}, 'N') for e in self.envs.unwrapped.envs]

    def record(self, actions=None):
        for i, (h, rows, case) in enumerate(zip(self.evaluators, self.rows, self.cases)):
            row = h.evaluate(None if actions is None else actions[i],
                             False if actions is None else self.terminateds[i],
                             False if actions is None else self.truncateds[i])
            row.update(step=self.steps, case_id=case['case_id'],
                       simulator_time=float(h.env._data.time))
            if not self.steps:
                row.update(reset_rng_seed=int(h.env.np_random_seed), mode=h.env._mode,
                           qpos=h.env._data.qpos.copy(), qvel=h.env._data.qvel.copy(),
                           ctrl=h.env._data.ctrl.copy())
            rows.append(row)
            append_jsonl(self.output / 'steps.jsonl', row)

    def step(self):
        if self.steps == 0:
            self.record()
        previous_plans = len(self.solver.plans) if self.solver else 0
        super().step()
        actions = self.policy.last_action
        if self.solver:
            if len(self.solver.plans) != previous_plans + int(self.steps % OFFSET == 0):
                raise RuntimeError('Reference policy replanning cadence changed')
            plan = self.solver.plans[-1]
            np.testing.assert_array_equal(actions, plan['raw_commands'][:, self.steps % OFFSET])
            if self.steps % OFFSET == 0:
                write_json(self.output / f'plan-{self.steps:02}.json', dict(step=self.steps, **plan))
        self.steps += 1
        self.record(actions)


def summarize(cases, trajectories):
    rows = []
    for case, traj in zip(cases, trajectories):
        if not traj:
            rows.append(dict(case=case, status='unfinished', steps=0))
            continue
        initial, *after = traj
        success = any(r['terminated'] for r in after)
        complete = len(after) == BUDGET
        rows.append(dict(case=case, status=('success' if success else 'failure') if complete else 'unfinished',
                         steps=len(after), initial_success=initial['success'],
                         reference_success=success if complete else None,
                         initial_distance_m=initial['distance'], final_distance_m=traj[-1]['distance'],
                         min_distance_m=min(r['distance'] for r in traj),
                         first_termination_step=next((r['step'] for r in after if r['terminated']), None),
                         first_geometric_success_step=next((r['step'] for r in after if r['success']), None),
                         contact_steps=[r['step'] for r in after if r['contacts']],
                         max_object_motion_m=max(float(np.linalg.norm(np.asarray(r['object_pos']) - initial['object_pos']))
                                                 for r in traj)))
    def metric(pool, key):
        known = [r[key] for r in pool if r.get(key) is not None]
        return dict(n=len(pool), observed=len(known), successes=sum(known),
                    rate=(sum(known) / len(pool)) if pool and len(known) == len(pool) else None)
    return dict(cases=rows, initial=metric(rows, 'initial_success'), overall=metric(rows, 'reference_success'),
                non_initial=metric([r for r in rows if r.get('initial_success') is False], 'reference_success'),
                unfinished=sum(r['status'] == 'unfinished' for r in rows))


def run_policy(data, cases, output, name, model=None, scaler=None, device=None, protocol=None):
    output.mkdir(exist_ok=False)
    solver = TracedSolver(model, scaler, device, protocol) if name == 'cem' else None
    world = None
    try:
        world = AuditedWorld(cases, output, solver)
        policy = make_policy(solver, scaler) if solver else Baseline(name)
        world.set_policy(LoggedPolicy(policy))
        result = world.evaluate_from_dataset(data, episodes_idx=[c['episode'] for c in cases],
                                             start_steps=[c['start_step'] for c in cases],
                                             goal_offset_steps=OFFSET, eval_budget=BUDGET,
                                             callables=EVAL['eval']['callables'], save_video=False)
        np.testing.assert_equal(result['episode_successes'],
                                [any(x['terminated'] for x in r[1:]) for r in world.rows])
        write_json(output / 'reference-metrics.json', result)
    finally:
        rows = world.rows if world is not None else [[] for _ in cases]
        write_json(output / 'summary.json', summarize(cases, rows))
        if world is not None:
            world.close()


def runtime_identity(device, bridge):
    files = {o.__name__: inspect.getfile(o) for o in
             (swm, swm.World, swm.data.HDF5Dataset, swm.policy.WorldModelPolicy,
              swm.solver.CEMSolver, type(bridge.base), make_policy)}
    files['eval.py'] = str(ROOT / 'eval.py')
    packages = ['stable-worldmodel', 'stable-pretraining', 'torch', 'torchvision', 'ogbench',
                'mujoco', 'gymnasium', 'h5py', 'hdf5plugin', 'numpy', 'scikit-learn',
                'transformers', 'hydra-core']
    return dict(python=sys.executable, python_version=sys.version,
                sources={k: dict(path=v, sha256=file_hash(v)) for k, v in files.items()},
                packages={p: version(p) for p in packages}, execution=execution_identity(device),
                model=bridge.identity)
