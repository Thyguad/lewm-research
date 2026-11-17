"""Server-side replay audit and plots. No ML imports at module import time."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from research_common import (
    append_jsonl, file_hash, finish_run, new_run, read_json, read_rows,
    require_complete, runtime_metadata, write_json,
)


def load_run(root):
    root = Path(root)
    require_complete(root)
    meta = read_json(root / 'run_metadata.json')
    if meta.get('schema') != 'lewm-run-v2':
        raise ValueError('Requires newly collected v2 data; old snapshots cannot be repaired retroactively')
    if file_hash(root / 'processors.json') != meta['processors_sha256']:
        raise ValueError('Preprocessor checksum mismatch')
    return meta


def snapshot_data(path, meta):
    import numpy as np

    path = Path(path)
    snap = read_json(path)
    if snap.get('schema') != 'lewm-snapshot-v2' or snap['run_id'] != meta['run_id']:
        raise ValueError(f'Invalid snapshot: {path}')
    if snap['case_id'] not in {c['case_id'] for c in meta['cases']}:
        raise ValueError('Snapshot case not present in run')
    if file_hash(path.parent / 'arrays.npz') != snap['arrays_sha256']:
        raise ValueError('Snapshot checksum mismatch')
    with np.load(path.parent / 'arrays.npz', allow_pickle=False) as handle:
        data = {k: handle[k].copy() for k in handle.files}
    candidates = snap['candidates']
    keys = [(r['iteration'], r['candidate_id']) for r in candidates]
    if len(set(keys)) != len(keys) or not keys:
        raise ValueError('Empty or duplicate candidate identities')
    selected = [i for i, r in enumerate(candidates) if r['selected']]
    if len(selected) != 1 or not np.array_equal(data['actions'][selected[0]], data['returned_actions'][0]):
        raise ValueError('Selected candidate must equal the actual returned mean')
    shape = (len(candidates), snap['horizon'], 2 * snap['action_block'])
    if data['actions'].shape != shape or data['J_pred'].shape != (len(candidates),):
        raise ValueError('Invalid action/cost dimensions')
    if len(data['prefix_actions']) != snap['environment_step']:
        raise ValueError('Action prefix length disagrees with environment_step')
    if not np.isfinite(data['actions']).all() or not np.isfinite(data['J_pred']).all():
        raise ValueError('Nonfinite snapshot values')
    return snap, data


def verify_runtime(meta, policy=None, cache_dir=None, device=None):
    from research_runtime import load_model, source_fingerprints

    cfg = meta['config'].copy()
    for name, value in (('policy', policy), ('cache_dir', cache_dir), ('device', device)):
        if value is not None:
            cfg[name] = value
    if source_fingerprints() != meta['sources']:
        raise ValueError('Installed stable-worldmodel source differs from collection')
    versions = runtime_metadata()['packages']
    for name in ('torch', 'torchvision', 'numpy', 'gymnasium', 'pymunk', 'pygame',
                 'Pillow', 'scikit-learn', 'transformers'):
        if versions[name] != meta['packages'][name]:
            raise ValueError(f'Runtime package changed since collection: {name}')
    for name in ('research_runtime.py', 'research_cem.py'):
        if file_hash(Path(__file__).with_name(name)) != meta['research_sources'][name]:
            raise ValueError(f'Collection/replay implementation changed: {name}')
    model, identity = load_model(cfg['policy'], cfg['cache_dir'], cfg['device'])
    for name in ('type', 'weights_sha256', 'config_sha256'):
        if identity[name] != meta['model'][name]:
            raise ValueError(f'Model identity mismatch: {name}')
    return cfg, model


def model_inputs(data, device):
    import torch
    return {name: torch.from_numpy(data['info_' + name].copy()).to(device)
            for name in ('pixels', 'goal', 'action')}


def replay_candidate(cfg, snap, data, actions, state_atol, pixel_atol):
    """Fresh environment per candidate/repeat; execute the full horizon once."""
    import numpy as np
    from research_runtime import (
        advance_world, assert_restored, physics_vector, render_frame,
        restore_world, task_measurements,
    )

    world = restore_world(cfg, snap['recipe'], data['prefix_actions'])
    try:
        restored = assert_restored(world, data, cfg['img_size'], state_atol, pixel_atol)
        ever_success = False  # Success on one of the executed steps, matching online evaluation.
        trajectory = []
        for action in actions:
            _, _, terminated, _ = advance_world(world, action)
            ever_success |= bool(terminated[0])
            trajectory.append(physics_vector(world))
        result = task_measurements(world)
        result.update(ever_success=ever_success, **restored)
        return result, np.stack(trajectory), render_frame(world, cfg['img_size'])
    finally:
        world.close()


def rank_correlation(rows, x, y):
    from scipy.stats import spearmanr

    if len(rows) < 3 or len({r[x] for r in rows}) < 2 or len({r[y] for r in rows}) < 2:
        return None
    result = float(spearmanr([r[x] for r in rows], [r[y] for r in rows]).statistic)
    return result if math.isfinite(result) else None


def audit(args):
    import numpy as np
    import torch
    from research_cem import expand_info
    from research_runtime import encode_frame, inverse_actions, execution_identity

    if min(args.state_atol, args.pixel_atol, args.cost_atol, args.cost_rtol) < 0:
        raise ValueError('Tolerances must be nonnegative')
    root = Path(args.run)
    meta = load_run(root)
    paths = sorted((root / 'snapshots').glob('*/plan_*/snapshot.json'))
    if not paths:
        raise ValueError('No snapshots; collect with --save-snapshots first')
    output = new_run(args.output)
    started = time.perf_counter()
    try:
        cfg, model = verify_runtime(meta, args.policy, args.cache_dir, args.device)
        stats = read_json(root / 'processors.json')['action']
        write_json(output / 'audit_metadata.json', {
            **runtime_metadata(), 'schema': 'lewm-audit-v2', 'source_run_id': meta['run_id'],
            'source_metadata_sha256': file_hash(root / 'run_metadata.json'),
            'config': cfg, 'tolerances': {k: getattr(args, k) for k in
                                        ('state_atol', 'pixel_atol', 'cost_atol', 'cost_rtol')},
            'execution': execution_identity(cfg['device']),
            'research_sources': {p.name: file_hash(p) for p in Path(__file__).parent.glob('research_*.py')},
            'repeats_per_candidate': 2,
            'task_cost': 'max(position_error / 20, wrapped_angle_error / (pi / 9))',
            'sampling_warning': 'Audited best/random/updated means are a selected subset, not the full search population.',
        })
        summaries, seen_plans = [], set()
        total_rows = total_steps = 0
        for path in paths:
            snap, data = snapshot_data(path, meta)
            plan_key = (snap['case_id'], snap['plan_id'])
            if plan_key in seen_plans:
                raise ValueError('Duplicate case/plan snapshot')
            seen_plans.add(plan_key)
            info = model_inputs(data, cfg['device'])
            with torch.inference_mode():
                goal = model.encode({'pixels': info['goal']})['emb'][0, -1].cpu().numpy()
            obs_error = int(np.abs(data['observation_pixels'].astype(int) -
                                   data['rendered_pixels'].astype(int)).max())
            # This can be nonzero at dataset initialization; expose it, do not silently substitute images.
            rows = []
            for index, candidate in enumerate(snap['candidates']):
                raw_actions = inverse_actions(data['actions'][index].copy(), stats)
                if len(raw_actions) != snap['horizon'] * snap['action_block']:
                    raise ValueError('Action block/time alignment error')
                with torch.inference_mode():
                    scored = expand_info(info, 1)
                    normalized = torch.from_numpy(data['actions'][index].copy())[None, None].to(cfg['device'])
                    cost = float(model.get_cost(scored, normalized)[0, 0].cpu())
                    predicted = scored['predicted_emb'][0, 0, -1].cpu().numpy()
                if not np.isclose(cost, data['J_pred'][index], rtol=args.cost_rtol, atol=args.cost_atol):
                    raise RuntimeError('Recomputed prediction cost differs from collection')
                first, trajectory, frame = replay_candidate(cfg, snap, data, raw_actions, args.state_atol, args.pixel_atol)
                second, again, frame_again = replay_candidate(cfg, snap, data, raw_actions, args.state_atol, args.pixel_atol)
                repeat_state = float(np.abs(trajectory - again).max())
                repeat_pixels = int(np.abs(frame.astype(int) - frame_again.astype(int)).max())
                if (not math.isfinite(repeat_state) or repeat_state > args.state_atol or
                        repeat_pixels > args.pixel_atol or first['ever_success'] != second['ever_success'] or
                        first['endpoint_success'] != second['endpoint_success']):
                    raise RuntimeError('Repeated reconstruction/candidate replay diverged')
                actual = encode_frame(model, frame, cfg['device'], cfg['img_size'])
                steps = 2 * (len(data['prefix_actions']) + len(raw_actions))
                row = {
                    'run_id': snap['run_id'], 'case_id': snap['case_id'], 'plan_id': snap['plan_id'],
                    'environment_step': snap['environment_step'], **candidate,
                    'J_pred': float(data['J_pred'][index]), 'J_pred_recomputed': cost,
                    'J_enc': float(np.square(actual - goal).sum()),
                    'prediction_endpoint_sse': float(np.square(predicted - actual).sum()),
                    **first, 'repeat_state_max_abs': repeat_state, 'repeat_pixel_max_abs': repeat_pixels,
                    'observation_render_pixel_max_abs': obs_error,
                    'offline_environment_steps': steps, 'environment_restores': 2,
                    'num_samples': snap['num_samples'], 'iterations': snap['iterations'],
                }
                append_jsonl(output / 'audit_records.jsonl', row)
                rows.append(row)
                total_rows += 1
                total_steps += steps
            selected = next(r for r in rows if r['selected'])
            best = min(rows, key=lambda r: r['J_task'])
            summary = {
                'run_id': snap['run_id'], 'case_id': snap['case_id'], 'plan_id': snap['plan_id'],
                'environment_step': snap['environment_step'], 'audited_count': len(rows),
                'selected_iteration': selected['iteration'], 'selected_candidate_id': selected['candidate_id'],
                'selected_J_pred': selected['J_pred'], 'selected_J_enc': selected['J_enc'],
                'selected_J_task': selected['J_task'],
                'best_audited_iteration': best['iteration'], 'best_audited_candidate_id': best['candidate_id'],
                'selection_loss_on_audited_set': selected['J_task'] - best['J_task'],
                'rho_pred_enc': rank_correlation(rows, 'J_pred', 'J_enc'),
                'rho_enc_task': rank_correlation(rows, 'J_enc', 'J_task'),
                'rho_pred_task': rank_correlation(rows, 'J_pred', 'J_task'),
                'observation_render_pixel_max_abs': obs_error,
            }
            summaries.append(summary)
            print('Audited', snap['case_id'], snap['plan_id'], len(rows), 'candidates')
        write_json(output / 'audit_summary.json', summaries)
        finish_run(output, 'complete', candidates=total_rows, offline_environment_steps=total_steps,
                   environment_restores=2 * total_rows, wall_seconds=time.perf_counter() - started)
    except BaseException as exc:
        finish_run(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def replan(args):
    """Counterfactual budget at one saved state, not another closed-loop run."""
    import copy
    import uuid
    import torch
    from types import SimpleNamespace
    import stable_worldmodel as swm
    from research_cem import LoggingCEMSolver
    from research_common import derived_seed
    from research_runtime import execution_identity

    root = Path(args.run)
    original = load_run(root)
    snap, data = snapshot_data(args.snapshot, original)
    if not 2 <= args.topk <= args.num_samples or args.iterations < 1:
        raise ValueError('Require 2 <= topk <= N and I >= 1')
    output = new_run(args.output)
    try:
        cfg, model = verify_runtime(original, args.policy, args.cache_dir, args.device)
        cfg.update(num_samples=args.num_samples, iterations=args.iterations, topk=args.topk,
                   diagnostics=True, save_snapshots=True)
        if args.planning_seed is not None:
            cfg['planning_seed'] = args.planning_seed
        run_id = uuid.uuid4().hex
        meta = copy.deepcopy(original)
        meta.update(**runtime_metadata(), run_id=run_id, config=cfg, kind='replan',
                    source_snapshot_sha256=file_hash(args.snapshot),
                    source_run_id=original['run_id'],
                    initialization='fixed source warm start; saved RNG state unless planning seed explicitly overridden')
        meta['cases'] = [c for c in original['cases'] if c['case_id'] == snap['case_id']]
        meta['execution'] = execution_identity(cfg['device'])
        meta['research_sources'] = {p.name: file_hash(p) for p in Path(__file__).parent.glob('research_*.py')}
        write_json(output / 'processors.json', read_json(root / 'processors.json'))
        if file_hash(output / 'processors.json') != original['processors_sha256']:
            raise ValueError('Preprocessor serialization changed')
        write_json(output / 'run_metadata.json', meta)
        solver = LoggingCEMSolver(
            model, num_samples=args.num_samples, n_steps=args.iterations, topk=args.topk,
            device=cfg['device'], seed=derived_seed(cfg['planning_seed'], snap['case_id']),
            diagnostics=True, snapshot_dir=output / 'snapshots',
            keep_best=cfg['keep_best'], keep_random=cfg['keep_random'],
            max_snapshot_plans=snap['plan_id'] + 1, run_id=run_id, case_id=snap['case_id'])
        solver.configure(action_space=SimpleNamespace(shape=(1, 2)), n_envs=1,
                         config=swm.PlanConfig(horizon=cfg['horizon'], receding_horizon=cfg['receding_horizon'],
                                               action_block=cfg['action_block'], history_len=1,
                                               warm_start=cfg['warm_start']))
        solver.plan_id, solver.environment_step = snap['plan_id'], snap['environment_step']
        if args.planning_seed is None:
            solver.torch_gen.set_state(torch.from_numpy(data['planner_rng_state'].copy()))
        solver.context_provider = lambda: (
            {'recipe': copy.deepcopy(snap['recipe'])},
            {k: data[k].copy() for k in ('prefix_actions', 'physics', 'rendered_pixels', 'observation_pixels')})
        solver.solve(model_inputs(data, cfg['device']), torch.from_numpy(data['init_action'].copy()))
        for row in solver.iteration_rows:
            append_jsonl(output / 'cem_iterations.jsonl', row)
        for row in solver.plan_rows:
            append_jsonl(output / 'plans.jsonl', row)
        finish_run(output, 'complete', replanned_snapshots=1)
    except BaseException as exc:
        finish_run(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def success_interval(case_means, single_seed):
    """Case is the independent unit. Never count repeated planner seeds as cases."""
    n = len(case_means)
    if not n:
        raise ValueError('No independent cases')
    p = sum(case_means) / n
    if single_seed:
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return p, max(0, center - radius), min(1, center + radius), 'Wilson 95%; one seed per case'
    # Hoeffding for independent bounded case means; conservative and valid at all-0/all-1.
    radius = math.sqrt(math.log(40) / (2 * n))
    return p, max(0, p - radius), min(1, p + radius), 'Hoeffding 95%; case clusters'


def comparable_metadata(meta):
    cfg = meta['config'].copy()
    for key in ('num_samples', 'iterations', 'planning_seed', 'policy', 'cache_dir',
                'diagnostics', 'save_snapshots', 'keep_best', 'keep_random', 'max_snapshot_plans'):
        cfg.pop(key, None)
    return {
        'config': cfg, 'cases': sorted(meta['cases'], key=lambda c: c['case_id']),
        'model': {k: meta['model'][k] for k in ('type', 'weights_sha256', 'config_sha256')},
        'processors': meta['processors_sha256'], 'sources': meta['sources'],
        'research_sources': meta['research_sources'], 'packages': meta['packages'],
        'dataset_sha256': meta['dataset_sha256'], 'protocol': meta['protocol'],
        'execution': meta['execution'], 'python': meta['python'], 'platform': meta['platform'],
    }


def budget_rows(roots):
    reference, groups, seen = None, {}, set()
    for root in map(Path, roots):
        meta = load_run(root)
        if meta['config']['diagnostics'] or meta.get('kind') == 'replan':
            raise ValueError('Budget plots require closed-loop runs with diagnostics disabled')
        identity = comparable_metadata(meta)
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError('Cannot pool runs with different cases/model/preprocessing/protocol/runtime')
        rows = read_rows(root / 'episodes.jsonl')
        if sorted(r['case_id'] for r in rows) != sorted(c['case_id'] for c in meta['cases']):
            raise ValueError('Missing/duplicate episode rows')
        for row in rows:
            cfg = meta['config']
            if (row['run_id'] != meta['run_id'] or row['diagnostics'] or
                    any(row[k] != cfg[k] for k in ('num_samples', 'iterations', 'planning_seed'))):
                raise ValueError('Episode identity disagrees with metadata')
            key = (row['num_samples'], row['iterations'], row['case_id'], row['planning_seed'])
            if key in seen:
                raise ValueError('Duplicate budget/case/seed observation')
            seen.add(key)
            groups.setdefault(key[:2], []).append(row)
    expected = None
    for rows in groups.values():
        pairs = {(r['case_id'], r['planning_seed']) for r in rows}
        if expected is None:
            expected = pairs
        elif pairs != expected:
            raise ValueError('Every budget must use exactly the same case/seed pairs')
        seeds = {}
        for row in rows:
            seeds.setdefault(row['case_id'], set()).add(row['planning_seed'])
        if len({tuple(sorted(s)) for s in seeds.values()}) != 1:
            raise ValueError('Every case must use the same planning seeds')
    if not groups:
        raise ValueError('No observations')
    return groups


def check_pair(args):
    """Compare real collected actions with diagnostics off/on, without inference."""
    import numpy as np

    roots = [Path(args.formal), Path(args.diagnostic)]
    metas = [load_run(root) for root in roots]
    if metas[0]['config']['diagnostics'] or not metas[1]['config']['diagnostics']:
        raise ValueError('Expected diagnostics-off formal run and diagnostics-on paired run')
    if any(meta.get('kind') == 'replan' for meta in metas):
        raise ValueError('Requires closed-loop runs')
    if comparable_metadata(metas[0]) != comparable_metadata(metas[1]):
        raise ValueError('Model/cases/protocol/runtime differ')
    for key in ('num_samples', 'iterations', 'planning_seed'):
        if metas[0]['config'][key] != metas[1]['config'][key]:
            raise ValueError(f'Paired configuration differs: {key}')
    for case in metas[0]['cases']:
        files = [root / 'cases' / case['case_id'] / 'executed_actions.npy' for root in roots]
        arrays = [np.load(path, allow_pickle=False) for path in files]
        if not np.array_equal(*arrays):
            raise RuntimeError(f"Diagnostics changed executed actions: {case['case_id']}")
        recipes = [read_json(root / 'cases' / case['case_id'] / 'recipe.json') for root in roots]
        if recipes[0] != recipes[1]:
            raise RuntimeError('Different environment initialization')
    print(f"PASS: {len(metas[0]['cases'])} paired cases have bitwise equal executed actions")


def plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = budget_rows(args.run)
    output = new_run(args.output)
    try:
        summaries = []
        for (n, iterations), rows in sorted(groups.items()):
            cases = {}
            for row in rows:
                cases.setdefault(row['case_id'], []).append(float(row['success']))
            means = [sum(values) / len(values) for values in cases.values()]
            p, lo, hi, method = success_interval(means, all(len(v) == 1 for v in cases.values()))
            summaries.append({
                'num_samples': n, 'iterations': iterations, 'success_rate': p,
                'ci_low': lo, 'ci_high': hi, 'ci_method': method, 'independent_cases': len(cases),
                'planning_seeds': len(rows) // len(cases),
                'mean_solver_seconds_per_episode': sum(r['solver_seconds'] for r in rows) / len(rows),
                'mean_candidate_prediction_steps': sum(r['candidate_predictions'] for r in rows) / len(rows),
            })
        write_json(output / 'budget_summary.json', summaries)
        labels = [f"({s['num_samples']}, {s['iterations']})" for s in summaries]
        rates = [s['success_rate'] for s in summaries]
        errors = [[s['success_rate'] - s['ci_low'] for s in summaries],
                  [s['ci_high'] - s['success_rate'] for s in summaries]]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].errorbar(labels, rates, yerr=errors, fmt='o')
        axes[0].set(xlabel='(N, I); case-level 95% interval', ylabel='Success rate', ylim=(0, 1))
        axes[1].plot(labels, [s['mean_solver_seconds_per_episode'] for s in summaries], 'o')
        axes[1].set(xlabel='(N, I)', ylabel='Solver seconds per episode (diagnostics off)')
        axes[2].plot([s['mean_candidate_prediction_steps'] for s in summaries], rates, 'o')
        axes[2].set(xlabel='Approximate candidate prediction steps', ylabel='Success rate', ylim=(0, 1))
        fig.tight_layout()
        fig.savefig(output / 'budget_success_time.png', dpi=160)
        plt.close(fig)
        finish_run(output, 'complete')
    except BaseException as exc:
        finish_run(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def diagnostic_plots(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    root = Path(args.run)
    meta = load_run(root)
    rows = read_rows(root / 'cem_iterations.jsonl')
    output = new_run(args.output)
    try:
        groups = {}
        for row in rows:
            if row['run_id'] != meta['run_id']:
                raise ValueError('Mixed run IDs')
            groups.setdefault((row['case_id'], row['plan_id']), []).append(row)
        if not groups:
            raise ValueError('No diagnostic iteration rows')
        for (case, plan), group in groups.items():
            group.sort(key=lambda r: r['iteration'])
            if [r['iteration'] for r in group] != list(range(meta['config']['iterations'])):
                raise ValueError('Incomplete or duplicate iteration rows')
            fig, ax = plt.subplots(figsize=(7, 4))
            for field in ('iteration_min_cost', 'elite_mean_cost', 'updated_mean_cost'):
                ax.plot([r['iteration'] for r in group], [r[field] for r in group], label=field)
            ax.scatter(group[-1]['iteration'], group[-1]['updated_mean_cost'], marker='*', s=120,
                       label='returned plan')
            ax.set(xlabel='CEM iteration', ylabel='Predicted terminal SSE', title=f'{case} / plan {plan}')
            ax.legend()
            fig.tight_layout()
            fig.savefig(output / f'cem_{case}_{plan:06d}.png', dpi=160)
            plt.close(fig)
        if args.audit:
            audit_root = Path(args.audit)
            require_complete(audit_root)
            audit_meta = read_json(audit_root / 'audit_metadata.json')
            if audit_meta['source_run_id'] != meta['run_id']:
                raise ValueError('Audit belongs to a different run')
            audited = read_rows(audit_root / 'audit_records.jsonl')
            audit_groups = {}
            for row in audited:
                audit_groups.setdefault((row['case_id'], row['plan_id']), []).append(row)
            for (case, plan), group in audit_groups.items():
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                for ax, (x, y) in zip(axes, [('J_pred', 'J_enc'), ('J_enc', 'J_task'), ('J_pred', 'J_task')]):
                    ax.scatter([r[x] for r in group], [r[y] for r in group], s=10)
                    chosen = next(r for r in group if r['selected'])
                    ax.scatter(chosen[x], chosen[y], marker='*', s=100)
                    ax.set(xlabel=x, ylabel=y)
                fig.suptitle(f'{case} / plan {plan}; selected subset, star = returned plan')
                fig.tight_layout()
                fig.savefig(output / f'audit_{case}_{plan:06d}.png', dpi=160)
                plt.close(fig)
        finish_run(output, 'complete')
    except BaseException as exc:
        finish_run(output, 'failed', error=f'{type(exc).__name__}: {exc}')
        raise


def parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--run', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--policy')
    a.add_argument('--cache-dir')
    a.add_argument('--device')
    a.add_argument('--state-atol', type=float, default=1e-7)
    a.add_argument('--pixel-atol', type=int, default=0)
    a.add_argument('--cost-atol', type=float, default=1e-4)
    a.add_argument('--cost-rtol', type=float, default=1e-4)
    g = sub.add_parser('plot')
    g.add_argument('--run', required=True, nargs='+')
    g.add_argument('--output', required=True)
    d = sub.add_parser('plot-diagnostics')
    d.add_argument('--run', required=True)
    d.add_argument('--audit')
    d.add_argument('--output', required=True)
    r = sub.add_parser('replan')
    r.add_argument('--run', required=True)
    r.add_argument('--snapshot', required=True, help='Path to one source snapshot.json')
    r.add_argument('--output', required=True)
    r.add_argument('--num-samples', required=True, type=int)
    r.add_argument('--iterations', required=True, type=int)
    r.add_argument('--topk', required=True, type=int)
    r.add_argument('--planning-seed', type=int)
    r.add_argument('--policy')
    r.add_argument('--cache-dir')
    r.add_argument('--device')
    c = sub.add_parser('check-pair')
    c.add_argument('--formal', required=True)
    c.add_argument('--diagnostic', required=True)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    {'audit': audit, 'plot': plots, 'plot-diagnostics': diagnostic_plots,
     'replan': replan, 'check-pair': check_pair}[args.command](args)
