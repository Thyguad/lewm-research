"""One preregistered E1 comparison from a newly verified step-25 snapshot."""

import copy
import hashlib
import inspect
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from research_common import derived_seed, file_hash, finish_run, new_run, read_json, require_complete, write_json
from .adaptation import Adapter
from .analysis import grid_rows
from .cohort import episode_metrics, load_cohort
from .cohort_pair import assert_commands
from .data import Dataset, local_inputs
from .experiments import execute, provenance, save_history, start_frozen
from .model import FrozenModel, Planner
from .perturbations import CASE_IDS, PROFILE, VERSION, assert_sealed, case_inputs, read_profile, seal_plan
from .protocol import Ledger, METHODS, digest, policy_config, validate_manifest
from .simulator import make_env


CASE, ARM, SEED = CASE_IDS[0], 'I_uniform_0.4', 142
PREFIX, BUDGET = 25, 50
ORACLE = 'Oracle-XYZ'


def compare_history(history, truth, goal, old, old_truth, stop):
    """Exact match, including issued versus applied commands and evaluator truth."""
    if len(truth) != stop + 1 or len(history.actions) != stop:
        raise ValueError(f'Expected {stop} executed steps, got {len(history.actions)}')
    for key, value in history.arrays().items():
        needed = stop if key == 'commands' else stop + 1
        if not np.array_equal(value, old[key][:needed]):
            mismatch = next((i for i, (a, b) in enumerate(zip(value, old[key][:needed]))
                             if not np.array_equal(a, b)), None)
            raise ValueError(f'Historical {key} mismatch (first recorded index {mismatch})')
    # Canonical JSON equates live geom-name tuples with stored JSON arrays.
    if not np.array_equal(goal, old['goal']) or digest(truth) != digest(old_truth[:stop + 1]):
        raise ValueError('Historical goal or evaluator trajectory mismatch')


def require_decision(ledger, planner, truth):
    if (ledger.limit != BUDGET or ledger.used != PREFIX or ledger.model.get('plan_calls') != 1
            or len(truth) != PREFIX + 1 or truth[-1]['success'] or truth[-1]['truncated']
            or planner.buffer):
        raise ValueError('Expected an unfinished step-25 decision on an empty action buffer after one plan')


def snapshot_digest(snapshot):
    return hashlib.sha256(pickle.dumps(snapshot, protocol=5)).hexdigest()


def load_parent(base, manifest, cohort, root):
    require_complete(root)
    parent = read_json(root / 'sealed-plan.json')
    assert_sealed(parent, parent)
    cfg = dict(base, experiment_protocol=VERSION, interface_parameterization='xyz')
    case, = [i['case'] for i in parent['inputs'] if i['case']['case_id'] == CASE]
    variant, = [v for v in read_profile()['variants'] if v['perturbation_id'] == ARM]
    aim = dict(variant, case_id=CASE, planning_seed=SEED, cohort_position=2)
    if (parent['experiment_protocol'] != VERSION or parent['config'] != cfg
            or parent['config_sha256'] != digest(cfg) or parent['profile_sha256'] != file_hash(PROFILE)
            or parent['cohort_sha256'] != file_hash(cohort) or parent['success_distance_m'] != .04
            or parent['source_cohort_seal'] != read_json(cohort)['seal_sha256']
            or aim not in parent['entries']):
        raise ValueError('E1 inputs do not match the sealed perturbation-grid plan')
    rows = grid_rows(root, parent)
    target, = [r for r in rows if (r['case_id'], r['planning_seed'], r['perturbation_id']) == (CASE, SEED, ARM)]
    if (rows != read_json(root / 'results.json') or not target['verified_task_loss']
            or not target['unfinished_at_decision'] or target['online_steps'] != BUDGET):
        raise ValueError('E0 receipt does not establish the fixed, audited loss and decision')
    pos = root / 'position-2'
    require_complete(pos)
    receipt, = [p for p in (pos / 'recoveries').glob('*/result.json')
                if read_json(p)['perturbation_id'] == ARM]
    if (read_json(receipt)['method'] != 'Frozen'
            or read_json(receipt)['termination'] != 'budget'):
        raise ValueError('Expected the completed Frozen E0 budget failure receipt')
    return parent, cfg, case, variant, receipt.parent


def runtime_identity(cfg, manifest, parent):
    from research_device import backend_metadata, resolve_device
    import stable_worldmodel
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    runtime = provenance(cfg, manifest)
    device = resolve_device(cfg['device'])
    imported = {}
    for name, obj in [('CubeEnv', CubeEnv), ('stable_worldmodel', stable_worldmodel), ('make_env', make_env)]:
        p = Path(inspect.getfile(obj)).resolve()
        imported[name] = dict(path=str(p), sha256=file_hash(p))
    runtime.update(python_executable=sys.executable, resolved_device=str(device),
                   device=backend_metadata(device), imported=imported)
    original = parent['runtime']
    keys = ('python_executable', 'python', 'platform', 'packages', 'resolved_device', 'device', 'imported',
            'weights_sha256', 'model_config_sha256', 'manifest_sha256')
    for k in keys:
        if runtime[k] != original[k]:
            raise ValueError(f'E0/E1 runtime mismatch: {k}')
    for name, sha in original['source_sha256'].items():
        # The entrypoint gains a subcommand; it does not change the E0 model or simulator.
        if name != 'le-wm/research_cube.py' and not name.startswith('le-wm/tests/'):
            if runtime['source_sha256'].get(name) != sha:
                raise ValueError(f'E0 executable source changed: {name}')
    return runtime


def plan_branch(model, public_cfg, ledger, method, history, known_mapping=None):
    if (method == ORACLE) != (known_mapping is not None):
        raise ValueError('Only the evaluator-only Oracle branch can receive the true mapping')
    adapter = Adapter(model, 'Frozen' if method == ORACLE else method, public_cfg, ledger)
    if method == ORACLE:
        adapter.gain = list(known_mapping)
    elif method != 'Frozen':
        embeddings = model.encode([o.pixels for o in history.observations], ledger)
        adapter.fit(embeddings, np.asarray(history.actions))
    return adapter, Planner(adapter, public_cfg, SEED)


def run_recovery(base, manifest, cohort, paired_e0, out):
    root, started, harness = new_run(out), perf_counter(), None
    paired_e0 = Path(paired_e0)
    prefix_steps, rows = 0, {}
    problem = None
    try:
        local_inputs(base)
        source = read_json(manifest)
        validate_manifest(source)
        parent, cfg, case, variant, receipt = load_parent(base, manifest, cohort, paired_e0)
        if (source['dataset_sha256'] != parent['dataset_sha256']
                or load_cohort(base, manifest, source, cohort, 'e0', 'dev', None, False, None, position=2) != [case]):
            raise ValueError('Dataset or cohort differs from E0')
        runtime = runtime_identity(cfg, manifest, parent)
        data = Dataset(cfg['dataset'])
        try:
            initial, target, goal = data.case(case)
            if case_inputs(data, [case]) != [i for i in parent['inputs'] if i['case'] == case]:
                raise ValueError('Exact input arrays differ from E0')
        finally:
            data.close()
        plan = seal_plan(dict(protocol='cube-perturb-recovery-step25-v1', experiment_protocol=VERSION,
                             e0_seal=parent['seal_sha256'], e0_results_sha256=file_hash(paired_e0 / 'results.json'),
                             e0_receipt={p.name: file_hash(p) for p in (receipt / name for name in
                                         ('allowed.npz', 'evaluation.json', 'result.json'))},
                             config=cfg, config_sha256=digest(cfg), case=case, variant=variant, seed=SEED,
                             shared_prefix_steps=PREFIX, branch_budget=BUDGET-PREFIX, fair_methods=list(METHODS),
                             diagnostic_only=ORACLE, runtime=runtime))
        write_json(root / 'sealed-plan.json', plan)
        with np.load(receipt / 'allowed.npz', allow_pickle=False) as f:
            old = {k: f[k] for k in ('pixels', 'sensors', 'commands', 'goal')}
        old_truth = read_json(receipt / 'evaluation.json')
        if len(old_truth) != BUDGET + 1 or len(old['commands']) != BUDGET:
            raise ValueError('Historical trace does not contain the full 50-step episode')
        model = FrozenModel(cfg, source['stats'])
        if model.identity != read_json(paired_e0 / 'position-2/model.json'):
            raise ValueError('Loaded LeWM identity differs from E0')
        env_cfg = dict(cfg, interface_scale=variant['xyz_gain'], friction_scale=variant['pad_friction_scale'])
        harness = make_env(env_cfg, variant['condition'], derived_seed(cfg['environment_seed'], CASE), initial, target)
        ledger = Ledger(BUDGET)
        history, truth, goal_z, _, planner = start_frozen(cfg, model, SEED, ledger, harness, goal)
        execute(harness, history, truth, planner, model, ledger, PREFIX, 'shared_prefix', goal_z)
        prefix_steps = ledger.used
        prefix_path = root / 'prefix'
        prefix_path.mkdir()
        save_history(prefix_path, history, truth, goal)
        require_decision(ledger, planner, truth)
        compare_history(history, truth, goal, old, old_truth, PREFIX)
        snap = harness.snapshot(planner)
        checksum = snapshot_digest(snap)
        fingerprint = history.fingerprint()
        with (prefix_path / 'evaluator_step25.pkl').open('xb') as handle:
            pickle.dump(snap, handle, protocol=5)
        write_json(prefix_path / 'verification.json', dict(
            matches_e0=True, state_step=harness.steps, snapshot_sha256=checksum, history_sha256=fingerprint,
            planner_rng_sha256=digest(snap['planner']['rng']), action_buffer_length=len(snap['planner']['buffer']),
            decision_distance_m=truth[-1]['distance'], ledger=asdict(ledger),
            comparison='observation/goal/issued commands/evaluator truth exact; E0 has no step-25 snapshot'))
        public_cfg = policy_config(cfg)
        for method in (*METHODS, ORACLE):
            assert_sealed(read_json(root / 'sealed-plan.json'), plan)
            privileged = method == ORACLE
            folder = root / ('offline-diagnostic' if privileged else 'fair') / method
            folder.mkdir(parents=True)
            h, t = copy.deepcopy(history), copy.deepcopy(truth)
            branch = Ledger(BUDGET, phases={'shared_prefix': PREFIX})
            start = perf_counter()
            row = dict(method=method, privileged=privileged, status='execution_error', prefix_sha256=fingerprint)
            rows[method] = row
            try:
                if h.fingerprint() != fingerprint:
                    raise ValueError('Branch prefix mutated before fitting')
                adapter, branch_planner = plan_branch(
                    model, public_cfg, branch, method, h,
                    known_mapping=variant['xyz_gain'] if privileged else None)
                harness.restore(snap, branch_planner)
                if (snapshot_digest(harness.snapshot(branch_planner)) != checksum
                        or harness.steps != PREFIX or not np.array_equal(harness.observe().pixels, h.observations[-1].pixels)
                        or harness.evaluate() != {**t[-1], 'applied_action': None}):
                    raise ValueError('Simulation, controller, RNG, buffer or observation restore mismatch')
                row['restore_verified'] = True
                execute(harness, h, t, branch_planner, model, branch, BUDGET-PREFIX, 'recovery', goal_z)
                assert_commands(h.actions, t, variant['xyz_gain'])
                if method == 'Frozen':
                    compare_history(h, t, goal, old, old_truth, BUDGET)
                    row['matches_e0_suffix_exactly'] = True
                metrics = episode_metrics(t)
                report = adapter.record()
                if privileged:
                    report.update(method=ORACLE, initialization='known mapping supplied by evaluator; no fitting')
                write_json(folder / 'adaptation.json', report)
                row.update(metrics, status='complete' if metrics['success'] or branch.used == BUDGET else 'unfinished',
                           termination='success' if metrics['success'] else 'budget' if branch.used == BUDGET
                           else 'environment_truncated', fitted_gain=None if privileged else adapter.gain,
                           supplied_gain=adapter.gain if privileged else None,
                           objective_evaluations=report['objective_evaluations'], parameter_count=report['parameter_count'],
                           residual_norm=float(np.linalg.norm(adapter.weights)))
            except BaseException as exc:
                row['reason'] = f'{type(exc).__name__}: {exc}'
                raise
            finally:
                row.update(branch_steps=branch.used-PREFIX, total_steps=branch.used,
                           snapshot_sha256=checksum, ledger=asdict(branch), wall_seconds=perf_counter()-start)
                save_history(folder, h, t, goal)
                write_json(folder / 'result.json', row)
    except BaseException as exc:
        problem = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if harness is not None:
            actual = harness.actual_steps
            harness.close()
        else:
            actual = 0
        write_json(root / 'fair-results.json', [rows.get(m, dict(method=m, status='not_run')) for m in METHODS])
        write_json(root / 'offline-diagnostic-result.json', rows.get(ORACLE, dict(method=ORACLE, status='not_run')))
        complete = all(rows.get(m, {}).get('status') == 'complete' for m in (*METHODS, ORACLE))
        finish_run(root, 'failed' if problem else 'complete' if complete else 'incomplete', reason=problem,
                   shared_prefix_steps=prefix_steps, physical_environment_steps=actual,
                   fair_suffix_steps=sum(rows.get(m, {}).get('branch_steps', 0) for m in METHODS),
                   diagnostic_steps=rows.get(ORACLE, {}).get('branch_steps', 0),
                   wall_seconds=perf_counter()-started)
    return complete
