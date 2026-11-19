"""Sealed, bounded E0 extension of an existing development cohort."""

import hashlib
import inspect
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from research_common import file_hash, finish_run, new_run, read_json, write_json
from .analysis import grid_rows
from .cohort import load_cohort
from .cohort_pair import contact_friction, position_delta, replay_matched
from .data import Dataset, local_inputs
from .experiments import provenance, run
from .protocol import digest, validate_manifest


PROFILE = Path(__file__).resolve().parents[1] / 'config/research/cube-perturbations.yaml'
VERSION = 'cube-dev-perturbation-grid-v1'
CASE_IDS = ('f98411fee6571747', 'cba4e92adcb900f3')
LIMITS = dict(total_steps=50, history_steps=25, goal_offset=25, action_block=5, horizon=5,
              execute_blocks=5, cem_samples=300, cem_iterations=10, cem_elites=30,
              score_batch_size=300, planning_seeds=[142], sensors='rgb')
IDENTITY_D = dict(perturbation_id='D_identity', condition='D', xyz_gain=[1, 1, 1], pad_friction_scale=1)
REUSE = ('No cross-protocol reuse: scalar-era receipts do not pin the current runtime device identity '
         'or execution source as required by this grid. All fourteen entries are fresh.')


def read_profile():
    spec = yaml.safe_load(PROFILE.read_text())
    id_gain = [1, 1, 1]
    expected = [('N', 'N', 'identity', id_gain, 1)]
    for family, suffix in [('xyz_uniform', 'uniform'), ('xyz_z_only', 'z')]:
        for s in (.6, .4):
            expected.append((f'I_{suffix}_{s}', 'I', family, [s]*3 if suffix == 'uniform' else [1, 1, s], 1))
    expected.extend((f'D_pad_{s}', 'D', 'pad_sliding', id_gain, s) for s in (.3, .1))
    keys = ('perturbation_id', 'condition', 'perturbation_family', 'xyz_gain', 'pad_friction_scale')
    if (spec.get('experiment_protocol') != VERSION or tuple(spec['cohort_cases']) != CASE_IDS
            or spec['planning_seed'] != 142 or spec['interface_parameterization'] != 'xyz'
            or spec['variants'] != [dict(zip(keys, v)) for v in expected]):
        raise ValueError('Versioned grid changed; do not alter arms or intensity after sealing')
    return spec


def array_fingerprint(a):
    a = np.asarray(a)
    return [str(a.dtype), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]


def case_inputs(data, cases):
    rows = []
    for case in cases:
        initial, target, goal = data.case(case)
        rows.append(dict(case=case, initial={k: array_fingerprint(v) for k, v in initial.items()},
                         target={k: array_fingerprint(v) for k, v in target.items()}, goal=array_fingerprint(goal)))
    return rows


def seal_plan(plan):
    return dict(plan, seal_sha256=digest(plan))


def assert_sealed(saved, expected):
    if saved != expected or seal_plan({k: v for k, v in saved.items() if k != 'seal_sha256'}) != saved:
        raise ValueError('Grid plan or its input identity changed')


def replay(cfg, variant, case, initial, target, commands, path):
    return replay_matched(dict(cfg, interface_scale=variant['xyz_gain'], friction_scale=variant['pad_friction_scale']),
                          variant['condition'], case, initial, target, commands, path)


def same_floor(a, b):
    # All original geom friction/priority stays unchanged, and no new explicit
    # pair is permitted to involve the floor, including cube-floor.
    return (a['floor'] == b['floor'] and a['geoms'] == b['geoms']
            and [p for p in a['pairs'] if 'floor' in (p['geom1'], p['geom2'])]
            == [p for p in b['pairs'] if 'floor' in (p['geom1'], p['geom2'])])


def pair_values(baseline, model, scale):
    from .simulator import PAD_NAMES

    wanted = dict()
    for geom in baseline['geoms']:
        if geom['name'] in PAD_NAMES:
            slide, twist, roll = geom['friction']
            wanted[frozenset((geom['name'], 'object_0'))] = [slide*scale, slide*scale, twist, roll, roll]
    pairs = [p for p in model['pairs'] if 'object_0' in (p['geom1'], p['geom2'])
             and ({p['geom1'], p['geom2']} & set(PAD_NAMES))]
    return len(pairs) == len(wanted) == len(PAD_NAMES) and all(
        np.array_equal(p['friction'], wanted.get(frozenset((p['geom1'], p['geom2'])))) for p in pairs)


def compare(variant, reference, sample, identity):
    n, mn, _ = reference
    t, mod, steps = sample
    kind = variant['condition']
    if (any(t[0][k] != n[0][k] for k in ('object_pos', 'effector_pos', 'distance'))
            or any(row['success'] != (row['distance'] <= .04) for row in t)):
        raise ValueError('Paired reset or 0.04 m success criterion mismatch')
    if not same_floor(mn, mod):
        raise ValueError('Unexpected floor or original geom change')
    if kind == 'N':
        return dict(parameters_verified=True, physical_response_verified=None)
    if kind == 'I':
        gain = np.asarray(variant['xyz_gain'])
        if not np.array_equal(mod['interface_scale'], gain) or mn['pairs'] != mod['pairs']:
            raise ValueError('Interface parameters mismatch or unintended contact-pair change')
        moving = gain != 1
        da = (np.asarray([x['applied_action'] for x in t[1:]]) -
              np.asarray([x['applied_action'] for x in n[1:]])) if steps else np.zeros((1, 5))
        delta = np.max(np.abs(da), axis=0)
        if np.any(delta[np.r_[~moving, True, True]]):
            raise ValueError('An unperturbed action channel changed')
        eff = np.max(np.abs(np.asarray([x['effector_pos'] for x in t]) -
                            np.asarray([x['effector_pos'] for x in n])), axis=0)
        verified = bool(np.any(delta[:3][moving] > 1e-6) and np.max(eff[moving]) > 1e-6)
        return dict(parameters_verified=True, physical_response_verified=verified,
                    command_max_abs_by_channel=delta.tolist(), effector_max_abs_m_by_axis=eff.tolist(),
                    reason=None if verified else 'insufficient_changed_axis_excitation_or_response')
    ident, mi, _ = identity
    same = position_delta(n, ident, 'object_pos')
    if (not same_floor(mn, mi) or not pair_values(mn, mi, 1)
            or mod['friction_scale'] != variant['pad_friction_scale']
            or not pair_values(mn, mod, variant['pad_friction_scale']) or same >= 1e-7):
        raise ValueError('D isolation, explicit pair coefficients or D-identity mismatch')
    contacts, identity_contacts = contact_friction(t, mod), contact_friction(ident, mi)
    if any(c['contacts'] and not c['coefficients_match'] for c in (contacts, identity_contacts)):
        raise ValueError('Solver contact coefficients disagree with the declared pad-object pairs')
    response = position_delta(ident, t, 'object_pos')
    verified = bool(contacts['verified'] and identity_contacts['verified'] and response > 1e-6)
    return dict(parameters_verified=True, physical_response_verified=verified, floor_unchanged=True,
                d_identity_object_max_abs_m=same, d_response_object_max_abs_m=response,
                d_contact_friction=contacts, d_identity_contact_friction=identity_contacts,
                reason=None if verified else 'insufficient_loaded_contact_or_response')


def audit_case(cfg, variants, case, data, commands, root):
    initial, target, _ = data.case(case)
    root.mkdir()
    def one(v):
        return replay(cfg, v, case, initial, target, commands, root / (v['perturbation_id'] + '.json'))
    reference, identity = one(variants[0]), one(IDENTITY_D)
    report = {}
    for v in variants:
        key = v['perturbation_id']
        sample = reference if key == 'N' else one(v)
        audit = compare(v, reference, sample, identity)
        receipt = read_json(root / f'{key}.json')
        report[key] = dict(audit, offline_steps=sample[2], offline_seconds=receipt['wall_seconds'],
                           matched_commands=len(commands))
    write_json(root / 'audit.json', report)
    return report


def run_grid(base, manifest, cohort, out):
    from .simulator import make_env  # Preserve the same simulator entry as legacy E0.
    from research_device import backend_metadata, resolve_device
    import stable_worldmodel
    from stable_worldmodel.envs.ogbench.cube_env import CubeEnv

    root, started, data = new_run(out), perf_counter(), None
    plan, problem = None, None
    try:
        local_inputs(base)
        source = read_json(manifest)
        validate_manifest(source)
        spec = read_profile()
        if any(base[k] != v for k, v in LIMITS.items()):
            raise ValueError('Grid differs from the sealed cohort planning/budget settings')
        admitted = load_cohort(base, manifest, source, cohort, 'collect', 'dev', None, False, None)
        cases = admitted[1:]
        if tuple(c['case_id'] for c in cases) != CASE_IDS:
            raise ValueError('Grid requires the exact sealed second and third cases, in order')
        variants = spec['variants']
        cfg = dict(base, experiment_protocol=VERSION, interface_parameterization='xyz')
        data = Dataset(cfg['dataset'])
        runtime = provenance(cfg, manifest)
        runtime.pop('timestamp_utc', None)
        imported = {k: Path(inspect.getfile(o)).resolve() for k, o in
                    [('CubeEnv', CubeEnv), ('stable_worldmodel', stable_worldmodel), ('make_env', make_env)]}
        device = resolve_device(cfg['device'])
        runtime.update(python_executable=sys.executable, resolved_device=str(device), device=backend_metadata(device),
                       imported={k: dict(path=str(p), sha256=file_hash(p)) for k, p in imported.items()})
        payload = dict(experiment_protocol=VERSION, config=cfg, config_sha256=digest(cfg),
                       profile_sha256=file_hash(PROFILE), cohort_sha256=file_hash(cohort),
                       source_cohort_seal=read_json(cohort)['seal_sha256'], dataset_sha256=source['dataset_sha256'],
                       runtime=runtime, inputs=case_inputs(data, cases),
                       selection='Both previously sealed nontrivial admissions (positions 2 and 3); keep position 1 unchanged.',
                       reuse_policy=REUSE, reused=0, success_distance_m=.04,
                       online_episode_limit=14, online_step_limit=700, offline_step_limit=916,
                       entries=[dict(v, case_id=c['case_id'], planning_seed=142, cohort_position=i)
                                for i, c in enumerate(cases, 2) for v in variants])
        plan = seal_plan(payload)
        write_json(root / 'sealed-plan.json', plan)
        assert_sealed(read_json(root / 'sealed-plan.json'), plan)
        # Four one-step real replays from one fixed start, before any online rollout.
        audit_case(cfg, [variants[0], variants[4], variants[-1]], cases[0], data,
                   [[.1, -.1, .2, .05, -.05]], root / 'preflight')
        if not read_json(root / 'preflight/audit.json')[variants[4]['perturbation_id']]['physical_response_verified']:
            raise RuntimeError('Real z-only preflight did not show changed-axis response')
        for i, case in enumerate(cases, 2):
            assert_sealed(read_json(root / 'sealed-plan.json'), plan)
            sub = root / f'position-{i}'
            ok = run(cfg, manifest, sub, 'e0', limit=1, cohort=cohort, cohort_position=i,
                     perturbations=variants, extension_seal=plan['seal_sha256'])
            if not ok:
                raise RuntimeError(f'Incomplete execution for sealed position {i}')
            paths = list((sub / 'recoveries').glob('*/result.json'))
            baseline, = [p.parent for p in paths if read_json(p)['perturbation_id'] == 'N']
            with np.load(baseline / 'allowed.npz', allow_pickle=False) as log:
                commands = log['commands']
            audit_case(cfg, variants, case, data, commands, sub / 'matched-command-audit')
    except BaseException as exc:
        problem = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if data is not None:
            data.close()
        try:
            rows = grid_rows(root, plan) if plan else []
            write_json(root / 'results.json', rows)
        except BaseException as exc:
            problem = problem or f'{type(exc).__name__}: {exc}'
            raise
        finally:
            counts = [read_json(p) for p in root.glob('position-*/STATUS.json')]
            snapshots = sum(s.get('offline_replay_steps', 0) for s in counts)
            replay_files = list(root.glob('position-*/matched-command-audit/*.json')) + list(root.glob('preflight/*.json'))
            replays = [r for p in replay_files if p.name != 'audit.json' for r in [read_json(p)]]
            finish_run(root, 'failed' if problem else 'complete', reason=problem,
                       online_steps=sum(s.get('evaluator_actual_environment_steps', 0) for s in counts)-snapshots,
                       offline_snapshot_steps=snapshots,
                       offline_matched_and_preflight_steps=sum(sum(r['ledger']['phases'].values()) for r in replays),
                       offline_matched_and_preflight_seconds=sum(r['wall_seconds'] for r in replays),
                       wall_seconds=perf_counter()-started)
    return True
