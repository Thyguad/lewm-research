"""Freeze a dev-only, N-solvable cohort before perturbation outcomes are known."""

import inspect
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from research_common import derived_seed, file_hash, finish_run, new_run, read_json, write_json
from .data import Dataset, local_inputs
from .experiments import execute, protocol_identity, provenance, save_history, start_frozen
from .protocol import History, Ledger, digest, validate_config, validate_manifest


PROFILE = Path(__file__).resolve().parents[1] / 'config/research/cube-cohort.yaml'


def profile(base):
    p = yaml.safe_load(PROFILE.read_text())
    cfg = dict(base, **p['planning'])
    rules = p['screening']
    validate_config(cfg)
    seeds = rules['seeds']
    if (len(seeds) != 2 or len(set(seeds)) != 2 or set(seeds) & set(cfg['planning_seeds'])
            or any(type(s) is not int or s < 0 for s in seeds)
            or cfg['history_steps'] != cfg['action_block'] * cfg['execute_blocks']
            or not 1 <= rules['candidates_per_batch'] <= 6 or not 1 <= rules['max_admitted'] <= 3
            or not np.isfinite(rules['minimum_object_motion_m']) or rules['minimum_object_motion_m'] <= 0):
        raise ValueError('Invalid screening budgets, seed separation or decision boundary')
    return cfg, rules


def candidates(manifest, rules, batch):
    if batch < 0:
        raise ValueError('Batch index cannot be negative')
    dev = [c for c in manifest['cases'] if c['split'] == 'dev']
    ordered = np.random.default_rng(rules['order_seed']).permutation(len(dev))
    size = rules['candidates_per_batch']
    selected = [dev[i] for i in ordered[batch * size:(batch + 1) * size]]
    if not selected:
        raise ValueError('Requested batch has no development cases')
    return selected


def dataset_stat(path):
    st = Path(path).stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def verify_dataset(path, expected, prior_stat=None):
    current = dataset_stat(path)
    if current != prior_stat and (file_hash(path) != expected or dataset_stat(path) != current):
        raise ValueError('Dataset content changed after split/stats freezing')
    return current


def frozen_source_identity():
    root = Path(__file__).parent
    return digest({
        **{name: file_hash(root / name) for name in
           ('model.py', 'adaptation.py', 'simulator.py', 'protocol.py', 'data.py')},
        'start_frozen': inspect.getsource(start_frozen), 'execute': inspect.getsource(execute),
    })


def contract(cfg, manifest, rules):
    return dict(source_manifest_sha256=file_hash(manifest), protocol_sha256=protocol_identity(cfg),
                score_batch_size=cfg['score_batch_size'], rules=rules,
                weights_sha256=file_hash(cfg['weights']),
                model_config_sha256=file_hash(Path(cfg['weights']).parent / 'config.json'),
                frozen_source_sha256=frozen_source_identity())


def compatible_contract(sealed, current):
    if sealed == current:
        return True
    # Exact source versions with compatible legacy Frozen behavior. The newest
    # adds opt-in scoring/archive callbacks; default CEM commands and counts are unchanged.
    previous = '1cb418cfe21e49441478733a08ff9f59dc69998265f7391ef385738ba8dcd7fe'
    reviewed = '8a3c8edf9cefc3e0cf9e78891b0d337a4d69dd45ddf302dd518807725236f2aa'
    with_scoring_hook = 'e9f24675f07ab97f3a50087ccc1cb5a4eb0d7cef399196ffd4dc0be281213cd3'
    with_archive_hook = '920cfbe13abb51ad75318353b424e80c6c76bf6fe702f77ebccc84def181a558'
    allowed = {reviewed: (previous,), with_scoring_hook: (previous, reviewed),
               with_archive_hook: (previous, reviewed, with_scoring_hook)}
    return any(dict(current, frozen_source_sha256=old) == sealed
               for old in allowed.get(current['frozen_source_sha256'], ()))


def episode_metrics(truth):
    first = truth[0]
    return dict(initial_distance_m=first['distance'], initial_success=bool(first['success']),
                final_distance_m=truth[-1]['distance'], min_distance_m=min(t['distance'] for t in truth),
                success=any(t['success'] for t in truth),
                first_success_step=next((i for i, t in enumerate(truth) if t['success']), None),
                max_motion_m=max(float(np.linalg.norm(np.asarray(t['object_pos']) - first['object_pos']))
                                 for t in truth),
                contact_steps=[i for i, t in enumerate(truth) if t['contacts']], steps=len(truth) - 1)


def decision(zero, frozen, rules):
    if zero['status'] != 'complete':
        return 'zero_unfinished'
    if zero['initial_success']:
        return 'initial_success'
    if zero['success']:
        return 'zero_command_success'
    if len(frozen) != len(rules['seeds']) or any(r['status'] != 'complete' for r in frozen):
        return 'frozen_unfinished'
    reasons = []
    if not all(r['success'] and not r['initial_success'] for r in frozen):
        reasons.append('frozen_not_2_of_2')
    if not all(r['max_motion_m'] >= rules['minimum_object_motion_m'] for r in frozen):
        reasons.append('insufficient_object_motion')
    if rules['require_pad_contact'] and not all(any(i > 0 for i in r['contact_steps']) for r in frozen):
        reasons.append('no_pad_contact_in_one_or_more_rollouts')
    return '+'.join(reasons) or 'eligible'


def rollout(cfg, model, case, initial, target, goal, seed, out):
    from .simulator import make_env

    started = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    ledger, env, history, truth = Ledger(cfg['total_steps']), None, None, None
    record = dict(seed=seed, status='unfinished')
    write_json(out / 'result.json', record)
    try:
        env = make_env(cfg, 'N', derived_seed(cfg['environment_seed'], case['case_id']), initial, target)
        if seed is None:
            history, truth = History([env.observe()]), [env.evaluate()]
            for _ in range(ledger.limit):
                if truth[-1]['truncated']:
                    break
                command = np.zeros(5, dtype=np.float32)
                obs, row = env.step(command, ledger, 'zero_command')
                history.append(command, obs)
                truth.append(row)
        else:
            history, truth, goal_z, _, planner = start_frozen(cfg, model, seed, ledger, env, goal)
            execute(env, history, truth, planner, model, ledger, ledger.limit, 'screen_frozen_n', goal_z)
        complete = ledger.used == ledger.limit or (seed is not None and truth[-1]['success'])
        record.update(status='complete' if complete else 'unfinished',
                      reason=None if complete else 'environment truncated before budget')
    except Exception as exc:
        record.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
    finally:
        if truth:
            record.update(episode_metrics(truth))
            save_history(out, history, truth, goal)
        record.update(ledger=asdict(ledger), wall_seconds=perf_counter() - started)
        write_json(out / 'result.json', record, overwrite=True)
        if env is not None:
            env.close()
    return record


def screen(cfg, rules, manifest, out, batch=0):
    from .model import FrozenModel

    root = new_run(out)
    data = None
    try:
        local_inputs(cfg)
        m = read_json(manifest)
        validate_manifest(m)
        if (cfg['protocol'], cfg['dataset_source']) != (m['protocol'], m['dataset_source']):
            raise ValueError('Manifest and config protocol/data identity mismatch')
        cases = candidates(m, rules, batch)
        if any(c['goal_offset'] != cfg['goal_offset'] for c in cases):
            raise ValueError('Screening cases have a different goal offset')
        spec = contract(cfg, manifest, rules)
        write_json(root / 'screen-plan.json', dict(contract=spec, contract_sha256=digest(spec),
                   config=cfg, batch=batch, cases=cases, order='seeded permutation of dev in source order'))
        rows = [dict(case=c, status='unfinished', eligible=False, admitted=False,
                     reason='not_run', zero=None, frozen=[dict(seed=s, status='unfinished') for s in rules['seeds']])
                for c in cases]
        results_path = root / 'screen-results.json'
        write_json(results_path, rows)
        print(f'Locked batch {batch}: {[c["case_id"] for c in cases]}; verifying dataset SHA-256', flush=True)
        stat = verify_dataset(cfg['dataset'], m['dataset_sha256'])
        write_json(root / 'metadata.json', provenance(cfg, manifest))
        data, model = Dataset(cfg['dataset']), FrozenModel(cfg, m['stats'])
        write_json(root / 'model.json', model.identity)
        admitted = []
        for row in rows:
            c = row['case']
            dest = root / c['case_id']
            initial, target, goal = data.case(c)
            zero = rollout(cfg, model, c, initial, target, goal, None, dest / 'zero')
            row['zero'] = zero
            skip = zero['status'] != 'complete' or zero['initial_success'] or zero['success']
            for i, seed in enumerate(rules['seeds']):
                row['frozen'][i] = (dict(seed=seed, status='skipped', reason=decision(zero, [], rules)) if skip
                                   else rollout(cfg, model, c, initial, target, goal, seed, dest / f'frozen-{seed}'))
                write_json(results_path, rows, overwrite=True)
            reason = decision(zero, row['frozen'], rules)
            row.update(status='unfinished' if reason.endswith('_unfinished') else 'complete',
                       reason=reason, eligible=reason == 'eligible')
            if row['eligible'] and len(admitted) < rules['max_admitted']:
                row.update(admitted=True, reason='admitted')
                admitted.append(c)
            elif row['eligible']:
                row['reason'] = 'eligible_over_cohort_cap'
            write_json(results_path, rows, overwrite=True)
            print(c['case_id'], row['reason'], flush=True)
        if dataset_stat(cfg['dataset']) != stat:
            raise RuntimeError('Dataset changed during screening')
        totals = {kind: sum(r.get('ledger', {}).get('phases', {}).get(phase, 0) for r in receipts)
                  for kind, phase, receipts in (
                      ('zero_steps', 'zero_command', [r['zero'] for r in rows]),
                      ('frozen_steps', 'screen_frozen_n', [x for r in rows for x in r['frozen']]))}
        payload = dict(schema='cube-dev-cohort-v1', batch=batch, contract=spec, cases=admitted,
                       screening=rows, dataset_stat=stat, dataset_sha256=m['dataset_sha256'],
                       dataset_verification='full SHA-256 before screening; stat match after',
                       screening_device=str(model.device), **totals)
        write_json(root / 'cohort.json', dict(payload, seal_sha256=digest(payload)))
        complete = all(r['status'] == 'complete' for r in rows)
        finish_run(root, 'complete' if complete else 'incomplete', candidates=len(rows), admitted=len(admitted), **totals)
        return complete
    except BaseException as exc:
        finish_run(root, 'failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if data is not None:
            data.close()


def load_cohort(cfg, manifest, source, cohort, stage, split, case_id, smoke, paired_e0, *, position=None):
    if split != 'dev' or stage not in ('e0', 'e1', 'collect') or case_id is not None or smoke:
        raise ValueError('Cohort is dev-only; no case override, smoke or classifier evaluation')
    if position is not None and (stage != 'e0' or type(position) is not int or position < 1):
        raise ValueError('Cohort position must be a positive integer and is supported only for E0')
    _, rules = profile(cfg)
    m = read_json(cohort)
    seal = m.pop('seal_sha256')
    if (m['schema'] != 'cube-dev-cohort-v1' or digest(m) != seal
            or not compatible_contract(m['contract'], contract(cfg, manifest, rules))):
        raise ValueError('Cohort, source manifest or the frozen research protocol/model changed')
    ordered = candidates(source, rules, m['batch'])
    if [r['case'] for r in m['screening']] != ordered or any(r['status'] != 'complete' for r in m['screening']):
        raise ValueError('Screening batch is incomplete or does not match the locked dev order')
    wanted = [r['case'] for r in m['screening']
              if [x['seed'] for x in r['frozen']] == rules['seeds'] and decision(r['zero'], r['frozen'], rules) == 'eligible']
    if not m['cases'] or m['cases'] != wanted[:rules['max_admitted']]:
        raise ValueError('Cohort is empty or its admitted cases do not match the seal')
    if position is not None and position > len(m['cases']):
        raise ValueError('Cohort position is outside the sealed admissions')
    if source['dataset_sha256'] != m['dataset_sha256']:
        raise ValueError('Source dataset does not match the cohort')
    verify_dataset(cfg['dataset'], source['dataset_sha256'], m['dataset_stat'])
    if stage == 'e1':
        from .cohort_pair import require_pair

        require_pair(paired_e0, cohort, m['cases'][0])
    if position is not None:
        return [m['cases'][position - 1]]
    return m['cases'][:1] if stage in ('e0', 'e1') else m['cases']
