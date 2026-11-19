"""Evaluator-only, matched-command audit for one cohort E0 triplet."""

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from research_common import derived_seed, file_hash, read_json, write_json
from .protocol import CONDITIONS, Ledger, transform_command


def contact_friction(traj, model):
    expected = {tuple(sorted((p['geom1'], p['geom2']))): p['friction'] for p in model['pairs']}
    contacts = [c for t in traj[1:] for c in t['contacts']]
    loaded = sum(float(c['force'][0]) > 1e-6 for c in contacts)
    matches = all(tuple(sorted(c['geoms'])) in expected
                  and np.allclose(c['friction'], expected[tuple(sorted(c['geoms']))]) for c in contacts)
    return dict(contacts=len(contacts), loaded_contacts=loaded,
                coefficients_match=matches, verified=bool(loaded and matches))


def position_delta(a, b, key):
    return float(np.max(np.abs(np.array([r[key] for r in a]) - np.array([r[key] for r in b]))))


def assert_commands(commands, trajectory, gain):
    if any(not np.array_equal(t['applied_action'], transform_command(a, gain))
           for t, a in zip(trajectory[1:], commands, strict=True)):
        raise ValueError('Actual action disagrees with the locked interface')


def replay_matched(cfg, condition, case, initial, target, commands, path):
    from .simulator import make_env

    started = perf_counter()
    ledger, env, traj, model = Ledger(len(commands)), None, [], None
    status = 'unfinished'
    try:
        env = make_env(cfg, condition, derived_seed(cfg['environment_seed'], case['case_id']), initial, target)
        model = env.audit_model()
        traj.append(env.evaluate())
        for a in commands:
            if traj[-1]['truncated']:
                raise RuntimeError('Matched-command audit truncated before all commands')
            _, t = env.step(a, ledger, 'offline_matched_command')
            traj.append(t)
        assert_commands(commands, traj, cfg['interface_scale'] if condition == 'I' else 1)
        status = 'complete'
    finally:
        if env is not None:
            env.close()
        write_json(path, dict(status=status, model=model, trajectory=traj, ledger=asdict(ledger),
                              wall_seconds=perf_counter()-started, role='offline_audit_only'))
    return traj, model, ledger.used


def e1_gate(rows, interface, dynamics, history_steps):
    if not rows['N']['success'] or rows['N']['initial_success']:
        return [], 'independent_N_not_successful_after_action'
    losses = [c for c, valid in (('I', interface), ('D', dynamics))
              if valid and not rows[c]['success'] and rows[c]['environment_steps'] >= history_steps]
    return losses, 'verified_loss_and_decision' if losses else 'no_verified_perturbation_loss_with_decision'


def verify_pair(cfg, manifest, case, root):
    from .cohort import episode_metrics
    from .data import Dataset

    cid = case['case_id']
    rows, online, commands = {}, {}, {}
    for p in (root / 'recoveries').glob('*/result.json'):
        r = read_json(p)
        if r['case_id'] != cid or r['status'] != 'complete':
            raise ValueError('Invalid E0 audit source')
        c = r['condition']
        rows[c] = r
        online[c] = read_json(p.parent / 'evaluation.json')
        with np.load(p.parent / 'allowed.npz', allow_pickle=False) as log:
            commands[c] = log['commands']
        assert_commands(commands[c], online[c], cfg['interface_scale'] if c == 'I' else 1)
    if set(rows) != set(CONDITIONS) or len({r['planning_seed'] for r in rows.values()}) != 1:
        raise ValueError('Expected one paired case and planning seed across N/I/D')
    counts = 0
    matched = commands['N']
    report = dict(case=case, planning_seed=rows['N']['planning_seed'], matched_commands=len(matched),
                  online={c: episode_metrics(t) for c, t in online.items()})
    interface = dynamics = False
    if len(matched):
        data = Dataset(cfg['dataset'])
        try:
            initial, target, _ = data.case(case)
        finally:
            data.close()
        dest = root / 'matched-command-audit'
        dest.mkdir(exist_ok=False)
        trails, models = {}, {}
        for name, condition, friction in [('N', 'N', 1.0), ('I', 'I', 1.0),
                                          ('D_identity', 'D', 1.0), ('D', 'D', cfg['friction_scale'])]:
            trails[name], models[name], used = replay_matched(
                dict(cfg, friction_scale=friction), condition, case, initial, target, matched, dest / f'{name}.json')
            counts += used
        interface_command = position_delta(trails['N'][1:], trails['I'][1:], 'applied_action')
        interface_response = position_delta(trails['N'], trails['I'], 'effector_pos')
        identity = position_delta(trails['N'], trails['D_identity'], 'object_pos')
        dynamics_response = position_delta(trails['D_identity'], trails['D'], 'object_pos')
        d0, d1 = [contact_friction(trails[c], models[c]) for c in ('D_identity', 'D')]
        scale = cfg['interface_scale']
        interface_matches = all(np.array_equal(t['applied_action'], transform_command(a, scale))
                                for t, a in zip(trails['I'][1:], matched, strict=True))
        interface = bool(interface_matches and interface_command > 1e-6 and interface_response > 1e-6)
        dynamics = bool(identity < 1e-7 and d0['verified'] and d1['verified'] and dynamics_response > 1e-6)
        report.update(interface_verified=interface, interface_command_max_abs=interface_command,
                      interface_effector_max_abs_m=interface_response, dynamics_verified=dynamics,
                      d_identity_object_max_abs_m=identity, d_response_object_max_abs_m=dynamics_response,
                      d_identity_contact_friction=d0, d_contact_friction=d1)
    losses, reason = e1_gate(rows, interface, dynamics, cfg['history_steps'])
    write_json(root / 'pair-audit.json', dict(report, offline_steps=counts, e1_allowed=bool(losses),
               verified_loss_conditions=losses, gate_reason=reason))
    return counts


def require_pair(path, cohort, case):
    if not path:
        raise ValueError('Cohort E1 requires --paired-e0 and a verified perturbation loss')
    path = Path(path)
    meta, audit = [read_json(path / n) for n in ('metadata.json', 'pair-audit.json')]
    if (meta.get('stage') != 'e0' or meta.get('cohort_sha256') != file_hash(cohort)
            or read_json(path / 'STATUS.json')['state'] != 'complete'
            or audit['case'] != case or not audit['e1_allowed']):
        raise ValueError('E0 incomplete, mismatched, or not eligible for E1; do not replace the case')
