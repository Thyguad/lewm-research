"""Cohort admission, pairing, and bounded planning contracts; no real rollouts."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research import cohort
from cube_research.cohort_pair import contact_friction, e1_gate
from cube_research.model import FrozenModel, Planner
from cube_research.protocol import Ledger, digest, grouped_cases, policy_config, read_config
from research_common import file_hash, read_json, write_json


def config():
    return cohort.profile(read_config(cohort.PROFILE.with_name('cube.yaml')))


def result(**kwargs):
    return dict(dict(status='complete', initial_success=False, success=True,
                     max_motion_m=.03, contact_steps=[1]), **kwargs)


def test_qualification_records_distinct_reasons():
    _, rules = config()
    n = result(success=False)
    assert cohort.decision(result(initial_success=True), [], rules) == 'initial_success'
    assert cohort.decision(result(), [], rules) == 'zero_command_success'
    assert cohort.decision(result(status='failed'), [], rules) == 'zero_unfinished'
    assert cohort.decision(n, [result()], rules) == 'frozen_unfinished'
    assert cohort.decision(n, [result(), n], rules) == 'frozen_not_2_of_2'
    assert 'insufficient_object_motion' in cohort.decision(n, [result(), result(max_motion_m=.005)], rules)
    assert 'no_pad_contact' in cohort.decision(n, [result(), result(contact_steps=[0])], rules)
    assert cohort.decision(n, [result(), result()], rules) == 'eligible'


def test_dev_order_and_seal_enforced(tmp_path):
    cfg, rules = config()
    assert set(rules['seeds']).isdisjoint(cfg['planning_seeds'])
    for name in ('weights.pt', 'data.h5', 'config.json'):
        (tmp_path / name).write_text('fixture')
    cfg.update(weights=str(tmp_path / 'weights.pt'), dataset=str(tmp_path / 'data.h5'))
    cases = grouped_cases(range(90), [100] * 90, 25, 21)
    source = dict(cases=cases, dataset_sha256=file_hash(cfg['dataset']))
    path = tmp_path / 'cases.json'
    write_json(path, source)
    original = file_hash(path)
    selected = cohort.candidates(source, rules, 0)
    assert len(selected) == 6 and {c['split'] for c in selected} == {'dev'}
    assert not set(c['case_id'] for c in selected) & {c['case_id'] for c in cohort.candidates(source, rules, 1)}
    frozen = [dict(result(), seed=s) for s in rules['seeds']]
    screening = [dict(case=c, status='complete', zero=result(success=False), frozen=frozen) for c in selected]
    payload = dict(schema='cube-dev-cohort-v1', batch=0, contract=cohort.contract(cfg, path, rules),
                   cases=selected[:3], screening=screening, dataset_stat=cohort.dataset_stat(cfg['dataset']),
                   dataset_sha256=source['dataset_sha256'])
    seal = tmp_path / 'cohort.json'
    write_json(seal, dict(payload, seal_sha256=digest(payload)))
    assert cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', None, False, None) == selected[:1]
    for position in (1, 2, 3):
        assert cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', None, False, None,
                                  position=position) == [selected[position - 1]]
    with pytest.raises(ValueError, match='outside the sealed admissions'):
        cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', None, False, None, position=4)
    with pytest.raises(ValueError, match='positive integer'):
        cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', None, False, None, position=0)
    with pytest.raises(ValueError, match='only for E0'):
        cohort.load_cohort(cfg, path, source, seal, 'e1', 'dev', None, False, None, position=2)
    assert cohort.load_cohort(cfg, path, source, seal, 'collect', 'dev', None, False, None) == selected[:3]
    assert file_hash(path) == original and read_json(path) == source
    with pytest.raises(ValueError, match='no case override'):
        cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', selected[1]['case_id'], False, None)
    with pytest.raises(ValueError, match='dev-only'):
        cohort.load_cohort(cfg, path, source, seal, 'collect', 'test', None, False, None)
    # Even re-sealing a different chosen subset does not change the admitted order.
    payload['cases'] = selected[1:4]
    write_json(seal, dict(payload, seal_sha256=digest(payload)), overwrite=True)
    with pytest.raises(ValueError, match='admitted cases'):
        cohort.load_cohort(cfg, path, source, seal, 'e0', 'dev', None, False, None)


def test_e1_gate_keeps_failed_independent_n_and_requires_loaded_contact():
    rows = {c: result(environment_steps=1) for c in ('N', 'I', 'D')}
    assert e1_gate(rows, True, False, 25)[0] == []
    rows['I'].update(success=False, environment_steps=50)
    assert e1_gate(rows, True, False, 25)[0] == ['I']
    rows['N']['success'] = False
    assert e1_gate(rows, True, False, 25)[1] == 'independent_N_not_successful_after_action'
    pair = dict(geom1='pad', geom2='object', friction=[.3, .3, .01, .02, .02])
    c = dict(geoms=['object', 'pad'], friction=pair['friction'], force=[0] * 6)
    traj = [{}, dict(contacts=[c])]
    assert not contact_friction(traj, dict(pairs=[pair]))['verified']
    c['force'][0] = .1
    assert contact_friction(traj, dict(pairs=[pair]))['verified']


def test_bounded_planner_and_frozen_bridge_use_distinct_temporal_commands():
    cfg, _ = config()
    cfg.update(cem_samples=4, cem_elites=2, cem_iterations=1)
    stats = dict(mean=[.1, -.2, .3, 0, .4], scale=[.4, .3, .2, .5, .6])
    ledger = Ledger(50)
    adapter = SimpleNamespace(model=SimpleNamespace(stats=stats, device='cpu'), ledger=ledger,
                              predict=lambda z, _: z[:, -1])
    planner = Planner(adapter, policy_config(cfg), 142)
    z = np.zeros(3, dtype='float32')
    actions = np.stack([planner.action(z, z) for _ in range(25)])
    expected_input = np.random.default_rng(142).normal(size=(4, 5, 5, 5)).astype('float32')
    expected_input[0] = 0
    expected = np.clip(expected_input[:2].mean(0) * stats['scale'] + stats['mean'], -1, 1)
    np.testing.assert_allclose(actions, expected.reshape(25, 5), atol=3e-8)
    assert actions.shape == (25, 5) and ledger.model['plan_calls'] == 1
    planner.action(z, z)
    assert ledger.model['plan_calls'] == 2

    bridge = object.__new__(FrozenModel)
    captured = []
    bridge.device, bridge.cfg, bridge.stats = 'cpu', cfg, stats
    bridge.base = SimpleNamespace(action_encoder=lambda t: captured.append(t) or t,
                                  predict=lambda z, a: z)
    grouped = actions.reshape(1, 5, 5, 5)
    bridge.predict(np.zeros((1, 5, 3)), grouped, ledger)
    assert captured[0].dtype == torch.float32 and captured[0].shape == (1, 5, 25)
    np.testing.assert_allclose(captured[0][0, 1], ((actions[5:10] - stats['mean']) / stats['scale']).ravel(), atol=1e-7)
