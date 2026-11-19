"""Focused checks of the step-25 replay gate and oracle boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research.cohort import profile
from cube_research.perturbation_recovery import (
    ORACLE, compare_history, plan_branch, require_decision, snapshot_digest,
)
from cube_research.protocol import History, Ledger, Observation, policy_config, read_config


def trace(steps):
    return History([Observation(np.array([i]), np.empty(0)) for i in range(steps + 1)],
                   [np.full(5, i, dtype=np.float32) for i in range(steps)])


def test_exact_prefix_and_full_trace_check_differentiate_lengths_and_channels():
    goal = np.array([3])
    old = dict(trace(50).arrays(), goal=goal)
    truth = [{'distance': 100-i} for i in range(51)]
    compare_history(trace(25), truth[:26], goal, old, truth, 25)
    compare_history(trace(50), truth, goal, old, truth, 50)
    bad = trace(50)
    bad.actions[-1][0] = -1
    with pytest.raises(ValueError, match='commands mismatch.*49'):
        compare_history(bad, truth, goal, old, truth, 50)
    with pytest.raises(ValueError, match='Expected 25'):
        compare_history(trace(26), truth[:27], goal, old, truth, 25)
    with pytest.raises(ValueError, match='evaluator trajectory'):
        compare_history(trace(25), truth[:25] + [{'distance': 0}], goal, old, truth, 25)


def test_evaluator_contact_names_match_their_json_representation_without_tolerance():
    goal = np.array([3])
    old = dict(trace(1).arrays(), goal=goal)
    live = [{}, {'contacts': [{'geoms': ('pad', 'object'), 'force': [1.]}]}]
    stored = [{}, {'contacts': [{'geoms': ['pad', 'object'], 'force': [1.]}]}]
    compare_history(trace(1), live, goal, old, stored, 1)
    stored[1]['contacts'][0]['force'][0] += 1e-12
    with pytest.raises(ValueError, match='evaluator trajectory'):
        compare_history(trace(1), live, goal, old, stored, 1)


def test_sealed_decision_requires_unfinished_exact_boundary_and_empty_buffer():
    ledger = Ledger(50, phases={'prefix': 25}, model={'plan_calls': 1})
    planner = SimpleNamespace(buffer=[])
    truth = [dict(success=False, truncated=False) for _ in range(26)]
    require_decision(ledger, planner, truth)
    planner.buffer.append(np.zeros(5))
    with pytest.raises(ValueError, match='empty action buffer'):
        require_decision(ledger, planner, truth)
    planner.buffer.clear()
    truth[-1]['success'] = True
    with pytest.raises(ValueError, match='unfinished'):
        require_decision(ledger, planner, truth)


def test_oracle_is_the_only_branch_given_a_mapping():
    cfg, _ = profile(read_config(Path(__file__).resolve().parents[1] / 'config/research/cube.yaml'))
    cfg = policy_config(dict(cfg, interface_parameterization='xyz', xyz_gain=[.4]*3))
    assert 'xyz_gain' not in cfg
    ledger, model = Ledger(50), SimpleNamespace(dim=5)
    for method in ('Frozen', 'R-I', 'R-D', 'R-U'):
        with pytest.raises(ValueError, match='Only the evaluator-only'):
            plan_branch(model, cfg, ledger, method, trace(25), known_mapping=[.4]*3)
    with pytest.raises(ValueError, match='Only the evaluator-only'):
        plan_branch(model, cfg, ledger, ORACLE, trace(25))
    frozen, _ = plan_branch(model, cfg, ledger, 'Frozen', trace(25))
    oracle, _ = plan_branch(model, cfg, ledger, ORACLE, trace(25), known_mapping=[.4]*3)
    assert frozen.gain == [1]*3 and oracle.gain == [.4]*3
    assert not oracle.weights.any() and ledger.updates == 0 and not ledger.model


def test_snapshot_digest_checks_rng_and_buffer():
    rng = np.random.default_rng(142)
    snap = {'rng': rng.bit_generator.state, 'buffer': []}
    assert snapshot_digest(snap) == snapshot_digest(dict(snap))
    rng.random()
    assert snapshot_digest(snap) != snapshot_digest(dict(snap, rng=rng.bit_generator.state))
    assert snapshot_digest(snap) != snapshot_digest(dict(snap, buffer=[np.zeros(5)]))
