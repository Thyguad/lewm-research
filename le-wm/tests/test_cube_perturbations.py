"""Synthetic checks for the versioned grid, adaptation and analysis."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research.adaptation import Adapter, GAIN_BOUNDS
from cube_research.analysis import grid_rows
from cube_research.cohort import compatible_contract, frozen_source_identity
from cube_research.perturbations import assert_sealed, read_profile, seal_plan
from cube_research.protocol import Ledger, policy_config, read_config, transform_command
from cube_research.simulator import Harness
from research_common import write_json


def test_xyz_commands_clip_before_and_after_gain_and_preserve_channel_order():
    commands = np.array([[2, -.7, -2., .25, -.8], [-.25, -.1, .3, .5, .9]])
    for scalar in (.4, 1., 1.8):
        np.testing.assert_array_equal(transform_command(commands, scalar), transform_command(commands, [scalar]*3))
    np.testing.assert_array_equal(transform_command(commands, [1]*3), np.clip(commands, -1, 1))
    np.testing.assert_array_equal(transform_command(commands, [.4, 2., .6]),
                                  [[.4, -1., -.6, .25, -.8], [-.1, -.2, .18, .5, .9]])
    with pytest.raises(ValueError):
        transform_command(commands, [1., -.3, 1.])


class TrackingModel:
    dim = 5

    def predict(self, z, effective, ledger):
        ledger.count('predictor', len(z))
        return z[:, -1] + effective[:, -1].sum(axis=1)


def config():
    return read_config(Path(__file__).resolve().parents[1] / 'config/research/cube.yaml')


@pytest.mark.parametrize('method', ['Frozen', 'R-I', 'R-D', 'R-U'])
def test_xyz_adapter_bounds_reset_freezing_cost_and_truth_boundary(method):
    cfg = dict(config(), interface_parameterization='xyz', adapt_regularization=1e-5)
    allowed = [policy_config(dict(cfg, **v, true_object_position='not policy input')) for v in read_profile()['variants']]
    assert all(a == allowed[0] for a in allowed)
    assert not {'condition', 'xyz_gain', 'pad_friction_scale', 'true_object_position'} & set(allowed[0])
    commands = np.tile([.31, -.22, .42, .02, .1], (12, 1))
    embeddings = np.vstack([np.zeros(5), np.cumsum(transform_command(commands, [1, 1, .4]), axis=0)])
    ledger = Ledger(50)
    model = TrackingModel()
    adapter = Adapter(model, method, allowed[0], ledger)
    assert adapter.gain == [1]*3 and not adapter.weights.any()
    adapter.fit(embeddings, commands)
    record = adapter.record()
    fit_gain, fit_residual = method in ('R-I', 'R-U'), method in ('R-D', 'R-U')
    assert record['parameter_count'] == 3*fit_gain + adapter.weights.size*fit_residual
    assert all(GAIN_BOUNDS[0] <= g <= GAIN_BOUNDS[1] for g in adapter.gain)
    if not fit_gain:
        assert adapter.gain == [1]*3
    if not fit_residual:
        assert not adapter.weights.any()
    if method == 'R-I':
        assert adapter.gain == pytest.approx([1, 1, .4], abs=.01)
    expected_updates = 20 if fit_gain else int(fit_residual)
    assert ledger.updates == expected_updates
    assert record['objective_evaluations'] == ledger.model.get('predictor_calls', 0) == (
        1 + 6*expected_updates if fit_gain else expected_updates)
    assert record['updated_backbone_parameters'] == 0
    fresh = Adapter(model, method, allowed[0], Ledger(50))
    assert fresh.gain == [1]*3 and not fresh.weights.any() and not fresh.trace
    assert Adapter(model, method, policy_config(config()), Ledger(50)).gain == 1


def test_vector_snapshot_restores_model_rng_and_buffer():
    env = SimpleNamespace(_data=np.arange(3), _ik=SimpleNamespace(_model=None),
                          _model=SimpleNamespace(**{k: np.ones(3) for k in
                          ('geom_friction', 'geom_priority', 'pair_friction', 'geom_rgba')}),
                          np_random=np.random.default_rng(2), variation_space=SimpleNamespace())
    harness = Harness(env, dict(interface_scale=[1, 1, .4]), 'I')
    planner = SimpleNamespace(buffer=[np.arange(5)], rng=np.random.default_rng(8))
    snap = harness.snapshot(planner)
    expected = (env.np_random.random(), planner.rng.random(), random.random(), np.random.random())
    harness.scale[2] = 1
    env._model.pair_friction[0] = .1
    planner.buffer[0][0] = 99
    harness.restore(snap, planner)
    assert expected == (env.np_random.random(), planner.rng.random(), random.random(), np.random.random())
    assert harness.scale == [1, 1, .4] and env._model.pair_friction[0] == 1 and planner.buffer[0][0] == 0


def test_sealed_grid_grouping_pairing_incomplete_and_refusal_to_reuse(tmp_path):
    spec = read_profile()
    plan = seal_plan(dict(config=dict(history_steps=25), cohort_sha256='cohort',
                          experiment_protocol=spec['experiment_protocol'], entries=[
                              dict(v, case_id=cid, planning_seed=142, cohort_position=i)
                              for i, cid in enumerate(spec['cohort_cases'], 2) for v in spec['variants']]))
    assert_sealed(plan, plan)
    with pytest.raises(ValueError, match='changed'):
        assert_sealed(dict(plan, cohort_sha256='other'), plan)
    sub = tmp_path / 'position-2'
    meta = dict(extension_seal=plan['seal_sha256'], experiment_protocol=plan['experiment_protocol'],
                cohort_sha256=plan['cohort_sha256'], config=plan['config'], cohort_position=2)
    write_json(sub / 'metadata.json', dict(meta, extension_seal='legacy'))
    for j, e in enumerate(plan['entries'][:6]):
        rp = sub / 'recoveries' / str(j)
        success = j != 5
        write_json(rp / 'result.json', dict(e, status='complete', success=success,
                   environment_steps=19+j if success else 50, termination='success' if success else 'budget',
                   first_success_step=19+j if success else None))
        write_json(rp / 'evaluation.json', [{'distance': .2}, {'distance': .03 if success else .1}])
    with pytest.raises(ValueError, match='Cannot pool or reuse'):
        grid_rows(tmp_path, plan)
    write_json(sub / 'metadata.json', meta, overwrite=True)
    rows = grid_rows(tmp_path, plan)
    assert len(rows) == 14 and len({r['perturbation_id'] for r in rows}) == 7
    assert [r['first_success_step_delta_vs_n'] for r in rows[:5]] == list(range(5))
    assert rows[5]['status'] == 'task_failure' and rows[5]['first_success_step'] is None
    assert rows[5]['success_delta_vs_n'] == -1 and rows[6]['status'] == 'unfinished'
    assert all(r['success_delta_vs_n'] is None for r in rows[7:])


def test_legacy_source_receipt_requires_exact_reviewed_source_pair():
    sealed = {'frozen_source_sha256': '1cb418cfe21e49441478733a08ff9f59dc69998265f7391ef385738ba8dcd7fe',
              'protocol_sha256': 'unchanged'}
    current = dict(sealed, frozen_source_sha256=frozen_source_identity())
    assert compatible_contract(sealed, current)
    assert not compatible_contract(sealed, dict(current, frozen_source_sha256='unknown'))
    assert not compatible_contract(sealed, dict(current, protocol_sha256='changed'))
