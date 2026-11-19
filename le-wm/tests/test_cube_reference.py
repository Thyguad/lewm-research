from pathlib import Path
from types import SimpleNamespace
import sys

import gymnasium as gym
import h5py
import numpy as np
import pytest
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_reference import AUDIT_IDS, verified_scores
from cube_research.reference import (BUDGET, OFFSET, PLAN, TracedSolver, action_blocks,
                                     command_rows, fit_actions, make_policy, open_dataset,
                                     sample_rows, summarize)
from research_common import write_json


def test_reference_sampling_preserves_last_row_exclusion_and_order():
    ep = np.repeat([0, 7, 11], [31, 23, 50])
    step = np.concatenate([np.arange(n) for n in (31, 23, 50)])
    ends = {i: step[ep == i].max() for i in set(ep)}
    valid = np.flatnonzero([t <= ends[e] - 25 for e, t in zip(ep, step)])
    expected = np.sort(valid[np.random.default_rng(42).choice(len(valid) - 1, 12, replace=False)])
    actual, actual_valid = sample_rows(ep, step, 12)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_valid, valid)
    assert valid[-1] not in actual


def test_real_hdf5_frameskip_and_per_step_scaler(tmp_path):
    path = tmp_path / 'cube.h5'
    rows = np.random.default_rng(5).normal(size=(27, 5)).astype('float32')
    rows[-1] = np.nan
    with h5py.File(path, 'w') as handle:
        handle['action'] = rows
        handle['qpos'] = np.zeros((27, 21), dtype='float64')
        handle['ep_len'], handle['ep_offset'] = [27], [0]
    data = open_dataset(path, tmp_path / 'cache')
    bad = swm.data.HDF5Dataset(str(path.with_suffix('')), frameskip=5, cache_dir=tmp_path / 'cache')
    try:
        data_rows = data.load_chunk([0], [0], [26])[0]
        assert data_rows['action'].shape == (26, 5)
        assert data_rows['qpos'].dtype == torch.float64
        scaler = fit_actions(data)
        np.testing.assert_array_equal(scaler.mean_, rows[:-1].mean(0, dtype='float64'))
        normalized = action_blocks(rows[None, :25], scaler)
        assert normalized.shape == (1, 5, 25)
        np.testing.assert_array_equal(normalized[0, 1], scaler.transform(rows[5:10]).ravel())
        np.testing.assert_allclose(command_rows(normalized, scaler)[0], rows[:25], atol=2e-7)
        policy = swm.policy.WorldModelPolicy(None, PLAN, process={'action': scaler})
        blocked = bad.load_chunk([0], [0], [25])[0]['action'][0].numpy()[None, None]
        with pytest.raises(ValueError, match='25 features.*expecting 5 features'):
            policy._prepare_info({'action': blocked})
    finally:
        data.h5_file.close()
        bad.h5_file.close()


@pytest.mark.parametrize('device', ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []))
def test_actual_reference_solver_policy_buffer_and_model_only_inputs(device):
    class Score:
        def get_cost(self, info, actions):
            assert set(info) == {'pixels', 'goal', 'action'}
            assert all(v.dtype == torch.float32 and v.device.type == device for v in info.values())
            assert actions.shape[-2:] == (5, 25)
            return actions.square().sum((-2, -1))

    # Inverse-transforming a normalized zero gives a deliberately unbounded raw command.
    scaler = StandardScaler().fit(np.array([[2]*5, [4]*5], dtype=np.float32))
    solver = TracedSolver(Score(), scaler, device, 'paper-300x10-top30')
    solver.num_samples, solver.n_steps, solver.topk = 4, 2, 2
    policy = make_policy(solver, scaler)
    policy.set_env(SimpleNamespace(num_envs=1, action_space=gym.spaces.Box(-1, 1, (1, 5))))
    c = np.zeros((1, 1, 224, 224, 3), dtype=np.uint8)
    truth = np.zeros((1, 1, 21), dtype=np.float64)
    info = {'pixels': c, 'goal': c, 'qpos': truth, 'action': np.ones((1, 1, 25), dtype=np.float64)}
    actions = np.stack([policy.get_action(info)[0] for _ in range(OFFSET)])
    assert len(solver.plans) == 1
    plan = solver.plans[0]
    np.testing.assert_array_equal(actions, plan['raw_commands'][0])
    assert actions.min() > 1 and info['qpos'] is truth and truth.dtype == np.float64
    assert plan['final_mean_cost'][0] == pytest.approx(float(np.square(plan['mean']).sum()), rel=1e-6)
    policy.get_action(info)
    assert len(solver.plans) == 2 and solver.plans[1]['warm_start_shape'] == [1, 0, 25]


def test_metrics_separate_initial_success_post_step_termination_and_unfinished():
    cases = [{'case_id': i} for i in range(3)]

    def trace(initial, success_step):
        return [dict(step=i, success=initial if i == 0 else i == success_step,
                     terminated=i == success_step, contacts=[], distance=.1, object_pos=[0, 0, 0])
                for i in range(BUDGET + 1)]

    result = summarize(cases, [trace(True, None), trace(False, 1), []])
    assert [r['status'] for r in result['cases']] == ['failure', 'success', 'unfinished']
    assert result['initial']['successes'] == 1 and result['overall']['successes'] == 1
    assert result['overall']['rate'] is None and result['non_initial']['rate'] == 1


def test_formal_gate_requires_first_plan_and_fixed_parity(tmp_path):
    cases = {cid: {'passed': True} for cid in AUDIT_IDS}
    write_json(tmp_path / 'fixed-score.json', {'passed': True, 'cases': cases})
    write_json(tmp_path / 'first-plan-score.json', cases)
    assert verified_scores(tmp_path)
    cases[AUDIT_IDS[0]]['passed'] = False
    write_json(tmp_path / 'first-plan-score.json', cases, overwrite=True)
    assert not verified_scores(tmp_path)
