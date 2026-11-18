"""CPU regressions and conditional real-MPS checks. No pretrained downloads.

On a Mac: python tests/test_research_devices.py --require-mps -v
--require-mps makes unavailable MPS a failure, not a misleading skipped pass.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_cem import LoggingCEMSolver, score_candidates
from research_device import prepare_model_info, resolve_device, probe_device
from research_diag import model_inputs

MPS_AVAILABLE = torch.backends.mps.is_available()
if '--require-mps' in sys.argv:
    sys.argv.remove('--require-mps')
    if not MPS_AVAILABLE:
        raise SystemExit('MPS required but unavailable; check native Python/PyTorch/macOS.')


class RecordingCost:
    def __init__(self):
        self.batches = []

    def get_cost(self, info, actions):
        assert info['pixels'].dtype == torch.float32
        assert info['pixels'].device == actions.device
        self.batches.append(actions.detach().cpu().clone())
        info.pop('goal')
        return (actions - 0.4).square().sum((-1, -2))


def raw_info():
    return {'pixels': torch.zeros(1, 1, 3, 8, 8, dtype=torch.float64),
            'goal': torch.ones(1, 1, 3, 8, 8, dtype=torch.float64),
            'action': np.zeros((1, 1, 2), dtype=np.float64),
            'state': np.array([1.123456789012345], dtype=np.float64)}


def make_solver(device, batch=3, diagnostics=False):
    model = RecordingCost()
    solver = LoggingCEMSolver(model, num_samples=8, n_steps=2, topk=3,
                              device=device, seed=12, candidate_batch_size=batch,
                              diagnostics=diagnostics)
    solver.configure(action_space=SimpleNamespace(shape=(1, 2)), n_envs=1,
                     config=SimpleNamespace(horizon=3, receding_horizon=2, action_block=5))
    return solver


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_priority_and_no_accelerator(self):
        for cuda, mps, expected in ((True, True, 'cuda:0'), (False, True, 'mps'), (False, False, 'cpu')):
            with self.subTest(expected=expected), patch.object(torch.cuda, 'is_available', return_value=cuda), \
                 patch.object(torch.cuda, 'current_device', return_value=0), \
                 patch.object(torch.cuda, 'device_count', return_value=1), \
                 patch.object(torch.backends.mps, 'is_built', return_value=True), \
                 patch.object(torch.backends.mps, 'is_available', return_value=mps):
                self.assertEqual(resolve_device('auto'), expected)

    def test_explicit_devices_never_silently_fall_back(self):
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device('cuda')
        with patch.object(torch.backends.mps, 'is_available', return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device('mps')
        for value in ('gpu', 'mps:1', 'cuda:-1', 'cpu:2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_device(value)

    def test_float32_model_boundary_keeps_physics_precision(self):
        data = raw_info()
        saved = data['state'].copy()
        info = prepare_model_info(data, 'cpu')
        self.assertEqual(set(info), {'pixels', 'goal', 'action'})
        self.assertTrue(all(t.dtype == torch.float32 for t in info.values()))
        np.testing.assert_array_equal(data['state'], saved)
        self.assertEqual(data['state'].dtype, np.float64)
        replay = model_inputs({f'info_{k}': np.asarray(data[k]) for k in info}, 'cpu')
        self.assertTrue(all(t.dtype == torch.float32 for t in replay.values()))

    def test_chunk_order_and_global_cem_result(self):
        full = make_solver('cpu', batch=0)
        chunked = make_solver('cpu', batch=3)
        a, b = full.solve(raw_info())['actions'], chunked.solve(raw_info())['actions']
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual([x.shape[1] for x in chunked.model.batches], [3, 3, 2] * 2)
        for iteration in range(2):
            torch.testing.assert_close(full.model.batches[iteration],
                                       torch.cat(chunked.model.batches[iteration*3:(iteration+1)*3], 1),
                                       rtol=0, atol=0)
        self.assertEqual(full.plan_rows[0]['candidate_predictions'], chunked.plan_rows[0]['candidate_predictions'])

    def test_chunked_diagnostics_leave_actions_unchanged(self):
        a, b = make_solver('cpu'), make_solver('cpu', diagnostics=True)
        for _ in range(2):
            x = a.solve(raw_info())['actions']
            y = b.solve(raw_info())['actions']
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_mutating_cost_cannot_remove_original_inputs(self):
        info = prepare_model_info(raw_info(), 'cpu')
        score_candidates(RecordingCost(), info, torch.zeros(1, 8, 3, 10), 3)
        self.assertIn('goal', info)

    def test_cpu_probe(self):
        self.assertEqual(probe_device('cpu')['probe'], 'passed')


@unittest.skipUnless(MPS_AVAILABLE, 'Requires a Mac with available MPS')
class MPSTests(unittest.TestCase):
    def test_mps_float64_boundary_and_sync(self):
        info = prepare_model_info(raw_info(), 'mps')
        self.assertTrue(all(t.dtype == torch.float32 and t.device.type == 'mps' for t in info.values()))
        self.assertEqual(probe_device('mps')['probe'], 'passed')

    def test_mps_solver_and_diagnostic_pair(self):
        a, b = make_solver('mps'), make_solver('mps', diagnostics=True)
        x, y = a.solve(raw_info())['actions'], b.solve(raw_info())['actions']
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        x, y = a.solve(raw_info(), x[:, 3:])['actions'], b.solve(raw_info(), y[:, 3:])['actions']
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(x.device.type, 'cpu')
        self.assertFalse(x.requires_grad)


if __name__ == '__main__':
    unittest.main()
