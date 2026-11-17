"""SERVER ONLY: tensor-level regressions; no checkpoint or dataset required."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_cem import LoggingCEMSolver
from research_common import sync_device
from research_diag import snapshot_data
from research_runtime import inverse_actions


class ToyCost:
    def __init__(self):
        self.candidates = []

    def get_cost(self, info, actions):
        # A mutating get_cost must not pollute the caller's model inputs.
        info['pixels'] = None
        info['predicted_emb'] = None
        self.candidates.append(actions.detach().cpu().clone())
        return (actions - 0.3).square().sum((-1, -2))


def inputs():
    return {'pixels': torch.zeros(1, 1, 3, 4, 4), 'goal': torch.zeros(1, 1, 3, 4, 4),
            'action': torch.zeros(1, 1, 2)}


def solver(model=None, **kwargs):
    instance = LoggingCEMSolver(model or ToyCost(), num_samples=8, n_steps=3, topk=3,
                                device='cpu', seed=42, **kwargs)
    instance.configure(action_space=SimpleNamespace(shape=(1, 2)), n_envs=1,
                       config=SimpleNamespace(horizon=3, receding_horizon=2, action_block=5))
    return instance


class SolverTests(unittest.TestCase):
    def test_warm_start_tail_and_empty_remainder(self):
        for k in (1, 2, 3):
            model = ToyCost()
            s = solver(model)
            first = s.solve(inputs())['actions']
            init = first[:, k:]
            model.candidates.clear()
            second = s.solve(inputs(), init)['actions']
            initial_mean = model.candidates[0][:, 0]
            torch.testing.assert_close(initial_mean[:, :3-k], init, rtol=0, atol=0)
            torch.testing.assert_close(initial_mean[:, 3-k:], torch.zeros(1, k, 10), rtol=0, atol=0)
            self.assertEqual(tuple(second.shape), (1, 3, 10))
            self.assertFalse(second.requires_grad)

    def test_diagnostics_do_not_change_rng_or_actions(self):
        formal, diagnostic = solver(), solver(diagnostics=True)
        a = b = None
        for _ in range(2):
            info = inputs()
            a = formal.solve(info, a)['actions']
            b = diagnostic.solve(inputs(), b)['actions']
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertTrue(torch.is_tensor(info['pixels']))
            self.assertNotIn('predicted_emb', info)
            a, b = a[:, 2:], b[:, 2:]
        self.assertEqual(len(formal.iteration_rows), 0)
        self.assertEqual(len(formal.model.candidates), 6)
        self.assertEqual(len(diagnostic.model.candidates), 12)
        self.assertEqual(formal.plan_rows[0]['extra_prediction_steps'], 0)
        self.assertEqual(diagnostic.plan_rows[0]['extra_prediction_steps'], 9)
        sync_device(torch.device('cpu'))

    def test_case_scoped_snapshots_and_actual_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in ('case_a', 'case_b'):
                s = solver(snapshot_dir=root, run_id='run', case_id=case, max_snapshot_plans=2)
                s.context_provider = lambda: ({'recipe': {}}, {
                    'prefix_actions': np.zeros((0, 2)), 'physics': np.zeros(22),
                    'rendered_pixels': np.zeros((4, 4, 3), dtype=np.uint8),
                    'observation_pixels': np.zeros((4, 4, 3), dtype=np.uint8)})
                returned = s.solve(inputs())['actions']
                path = root / case / 'plan_000000' / 'snapshot.json'
                meta = {'run_id': 'run', 'cases': [{'case_id': case}]}
                snap, arrays = snapshot_data(path, meta)
                selected = [i for i, r in enumerate(snap['candidates']) if r['selected']]
                self.assertEqual(len(selected), 1)
                np.testing.assert_array_equal(arrays['actions'][selected[0]], returned[0].numpy())
                self.assertEqual(snap['candidates'][selected[0]]['candidate_id'], 'mean')
                self.assertEqual(s.plan_rows[0]['returned_mean_cost'], float((returned - 0.3).square().sum()))
                s.solve(inputs(), returned[:, 3:])
                self.assertTrue((root / case / 'plan_000001' / 'snapshot.json').exists())
                with (path.parent / 'arrays.npz').open('ab') as handle:
                    handle.write(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    snapshot_data(path, meta)

    def test_inverse_transform_and_block_order(self):
        from sklearn.preprocessing import StandardScaler
        original = StandardScaler().fit(np.array([[10., 30.], [50., 90.], [40., 60.]]))
        stats = {'mean': original.mean_.tolist(), 'scale': original.scale_.tolist()}
        normalized = np.arange(30, dtype=np.float32).reshape(3, 10) / 10
        raw = inverse_actions(normalized, stats)
        expected = np.stack([original.inverse_transform(row[None])[0] for row in normalized.reshape(-1, 2)])
        np.testing.assert_array_equal(raw, expected)
        self.assertEqual(raw.shape, (15, 2))

    def test_nonfinite_cost_rejected(self):
        class Bad:
            def get_cost(self, info, actions):
                return torch.full(actions.shape[:2], float('nan'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            solver(Bad()).solve(inputs())


if __name__ == '__main__':
    unittest.main()
