"""Exercise actual LeWM rollout/Predictor with a tiny local encoder (no downloads)."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_device import prepare_model_info
from research_cem import score_candidates
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor, Embedder

MPS_AVAILABLE = torch.backends.mps.is_available()
if '--require-mps' in sys.argv:
    sys.argv.remove('--require-mps')
    if not MPS_AVAILABLE:
        raise SystemExit('MPS required but unavailable')


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(3, 8)

    def forward(self, pixels, interpolate_pos_encoding=True):
        return SimpleNamespace(last_hidden_state=self.project(pixels.mean((-1, -2)))[:, None])


def model():
    torch.manual_seed(42)
    return LeWM(TinyEncoder(), Predictor(num_frames=3, depth=1, heads=2, mlp_dim=16,
                                         input_dim=8, hidden_dim=8, dim_head=4),
                Embedder(input_dim=10, smoothed_dim=8, emb_dim=8)).eval().requires_grad_(False)


class LeWMDeviceTests(unittest.TestCase):
    def check_rollout(self, device):
        wm = model().to(device)
        gen = torch.Generator().manual_seed(123)
        raw = {'pixels': torch.randn(1, 1, 3, 8, 8, generator=gen, dtype=torch.float64),
               'goal': torch.randn(1, 1, 3, 8, 8, generator=gen, dtype=torch.float64),
               'action': torch.zeros(1, 1, 2, dtype=torch.float64)}
        info = prepare_model_info(raw, device)
        actions = torch.randn(1, 7, 5, 10, generator=gen).to(device)
        with torch.inference_mode():
            full = score_candidates(wm, info, actions, 0)
            chunked = score_candidates(wm, info, actions, 3)
        self.assertEqual(tuple(chunked.shape), (1, 7))
        self.assertTrue(torch.isfinite(chunked).all())
        torch.testing.assert_close(chunked, full, rtol=1e-4, atol=1e-5)
        self.assertNotIn('predicted_emb', info)

    def test_cpu_real_lewm_rollout_and_chunking(self):
        self.check_rollout('cpu')

    @unittest.skipUnless(MPS_AVAILABLE, 'Requires Apple MPS')
    def test_mps_real_lewm_rollout_and_chunking(self):
        self.check_rollout('mps')


if __name__ == '__main__':
    unittest.main()
