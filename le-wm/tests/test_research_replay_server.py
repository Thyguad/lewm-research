"""Real PushT reset/prefix replay on Mac/Linux, no checkpoint or dataset."""
import json
import os
from pathlib import Path
import sys
import unittest

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_cem import DEFAULTS
from research_common import json_default
from research_runtime import (
    advance_world, assert_restored, base_env, install_reset_recorder, make_world,
    physics_vector, render_frame, restore_world,
)


class ReplayTests(unittest.TestCase):
    def test_reset_json_roundtrip_prefix_and_continuation(self):
        cfg = {**DEFAULTS, 'device': 'cpu'}
        world = make_world(cfg)
        try:
            recipe = install_reset_recorder(world, 123)
            world.reset(seed=[123], options=[{}])
            # A repeatable contact-rich scene; mirror the online initialization order.
            recipe.update(initial_state=np.array([220., 280., 256., 256., 0., 0., 0.]),
                          goal_state=np.array([270., 230., 280., 245., 0.1, 0., 0.]))
            env = base_env(world)
            env._set_state(recipe['initial_state'])
            env._set_goal_state(recipe['goal_state'])
            prefix = np.linspace([0.12, -0.02], [0.08, -0.04], 15, dtype=np.float32)
            for action in prefix:
                advance_world(world, action)
            snapshot = {'physics': physics_vector(world), 'rendered_pixels': render_frame(world, 224)}
            recipe = json.loads(json.dumps(recipe, default=json_default))
            restored = restore_world(cfg, recipe, prefix)
            try:
                assert_restored(restored, snapshot, 224)
                for action in np.linspace([0.08, -0.04], [0.04, -0.05], 15, dtype=np.float32):
                    advance_world(world, action)
                    advance_world(restored, action)
                    np.testing.assert_allclose(physics_vector(world), physics_vector(restored), rtol=0, atol=1e-7)
                    np.testing.assert_array_equal(render_frame(world, 224), render_frame(restored, 224))
                bad = {'physics': snapshot['physics'] + 1, 'rendered_pixels': snapshot['rendered_pixels']}
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    assert_restored(restored, bad, 224)
            finally:
                restored.close()
        finally:
            world.close()


if __name__ == '__main__':
    unittest.main()
