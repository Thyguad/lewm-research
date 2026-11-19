"""Real MuJoCo check, independent of weights/data. Optional isolated package loading
avoids stable_worldmodel's torch-heavy __init__, without replacing physics or Cube.
This is not the end-to-end checkpoint smoke command.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import types
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--isolated-imports", action="store_true")
p.add_argument("--render-backend", default="egl")
a = p.parse_args()
if a.render_backend != "auto":
    os.environ["MUJOCO_GL"] = a.render_backend
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "le-wm"))
sys.path.insert(0, str(root / "stable-worldmodel"))
if a.isolated_imports:
    # Namespace only: actual repository spaces/env utils/Cube source are loaded.
    for name, path in [
        ("stable_worldmodel", "stable_worldmodel"),
        ("stable_worldmodel.envs", "stable_worldmodel/envs"),
        ("stable_worldmodel.envs.ogbench", "stable_worldmodel/envs/ogbench"),
    ]:
        package = types.ModuleType(name)
        package.__path__ = [str(root / "stable-worldmodel" / path)]
        sys.modules[name] = package
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from cube_research.simulator import research_env_class, Harness
from cube_research.protocol import read_config, Ledger
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

cfg = read_config(root / "le-wm/config/research/cube.yaml")
cfg["render_backend"] = a.render_backend
cls = research_env_class(CubeEnv)


def create(scale=None):
    e = cls(
        env_type="single",
        ob_type="states",
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=False,
        mode="data_collection",
        contact_scale=scale,
    )
    e.reset(seed=42, options={"variation": []})
    e.set_target_pos(0, [0.45, 0.12, 0.02], [1, 0, 0, 0])
    e.pre_step()
    e.post_step()
    return e


np.random.seed(42)
env = create()
harness = Harness(env, cfg, "N")
commands = []
baseline = []
contact_snapshot = None
try:
    obs = harness.observe()
    assert obs.pixels.shape == (224, 224, 3)
    assert env._np_random_seed == 42
    initial = harness.snapshot()
    effector = []
    for scale in (1.0, 0.6):
        harness.restore(initial)
        harness.scale = scale
        _, truth = harness.step(np.array([0.5, 0, 0, 0, 0]), Ledger(1), "interface_check")
        effector.append(truth["effector_pos"])
    interface_response = float(np.max(np.abs(np.asarray(effector[0]) - effector[1])))
    assert interface_response > 1e-6
    harness.restore(initial)
    oracle = CubeMarkovOracle(env=env, min_norm=0.4)
    oracle.reset(None, env.get_reset_info())
    for _ in range(70):
        action = oracle.select_action(None, env.get_step_info())
        commands.append(action)
        _, truth = harness.step(action, Ledger(1), "offline_oracle")
        baseline.append(truth)
        if truth["contacts"] and contact_snapshot is None:
            contact_snapshot = harness.snapshot()
    assert contact_snapshot is not None, "Fixture failed to reach gripper-object contact"
    harness.restore(contact_snapshot)
    from types import SimpleNamespace

    planner = SimpleNamespace(buffer=[np.ones(5) * 0.02], rng=np.random.default_rng(7))
    ledger = Ledger(0)
    replay = harness.replay_check(
        np.array([[0.1, 0, 0.2, 0, 1], [-0.1, 0, 0, 0, 0]]), ledger, planner
    )
    assert len(planner.buffer) == 1 and np.allclose(planner.buffer[0], 0.02)
    assert ledger.offline_replay_steps == 4 and ledger.used == 0
    # Persist/restore supported fields via pickle (MjData carries exact state).
    import pickle

    snap = harness.snapshot(planner)
    try:
        encoded = pickle.dumps(snap)
        decoded = pickle.loads(encoded)
        harness.restore(decoded, planner)
        serializable = True
    except Exception as exc:
        raise AssertionError(f"Snapshot persistence failed: {exc}") from exc
finally:
    harness.close()

trajectories = {}
for label, scale in [("identity", 1.0), ("changed", 0.3)]:
    env = create(scale)
    harness = Harness(env, cfg, "D")
    try:
        values = []
        for action in commands:
            _, truth = harness.step(action, Ledger(1), "offline_replay")
            values.append(truth)
        trajectories[label] = values
        assert any(x["contacts"] for x in values)
        for x in values:
            for c in x["contacts"]:
                pad = next(n for n in c["geoms"] if "pad" in n)
                mu = float(env._model.geom(pad).friction[0])
                assert np.allclose(c["friction"][:2], mu * scale)
    finally:
        harness.close()

base = np.array([x["object_pos"] for x in baseline])
identity = np.array([x["object_pos"] for x in trajectories["identity"]])
changed = np.array([x["object_pos"] for x in trajectories["changed"]])
identity_error = float(np.max(np.abs(base - identity)))
response = float(np.max(np.abs(changed - identity)))
assert identity_error < 1e-7, identity_error
assert response > 1e-6, response
print(
    json.dumps(
        dict(
            kind="real_cube_simulator_only",
            isolated_imports=a.isolated_imports,
            steps_per_replay=len(commands),
            identity_max_abs=identity_error,
            contact_response_max_abs=response,
            interface_response_max_abs=interface_response,
            replay=replay,
            snapshot_pickle=serializable,
            weights_forward=False,
            expert_dataset=False,
        ),
        indent=2,
    )
)
