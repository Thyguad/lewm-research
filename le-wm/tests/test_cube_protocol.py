"""No checkpoint required: scientific invariants, not synthetic task results."""

import copy
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research.protocol import (
    History,
    Observation,
    Ledger,
    transform_command,
    observation,
    grouped_cases,
    validate_manifest,
    digest,
    read_config,
    probe_actions,
    policy_config,
)
from cube_research.adaptation import Adapter
from cube_research.diagnosis import Diagnostic

CFG = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")


def test_action_direction_bounds_and_identity():
    a = np.array([0.8, -0.8, 0.2, 0.6, -0.3])
    assert np.array_equal(transform_command(a), a)
    assert np.allclose(transform_command(a, 2), [1, -1, 0.4, 0.6, -0.3])
    assert np.allclose(transform_command(a, 0.5), [0.4, -0.4, 0.1, 0.6, -0.3])
    assert np.allclose(transform_command([2, 0, 0, 0, 0], 0.5), [0.5, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        transform_command([1, 2])
    with pytest.raises(ValueError):
        transform_command([0, 0, np.nan, 0, 0])


def test_no_privileged_info_and_no_truth_config():
    info = {
        "condition": "D",
        "scale": 0.3,
        "qpos": np.ones(19),
        "privileged/block_0_pos": np.ones(3),
    }
    o = observation(np.zeros((224, 224, 3), np.uint8), info, "rgb")
    assert set(vars(o)) == {"pixels", "sensors"} and o.sensors.size == 0
    c = policy_config(CFG)
    assert not {"friction_scale", "interface_scale", "weights", "dataset", "condition"} & set(c)


def test_whole_episode_splits_and_tampering():
    cases = grouped_cases(range(50), [100] * 50, 25, 42)
    m = dict(schema="cube-cases-v1", cases=cases, cases_sha256=digest(cases))
    validate_manifest(m)
    assert len({c["episode"] for c in cases}) == 50
    assert {c["split"] for c in cases} == {"dev", "train", "test"}
    m["cases"][0]["split"] = "test"
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_budget_and_probe_return_are_real_actions():
    h = History([Observation(np.zeros((4, 4, 3), np.uint8), np.empty(0))])
    h.append([0, 0, 0, 0, 1], h.observations[0])
    a, phase = probe_actions("lift_cycle", 5, 0.2, h)
    assert phase == "commanded_closing" and len(a) == 5 and np.all(a[:, 4] == 0)
    assert np.allclose(a.sum(0), 0)
    ledger = Ledger(5)
    for _ in a:
        ledger.step("probe_including_return")
    assert ledger.used == 5
    with pytest.raises(RuntimeError):
        ledger.step("recovery")
    clone = copy.deepcopy(h)
    clone.actions[0][0] = 0.2
    assert h.actions[0][0] == 0 and h.fingerprint() != clone.fingerprint()


class LinearFrozen:
    """Synthetic model exclusively for checking adapter math."""

    dim = 3

    def __init__(self):
        self.matrix = np.eye(3)

    def predict(self, z, a, ledger):
        ledger.count("predictor", len(z))
        return z[:, -1] + a[:, -1, :, :3].sum(1) @ self.matrix


def test_adaptation_updates_frozen_scope_reset_and_unified_no_label():
    rng = np.random.default_rng(6)
    commands = rng.uniform(-0.2, 0.2, (40, 5))
    # Exact linear interface shift. All overlapping block targets consistent.
    z = np.vstack((np.zeros(3), np.cumsum(commands[:, :3] * 0.6, axis=0))).astype(np.float32)
    model = LinearFrozen()
    original = model.matrix.copy()
    outputs = {}
    for method in ("Frozen", "R-I", "R-D", "R-U"):
        ledger = Ledger(40)
        adapter = Adapter(model, method, policy_config(CFG), ledger)
        adapter.fit(z, commands)
        outputs[method] = adapter
        assert np.array_equal(model.matrix, original)
        fresh = Adapter(model, method, policy_config(CFG), Ledger(40))
        assert fresh.gain == 1 and not fresh.weights.any() and fresh.trace == []
        assert adapter.record()["updated_backbone_parameters"] == 0
    assert abs(outputs["R-I"].gain - 0.6) < 0.1
    assert np.linalg.norm(outputs["R-D"].weights) > 0
    assert outputs["R-U"].gain != 1 and np.linalg.norm(outputs["R-U"].weights) > 0
    assert outputs["Frozen"].trace == []
    assert outputs["R-D"].ledger.updates == 1


def test_diagnostic_group_calibration_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 3)
    labels = np.tile(np.arange(3), 10)
    x = np.eye(3)[labels] + rng.normal(0, 0.01, (30, 3))
    y = np.ones((30, 4))
    y[np.arange(30), labels] = 0
    d = Diagnostic()
    meta = d.fit(x, labels, y, groups)
    assert not set(meta["training_groups"]) & set(meta["calibration_groups"])
    p = d.predict(x[0])
    assert p["predicted_condition"] == "N"
    d.save(tmp_path / "d.npz")
    assert Diagnostic.load(tmp_path / "d.npz").predict(x[0]) == p


def test_full_history_feature_includes_every_frame_and_sensor():
    from cube_research.diagnosis import features

    h = History(
        [
            Observation(np.zeros((224, 224, 3), np.uint8), np.array([i], np.float32))
            for i in range(4)
        ],
        [np.zeros(5)] * 3,
    )
    z = np.ones((4, 3))
    goal = np.zeros(3)
    f = features(z, h, goal, 5)
    for i in range(4):
        hh = copy.deepcopy(h)
        hh.observations[i].pixels[0, 0, 0] = 255
        assert not np.array_equal(f, features(z, hh, goal, 5))
    hh = copy.deepcopy(h)
    hh.observations[1].sensors[0] += 10
    assert not np.array_equal(f, features(z, hh, goal, 5))
