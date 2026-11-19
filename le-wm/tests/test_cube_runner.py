"""Synthetic orchestration contract test; NOT a Cube/LeWM performance test."""

import copy
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research import experiments, model, simulator
from cube_research.protocol import read_config, Ledger, Observation, digest
from research_common import write_json, read_json, file_hash


class Model:
    dim = 3
    device = "cpu"
    identity = {"weights_sha256": "synthetic"}

    def __init__(self, cfg, stats):
        pass

    def encode(self, frames, ledger):
        ledger.count("encoder", len(frames))
        return np.array([f.mean((0, 1)) / 255 for f in frames], np.float32)

    def predict(self, z, a, ledger):
        ledger.count("predictor", len(z))
        return z[:, -1] + a[:, -1, :, :3].sum(1) * 0.001


class Env:
    def __init__(self):
        self.t = 0

    def observe(self):
        return Observation(np.full((224, 224, 3), self.t, np.uint8), np.empty(0))

    def evaluate(self):
        return dict(success=False, truncated=False, distance=1.0)

    def step(self, a, ledger, phase):
        ledger.step(phase)
        self.t += 1
        return self.observe(), self.evaluate()

    def snapshot(self, planner=None):
        return dict(
            t=self.t,
            planner=None
            if planner is None
            else copy.deepcopy((planner.buffer, planner.rng.bit_generator.state)),
        )

    def restore(self, snap, planner=None):
        self.t = snap["t"]
        if snap["planner"] is not None:
            planner.buffer, planner.rng.bit_generator.state = copy.deepcopy(snap["planner"])

    def replay_check(self, a, ledger, planner=None):
        ledger.offline_replay_steps += 2 * len(a)
        ledger.restore_checks += 1
        return {"status": "synthetic"}

    def audit_model(self):
        return {"condition": "synthetic"}

    def close(self):
        pass


class Data:
    def __init__(self, path):
        pass

    def case(self, c):
        return {}, {}, np.zeros((224, 224, 3), np.uint8)

    def close(self):
        pass


def test_cohort_frozen_branch_preserves_rng_and_early_success_does_not_adapt():
    cfg = read_config(Path(__file__).resolve().parents[1] / 'config/research/cube.yaml')
    cfg.update(total_steps=12, history_steps=5, horizon=1, execute_blocks=1,
               cem_samples=4, cem_elites=2, cem_iterations=1, adapt_steps=1)
    bridge, env = Model(cfg, {}), Env()
    goal = np.full((224, 224, 3), 50, np.uint8)
    ledger, seed = Ledger(12), 142
    history, truth, goal_z, _, planner = experiments.start_frozen(cfg, bridge, seed, ledger, env, goal)
    experiments.execute(env, history, truth, planner, bridge, ledger, 5, 'history', goal_z)
    prefix = copy.deepcopy((history, truth, ledger))
    snap = env.snapshot(planner)
    rng = copy.deepcopy(planner.rng.bit_generator.state)
    experiments.execute(env, history, truth, planner, bridge, ledger, 7, 'recovery', goal_z)
    for method in ('Frozen', 'R-I', 'R-D', 'R-U'):
        env.restore(snap, planner)
        h, t, ll = copy.deepcopy(prefix)
        assert env.t == 5 and h.fingerprint() == prefix[0].fingerprint()
        adapter = experiments.recover(env, h, t, bridge, goal_z, method, cfg, seed, ll, rng)
        assert ll.used == 12 and adapter.record()['training_rows'] == (0 if method == 'Frozen' else 1)
        if method == 'Frozen':
            np.testing.assert_array_equal(h.actions, history.actions)
        env.restore(snap, planner)
        h, t, ll = copy.deepcopy(prefix)
        t[-1]['success'] = True
        fresh = experiments.recover(env, h, t, bridge, goal_z, method, cfg, seed, ll, rng)
        assert env.t == ll.used == 5 and ll.updates == 0 and not fresh.trace and fresh.gain == 1


def test_real_runner_orchestration_shared_history_budgets_and_artifacts(tmp_path, monkeypatch):
    cfg = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")
    cfg.update(
        total_steps=12,
        history_steps=5,
        extra_budgets=[0, 2],
        horizon=1,
        cem_samples=4,
        cem_elites=2,
        cem_iterations=1,
        adapt_steps=1,
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"synthetic")
    (tmp_path / "config.json").write_text("{}")
    dataset = tmp_path / "d.h5"
    dataset.write_bytes(b"synthetic")
    cfg.update(weights=str(weights), dataset=str(dataset))
    cases = [
        dict(case_id=digest([0, 0, 25])[:16], episode=0, start_step=0, goal_offset=25, split="dev")
    ]
    manifest = tmp_path / "cases.json"
    write_json(
        manifest,
        dict(
            schema="cube-cases-v1",
            cases=cases,
            cases_sha256=digest(cases),
            protocol=cfg["protocol"],
            dataset_source=cfg["dataset_source"],
            dataset_sha256=file_hash(dataset),
            stats={},
        ),
    )
    monkeypatch.setattr(experiments, "Dataset", Data)
    monkeypatch.setattr(model, "FrozenModel", Model)
    monkeypatch.setattr(simulator, "make_env", lambda *args: Env())
    out = tmp_path / "run"
    assert experiments.run(cfg, manifest, out, "collect", limit=1)
    assert read_json(out / "STATUS.json")["completed"] == 48
    rows = [read_json(p) for p in (out / "recoveries").glob("*/result.json")]
    for hid in {r["history_id"] for r in rows}:
        rs = [r for r in rows if r["history_id"] == hid]
        assert len(rs) == 4 and len({r["history_sha256"] for r in rs}) == 1
        assert all(r["environment_steps"] == 12 for r in rs)
        assert all(sum(r["ledger"]["phases"].values()) == 12 for r in rs)
        assert all(r["ledger"]["offline_replay_steps"] > 0 for r in rs)
        for r in rs:
            if r["strategy"] != "passive":
                assert r["actual_extra_steps"] == 2
    for p in (out / "recoveries").glob("*/allowed.npz"):
        with np.load(p, allow_pickle=False) as f:
            assert len(f["pixels"]) == 13 and len(f["commands"]) == 12
    with pytest.raises(FileExistsError):
        experiments.run(cfg, manifest, out, "collect", limit=1)


def test_lock_rejects_test_time_retuning(tmp_path, monkeypatch):
    cfg = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")
    for name in ("data.h5", "weights.pt", "config.json"):
        (tmp_path / name).write_text("x")
    cfg.update(dataset=str(tmp_path / "data.h5"), weights=str(tmp_path / "weights.pt"))
    c = dict(case_id=digest([0, 0, 25])[:16], episode=0, start_step=0, goal_offset=25, split="test")
    m = tmp_path / "m.json"
    write_json(
        m,
        dict(
            schema="cube-cases-v1",
            cases=[c],
            cases_sha256=digest([c]),
            dataset_sha256=file_hash(cfg["dataset"]),
        ),
    )
    lock = tmp_path / "lock.json"
    experiments.lock(cfg, m, lock)
    experiments.check_lock(cfg, m, lock)
    cfg["friction_scale"] = 0.2
    with pytest.raises(ValueError):
        experiments.check_lock(cfg, m, lock)


def test_collection_training_calibration_and_locked_evaluation(tmp_path, monkeypatch):
    cfg = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")
    cfg.update(
        total_steps=7,
        history_steps=5,
        extra_budgets=[0],
        horizon=1,
        cem_samples=2,
        cem_elites=1,
        cem_iterations=1,
        adapt_steps=1,
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"synthetic")
    (tmp_path / "config.json").write_text("{}")
    dataset = tmp_path / "d.h5"
    dataset.write_bytes(b"synthetic")
    cfg.update(weights=str(weights), dataset=str(dataset))
    cases = [
        dict(
            case_id=digest([i, 0, 25])[:16],
            episode=i,
            start_step=0,
            goal_offset=25,
            split="train" if i < 5 else "test",
        )
        for i in range(6)
    ]
    manifest = tmp_path / "cases.json"
    write_json(
        manifest,
        dict(
            schema="cube-cases-v1",
            cases=cases,
            cases_sha256=digest(cases),
            protocol=cfg["protocol"],
            dataset_source=cfg["dataset_source"],
            dataset_sha256=file_hash(dataset),
            stats={},
        ),
    )
    monkeypatch.setattr(experiments, "Dataset", Data)
    monkeypatch.setattr(model, "FrozenModel", Model)
    monkeypatch.setattr(simulator, "make_env", lambda *args: Env())
    collection = tmp_path / "collection"
    assert experiments.run(cfg, manifest, collection, "collect", split="train", limit=5)
    diagnostic = tmp_path / "diagnostic"
    experiments.train(cfg, manifest, [collection], diagnostic)
    assert read_json(diagnostic / "STATUS.json")["state"] == "complete"
    lock = tmp_path / "lock.json"
    experiments.lock(cfg, manifest, lock, diagnostic)
    evaluation = tmp_path / "evaluation"
    assert experiments.run(
        cfg,
        manifest,
        evaluation,
        "evaluate",
        split="test",
        limit=1,
        lock_path=lock,
        diagnostic_path=diagnostic,
    )
    rows = [read_json(p) for p in (evaluation / "recoveries").glob("*/result.json")]
    assert len(rows) == 3 and all(r["method"] == "Selected" for r in rows)
    assert all(r["diagnostic"]["method"] == "Frozen" for r in rows)
    from cube_research.analysis import summarize

    summarize([evaluation], tmp_path / "summary")
    assert len(read_json(tmp_path / "summary/summary.json")["diagnostic"]) == 1


def test_failure_keeps_partial_history_and_does_not_complete(tmp_path, monkeypatch):
    cfg = read_config(Path(__file__).resolve().parents[1] / "config/research/cube.yaml")
    cfg.update(
        total_steps=7,
        history_steps=5,
        extra_budgets=[0],
        horizon=1,
        cem_samples=2,
        cem_elites=1,
        cem_iterations=1,
        adapt_steps=1,
    )
    for name in ("weights.pt", "config.json", "d.h5"):
        (tmp_path / name).write_text("synthetic")
    cfg.update(weights=str(tmp_path / "weights.pt"), dataset=str(tmp_path / "d.h5"))
    cases = [
        dict(case_id=digest([0, 0, 25])[:16], episode=0, start_step=0, goal_offset=25, split="dev")
    ]
    manifest = tmp_path / "cases.json"
    write_json(
        manifest,
        dict(
            schema="cube-cases-v1",
            cases=cases,
            cases_sha256=digest(cases),
            protocol=cfg["protocol"],
            dataset_source=cfg["dataset_source"],
            dataset_sha256=file_hash(cfg["dataset"]),
            stats={},
        ),
    )

    class BrokenRecovery(Env):
        def step(self, a, ledger, phase):
            if phase == "recovery":
                ledger.step(phase)
                raise RuntimeError("intentional simulator failure")
            return super().step(a, ledger, phase)

    monkeypatch.setattr(experiments, "Dataset", Data)
    monkeypatch.setattr(model, "FrozenModel", Model)
    monkeypatch.setattr(simulator, "make_env", lambda *args: BrokenRecovery())
    root = tmp_path / "failed"
    assert not experiments.run(cfg, manifest, root, "e1", limit=1)
    assert read_json(root / "STATUS.json")["state"] == "incomplete"
    rows = [p for p in (root / "recoveries").glob("*/result.json")]
    assert len(rows) == 12
    for p in rows:
        assert read_json(p)["status"] == "failed"
        with np.load(p.parent / "allowed.npz") as f:
            assert len(f["commands"]) == 5 and len(f["pixels"]) == 6
        assert read_json(p)["environment_steps"] == 6  # failed attempt is counted too
