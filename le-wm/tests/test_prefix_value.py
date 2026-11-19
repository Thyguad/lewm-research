"""Contract/numerical tests; synthetic harnesses here are NOT Cube experiments."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cube_research.model import Planner
from cube_research.protocol import Ledger, Observation
from planning_ideas.prefix_contracts import (METHODS, features, sample_cases, signature,
                                            utility, validate_config, weights_identity)
from planning_ideas.prefix_data import budget, rollout_branch
from planning_ideas.prefix_evaluation import summarize, trial
from planning_ideas.prefix_pool import (Predictor, PrefixPolicy, ProposalPlanner, branch_indices,
                                      choose, local_edits)
from research_common import read_json
from research_prefix_value import main, parser, read_config


@pytest.fixture
def cfg():
    return read_config(parser().parse_args(["--smoke", "prepare", "--goal-cache", "unused", "--out", "unused"]))


class FakeModel:
    device = "cpu"
    dim = 3
    stats = dict(mean=[0.] * 5, scale=[.4] * 5)
    identity = {"validation_only": "synthetic"}

    def __init__(self, cfg=None, stats=None):
        if stats is not None:
            self.stats = stats

    def encode(self, frames, ledger):
        ledger.count("encoder", len(frames))
        return np.array([np.asarray(f).mean((0, 1)) / 100 - 1.28 for f in frames], np.float32)

    def predict(self, z, commands, ledger):
        ledger.count("predictor", len(z))
        return (z[:, -1] + np.asarray(commands)[:, -1, :, :3].sum(1) * .04).astype(np.float32)


def pixels(position):
    return np.broadcast_to(np.clip(np.rint((np.asarray(position) + 1.28) * 100), 0, 255).astype(np.uint8),
                           (4, 4, 3)).copy()


class FakeHarness:
    def __init__(self, initial, target):
        self.position = np.asarray(initial["qpos"], float).copy()
        self.target = np.asarray(target["pos"], float).copy()
        self.controller = np.zeros(5)
        self.actual_steps = 0
        self.steps = 0
        self.rng = np.random.default_rng(9)

    def snapshot(self):
        return copy.deepcopy((self.position, self.controller, self.steps, self.rng.bit_generator.state))

    def restore(self, snapshot):
        self.position, self.controller, self.steps, state = copy.deepcopy(snapshot)
        self.rng.bit_generator.state = state

    def observe(self):
        return Observation(pixels(self.position), np.empty(0))

    def evaluate(self):
        return dict(distance=float(np.linalg.norm(self.position - self.target)),
                    object_pos=self.position.tolist(), terminated=False, truncated=False)

    def step(self, action, ledger, phase):
        ledger.step(phase)
        self.actual_steps += 1
        self.steps += 1
        self.controller = np.array(action, copy=True)
        self.position += np.asarray(action)[:3] * .04
        return self.observe(), self.evaluate()

    def replay_check(self, actions, ledger):
        snap = self.snapshot()
        paths = []
        for _ in range(2):
            self.restore(snap)
            one = []
            for action in actions:
                obs, _ = self.step(action, Ledger(1), "test_replay")
                ledger.offline_replay_steps += 1
                one.append(obs.pixels)
            paths.append(one)
        self.restore(snap)
        np.testing.assert_array_equal(paths[0], paths[1])
        ledger.restore_checks += 1
        return dict(status="passed", validation_only="synthetic harness")

    def close(self):
        pass

    def audit_model(self):
        return dict(validation_only="synthetic harness", condition="N")


class FakeDataset:
    lengths = np.full(24, 100)
    offsets = np.arange(24) * 100
    f = {"privileged_block_0_pos": np.c_[np.tile(np.arange(100) * .01, 24), np.zeros((2400, 2))]}

    def __init__(self, path):
        pass

    def case(self, case):
        start = case["start_step"] * .01
        target = np.array([start + .25, 0., 0.])
        return dict(qpos=np.array([start, 0., 0.]), qvel=np.zeros(3)), dict(pos=target, quat=np.array([1., 0, 0, 0])), pixels(target)

    def close(self):
        pass


def test_observer_preserves_legacy_commands_and_rng(cfg):
    model = FakeModel()
    legacy_ledger, pool_ledger = Ledger(10), Ledger(10)
    baseline = Planner(Predictor(model, legacy_ledger), cfg, 42)
    z, g = np.zeros(3, np.float32), np.array([.2, .1, 0], np.float32)
    expected = np.stack([baseline.action(z, g) for _ in range(5)])
    proposer = ProposalPlanner(model, cfg, pool_ledger, 42)
    pool = proposer.propose(z, g, 10)
    np.testing.assert_array_equal(pool["inputs"]["commands"][0], expected)
    assert proposer.cem.rng.bit_generator.state == baseline.rng.bit_generator.state
    assert legacy_ledger.model["predictor_items"] == 12 * 2 * 5
    assert pool_ledger.model["predictor_items"] == 12 * 2 * 5 + 3 * 5 + len(pool["terminal_l2"])
    assert pool_ledger.used == 0
    index, _ = choose(pool, "pool-l2")
    assert pool["terminal_l2"][index] <= pool["terminal_l2"][0]


def test_prefix_features_exclude_truth_and_plan_suffix(cfg):
    pool = ProposalPlanner(FakeModel(), cfg, Ledger(0), 42).propose(np.zeros(3, np.float32), np.ones(3, np.float32), 10)
    x = features(pool["inputs"], FakeModel.stats, future=True)
    pool["full_plans"][:, 1:] = -1
    np.testing.assert_array_equal(x, features(pool["inputs"], FakeModel.stats, future=True))
    changed = copy.deepcopy(pool["inputs"])
    changed["predicted"] += 9
    np.testing.assert_array_equal(features(changed, FakeModel.stats, future=False),
                                  features(pool["inputs"], FakeModel.stats, future=False))
    assert not np.array_equal(features(changed, FakeModel.stats, future=True), x)
    with pytest.raises(ValueError, match="declared"):
        features(dict(changed, object_pos=np.ones((len(x), 3))), FakeModel.stats, future=True)
    with pytest.raises(ValueError, match="executed prefix"):
        features(dict(changed, commands=pool["full_plans"]), FakeModel.stats, future=True)


def test_edits_preserve_declared_action_channels():
    mean = np.zeros((5, 5, 5), np.float32)
    mean[0, :, 4] = np.linspace(-1, 1, 5)
    mean[:, :, 3] = .3
    edits = local_edits(mean, np.ones(5), 4, .5)
    for edit in edits:
        np.testing.assert_array_equal(edit[1:], mean[1:])
        np.testing.assert_array_equal(edit[:, :, 3], mean[:, :, 3])
    for edit in edits[:3]:
        np.testing.assert_array_equal(edit[:, :, 4], mean[:, :, 4])
    np.testing.assert_array_equal(edits[3][0, 1:, 4], mean[0, :-1, 4])


def test_realized_utility_success_dominates_failure_and_counts_prefix():
    assert utility([.2] * 50 + [.04], 50) == pytest.approx(.9)
    assert utility([.2, .03], 50) > utility([.2] * 50 + [.04], 50)
    assert 0 < utility([.2, .041], 50) < .2
    assert utility([.2, .3], 50) == 0
    with pytest.raises(ValueError, match="exceeds"):
        utility([.2] * 52, 50)


def test_sampling_is_grouped_reproducible_and_does_not_use_labels(cfg):
    a = sample_cases([1, 3, 5], np.full(6, 50), 9, cfg, np.random.default_rng(1))
    b = sample_cases([1, 3, 5], np.full(6, 50), 9, cfg, np.random.default_rng(1))
    assert a == b and len({c["case_id"] for c in a}) == 9
    assert [sum(c["episode"] == ep for c in a) for ep in (1, 3, 5)] == [3, 3, 3]
    assert all(c["warmup_steps"] % 5 == 0 for c in a)
    with pytest.raises(ValueError, match="Insufficient"):
        sample_cases([1], np.full(2, 26), 3, cfg, np.random.default_rng(1))


def test_protocol_device_portability_and_smoke_separation(cfg):
    assert signature(cfg) == signature(dict(cfg, device="cuda:1", weights="elsewhere", render_backend="egl"))
    assert signature(cfg) != signature(dict(cfg, smoke=False))
    with pytest.raises(ValueError):
        validate_config(dict(cfg, warmup_steps=[3]))
    with pytest.raises(ValueError):
        validate_config(dict(cfg, total_steps=50))


def test_collect_branches_restore_controller_and_use_own_continuation_images(cfg):
    harness = FakeHarness(dict(qpos=np.zeros(3)), dict(pos=np.array([.3, 0, 0])))
    snapshot = harness.snapshot()
    commands = np.zeros((5, 5), np.float32)
    commands[:, 0] = .2
    a, b = Ledger(10), Ledger(10)
    first = rollout_branch(harness, snapshot, commands, 10, cfg, FakeModel(), pixels([.3, 0, 0]), 99, a)
    harness.controller[:] = 999
    second = rollout_branch(harness, snapshot, commands, 10, cfg, FakeModel(), pixels([.3, 0, 0]), 99, b)
    np.testing.assert_array_equal(first["pixels"], second["pixels"])
    np.testing.assert_array_equal(first["commands"], second["commands"])
    assert a.phases["branch_prefix"] == 5 and a.phases["branch_continuation"] == 5
    assert budget({"a": a, "b": b}, harness.actual_steps)["accounted"]
    changed = commands.copy()
    changed[:, 0] = -.2
    third = rollout_branch(harness, snapshot, changed, 10, cfg, FakeModel(), pixels([.3, 0, 0]), 99, Ledger(10))
    assert not np.array_equal(first["commands"][5:], third["commands"][5:])


def test_policy_buffer_and_selected_candidate_are_executed(cfg):
    ledger = Ledger(10)
    score = lambda inputs, method: np.arange(len(inputs["remaining"]), dtype=float)
    policy = PrefixPolicy(FakeModel(), cfg, ledger, pixels([.3, 0, 0]), 42, "value", score)
    commands = [policy.action(pixels([0, 0, 0]), 10)]
    counts = copy.deepcopy(ledger.model)
    for remaining in (9, 8, 7, 6):
        commands.append(policy.action(pixels([0, 0, 0]), remaining))
    assert counts == ledger.model
    decision = policy.decisions[0]
    np.testing.assert_array_equal(commands, decision["commands"][decision["chosen"]])
    with pytest.raises(TypeError):
        policy.action(pixels([0, 0, 0]), 5, object_position=[0, 0, 0])


def test_branch_selection_has_no_truth_dependency(cfg):
    pool = ProposalPlanner(FakeModel(), cfg, Ledger(0), 42).propose(np.zeros(3, np.float32), np.ones(3, np.float32), 10)
    n = len(pool["terminal_l2"])
    score = lambda inputs, method: np.arange(n)
    ids = branch_indices(pool, 3, np.random.default_rng(9), scorer=score)
    assert len(set(ids)) == 3 and ids[0] == n - 1


def test_initial_success_does_not_plan_or_execute(cfg):
    harness = FakeHarness(dict(qpos=np.zeros(3)), dict(pos=np.zeros(3)))
    ledger = Ledger(10)
    result = trial(cfg, harness, harness.snapshot(), harness.observe().pixels,
                   pixels(np.zeros(3)), FakeModel(), None, "cem-mean", 42, ledger)
    assert result["initially_successful"] and result["first_hit_step"] == 0
    assert not ledger.model and ledger.used == harness.actual_steps == 0


def test_summary_exposes_failures_and_pairs_independent_cases():
    cases = [dict(case_id="a"), dict(case_id="b")]
    rows = []
    for c in cases:
        for method in METHODS:
            for seed in (42, 43):
                success = method == "value" and c["case_id"] == "b"
                rows.append(dict(case_id=c["case_id"], method=method, seed=seed, state="complete",
                                 success=success, penalized_first_hit_step=10 if success else 51,
                                 min_distance_m=.01, final_distance_m=.01, executed_steps=10,
                                 elapsed_seconds=1., scoring_seconds=.1))
    summary = summarize(rows, cases, [42, 43], 9)
    assert summary["methods"]["value"]["paired_delta_vs_cem_mean"]["success"] == .5
    rows.pop()
    rows[0]["state"] = "failed"
    incomplete = summarize(rows, cases, [42, 43], 9)
    assert len(incomplete["failed"]) == len(incomplete["missing"]) == 1
    assert incomplete["methods"]["value"]["success_rate"] is None
    assert incomplete["paired_complete_cases"] == []
    with pytest.raises(ValueError, match="duplicate"):
        summarize(rows + [rows[1]], cases, [42, 43], 9)


def test_value_gradients_and_preference_learning():
    torch = pytest.importorskip("torch")
    from planning_ideas.prefix_value import ValueHead, objective
    torch.set_num_threads(2)
    torch.manual_seed(9)
    head = ValueHead(np.zeros(2), np.ones(2), 16)
    x, y = torch.tensor([[0., 0.], [1., 1.]]), torch.tensor([0., 1.])
    spec = dict(tie_tolerance=.02, ranking_temperature=.1, ranking_weight=.2)
    opt = torch.optim.AdamW(head.parameters(), lr=.02)
    before = objective(head(x), y, spec)[0].item()
    for _ in range(40):
        opt.zero_grad()
        loss = objective(head(x), y, spec)[0]
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        opt.step()
    assert objective(head(x), y, spec)[0].item() < before / 4
    assert head(x)[1] > head(x)[0]


def test_cli_pipeline_and_one_refresh_with_synthetic_physics(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import cube_research.data as data
    import cube_research.simulator as sim
    import planning_ideas.prefix_data as collection
    import planning_ideas.prefix_evaluation as evaluation
    from planning_ideas.prefix_contracts import load_manifest
    from planning_ideas.prefix_training import load_groups, load_scorer
    monkeypatch.setattr(data, "Dataset", FakeDataset)
    monkeypatch.setattr(collection, "FrozenModel", FakeModel)
    monkeypatch.setattr(evaluation, "FrozenModel", FakeModel)
    monkeypatch.setattr(sim, "make_env", lambda cfg, condition, seed, initial, target: FakeHarness(initial, target))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test-only fake backbone; never a Cube checkpoint")
    (tmp_path / "config.json").write_text("{}")
    dataset = tmp_path / "dataset.h5"
    dataset.write_bytes(b"synthetic fixture")
    local = tmp_path / "local.yaml"
    local.write_text(yaml.safe_dump(dict(dataset=str(dataset), weights=str(weights), device="cpu", render_backend="auto")))
    cache = tmp_path / "goal-cache"
    cache.mkdir()
    (cache / "STATUS.json").write_text('{"state":"complete"}')
    source = dict(schema="cube-goal-set-cache-v1", inputs=dict(dataset_bytes=dataset.stat().st_size,
                  **weights_identity(dict(weights=weights))), data_config=dict(goal_offset=25),
                  splits=dict(train=list(range(20)), dev=[20, 21], test=[22, 23], donor=[]),
                  cases={s: [dict(case_id=f"{s}{ep}", episode=ep, start_step=0, goal_offset=25) for ep in eps]
                         for s, eps in (("dev", [20, 21]), ("test", [22, 23]))}, action_stats=FakeModel.stats)
    (cache / "manifest.json").write_text(json.dumps(source))
    def run(command, *args, out):
        main(["--inputs", str(local), "--smoke", command, *map(str, args), "--out", str(out)])
        assert read_json(out / "STATUS.json")["state"] == "complete"
    manifest = tmp_path / "manifest"
    run("prepare", "--goal-cache", cache, out=manifest)
    train_data, cal_data = tmp_path / "train-data", tmp_path / "cal-data"
    run("collect", "--manifest", manifest, "--split", "train", out=train_data)
    run("collect", "--manifest", manifest, "--split", "calibration", out=cal_data)
    heads = tmp_path / "heads"
    run("train", "--manifest", manifest, "--train-data", train_data, "--calibration-data", cal_data, out=heads)
    cfg = read_config(parser().parse_args(["--inputs", str(local), "--smoke", "train", "--manifest", str(manifest),
                    "--train-data", str(train_data), "--calibration-data", str(cal_data), "--out", "unused"]))
    m = load_manifest(cfg, manifest)
    retrain_cfg = copy.deepcopy(cfg)
    retrain_cfg["training"]["steps"] += 10
    assert load_manifest(retrain_cfg, manifest) == m
    assert len(load_groups(retrain_cfg, m, manifest, [train_data], "train")[0]) > 0
    from research_common import file_hash
    scorer = load_scorer(cfg, heads, file_hash(manifest / "manifest.json"))
    assert scorer.round == 0
    assert all(not p.requires_grad for h in scorer.heads.values() for p in h.parameters())
    with pytest.raises(ValueError, match="checkpoint"):
        load_scorer(retrain_cfg, heads, file_hash(manifest / "manifest.json"))
    with pytest.raises(ValueError, match="Duplicate state"):
        load_groups(cfg, m, manifest, [train_data, train_data], "train")
    with pytest.raises(ValueError, match="split"):
        load_groups(cfg, m, manifest, [cal_data], "train")
    refresh = tmp_path / "refresh"
    run("collect", "--manifest", manifest, "--split", "train", "--round", 1, "--trained", heads, "--limit", 2, out=refresh)
    refined = tmp_path / "refined"
    run("train", "--manifest", manifest, "--train-data", train_data, refresh, "--calibration-data", cal_data, out=refined)
    dev = tmp_path / "dev"
    run("evaluate", "--manifest", manifest, "--trained", refined, out=dev)
    summary = read_json(dev / "summary.json")
    assert summary["all"]["expected_trials"] == summary["all"]["completed_trials"] == 4
    assert len(summary["all"]["paired_complete_cases"]) == 1
    run("analyze", "--run", dev, out=tmp_path / "summary")
    with pytest.raises(ValueError, match="one refresh"):
        run("collect", "--manifest", manifest, "--split", "train", "--round", 1, "--trained", refined, out=tmp_path / "bad-refresh")
    assert read_json(tmp_path / "bad-refresh" / "STATUS.json")["state"] == "failed"
    # Tampered labels must not silently change training evidence.
    state = json.loads((train_data / "states.jsonl").read_text().splitlines()[0])["state_id"]
    (train_data / "states" / state / "labels.json").write_text('{"utility":[1,1,1]}')
    with pytest.raises(ValueError, match="Modified labels"):
        load_groups(cfg, m, manifest, [train_data], "train")
