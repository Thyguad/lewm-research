import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cube_research.model import Planner
from cube_research.protocol import Ledger
from planning_ideas.goal_eval import RGBPolicy, summarize
from planning_ideas.goal_pairs import matched_pairs, split_episodes
from planning_ideas.goal_set import (METHODS, GoalSetHead, bind_goal, bounded_distance,
                                     object_position_distance, projector_gap, training_loss)


@pytest.fixture
def cfg():
    return yaml.safe_load((ROOT / "config/research/ideas/goal-set.yaml").read_text())


def test_bounded_projection_matches_independent_constrained_solver():
    from scipy.optimize import minimize

    head = GoalSetHead(8, 3, 16)
    rng = np.random.default_rng(9)
    goal, candidate = torch.from_numpy(rng.normal(size=(2, 8)).astype(np.float32))
    q = head(goal).detach()
    torch.testing.assert_close(q.T @ q, torch.eye(3), atol=1e-6, rtol=1e-6)
    d = candidate - goal
    for radius in (0., .4, 5.):
        actual = bounded_distance(candidate[None], goal, q, radius, 2.5).item()
        if radius == 0.:
            assert actual == pytest.approx(d.square().sum().item() / 2.5)
            continue
        q_np, d_np = q.numpy().astype(float), d.numpy().astype(float)
        result = minimize(lambda u: np.sum((d_np - q_np @ u) ** 2) / 2.5, np.zeros(3),
                          constraints=[dict(type="ineq", fun=lambda u: radius**2 - u @ u)],
                          method="SLSQP", options={"ftol": 1e-11})
        assert result.success
        assert actual == pytest.approx(result.fun, abs=2e-6)
    torch.testing.assert_close(projector_gap(q, -q), torch.tensor(0.), atol=2e-6, rtol=0.)


def test_latent_gradients_stop_at_frozen_backbone_and_sets_do_not_collapse(cfg):
    torch.manual_seed(4)
    backbone = torch.nn.Linear(8, 8).requires_grad_(False)
    with torch.inference_mode():
        z_np = backbone(torch.randn(8, 8)).numpy().copy()
    a, b = torch.from_numpy(z_np).chunk(2)
    head = GoalSetHead(8, 2, 16)
    cal = dict(radius=.1, scale2=1.)
    loss = training_loss("goal-set", head, a, b, None, torch.tensor([True, True, False, False]),
                         cal, cfg["training"])
    loss.backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert sum(float(p.grad.norm()) for p in head.parameters()) > 0
    score = bind_goal("goal-set", head, b[0].numpy(), cal)
    g = b[0].numpy()
    assert score(np.stack((g, g + 5)), None)[0] == 0.
    assert score(np.stack((g, g + 5)), None)[1] > 150.


def test_group_splits_and_paired_labels(cfg):
    lengths = np.arange(201, 501)
    s, cases = split_episodes(lengths, cfg)
    assert (s, cases) == split_episodes(lengths, cfg)
    joined = [ep for group in s.values() for ep in group]
    assert len(joined) == len(set(joined))
    for key, values in cases.items():
        assert {c["episode"] for c in values} <= set(s[key])
        assert all(c["start_step"] + c["goal_offset"] < lengths[c["episode"]] for c in values)
    p = np.array([[0, 0, 0], [0, 0, 0], [.04, 0, 0], [.075, 0, 0]])
    j = np.array([[0., 0], [.8, .8], [0., 0.], [0., 0.]])
    pairs, hard = matched_pairs(p, j, np.zeros((4, 1)), cfg["pairing"], np.random.default_rng(1))
    assert pairs.tolist() == [[1, 3, 0]]
    assert hard == 1


def test_position_cost_is_isotropic_in_metres():
    spec = dict(position_axis_scale=[.1, .2, .04], position_distance_unit=.06)
    axis_delta = torch.diag(torch.tensor([.6, .3, 1.5]))
    torch.testing.assert_close(object_position_distance(axis_delta, torch.zeros(3), spec), torch.ones(3))


class FakeFrozen:
    device = "cpu"
    stats = {"mean": [0.] * 5, "scale": [.5] * 5}

    def encode(self, frames, ledger):
        assert len(frames) == 1 and frames[0].shape == (2, 2, 3)
        ledger.count("encoder")
        return np.full((1, 3), frames[0].mean(), np.float32)

    def predict(self, z, commands, ledger):
        assert commands.shape[-2:] == (5, 5)
        ledger.count("predictor", len(z))
        return z[:, -1] + commands[:, -1, 0, :3] / 4


def test_policy_only_needs_rgb_and_reuses_execution_prefix(cfg):
    c = {**cfg, "cem_samples": 12, "cem_iterations": 2, "cem_elites": 3}
    model = FakeFrozen()
    ledger = Ledger(10)
    rgb, goal_rgb = np.zeros((2, 2, 3)), np.ones((2, 2, 3))
    cal = dict(scale2=1.)
    policy = RGBPolicy(model, ledger, goal_rgb, "l2", None, cal, c, 42)
    default = Planner(policy.planner.adapter, c, 42)
    np.testing.assert_array_equal(policy.action(rgb), default.action(np.zeros(3, np.float32), policy.goal))
    ledger.model.clear()  # Isolate the next executed steps from baseline calls above.
    for _ in range(4):
        policy.action(rgb)
    assert not ledger.model
    policy.action(rgb)
    assert ledger.model["encoder_items"] == 1
    assert ledger.model["predictor_items"] == 12 * 2 * 5
    with pytest.raises(TypeError):
        policy.action(rgb, object_position=np.zeros(3))


def test_summary_weights_cases_not_seeds():
    rows = []
    for method in METHODS:
        for case, seeds in (("a", (42, 43, 44)), ("b", (42,))):
            for seed in seeds:
                success = method != "l2" and case == "b"
                rows.append(dict(case_id=case, seed=seed, method=method, success=success,
                                 first_hit_step=3 if success else None,
                                 penalized_first_hit_step=3 if success else 51, min_distance_m=.02,
                                 final_distance_m=.1, elapsed_seconds=1., scoring_seconds=.1,
                                 ledger={"model": {}}))
    score = summarize(rows, 99)["methods"]["goal-set"]
    assert score["means"]["success"] == .5
    assert score["delta_vs_l2"]["success"] == .5
    assert score["means"]["penalized_first_hit_step"] == 27
    assert score["first_hit_step_given_success"] == 3
