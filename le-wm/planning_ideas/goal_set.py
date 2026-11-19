"""Terminal costs. Training heads and QR run in float32 on CPU."""

import numpy as np
import torch
from torch import nn

METHODS = ("l2", "scalar", "position", "global-set", "goal-set")


def mlp(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
        nn.SiLU(), nn.Linear(hidden, outputs),
    )


class GoalSetHead(nn.Module):
    def __init__(self, dim, rank, hidden, *, conditional=True):
        super().__init__()
        if not 0 < rank < dim:
            raise ValueError("Expected 0 < rank < latent dimension")
        self.dim, self.rank = dim, rank
        basis = torch.linalg.qr(torch.randn(dim, rank), mode="reduced").Q
        self.register_buffer("initial_basis", basis)
        if conditional:
            self.network = mlp(dim, hidden, dim * rank)
            nn.init.normal_(self.network[-1].weight, std=1e-3)
            nn.init.normal_(self.network[-1].bias, std=1e-3)
        else:
            self.network = None
            self.delta = nn.Parameter(torch.randn(dim, rank) * 1e-3)

    def forward(self, goal):
        if goal.shape[-1] != self.dim:
            raise ValueError("Unexpected goal latent dimension")
        delta = (
            self.delta.expand(*goal.shape[:-1], self.dim, self.rank)
            if self.network is None
            else self.network(goal.float()).reshape(*goal.shape[:-1], self.dim, self.rank)
        )
        return torch.linalg.qr(self.initial_basis + delta.float(), mode="reduced").Q


def squared_l2(a, b):
    return (a - b).square().sum(dim=-1)


def bounded_distance(candidate, goal, basis, radius, scale2):
    """Squared distance to a radius-bounded affine subspace, in latent units."""
    delta = candidate - goal
    coords = (basis.transpose(-1, -2) @ delta.unsqueeze(-1)).squeeze(-1)
    norm = coords.norm(dim=-1, keepdim=True)
    clipped = coords * (radius / norm.clamp_min(1e-12)).clamp(max=1)
    residual = delta - (basis @ clipped.unsqueeze(-1)).squeeze(-1)
    return residual.square().sum(dim=-1) / scale2


def projector_gap(a, b):
    # ||aa^T - bb^T||_F^2 for two orthonormal bases of equal rank.
    return (2 * (a.shape[-1] - (a.transpose(-1, -2) @ b).square().sum((-2, -1)))).clamp_min(0)


def build_head(method, dim, spec):
    if method == "l2":
        return None
    if method in ("global-set", "goal-set"):
        return GoalSetHead(dim, spec["rank"], spec["hidden"], conditional=method == "goal-set")
    if method == "position":
        return mlp(dim, spec["hidden"], 3)
    if method == "scalar":
        return mlp(2 * dim, spec["hidden"], 1)
    raise ValueError(f"Unknown method: {method}")


def scalar_distance(head, a, b):
    raw = head(torch.cat((b.expand_as(a), a - b), dim=-1)).squeeze(-1)
    return nn.functional.softplus(raw)


def object_position_distance(a, b, calibration):
    # Heads predict standardized coordinates. Undo axis scaling before measuring
    # geometric distance; the negative-pair threshold just sets the cost unit.
    axis_scale = a.new_tensor(calibration["position_axis_scale"])
    return (((a - b) * axis_scale) / calibration["position_distance_unit"]).square().sum(-1)


def costs(method, head, a, b, calibration):
    if method == "l2":
        return squared_l2(a, b) / calibration["scale2"]
    if method == "position":
        return object_position_distance(head(a), head(b), calibration)
    if method == "scalar":
        return scalar_distance(head, a, b)
    return bounded_distance(a, b, head(b), calibration["radius"], calibration["scale2"])


def training_loss(method, head, a, b, positions, is_positive, calibration, spec):
    if method == "position":
        # Coordinates are standardized using only the matched training frames.
        mu, sigma = positions[2:]
        predicted = torch.cat((head(a), head(b)))
        target = (torch.cat(positions[:2]) - mu) / sigma
        return nn.functional.mse_loss(predicted, target)

    basis = head(b) if method.endswith("-set") else None
    values = (
        bounded_distance(a, b, basis, calibration["radius"], calibration["scale2"])
        if basis is not None else scalar_distance(head, a, b)
    )
    loss = values[is_positive].mean() + spec["negative_weight"] * (
        spec["margin"] - values[~is_positive]
    ).clamp_min(0).mean()
    if method == "goal-set":
        loss = loss + spec["projector_weight"] * projector_gap(
            basis[is_positive], head(a[is_positive])
        ).mean()
    return loss


@torch.inference_mode()
def bind_goal(method, head, goal, calibration):
    """Cache representations of one RGB-encoded goal for all CEM plans."""
    if method == "l2":
        def l2_score(candidates, _goal):
            # Preserve exactly the ranking of the legacy NumPy reduction.
            return np.sum((candidates - goal) ** 2, axis=1).astype(np.float64) / calibration["scale2"]
        return l2_score
    target = torch.as_tensor(goal, dtype=torch.float32)
    basis = head(target) if method.endswith("-set") else None
    position = head(target) if method == "position" else None

    @torch.inference_mode()
    def score(candidates, _goal):
        a = torch.as_tensor(candidates, dtype=torch.float32)
        if basis is not None:
            value = bounded_distance(a, target, basis, calibration["radius"], calibration["scale2"])
        elif position is not None:
            value = object_position_distance(head(a), position, calibration)
        else:
            value = costs(method, head, a, target, calibration)
        return value.numpy()

    return score
