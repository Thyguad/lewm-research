"""Small CPU float32 value heads. The LeWM backbone is not an optimizer member."""
import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, mean, scale, hidden):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        dim = len(mean)
        self.network = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.network((x.float() - self.mean) / self.scale).squeeze(-1).sigmoid()


def objective(prediction, target, spec):
    # All candidates in a group share a physical state and the continuation seed.
    regression = nn.functional.smooth_l1_loss(prediction, target, beta=.1)
    difference = target[:, None] - target[None, :]
    preferred = difference > spec["tie_tolerance"]
    ranking = prediction.sum() * 0
    if preferred.any():
        logits = (prediction[:, None] - prediction[None, :]) / spec["ranking_temperature"]
        ranking = nn.functional.softplus(-logits[preferred]).mean()
    return regression + spec["ranking_weight"] * ranking, regression, ranking
