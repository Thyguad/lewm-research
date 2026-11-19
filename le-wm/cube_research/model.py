"""Frozen LeWM bridge and bounded CEM over executable Cube commands."""

from __future__ import annotations
import time
import numpy as np
from research_common import sync_device


class FrozenModel:
    def __init__(self, cfg, action_stats):
        from research_runtime import load_model
        from research_device import resolve_device
        self.cfg, self.stats = {"score_batch_size": cfg["score_batch_size"]}, action_stats
        self.device = resolve_device(cfg["device"])
        self.base, self.identity = load_model(cfg["weights"], None, self.device)
        import json
        from pathlib import Path

        spec = json.loads((Path(cfg["weights"]).parent / "config.json").read_text())
        if spec["action_encoder"]["input_dim"] != cfg["action_dim"] * cfg["action_block"]:
            raise ValueError("Checkpoint action input does not match Cube 5x5")
        self.dim = int(spec["predictor"]["input_dim"])
        if self.dim <= 0:
            raise ValueError("Invalid predictor latent dimension")
        self.identity.update(
            preprocessing="ImageNet normalize then resize 224; dataset StandardScaler action",
            action_stats=action_stats,
        )

    def encode(self, frames, ledger):
        import torch
        from research_runtime import image_transform
        from torchvision import tv_tensors

        result = []
        for start in range(0, len(frames), self.cfg["score_batch_size"]):
            part = np.asarray(frames[start : start + self.cfg["score_batch_size"]])
            x = torch.from_numpy(part.copy()).permute(0, 3, 1, 2)
            x = image_transform(224)(tv_tensors.Image(x)).unsqueeze(1).to(self.device)
            with torch.inference_mode():
                emb = self.base.encode({"pixels": x})["emb"][:, -1]
            if emb.shape != (len(part), self.dim):
                raise ValueError("Encoder output does not match predictor latent dimension")
            result.append(emb.cpu().numpy())
            ledger.count("encoder", len(part))
        return np.concatenate(result)

    def predict(self, z, effective_commands, ledger):
        import torch

        result = []
        normalized = (effective_commands - np.asarray(self.stats["mean"])) / np.asarray(
            self.stats["scale"]
        )
        actions = normalized.reshape(*normalized.shape[:2], -1)
        for start in range(0, len(z), self.cfg["score_batch_size"]):
            end = start + self.cfg["score_batch_size"]
            tz = torch.as_tensor(z[start:end], dtype=torch.float32, device=self.device)
            ta = torch.as_tensor(actions[start:end], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                pred = self.base.predict(tz, self.base.action_encoder(ta))[:, -1]
            result.append(pred.cpu().numpy())
            ledger.count("predictor", len(tz))
        return np.concatenate(result)


def terminal_embedding(adapter, embedding, candidates):
    """Roll out one shared initial embedding against a batch of action blocks."""
    z = np.repeat(embedding[None, None], len(candidates), axis=0)
    for h in range(candidates.shape[1]):
        # Match the predictor's at most three blocks of autoregressive context.
        a = candidates[:, max(0, h - 2) : h + 1]
        pred = adapter.predict(z[:, -3:], a)
        z = np.concatenate((z, pred[:, None]), axis=1)
    return z[:, -1]


class Planner:
    def __init__(self, adapter, cfg, seed, *, terminal_cost=None):
        self.adapter, self.cfg = adapter, cfg
        self.rng = np.random.default_rng(seed)
        self.buffer = []
        self.terminal_cost = terminal_cost
        self.scoring_seconds = 0.0

    def reset_buffer(self):
        self.buffer = []

    def action(self, embedding, goal):
        if self.buffer:
            return self.buffer.pop(0)
        c, ledger = self.cfg, self.adapter.ledger
        started = time.perf_counter()
        shape = (c["horizon"], c["action_block"], c["action_dim"])
        # CEM is defined in the model's normalized action space. Sampling raw
        # environment commands and normalizing afterward produces values far
        # outside the training distribution because the dataset scales are
        # much smaller than the executable [-1, 1] range.
        mean, std = np.zeros(shape), np.ones(shape)
        stats = getattr(self.adapter.model, "stats", None) or {}
        action_mean = np.asarray(stats.get("mean", np.zeros(5)))
        action_scale = np.asarray(stats.get("scale", np.ones(5)))
        for _ in range(c["cem_iterations"]):
            normalized = self.rng.normal(
                mean, std, (c["cem_samples"], *shape)
            ).astype(np.float32)
            # Keep the current distribution mean as an evaluated candidate,
            # matching the reference CEM and preventing a regression to a
            # random final sample when the elite distribution has converged.
            normalized[0] = mean
            candidates = np.clip(
                normalized * action_scale + action_mean, -1, 1
            ).astype(np.float32)
            terminal = terminal_embedding(self.adapter, embedding, candidates)
            score_started = time.perf_counter()
            costs = (
                np.sum((terminal - goal) ** 2, axis=1)
                if self.terminal_cost is None
                else self.terminal_cost(terminal, goal)
            )
            self.scoring_seconds += time.perf_counter() - score_started
            elite = normalized[np.argsort(costs)[: c["cem_elites"]]]
            mean, std = elite.mean(0), np.maximum(elite.std(0), 0.03)
            ledger.count("cem_score", len(candidates))
        # Execute the final CEM mean, as in the reference solver. A single
        # sampled elite can contain a one-step discontinuity that is unsafe
        # for contact-rich manipulation even when its predicted cost is low.
        plan = np.clip(
            mean[: c["execute_blocks"]] * action_scale + action_mean, -1, 1
        ).astype(np.float32).reshape(-1, 5)
        self.buffer = [v.copy() for v in plan]
        sync_device(self.adapter.model.device)
        ledger.planning_seconds += time.perf_counter() - started
        ledger.count("plan")
        return self.buffer.pop(0)
