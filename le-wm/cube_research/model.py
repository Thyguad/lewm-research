"""Frozen LeWM bridge and bounded CEM over executable Cube commands."""

from __future__ import annotations
import time
import numpy as np
from research_common import sync_device


class FrozenModel:
    def __init__(self, cfg, action_stats):
        from research_runtime import load_model
        from research_device import resolve_device
        from .protocol import policy_config

        self.cfg, self.stats = policy_config(cfg), action_stats
        self.device = resolve_device(cfg["device"])
        self.base, self.identity = load_model(cfg["weights"], None, self.device)
        self.dim = 192
        import json
        from pathlib import Path

        spec = json.loads((Path(cfg["weights"]).parent / "config.json").read_text())
        if spec["action_encoder"]["input_dim"] != cfg["action_dim"] * cfg["action_block"]:
            raise ValueError("Checkpoint action input does not match Cube 5x5")
        if spec["predictor"]["input_dim"] != self.dim:
            raise ValueError("Expected official Cube 192-dimensional latent")
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


class Planner:
    def __init__(self, adapter, cfg, seed):
        self.adapter, self.cfg = adapter, cfg
        self.rng = np.random.default_rng(seed)
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def action(self, embedding, goal):
        if self.buffer:
            return self.buffer.pop(0)
        c, ledger = self.cfg, self.adapter.ledger
        started = time.perf_counter()
        shape = (c["horizon"], c["action_block"], c["action_dim"])
        mean, std = np.zeros(shape), np.ones(shape)
        for _ in range(c["cem_iterations"]):
            candidates = np.clip(
                self.rng.normal(mean, std, (c["cem_samples"], *shape)), -1, 1
            ).astype(np.float32)
            z = np.repeat(embedding[None, None], len(candidates), axis=0)
            for h in range(c["horizon"]):
                # Preserve the reference predictor's <=3 block autoregressive context.
                a = candidates[:, max(0, h - 2) : h + 1]
                pred = self.adapter.predict(z[:, -3:], a)
                z = np.concatenate((z, pred[:, None]), axis=1)
            costs = np.sum((z[:, -1] - goal) ** 2, axis=1)
            elite = candidates[np.argsort(costs)[: c["cem_elites"]]]
            mean, std = elite.mean(0), np.maximum(elite.std(0), 0.03)
            ledger.count("cem_score", len(candidates))
        # Execute best evaluated sequence; mean was not itself evaluated.
        plan = candidates[int(np.argmin(costs))][: c["execute_blocks"]].reshape(-1, 5)
        self.buffer = [v.copy() for v in plan]
        sync_device(self.adapter.model.device)
        ledger.planning_seconds += time.perf_counter() - started
        ledger.count("plan")
        return self.buffer.pop(0)
