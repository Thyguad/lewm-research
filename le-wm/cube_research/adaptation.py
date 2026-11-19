"""Per-episode adaptations with frozen backbone and explicit training costs.

R-D is a ridge-fitted low-rank latent/action residual. R-U uses variable
projection: fit that residual for each candidate interface gain, then minimize
the joint regularized objective. No condition labels enter this module.
"""

from __future__ import annotations
import time
import numpy as np
from .protocol import METHODS, transform_command

GAIN_BOUNDS = (0.25, 4.0)


class Adapter:
    def __init__(self, model, method, cfg, ledger):
        if method not in METHODS:
            raise ValueError(method)
        self.model, self.method, self.cfg, self.ledger = model, method, cfg, ledger
        self.xyz = cfg.get("interface_parameterization", "scalar") == "xyz"
        self.gain = [1.0] * 3 if self.xyz else 1.0
        self.objective_evaluations = 0
        self.projection = np.random.default_rng(cfg["adapt_seed"]).normal(
            size=(model.dim, cfg["residual_rank"])
        ) / np.sqrt(model.dim)
        self.weights = np.zeros((cfg["residual_rank"] + 6, model.dim), dtype=np.float64)
        self.trace = []

    def features(self, z, actions):
        return np.concatenate(
            (z[:, -1] @ self.projection, actions[:, -1].mean(axis=-2), np.ones((len(z), 1))), axis=1
        )

    def predict(self, z, commands):
        effective = transform_command(commands, self.gain)
        pred = self.model.predict(z, effective, self.ledger)
        if self.method in ("R-D", "R-U"):
            pred = pred + self.features(z, effective) @ self.weights
        return pred.astype(np.float32)

    def fit(self, embeddings, commands):
        started = time.perf_counter()
        n, block = len(commands), self.cfg["action_block"]
        if self.method == "Frozen":
            return
        if n < block:
            raise ValueError("Insufficient history: cannot fit a complete model transition")
        # Every allowed transition is used, including overlapping windows. They
        # are fitting rows, never independent statistical cases.
        z = np.asarray(embeddings[:-block])[:, None, :]
        a = np.stack([commands[t : t + block] for t in range(n - block + 1)])[:, None]
        y = np.asarray(embeddings[block:])
        reg = self.cfg["adapt_regularization"]
        self.training_rows = len(y)

        def objective(log_gain):
            self.objective_evaluations += 1
            gain = np.exp(log_gain) if self.xyz else float(np.exp(log_gain))
            effective = transform_command(a, gain)
            p = self.model.predict(z, effective, self.ledger).astype(np.float64)
            w = np.zeros_like(self.weights)
            if self.method in ("R-D", "R-U"):
                x = self.features(z, effective)
                # mean squared latent error + lambda * sum(W^2) / latent_dim
                w = np.linalg.solve(x.T @ x + len(x) * reg * np.eye(x.shape[1]), x.T @ (y - p))
                p += x @ w
            loss = float(
                np.mean((p - y) ** 2) + reg * np.sum(w * w) / self.model.dim
                + reg * (np.sum(log_gain**2) if self.xyz else log_gain**2)
            )
            if not np.isfinite(loss):
                raise FloatingPointError("Non-finite adaptation objective")
            return loss, w

        log_gain = np.zeros(3) if self.xyz else 0.0
        loss, weights = objective(log_gain)
        steps = 1 if self.method == "R-D" else self.cfg["adapt_steps"]
        radius = self.cfg["adapt_search_radius"]
        for step in range(steps):
            if self.method != "R-D":
                choices = [(loss, log_gain, weights)]
                axes = np.eye(3) if self.xyz else (1,)
                for axis in axes:
                    for direction in (-1, 1):
                        candidate = np.clip(log_gain + direction * radius * axis, *np.log(GAIN_BOUNDS))
                        if not self.xyz:
                            candidate = float(candidate)
                        value, w = objective(candidate)
                        choices.append((value, candidate, w))
                value, gain_candidate, weights = min(choices, key=lambda x: x[0])
                if np.array_equal(gain_candidate, log_gain):
                    radius *= 0.5
                log_gain, loss = gain_candidate, value
            self.gain = np.exp(log_gain).tolist() if self.xyz else float(np.exp(log_gain))
            self.weights = weights
            self.ledger.updates += 1
            self.trace.append(
                dict(
                    step=step + 1,
                    loss=loss,
                    gain=self.gain,
                    residual_norm=float(np.linalg.norm(weights)),
                )
            )
        self.ledger.adaptation_seconds += time.perf_counter() - started

    def record(self):
        return dict(
            method=self.method,
            gain=self.gain,
            residual=self.weights.tolist(),
            projection_seed=self.cfg["adapt_seed"],
            parameter_count=(
                int(self.method in ("R-I", "R-U")) * (3 if self.xyz else 1)
                + (self.weights.size if self.method in ("R-D", "R-U") else 0)
            ),
            updated_backbone_parameters=0,
            extra_training_data=0,
            training_rows=getattr(self, "training_rows", 0),
            updates=self.trace,
            initialization="gain=1; residual=0; fixed seeded projection; no persistent optimizer",
            interface_parameterization="xyz" if self.xyz else "scalar",
            interface_bounds=GAIN_BOUNDS,
            objective_evaluations=self.objective_evaluations,
            regularization=self.cfg["adapt_regularization"],
            initial_search_radius=self.cfg["adapt_search_radius"],
            update_definition="one accepted-or-retained search iterate; R-D one ridge fit",
        )
