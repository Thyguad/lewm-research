"""Full-history diagnostic reference and task-outcome routing, fitted offline."""

from __future__ import annotations
import numpy as np
from .protocol import CONDITIONS, METHODS


def features(embeddings, history, goal, total_steps):
    """Padded full sequence, no truncation/hidden frames. Raw RGB spatial pools
    supplement frozen LeWM features; all declared proprio is included.
    """
    a = history.arrays()
    n = len(a["commands"])
    if n > total_steps:
        raise ValueError("History exceeds locked protocol")
    # RGB 14x14 spatial means preserve a second visual diagnostic input.
    rgb = a["pixels"].reshape(n + 1, 14, 16, 14, 16, 3).mean((2, 4)).reshape(n + 1, -1) / 255
    actions = np.concatenate((a["commands"], np.zeros((1, 5))), axis=0)
    values = np.concatenate((embeddings, rgb, a["sensors"], actions, np.ones((n + 1, 1))), axis=1)
    pad = np.zeros((total_steps + 1, values.shape[1]), dtype=np.float32)
    pad[: n + 1] = values
    return np.concatenate((pad.ravel(), goal.ravel())).astype(np.float64)


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Diagnostic:
    """Dual ridge avoids a huge feature x feature solve for short datasets."""

    def fit(self, x, labels, losses, groups, seed=0):
        groups = np.asarray(groups)
        unique = np.unique(groups)
        if len(unique) < 5:
            raise ValueError("Need >=5 independent training episodes, not planning seeds")
        rng = np.random.default_rng(seed)
        rng.shuffle(unique)
        cal_groups = set(unique[: max(1, len(unique) // 5)])
        cal = np.array([g in cal_groups for g in groups])
        train = ~cal
        if set(np.asarray(labels)[train]) != set(range(3)) or set(np.asarray(labels)[cal]) != set(
            range(3)
        ):
            raise ValueError("Training and calibration groups both need N/I/D")
        self.mean = x[train].mean(0)
        self.scale = x[train].std(0)
        self.scale[self.scale < 1e-5] = 1
        xx = (x - self.mean) / self.scale
        target = np.concatenate((np.eye(3)[labels], losses), axis=1)
        self.target_mean = target[train].mean(0)
        self.train_x = xx[train]
        self.alpha = np.linalg.solve(
            self.train_x @ self.train_x.T + len(self.train_x) * np.eye(train.sum()),
            target[train] - self.target_mean,
        )
        pred = xx[cal] @ self.train_x.T @ self.alpha + self.target_mean
        candidates = np.geomspace(0.1, 10, 31)
        y = np.asarray(labels)[cal]
        scores = [
            -np.log(softmax(pred[:, :3] / t)[np.arange(len(y)), y].clip(1e-12)).mean()
            for t in candidates
        ]
        self.temperature = float(candidates[int(np.argmin(scores))])
        return dict(
            training_groups=sorted(set(groups[train])),
            calibration_groups=sorted(cal_groups),
            training_rows=int(train.sum()),
            calibration_rows=int(cal.sum()),
            extra_data="Recorded training-only histories and offline paired recovery outcomes",
            class_order=CONDITIONS,
            method_order=METHODS,
        )

    def predict(self, x):
        # No label, condition, evaluation channel, or simulator argument.
        xx = (np.asarray(x) - self.mean) / self.scale
        y = xx @ self.train_x.T @ self.alpha + self.target_mean
        p = softmax(y[:3] / self.temperature)
        return dict(
            probabilities=p.tolist(),
            predicted_condition=CONDITIONS[int(np.argmax(p))],
            method=METHODS[int(np.argmin(y[3:]))],
            predicted_losses=y[3:].tolist(),
        )

    def save(self, path):
        np.savez_compressed(path, **vars(self))

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as f:
            obj.__dict__.update({k: f[k] for k in f.files})
        obj.temperature = float(obj.temperature)
        return obj
