"""Explicit local dataset access. Truth is used only by case initialization."""

from pathlib import Path
import numpy as np
from research_common import file_hash, write_json
from .protocol import grouped_cases, digest, validate_manifest


def local_inputs(cfg):
    for key in ("dataset", "weights"):
        if not cfg.get(key) or not Path(cfg[key]).is_file():
            raise FileNotFoundError(
                f"{key}: set an existing local path; automatic downloads are disabled"
            )
    if not (Path(cfg["weights"]).parent / "config.json").is_file():
        raise FileNotFoundError("config.json must be next to weights.pt")


class Dataset:
    def __init__(self, path):
        import h5py
        import hdf5plugin  # noqa: F401

        self.path = str(Path(path).resolve())
        self.f = h5py.File(path, "r")
        self.keys = list(self.f)
        required = (
            "pixels",
            "action",
            "qpos",
            "qvel",
            "ep_len",
            "ep_offset",
            "privileged_block_0_pos",
            "privileged_block_0_quat",
        )
        missing = set(required) - set(self.keys)
        if missing:
            raise ValueError(
                f"Dataset schema mismatch, missing {sorted(missing)}; keys={self.keys}"
            )
        if self.f["action"].shape[-1] != 5 or self.f["pixels"].shape[1:] != (224, 224, 3):
            raise ValueError("Dataset must be cube-single RGB224 and 5D action")
        self.lengths = self.f["ep_len"][:]
        self.offsets = self.f["ep_offset"][:]

    def stats(self):
        # Streaming population statistics, same StandardScaler convention as
        # official evaluation. These are frozen offline pretraining-data stats.
        count = 0
        total = np.zeros(5)
        square = np.zeros(5)
        for start in range(0, len(self.f["action"]), 100000):
            a = self.f["action"][start : start + 100000].astype(np.float64)
            a = a[np.isfinite(a).all(axis=1)]
            count += len(a)
            total += a.sum(0)
            square += (a * a).sum(0)
        if not count:
            raise ValueError("No finite offline actions")
        mean = total / count
        std = np.sqrt(np.maximum(square / count - mean * mean, 0))
        std[std < 1e-12] = 1
        return dict(mean=mean.tolist(), scale=std.tolist(), rows=count)

    def case(self, case):
        ep, step, offset = case["episode"], case["start_step"], case["goal_offset"]
        if step + offset >= self.lengths[ep]:
            raise ValueError("Goal outside episode")
        row = int(self.offsets[ep] + step)
        goal = row + offset
        return (
            dict(qpos=self.f["qpos"][row], qvel=self.f["qvel"][row]),
            dict(
                pos=self.f["privileged_block_0_pos"][goal],
                quat=self.f["privileged_block_0_quat"][goal],
            ),
            self.f["pixels"][goal],
        )

    def close(self):
        self.f.close()


def make_manifest(cfg, out):
    if not cfg.get("dataset") or not Path(cfg["dataset"]).is_file():
        raise FileNotFoundError("Prepare local Cube HDF5 and set dataset first")
    d = Dataset(cfg["dataset"])
    try:
        rows = grouped_cases(
            range(len(d.lengths)), d.lengths, cfg["goal_offset"], cfg["split_seed"]
        )
        payload = dict(
            schema="cube-cases-v1",
            cases=rows,
            cases_sha256=digest(rows),
            dataset_sha256=file_hash(d.path),
            stats=d.stats(),
            split_seed=cfg["split_seed"],
            split_unit="whole source episode; one start-goal case per episode",
            dataset_source=cfg["dataset_source"],
            protocol=cfg["protocol"],
        )
        validate_manifest(payload)
        write_json(out, payload)
    finally:
        d.close()
