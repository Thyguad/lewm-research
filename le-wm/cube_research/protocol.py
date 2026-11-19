"""Information boundary, protocol validation, case splits, and interaction ledger."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONDITIONS = ("N", "I", "D")
METHODS = ("Frozen", "R-I", "R-D", "R-U")
STRATEGIES = ("passive", "continue", "fixed", "random")
SENSORS = (
    "proprio/joint_pos",
    "proprio/joint_vel",
    "proprio/effector_pos",
    "proprio/effector_yaw",
    "proprio/gripper_opening",
    "proprio/gripper_vel",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_config(path):
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(c):
    if c["protocol"] != "cube-single-expert-segment-v1":
        raise ValueError("Only the expert-segment protocol is implemented")
    if c["action_dim"] != 5 or c["action_block"] != 5 or c["image_size"] != 224:
        raise ValueError("Official Cube checkpoint requires 5 actions x 5 steps, RGB 224")
    if c["sensors"] not in ("rgb", "rgb_proprio"):
        raise ValueError("Sensors must be rgb or rgb_proprio (no contact force)")
    for k in (
        "total_steps",
        "history_steps",
        "goal_offset",
        "horizon",
        "execute_blocks",
        "cem_samples",
        "cem_iterations",
        "cem_elites",
        "score_batch_size",
        "adapt_steps",
        "residual_rank",
    ):
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError(f"{k} must be a positive integer")
    if c["execute_blocks"] > c["horizon"] or c["cem_elites"] > c["cem_samples"]:
        raise ValueError("Invalid planning dimensions")
    if c["history_steps"] < c["action_block"]:
        raise ValueError("History must contain at least one complete action block")
    if any(
        type(b) is not int or b < 0 or b + c["history_steps"] >= c["total_steps"]
        for b in c["extra_budgets"]
    ):
        raise ValueError("Each extra budget must leave recovery steps")
    if c["fixed_probe"] not in ("x_cycle", "y_cycle", "lift_cycle"):
        raise ValueError("Fixed probe must come from the locked small action library")
    if not 0 < c["probe_amplitude"] <= 0.3:
        raise ValueError("Probe amplitude must be in (0, .3]")
    for key in ("interface_scale", "friction_scale", "adapt_search_radius", "adapt_regularization"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if len(set(c["planning_seeds"])) != len(c["planning_seeds"]) or not c["planning_seeds"]:
        raise ValueError("Planning seeds must be nonempty and unique")
    if any(type(v) is not int or v < 0 for v in c["planning_seeds"]):
        raise ValueError("Invalid planning seed")
    if c["render_backend"] not in ("auto", "egl", "osmesa", "glfw"):
        raise ValueError("Unsupported MuJoCo rendering backend")
    if c.get("interface_parameterization", "scalar") not in ("scalar", "xyz"):
        raise ValueError("Unsupported interface parameterization")


def transform_command(command, scale=1.0):
    """Command in [-1,1] -> clamp -> xyz gain -> clamp; yaw/gripper unchanged.

    The environment subsequently converts to metres/radians/gripper increments.
    Statistical dataset standardization belongs only to model input.
    """
    a = np.asarray(command, dtype=np.float64)
    gain = np.asarray(scale, dtype=np.float64)
    if (a.shape[-1:] != (5,) or not np.isfinite(a).all() or gain.shape not in ((), (3,))
            or not np.isfinite(gain).all() or np.any(gain <= 0)):
        raise ValueError("Expected finite five-dimensional actions and positive scale")
    out = np.clip(a, -1, 1).copy()
    out[..., :3] = np.clip(out[..., :3] * gain, -1, 1)
    return out


@dataclass(frozen=True)
class Observation:
    pixels: np.ndarray
    sensors: np.ndarray


def observation(pixels, info, sensor_protocol):
    # Allowlist: never copy env.info, raw state, contact, label, or perturbation.
    values = (
        np.concatenate([np.asarray(info[k]).reshape(-1) for k in SENSORS])
        if sensor_protocol == "rgb_proprio"
        else np.empty(0)
    )
    return Observation(np.array(pixels, copy=True), np.array(values, dtype=np.float32))


@dataclass
class History:
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    def append(self, command, obs):
        if len(self.observations) != len(self.actions) + 1:
            raise ValueError("History must start with an observation")
        self.actions.append(np.asarray(command, dtype=np.float32).copy())
        self.observations.append(obs)

    def arrays(self):
        return dict(
            pixels=np.stack([o.pixels for o in self.observations]),
            sensors=np.stack([o.sensors for o in self.observations]),
            commands=np.asarray(self.actions, dtype=np.float32).reshape(-1, 5),
        )

    def fingerprint(self):
        h = hashlib.sha256()
        for k, v in self.arrays().items():
            h.update(k.encode())
            h.update(str(v.shape).encode())
            h.update(v.tobytes())
        return h.hexdigest()


@dataclass
class Ledger:
    limit: int
    phases: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    updates: int = 0
    adaptation_seconds: float = 0.0
    planning_seconds: float = 0.0
    restore_checks: int = 0
    offline_replay_steps: int = 0

    @property
    def used(self):
        return sum(self.phases.values())

    def step(self, phase):
        if self.used >= self.limit:
            raise RuntimeError("Interaction budget exhausted")
        self.phases[phase] = self.phases.get(phase, 0) + 1

    def count(self, name, batch=1):
        self.model[name + "_calls"] = self.model.get(name + "_calls", 0) + 1
        self.model[name + "_items"] = self.model.get(name + "_items", 0) + int(batch)


def grouped_cases(episodes, lengths, offset, seed, fractions=(0.2, 0.6, 0.2)):
    """One case per complete source episode. No adjacent-frame leakage."""
    rng = np.random.default_rng(seed)
    eligible = [(int(e), int(n)) for e, n in zip(episodes, lengths) if n > offset + 1]
    if len(eligible) < 5:
        raise ValueError("Need >=5 independent episodes for dev/train/test")
    rng.shuffle(eligible)
    ndev = max(1, int(len(eligible) * fractions[0]))
    ntrain = max(1, int(len(eligible) * fractions[1]))
    rows = []
    for i, (ep, n) in enumerate(eligible):
        start = int(rng.integers(0, n - offset))
        rows.append(
            dict(
                case_id=digest([ep, start, offset])[:16],
                episode=ep,
                start_step=start,
                goal_offset=offset,
                split="dev" if i < ndev else "train" if i < ndev + ntrain else "test",
            )
        )
    return rows


def validate_manifest(m):
    if m["schema"] != "cube-cases-v1":
        raise ValueError("Wrong manifest schema")
    if digest(m["cases"]) != m["cases_sha256"]:
        raise ValueError("Case manifest changed")
    ids, episodes = set(), set()
    for c in m["cases"]:
        if c["case_id"] in ids or c["episode"] in episodes:
            raise ValueError("Duplicate case or episode: split leakage")
        if (
            any(type(c[k]) is not int or c[k] < 0 for k in ("episode", "start_step", "goal_offset"))
            or c["goal_offset"] == 0
        ):
            raise ValueError("Invalid case index")
        if c["case_id"] != digest([c["episode"], c["start_step"], c["goal_offset"]])[:16]:
            raise ValueError("Case ID must match episode/start/goal")
        ids.add(c["case_id"])
        episodes.add(c["episode"])
        if c["split"] not in ("dev", "train", "test"):
            raise ValueError("Invalid split")


def probe_actions(name, budget, amplitude, history):
    """Low-amplitude out-and-back translations; never open a held gripper.

    Library is valid in all observable stages. A commanded closing phase is
    detected from issued gripper actions only; no simulator contact labels.
    The second half is the charged return/re-approach, not a free reset.
    """
    if name not in ("x_cycle", "y_cycle", "lift_cycle") or budget < 0:
        raise ValueError("Unknown probe/budget")
    out = np.zeros((budget, 5), dtype=np.float32)
    axis = {"x_cycle": 0, "y_cycle": 1, "lift_cycle": 2}[name]
    phase = (
        "commanded_closing" if history.actions and history.actions[-1][4] > 0.1 else "unconfirmed"
    )
    # Relative gripper command zero requests the current opening, not release.
    half = budget // 2
    out[:half, axis] = amplitude
    out[half : 2 * half, axis] = -amplitude
    return out, phase


POLICY_KEYS = (
    "action_dim",
    "action_block",
    "image_size",
    "total_steps",
    "horizon",
    "execute_blocks",
    "cem_samples",
    "cem_iterations",
    "cem_elites",
    "score_batch_size",
    "adapt_steps",
    "adapt_search_radius",
    "adapt_regularization",
    "residual_rank",
    "adapt_seed",
)


def policy_config(cfg):
    """Remove evaluator truth, data paths, splits and condition before algorithms."""
    out = {k: cfg[k] for k in POLICY_KEYS}
    if "interface_parameterization" in cfg:
        out["interface_parameterization"] = cfg["interface_parameterization"]
    return out
