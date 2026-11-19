"""Portable manifests and information boundaries for prefix-value research."""
from pathlib import Path

import numpy as np

from cube_research.protocol import digest
from research_common import file_hash, read_json, require_complete, write_json

METHODS = ("cem-mean", "pool-l2", "value-no-future", "value")
INPUT_KEYS = ("current", "goal", "predicted", "commands", "remaining")
SCHEMA = "cube-prefix-value-v1"


def scientific_config(cfg):
    """Hardware and local paths do not change the scientific contract."""
    return {k: v for k, v in cfg.items() if k not in
            ("dataset", "weights", "device", "render_backend", "score_batch_size")}


def signature(cfg):
    return digest(scientific_config(cfg))


def data_signature(cfg):
    # Cached branch outcomes do not depend on subsequent optimizer settings.
    return digest({k: v for k, v in scientific_config(cfg).items() if k != "training"})


def validate_config(cfg):
    fixed = dict(protocol=SCHEMA, sensors="rgb", image_size=224, action_dim=5,
                 action_block=5, horizon=5, execute_blocks=1, goal_offset=25,
                 success_distance_m=.04)
    if any(cfg.get(k) != v for k, v in fixed.items()):
        raise ValueError("Prefix-value v1 requires RGB Cube, five-step prefixes and offset 25")
    formal = dict(total_steps=50, cem_samples=300, cem_iterations=10, cem_elites=30)
    smoke = dict(total_steps=10, cem_samples=12, cem_iterations=2, cem_elites=3)
    if any(cfg[k] != v for k, v in (smoke if cfg.get("smoke") else formal).items()):
        raise ValueError("Use the formal protocol or explicit --smoke")
    for k in ("score_batch_size", "train_states", "calibration_states", "calibration_trajectories"):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError(f"Invalid positive integer: {k}")
    if not cfg["warmup_steps"] or any(type(t) is not int or t < 0 or t % 5 or
                                      t >= cfg["total_steps"] for t in cfg["warmup_steps"]):
        raise ValueError("Warmup times must leave a full block and be multiples of five")
    p, t = cfg["pool"], cfg["training"]
    for obj, keys in ((p, ("size", "archive_per_iteration", "branches")),
                      (t, ("hidden", "steps", "evaluate_every", "batch_states", "cpu_threads"))):
        if any(type(obj[k]) is not int or obj[k] <= 0 for k in keys):
            raise ValueError("Pool/training sizes must be positive integers")
    if not 2 <= p["branches"] <= p["size"] or not 0 <= p["edits"] <= min(4, p["size"] - 2):
        raise ValueError("Invalid pool/branch/edit sizes")
    for value in (p["edit_scale"], t["learning_rate"], t["ranking_temperature"]):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Invalid learning/edit scale")
    for key in ("weight_decay", "ranking_weight", "tie_tolerance"):
        if not np.isfinite(t[key]) or t[key] < 0:
            raise ValueError(f"Invalid {key}")
    for key in ("dev_seeds", "test_seeds"):
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]) or any(
                type(s) is not int or s < 0 for s in cfg[key]):
            raise ValueError(f"Invalid {key}")


def weights_identity(cfg):
    path = Path(cfg["weights"])
    return dict(weights_sha256=file_hash(path), model_config_sha256=file_hash(path.parent / "config.json"))


def case_fingerprint(initial, target, rgb):
    import hashlib
    h = hashlib.sha256()
    for value in (initial["qpos"], initial["qvel"], target["pos"], target["quat"], rgb):
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def checked_case(dataset, case):
    initial, target, rgb = dataset.case(case)
    if case_fingerprint(initial, target, rgb) != case["source_sha256"]:
        raise ValueError(f"Dataset content changed at case {case['case_id']}")
    return initial, target, rgb


def sample_cases(episodes, lengths, count, cfg, rng):
    """Balanced source trajectories; unique start/goal cases, without success selection."""
    pools = {ep: rng.permutation(int(lengths[ep]) - cfg["goal_offset"]).tolist() for ep in episodes}
    cases = []
    while pools and len(cases) < count:
        for ep in rng.permutation(list(pools)).tolist():
            start = pools[ep].pop()
            case = dict(episode=ep, start_step=start, goal_offset=cfg["goal_offset"])
            case["case_id"] = digest([ep, start, cfg["goal_offset"]])[:16]
            case["warmup_steps"] = int(cfg["warmup_steps"][len(cases) % len(cfg["warmup_steps"])])
            cases.append(case)
            if not pools[ep]:
                del pools[ep]
            if len(cases) == count:
                break
    if len(cases) != count:
        raise ValueError("Insufficient independent start/goal rows for requested state manifest")
    return cases


def prepare(cfg, goal_cache, out):
    from cube_research.data import Dataset
    require_complete(goal_cache)
    source = read_json(Path(goal_cache) / "manifest.json")
    if source["schema"] != "cube-goal-set-cache-v1":
        raise ValueError("Expected the existing idea1 feature-cache manifest")
    weights = weights_identity(cfg)
    if any(source["inputs"][k] != v for k, v in weights.items()):
        raise ValueError("Source goal-cache weights differ")
    if source["inputs"]["dataset_bytes"] != Path(cfg["dataset"]).stat().st_size:
        raise ValueError("Source goal-cache dataset size differs")
    if source["data_config"]["goal_offset"] != cfg["goal_offset"]:
        raise ValueError("Goal offset differs from the source study")
    groups = source["splits"]
    joined = [ep for group in groups.values() for ep in group]
    if len(joined) != len(set(joined)):
        raise ValueError("Source trajectory split overlap")
    rng = np.random.default_rng(cfg["split_seed"])
    available = rng.permutation(groups["train"]).tolist()
    ncal = cfg["calibration_trajectories"]
    if len(available) <= ncal:
        raise ValueError("Need training trajectories remaining after calibration split")
    splits = dict(train=available[ncal:], calibration=available[:ncal],
                  dev=groups["dev"], test=groups["test"])
    dataset = Dataset(cfg["dataset"])
    try:
        cases = {s: sample_cases(splits[s], dataset.lengths, cfg[s + "_states"], cfg, rng)
                 for s in ("train", "calibration")}
        for s in ("dev", "test"):
            cases[s] = [dict(c, warmup_steps=0) for c in source["cases"][s]]
            if any(c["episode"] not in splits[s] for c in cases[s]):
                raise ValueError("Source evaluation cases are outside their trajectory split")
            if len({c["episode"] for c in cases[s]}) != len(cases[s]) or len({c["case_id"] for c in cases[s]}) != len(cases[s]):
                raise ValueError("Evaluation needs one independent source trajectory per case")
        for group in cases.values():
            for case in group:
                case["source_sha256"] = case_fingerprint(*dataset.case(case))
        payload = dict(schema=SCHEMA, config_sha256=data_signature(cfg), smoke=cfg.get("smoke", False),
                       source_manifest_sha256=file_hash(Path(goal_cache) / "manifest.json"),
                       weights=weights, dataset_bytes=Path(cfg["dataset"]).stat().st_size,
                       action_stats=source["action_stats"], splits=splits, cases=cases,
                       continuation="frozen L2 CEM 300/10/30, fresh buffer and common seed per state"
                       if not cfg.get("smoke") else "smoke-only frozen L2 CEM 12/2/3",
                       labels="simulator truth only in training labels and evaluation outputs")
        write_json(out / "manifest.json", payload)
    finally:
        dataset.close()


def load_manifest(cfg, path, *, check_assets=True):
    require_complete(path)
    m = read_json(Path(path) / "manifest.json")
    if m["schema"] != SCHEMA or m["config_sha256"] != data_signature(cfg):
        raise ValueError("Manifest/config mismatch (smoke and formal artifacts cannot mix)")
    if check_assets and (m["weights"] != weights_identity(cfg) or
                         m["dataset_bytes"] != Path(cfg["dataset"]).stat().st_size):
        raise ValueError("Dataset/checkpoint identity changed")
    return m


def utility(distances, remaining, threshold=.04):
    """Success dominates progress; all times include the prefix and continuation."""
    d = np.asarray(distances, dtype=np.float64)
    if remaining <= 0 or d.ndim != 1 or not len(d) or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("Invalid utility trajectory")
    if len(d) > remaining + 1:
        raise ValueError("Utility trajectory exceeds remaining budget")
    hits = np.flatnonzero(d <= threshold)
    if len(hits):
        return float(1 - .1 * hits[0] / remaining)
    return float(.2 * np.clip((d[0] - d.min()) / max(d[0], threshold), 0, 1))


def features(inputs, stats, *, future):
    """Explicit allowlist. No simulator fields, labels, future suffix or case IDs."""
    if set(inputs) != set(INPUT_KEYS):
        raise ValueError("Only declared policy inputs are allowed")
    z, g, p = (np.asarray(inputs[k], dtype=np.float32) for k in INPUT_KEYS[:3])
    a, b = np.asarray(inputs["commands"], dtype=np.float32), np.asarray(inputs["remaining"], dtype=np.float32)
    n = len(z)
    if z.ndim != 2 or g.shape != z.shape or p.shape != z.shape or a.shape != (n, 5, 5) or b.shape != (n,):
        raise ValueError("Expected [N,D] latents, [N,5,5] executed prefix, [N] remaining")
    if any(not np.isfinite(x).all() for x in (z, g, p, a, b)) or (np.abs(a) > 1).any():
        raise ValueError("Nonfinite inputs or non-executable commands")
    if (b < 5).any() or (b > 50).any() or (b % 5 != 0).any():
        raise ValueError("v1 evaluates complete five-step blocks only")
    scale = np.asarray(stats["scale"], dtype=np.float32)
    if scale.shape != (5,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid action statistics")
    normalized = (a - np.asarray(stats["mean"], dtype=np.float32)) / scale
    pieces = [z, g, normalized.reshape(n, 25), b[:, None] / 50]
    if future:
        pieces.append(p)
    return np.concatenate(pieces, axis=1).astype(np.float32)
