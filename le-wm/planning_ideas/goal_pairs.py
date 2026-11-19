"""Offline Cube supervision and deterministic trajectory-first sampling."""

import numpy as np

from cube_research.protocol import digest

POSITION = "privileged_block_0_pos"
JOINTS = "proprio_joint_pos"
GRIPPER = "proprio_gripper_opening"


def split_episodes(lengths, cfg):
    rng = np.random.default_rng(cfg["split_seed"])
    pool = iter(rng.permutation(np.flatnonzero(lengths > cfg["goal_offset"])).tolist())
    from itertools import islice

    splits = {key: list(islice(pool, cfg[f"{key}_episodes"]))
              for key in ("train", "dev", "test", "donor")}
    if not splits["train"] or not splits["dev"]:
        raise ValueError("Need separate training and development trajectories")
    cases = {}
    offset = cfg["goal_offset"]
    for split in ("dev", "test"):
        episodes = splits[split][:cfg["dev_cases"]] if split == "dev" else splits[split]
        cases[split] = []
        for ep in episodes:
            start = int(rng.integers(int(lengths[ep]) - offset))
            cases[split].append(dict(case_id=digest([ep, start, offset])[:16], episode=ep,
                                     start_step=start, goal_offset=offset))
    return splits, cases


def frame_rows(dataset, episodes, stride):
    episodes = np.repeat(episodes, [len(range(0, dataset.lengths[i], stride)) for i in episodes])
    if not len(episodes):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    counts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, len(episodes)]
    steps = (np.arange(len(episodes)) - np.repeat(counts[:-1], np.diff(counts))) * stride
    return dataset.offsets[episodes].astype(int) + steps, episodes


def read_rows(column, rows):
    """Read sparse rows in storage-chunk order without HDF5 fancy indexing."""
    rows = np.asarray(rows, dtype=int)
    if not len(rows):
        return np.empty((0, *column.shape[1:]), dtype=column.dtype)
    chunk = column.chunks[0] if column.chunks else 256
    order = np.argsort(rows)
    sorted_rows = rows[order]
    keys = sorted_rows // chunk
    edges = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(rows)]
    out = np.empty((len(rows), *column.shape[1:]), dtype=column.dtype)
    for lo, hi in zip(edges[:-1], edges[1:]):
        start, end = sorted_rows[lo], sorted_rows[hi - 1] + 1
        out[order[lo:hi]] = column[start:end][sorted_rows[lo:hi] - start]
    return out


def labels(dataset, rows):
    return (read_rows(dataset.f[POSITION], rows), read_rows(dataset.f[JOINTS], rows),
            read_rows(dataset.f[GRIPPER], rows).reshape(len(rows), 1))


def robot_distance(joints, gripper, goal_joints, goal_gripper, spec):
    wrapped = (joints - goal_joints + np.pi) % (2 * np.pi) - np.pi
    angular = np.mean((wrapped / spec["joint_unit_rad"]) ** 2, axis=-1)
    opening = ((gripper - goal_gripper) / spec["gripper_unit"]).ravel() ** 2
    return np.sqrt(angular + opening)


def matched_pairs(pos, joints, gripper, spec, rng):
    """Each sampled target occurs equally often in the positive and negative labels."""
    pools = {}
    for goal in range(len(pos)):
        d = np.linalg.norm(pos - pos[goal], axis=-1)
        arm = robot_distance(joints, gripper, joints[goal], gripper[goal], spec)
        p = np.flatnonzero((d <= spec["positive_m"]) & (arm >= spec["min_positive_robot"]))
        n = (d > spec["negative_m"]) & (arm <= spec["max_negative_robot"])
        hard = np.flatnonzero(n & (d <= spec["hard_max_m"]))
        other = np.flatnonzero(n & (d > spec["hard_max_m"]))
        if len(p) and len(hard) + len(other):
            pools[goal] = [rng.permutation(items).tolist() for items in (p, hard, other)]

    result, hard_count = [], 0
    while pools and len(result) < spec["max_per_label"]:
        # Take at most one pair bundle per goal per pass, with scarce hard
        # negatives used early enough to meet the quota when possible.
        for goal in rng.permutation(list(pools)).tolist():
            p, hard, other = pools[goal]
            use_hard = bool(hard and (not other or 2 * hard_count <= len(result)))
            result.append((p.pop(), (hard if use_hard else other).pop(), goal))
            hard_count += use_hard
            if not p or not (hard or other):
                del pools[goal]
            if len(result) == spec["max_per_label"]:
                break
    return np.asarray(result, dtype=int).reshape(-1, 3), int(hard_count)


def mechanism_cases(dataset, dev_cases, donor_episodes, cfg):
    """Preselect real goal images. Never reuse a donor trajectory across cases."""
    spec = cfg["pairing"]
    rows, episodes = frame_rows(dataset, donor_episodes, spec["frame_stride"])
    if not len(rows):
        return [], [case["case_id"] for case in dev_cases[:cfg["mechanism_cases"]]]
    pos, joints, grip = labels(dataset, rows)
    picked, missing, used = [], [], set()
    for case in dev_cases[:cfg["mechanism_cases"]]:
        row = int(dataset.offsets[case["episode"]]) + case["start_step"] + case["goal_offset"]
        p, j, g = labels(dataset, [row])
        distance = np.linalg.norm(pos - p, axis=-1)
        arm = robot_distance(joints, grip, j[0], g[0], spec)
        available = ~np.isin(episodes, list(used))
        equivalents = np.flatnonzero(available & (distance <= spec["positive_m"])
                                     & (arm >= spec["min_positive_robot"]))
        wrong = np.flatnonzero(available & (distance > 2 * cfg["success_distance_m"])
                               & (arm <= spec["max_negative_robot"]))
        found = []
        for i in equivalents[np.argsort(-arm[equivalents], kind="stable")]:
            if all(episodes[i] != episodes[k] and
                   robot_distance(joints[i:i+1], grip[i:i+1], joints[k], grip[k], spec)[0]
                   >= spec["min_positive_robot"] for k in found):
                found.append(i)
            if len(found) == 2:
                break
        wrong = wrong[~np.isin(episodes[wrong], episodes[found])]
        if len(found) != 2 or not len(wrong):
            missing.append(case["case_id"])
            continue
        found.append(wrong[np.argmin(distance[wrong])])
        used.update(episodes[found].tolist())
        picked.append(dict(case_id=case["case_id"], donor_rows=rows[found].tolist(),
                           donor_episodes=episodes[found].tolist(),
                           original_goal_distance_m=distance[found].tolist(),
                           original_goal_robot_distance=arm[found].tolist()))
    return picked, missing
