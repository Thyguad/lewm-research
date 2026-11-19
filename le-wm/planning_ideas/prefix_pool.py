"""Shared CEM proposal pools. All four selectors receive the same candidate procedure."""
import time

import numpy as np

from cube_research.model import Planner, terminal_embedding
from cube_research.protocol import transform_command
from research_common import sync_device
from .prefix_contracts import METHODS


class Predictor:
    def __init__(self, model, ledger):
        self.model, self.ledger = model, ledger

    def predict(self, z, commands):
        return self.model.predict(z, transform_command(commands), self.ledger)


class Archive:
    def __init__(self, keep):
        self.keep, self.plans, self.costs, self.mean = keep, [], [], None

    def __call__(self, plans, costs, final):
        if final:
            self.mean = plans[0].copy()
        else:
            indices = np.argsort(costs, kind="stable")[:self.keep]
            self.plans.extend(plans[indices].copy())
            self.costs.extend(np.asarray(costs)[indices].tolist())


def diverse_indices(prefixes, costs, size, scale, *, include=()):
    """Half quality, half farthest-prefix selection; collapse identical executed prefixes."""
    a = np.asarray(prefixes, np.float32)
    ranked = np.argsort(costs, kind="stable").tolist()
    chosen = list(include)
    normalized = (a / np.asarray(scale)).reshape(len(a), -1)

    def distinct(i):
        return all(np.max(np.abs(a[i] - a[j])) > 1e-6 for j in chosen)

    for i in ranked:
        if len(chosen) >= max(1, size // 2):
            break
        if distinct(i):
            chosen.append(i)
    while len(chosen) < size:
        available = [i for i in ranked if distinct(i)]
        if not available:
            break
        distances = ((normalized[available, None] - normalized[chosen]) ** 2).mean(-1).min(-1)
        chosen.append(available[int(np.argmax(distances))])
    return chosen


def local_edits(mean, scale, count, amplitude):
    """Edit only issued xyz or gripper timing; never inspect contact/state labels."""
    edits = []
    for i in range(count):
        plan = mean.copy()
        if i < 3:
            plan[0, :, i] += (1 if i != 1 else -1) * amplitude * scale[i]
        else:
            plan[0, 1:, 4] = mean[0, :-1, 4]  # Delay gripper changes by one control step.
        edits.append(np.clip(plan, -1, 1).astype(np.float32))
    return edits


class ProposalPlanner:
    def __init__(self, model, cfg, ledger, seed):
        self.model, self.cfg, self.ledger = model, cfg, ledger
        self.archive = Archive(cfg["pool"]["archive_per_iteration"])
        self.adapter = Predictor(model, ledger)
        self.cem = Planner(self.adapter, cfg, seed, candidate_observer=self.archive)

    def propose(self, current, goal, remaining):
        if remaining < 5 or remaining % 5:
            raise ValueError("Prefix-value planning requires at least one complete block")
        self.archive.plans.clear()
        self.archive.costs.clear()
        self.cem.reset_buffer()
        self.cem.action(current, goal)  # Identical legacy CEM; observer cannot affect RNG or elites.
        self.cem.reset_buffer()
        started = time.perf_counter()
        p = self.cfg["pool"]
        mean = self.archive.mean
        scale = np.asarray(self.model.stats["scale"], dtype=np.float32)
        archive = np.asarray(self.archive.plans)
        kept = diverse_indices(archive[:, 0], self.archive.costs,
                               p["size"] - p["edits"] - 1, scale)
        # Include and explicitly predict the final mean, which legacy CEM does not score.
        extra = np.stack([mean, *local_edits(mean, scale, p["edits"], p["edit_scale"])])
        end = terminal_embedding(self.adapter, current, extra)
        extra_cost = ((end - goal) ** 2).sum(-1)
        self.ledger.count("pool_extra_terminal_score", len(extra))
        plans = np.concatenate((extra[:1], archive[kept], extra[1:]))
        costs = np.r_[extra_cost[:1], np.asarray(self.archive.costs)[kept], extra_cost[1:]]
        if not np.isfinite(costs).all():
            raise FloatingPointError("Nonfinite world-model terminal scores")
        ids = diverse_indices(plans[:, 0], costs, p["size"], scale, include=(0,))
        plans, costs = plans[ids], costs[ids]
        prefixes = plans[:, 0].astype(np.float32)
        # Prediction and features depend ONLY on executed commands, never on the plan tail.
        predicted = self.adapter.predict(np.repeat(current[None, None], len(plans), axis=0),
                                         prefixes[:, None])
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Nonfinite world-model prefix prediction")
        inputs = dict(current=np.repeat(current[None], len(plans), axis=0),
                      goal=np.repeat(goal[None], len(plans), axis=0), predicted=predicted,
                      commands=prefixes, remaining=np.full(len(plans), remaining, np.float32))
        sync_device(self.model.device)
        self.ledger.planning_seconds += time.perf_counter() - started
        return dict(inputs=inputs, terminal_l2=costs, full_plans=plans, mean_index=0)


def choose(pool, method, scorer=None):
    if method not in METHODS:
        raise ValueError(f"Unknown selector {method}")
    if method == "cem-mean":
        return 0, None
    if method == "pool-l2":
        return int(np.argmin(pool["terminal_l2"])), None
    if scorer is None:
        raise ValueError("Learned selector requires a fitted scorer")
    values = np.asarray(scorer(pool["inputs"], method), dtype=np.float64)
    if values.shape != (len(pool["terminal_l2"]),) or not np.isfinite(values).all():
        raise ValueError("Invalid prefix value scores")
    return int(np.argmax(values)), values


class PrefixPolicy:
    """RGB-only policy. No environment, evaluation info or label loading interface."""
    def __init__(self, model, cfg, ledger, goal_rgb, seed, method, scorer=None):
        self.model, self.ledger = model, ledger
        self.cfg = {k: cfg[k] for k in ("pool", "horizon", "action_block", "action_dim", "execute_blocks",
                                         "cem_samples", "cem_iterations", "cem_elites")}
        self.goal = model.encode([goal_rgb], ledger)[0]
        self.proposer = ProposalPlanner(model, self.cfg, ledger, seed)
        self.method, self.scorer = method, scorer
        self.buffer, self.decisions = [], []
        self.scoring_seconds = 0.

    def action(self, pixels, remaining):
        if not self.buffer:
            z = self.model.encode([pixels], self.ledger)[0]
            pool = self.proposer.propose(z, self.goal, remaining)
            started = time.perf_counter()
            index, values = choose(pool, self.method, self.scorer)
            self.scoring_seconds += time.perf_counter() - started
            self.ledger.count("prefix_score", len(pool["terminal_l2"]))
            self.buffer = list(pool["inputs"]["commands"][index].copy())
            self.decisions.append(dict(remaining=remaining, chosen=index,
                                       commands=pool["inputs"]["commands"].tolist(),
                                       terminal_l2=pool["terminal_l2"].tolist(),
                                       values=None if values is None else values.tolist()))
        return self.buffer.pop(0)


def branch_indices(pool, count, rng, *, scorer=None):
    """Base: mean/quality/diversity; refresh: high values/disagreement/random coverage."""
    n = len(pool["terminal_l2"])
    if n < count:
        raise ValueError(f"Only {n} distinct prefixes for {count} branches")
    l2_order = np.argsort(pool["terminal_l2"], kind="stable")
    selected = []
    def add(order, limit):
        if len(selected) >= limit:
            return
        for i in order:
            if int(i) not in selected:
                selected.append(int(i))
            if len(selected) >= limit:
                break
    if scorer is None:
        add([0, *l2_order], min(2, count))
        a = pool["inputs"]["commands"].reshape(n, -1)
        if len(selected) < count - 1:
            distances = ((a[:, None] - a[selected]) ** 2).mean(-1).min(-1)
            add(np.argsort(-distances), min(4, count - 1))
    else:
        _, values = choose(pool, "value", scorer)
        order = np.argsort(-values, kind="stable")
        add(order, min(2, count))
        # Positive disagreement: new head prefers a candidate L2 ranks poorly.
        ranks_l2, ranks_value = np.argsort(l2_order), np.argsort(order)
        add(np.argsort(-(ranks_l2 - ranks_value), kind="stable"), min(4, count - 1))
    add(rng.permutation(n), count)
    return selected
