"""Evaluator-only branch execution; offline labels never enter an online policy."""
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cube_research.model import FrozenModel, Planner
from cube_research.protocol import Ledger
from research_common import append_jsonl, derived_seed, file_hash, write_json
from .prefix_contracts import checked_case, load_manifest, utility
from .prefix_pool import Predictor, PrefixPolicy, ProposalPlanner, branch_indices


class Continuation:
    """Always legacy L2 + final CEM mean; reset optimizer/RNG/buffer for each branch."""
    def __init__(self, model, cfg, ledger, goal_rgb, seed):
        self.model, self.ledger = model, ledger
        self.goal = model.encode([goal_rgb], ledger)[0]
        self.planner = Planner(Predictor(model, ledger), cfg, seed)

    def action(self, rgb):
        z = None if self.planner.buffer else self.model.encode([rgb], self.ledger)[0]
        return self.planner.action(z, self.goal)


def stopped(truth, cfg):
    return truth["distance"] <= cfg["success_distance_m"] or truth["terminated"] or truth["truncated"]


def rollout_branch(harness, snapshot, prefix, remaining, cfg, model, goal_rgb, seed, ledger):
    harness.restore(snapshot)
    obs, truth = harness.observe(), harness.evaluate()
    initial_pixels = obs.pixels.copy()
    if truth["distance"] <= cfg["success_distance_m"]:
        raise ValueError("Do not label a branch from an already successful state")
    images, evaluations, commands = [initial_pixels], [truth], []
    continuation = None
    for step in range(remaining):
        if step < len(prefix):
            action, phase = prefix[step], "branch_prefix"
        else:
            if continuation is None:
                continuation = Continuation(model, cfg, ledger, goal_rgb, seed)
            action, phase = continuation.action(obs.pixels), "branch_continuation"
        obs, truth = harness.step(action, ledger, phase)
        images.append(obs.pixels.copy())
        evaluations.append(truth)
        commands.append(np.asarray(action).tolist())
        if stopped(truth, cfg):
            break
    distances = [r["distance"] for r in evaluations]
    hits = np.flatnonzero(np.array(distances) <= cfg["success_distance_m"])
    return dict(utility=utility(distances, remaining, cfg["success_distance_m"]),
                success=bool(len(hits)), first_hit_step=int(hits[0]) if len(hits) else None,
                distances_m=distances, executed_steps=len(commands),
                termination="success" if len(hits) else "terminated" if truth["terminated"] else
                "truncated" if truth["truncated"] else "budget", commands=commands,
                evaluations=evaluations, pixels=np.stack(images), ledger=asdict(ledger))


def budget(ledgers, actual):
    rows = {k: asdict(v) for k, v in ledgers.items()}
    attempted = sum(v.used + v.offline_replay_steps for v in ledgers.values())
    return dict(ledgers=rows, attempted_environment_steps=attempted,
                actual_environment_steps=actual, accounted=attempted == actual,
                model_prediction_items=sum(v.model.get("predictor_items", 0) for v in ledgers.values()),
                planning_seconds=sum(v.planning_seconds for v in ledgers.values()))


def collect_one(cfg, case, split, round_id, dataset, model, scorer, out, manifest_sha):
    from cube_research.simulator import make_env
    state_id = f"{case['case_id']}-t{case['warmup_steps']}-r{round_id}"
    path = out / "states" / state_id
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    initial, target, goal_rgb = checked_case(dataset, case)
    harness = make_env(cfg, "N", cfg["environment_seed"], initial, target)
    ledgers = dict(warmup=Ledger(case["warmup_steps"]), proposals=Ledger(0), audit=Ledger(0))
    meta = dict(state_id=state_id, case_id=case["case_id"], episode=case["episode"], split=split,
                round=round_id, warmup_steps=case["warmup_steps"], manifest_sha256=manifest_sha,
                state="running")
    seed = derived_seed(cfg["collection_seed"], state_id)
    write_json(path / "state.json", meta)
    try:
        write_json(path / "environment.json", harness.audit_model())
        obs, truth = harness.observe(), harness.evaluate()
        if truth["distance"] <= cfg["success_distance_m"]:
            meta.update(state="skipped", reason="initially_successful")
            return meta
        policy = (Continuation(model, cfg, ledgers["warmup"], goal_rgb, seed) if scorer is None else
                  PrefixPolicy(model, cfg, ledgers["warmup"], goal_rgb, seed, "value", scorer))
        warm_images, warm_actions, warm_truth = [obs.pixels.copy()], [], [truth]
        for t in range(case["warmup_steps"]):
            action = policy.action(obs.pixels) if scorer is None else policy.action(obs.pixels, cfg["total_steps"] - t)
            obs, truth = harness.step(action, ledgers["warmup"], "state_generation")
            warm_images.append(obs.pixels.copy())
            warm_actions.append(np.asarray(action).tolist())
            warm_truth.append(truth)
            if stopped(truth, cfg):
                break
        np.savez_compressed(path / "warmup.npz", pixels=np.stack(warm_images), commands=np.asarray(warm_actions))
        write_json(path / "warmup-evaluation.json", warm_truth)
        if stopped(truth, cfg):
            meta.update(state="skipped", reason="task_ended_during_state_generation")
            return meta
        remaining = cfg["total_steps"] - ledgers["warmup"].used
        z, goal = model.encode([obs.pixels, goal_rgb], ledgers["proposals"])
        proposer = ProposalPlanner(model, cfg, ledgers["proposals"], seed)
        pool = proposer.propose(z, goal, remaining)
        rng = np.random.default_rng(seed)
        ids = branch_indices(pool, cfg["pool"]["branches"], rng, scorer=scorer)
        inputs = {k: v[ids].copy() for k, v in pool["inputs"].items()}
        np.savez(path / "inputs.npz", **inputs)
        np.savez_compressed(path / "start.npz", pixels=obs.pixels, goal_rgb=goal_rgb)
        write_json(path / "proposal.json", dict(selected_indices=ids,
                   terminal_l2=pool["terminal_l2"].tolist(), commands=pool["inputs"]["commands"].tolist(),
                   predicted_values=None if scorer is None else scorer(pool["inputs"], "value").tolist()))
        snapshot = harness.snapshot()
        audit = harness.replay_check(inputs["commands"][0], ledgers["audit"])
        write_json(path / "replay.json", audit)
        outcomes = []
        # A common seed and a fresh continuation buffer in EVERY branch. Each branch
        # observes its own images; continuation actions are not copied across branches.
        continuation_seed = derived_seed(seed, "fixed-l2-continuation")
        for i, prefix in enumerate(inputs["commands"]):
            ledger = ledgers[f"branch_{i}"] = Ledger(remaining)
            result = rollout_branch(harness, snapshot, prefix, remaining, cfg, model, goal_rgb,
                                    continuation_seed, ledger)
            if not np.array_equal(result["pixels"][0], obs.pixels):
                raise RuntimeError("Branch start image differs after full-state restore")
            np.savez_compressed(path / f"branch-{i}.npz", pixels=result.pop("pixels"))
            write_json(path / f"branch-{i}-evaluation.json", result)
            outcomes.append(result["utility"])
        # Labels are physically separate from the deployment input file.
        write_json(path / "labels.json", dict(utility=outcomes, continuation_seed=continuation_seed,
                                               continuation="frozen-l2-cem"))
        meta.update(state="complete", remaining=remaining, branches=len(ids),
                    inputs_sha256=file_hash(path / "inputs.npz"), labels_sha256=file_hash(path / "labels.json"))
        return meta
    except BaseException as error:
        meta.update(state="failed", error=type(error).__name__, reason=str(error))
        raise
    finally:
        meta.update(elapsed_seconds=time.perf_counter() - started,
                    budget=budget(ledgers, harness.actual_steps))
        write_json(path / "state.json", meta, overwrite=True)
        append_jsonl(out / "states.jsonl", meta)
        harness.close()


def collect(cfg, args, out):
    from cube_research.data import Dataset
    from .prefix_training import load_scorer
    manifest = load_manifest(cfg, args.manifest)
    manifest_sha = file_hash(args.manifest / "manifest.json")
    if args.round == 1 and (args.split != "train" or args.trained is None):
        raise ValueError("Refresh requires --split train --round 1 --trained BASE_TRAINING_RUN")
    if args.round == 0 and args.trained is not None:
        raise ValueError("Base collection must not depend on a learned scorer")
    scorer = None if args.round == 0 else load_scorer(cfg, args.trained, manifest_sha)
    if scorer is not None and scorer.round != 0:
        raise ValueError("v1 permits one refresh, collected by a base-round model")
    cases = manifest["cases"][args.split]
    cases = cases[args.offset: None if args.limit is None else args.offset + args.limit]
    if not cases:
        raise ValueError("No collection states in requested range")
    write_json(out / "collection.json", dict(manifest_sha256=manifest_sha, split=args.split, round=args.round,
               requested_states=[f"{c['case_id']}-t{c['warmup_steps']}-r{args.round}" for c in cases],
               collector_sha256=None if scorer is None else scorer.sha256,
               continuation="fixed frozen L2; fresh common RNG/buffer per branch"))
    dataset = Dataset(cfg["dataset"])
    rows = []
    try:
        model = FrozenModel(cfg, manifest["action_stats"])
        write_json(out / "model.json", dict(device=model.device, identity=model.identity))
        for i, case in enumerate(cases):
            row = collect_one(cfg, case, args.split, args.round, dataset, model, scorer, out, manifest_sha)
            rows.append(row)
            print(f"{i+1}/{len(cases)} {row['state_id']}: {row['state']}", flush=True)
        write_json(out / "collection-summary.json", dict(
            requested=len(cases), complete=sum(r["state"] == "complete" for r in rows),
            skipped=[dict(state_id=r["state_id"], reason=r["reason"]) for r in rows if r["state"] == "skipped"],
            actual_environment_steps=sum(r["budget"]["actual_environment_steps"] for r in rows),
            model_prediction_items=sum(r["budget"]["model_prediction_items"] for r in rows),
            elapsed_seconds=sum(r["elapsed_seconds"] for r in rows)))
        if not any(r["state"] == "complete" for r in rows):
            raise ValueError("No usable labeled states; inspect explicit skip records")
        if not all(r["budget"]["accounted"] for r in rows):
            raise RuntimeError("Environment budget accounting mismatch")
    finally:
        dataset.close()
