"""Paired Cube evaluation. Physics labels never enter the RGB policy."""

import resource
import sys
import time
from dataclasses import asdict

import numpy as np
import torch

from cube_research.data import Dataset
from cube_research.model import FrozenModel, Planner, terminal_embedding
from cube_research.protocol import Ledger, transform_command
from research_common import append_jsonl, file_hash, read_json, require_complete, write_json
from research_runtime import execution_identity
from .goal_pairs import read_rows
from .goal_set import METHODS, bind_goal, build_head
from .goal_training import load_manifest

PLANNER_KEYS = ("horizon", "action_block", "action_dim", "execute_blocks", "cem_samples",
                "cem_iterations", "cem_elites")


class FrozenPredictor:
    def __init__(self, model, ledger):
        self.model, self.ledger = model, ledger
        self.mps_allocated_sampled_peak = 0

    def predict(self, z, commands):
        out = self.model.predict(z, transform_command(commands), self.ledger)
        if self.model.device == "mps":
            self.mps_allocated_sampled_peak = max(self.mps_allocated_sampled_peak,
                                                  torch.mps.current_allocated_memory())
        return out


class RGBPolicy:
    def __init__(self, model, ledger, goal_rgb, method, head, calibration, planner_cfg, seed):
        self.model, self.ledger = model, ledger
        self.goal = model.encode([goal_rgb], ledger)[0]
        started = time.perf_counter()
        score = bind_goal(method, head, self.goal, calibration)
        self.setup_seconds = time.perf_counter() - started
        self.planner = Planner(FrozenPredictor(model, ledger), planner_cfg, seed, terminal_cost=score)

    def action(self, rgb):
        # Encodings are needed only on replanning steps, not during the prefix.
        z = None if self.planner.buffer else self.model.encode([rgb], self.ledger)[0]
        return self.planner.action(z, self.goal)


def trial(cfg, case, variant, initial, target, original_pos, goal_rgb, model, head, calibration,
          method, seed):
    from cube_research.simulator import make_env

    started = time.perf_counter()
    harness = make_env(cfg, "N", cfg["environment_seed"], initial, target)
    try:
        if model.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(model.device)
        ledger = Ledger(cfg["total_steps"])
        observations, actions = [harness.evaluate()], []
        policy = None
        threshold = cfg["success_distance_m"]
        if observations[0]["distance"] > threshold:
            obs = harness.observe()
            policy = RGBPolicy(model, ledger, goal_rgb, method, head, calibration,
                               {k: cfg[k] for k in PLANNER_KEYS}, seed)
            for _ in range(ledger.limit):
                command = policy.action(obs.pixels)
                obs, truth = harness.step(command, ledger, "evaluation")
                observations.append(truth)
                actions.append(command.tolist())
                if truth["distance"] <= threshold or truth["truncated"] or truth["terminated"]:
                    break
        dist = np.array([x["distance"] for x in observations])
        original_dist = np.linalg.norm(np.array([x["object_pos"] for x in observations]) - original_pos, axis=1)
        hit = np.flatnonzero(dist <= threshold)
        first = int(hit[0]) if len(hit) else None
        return dict(case_id=case["case_id"], episode=case["episode"], method=method, seed=seed,
                    variant=variant, initially_successful=bool(dist[0] <= threshold),
                    success=first is not None, first_hit_step=first,
                    penalized_first_hit_step=ledger.limit + 1 if first is None else first,
                    executed_steps=len(actions), min_distance_m=float(dist.min()),
                    initial_distance_m=float(dist[0]), final_distance_m=float(dist[-1]),
                    original_min_distance_m=float(original_dist.min()),
                    original_success=bool(original_dist.min() <= threshold), actions=actions,
                    distances_m=dist.tolist(), original_distances_m=original_dist.tolist(),
                    elapsed_seconds=time.perf_counter() - started, ledger=asdict(ledger),
                    scoring_seconds=0 if policy is None else policy.planner.scoring_seconds + policy.setup_seconds,
                    rss_process_high_water_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024),
                    cuda_torch_peak_allocated_bytes=(torch.cuda.max_memory_allocated(model.device)
                                                    if model.device.startswith("cuda") else None),
                    mps_torch_sampled_peak_allocated_bytes=(policy.planner.adapter.mps_allocated_sampled_peak
                                                            if policy and model.device == "mps" else None))
    finally:
        harness.close()


SUMMARY_KEYS = ("success", "penalized_first_hit_step", "min_distance_m", "final_distance_m",
                "elapsed_seconds", "scoring_seconds", "encoded_images", "candidate_prediction_steps")


def metrics(row):
    model_calls = row["ledger"]["model"]
    return {**row, "encoded_images": model_calls.get("encoder_items", 0),
            "candidate_prediction_steps": model_calls.get("predictor_items", 0)}


def summarize(rows, seed):
    cases = sorted({r["case_id"] for r in rows})
    if not cases:
        return dict(cases=0, trials=0, methods={})
    groups = {(method, case): [metrics(r) for r in rows if r["method"] == method and r["case_id"] == case]
              for method in METHODS for case in cases}
    values = {method: np.array([[np.mean([r[k] for r in groups[method, c]]) for k in SUMMARY_KEYS]
                               for c in cases]) for method in METHODS}
    samples = np.random.default_rng(seed).integers(len(cases), size=(2000, len(cases)))
    methods = {}
    for method in METHODS:
        means = dict(zip(SUMMARY_KEYS, values[method].mean(axis=0).tolist()))
        hits = [[r["first_hit_step"] for r in groups[method, c] if r["success"]] for c in cases]
        hit_means = [np.mean(h) for h in hits if h]
        diff = values[method] - values["l2"]
        cis = np.quantile(diff[samples].mean(axis=1), [.025, .975], axis=0).T.tolist()
        methods[method] = dict(means=means,
                               first_hit_step_given_success=float(np.mean(hit_means)) if hit_means else None,
                               delta_vs_l2=dict(zip(SUMMARY_KEYS, diff.mean(axis=0).tolist())),
                               paired_case_bootstrap_95=dict(zip(SUMMARY_KEYS, cis)) if len(cases) > 1 else None)
    return dict(cases=len(cases), trials=len(rows), methods=methods)


def load_study(cfg, args):
    manifest = load_manifest(cfg, args.cache)
    require_complete(args.trained)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    path = args.trained / "heads.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    identity = dict(manifest_sha256=file_hash(args.cache / "manifest.json"), heads_sha256=file_hash(path))
    if (checkpoint.get("schema") != "cube-goal-set-heads-v1" or
            checkpoint["manifest_sha256"] != identity["manifest_sha256"] or
            checkpoint["dim"] != manifest["latent_dim"] or checkpoint["training"] != cfg["training"]):
        raise ValueError("Checkpoint must come from this feature cache and training configuration")
    heads = {m: build_head(m, checkpoint["dim"], cfg["training"]) for m in METHODS}
    for method, head in heads.items():
        if head is not None:
            head.load_state_dict(checkpoint["states"][method])
            head.eval().requires_grad_(False)
    model = FrozenModel(cfg, manifest["action_stats"])
    if model.dim != checkpoint["dim"]:
        raise ValueError("Checkpoint and backbone latent dimensions differ")
    identity.update(device=execution_identity(model.device), model=model.identity)
    return manifest, model, heads, checkpoint["calibration"], identity


def replacement_goals(dataset, case, variants, out):
    from PIL import Image

    initial, target, rgb = dataset.case(case)
    results = [("original", target, rgb)]
    if variants is not None:
        rows = variants["donor_rows"]
        images = read_rows(dataset.f["pixels"], rows)
        labels = ("equivalent-1", "equivalent-2", "wrong-object")
        for variant, row, image in zip(labels, rows, images):
            results.append((variant, dict(pos=dataset.f["privileged_block_0_pos"][row],
                                          quat=dataset.f["privileged_block_0_quat"][row])
                            if variant == "wrong-object" else target, image))
        for variant, _, image in results:
            Image.fromarray(image).save(out / f"{case['case_id']}-{variant}.png")
    return initial, target, results


def evaluate(cfg, args, out):
    if args.smoke and (args.split != "dev" or args.dev_prefix):
        raise ValueError("Smoke is a standalone reduced-budget dev check")
    if args.dev_prefix is not None and (args.split != "dev" or args.dev_prefix <= 0):
        raise ValueError("Dev prefix must be positive and dev-only")
    if args.mechanism and args.split != "dev":
        raise ValueError("Goal replacement is a development-only mechanism study")
    manifest, model, heads, calibration, identity = load_study(cfg, args)
    settings = {k: cfg[k] for k in (*PLANNER_KEYS, "protocol", "total_steps", "success_distance_m",
                                    "environment_seed", "score_batch_size", "test_seeds", "dev_seeds")}
    if args.split == "test":
        if args.dev_run is None:
            raise ValueError("Test requires --dev-run pointing to completed full development evaluation")
        require_complete(args.dev_run)
        dev = read_json(args.dev_run / "study.json")
        keys = ("heads_sha256", "manifest_sha256")
        if not (dev["full_dev"] and dev["settings"] == settings and all(dev[k] == identity[k] for k in keys)):
            raise ValueError("Test requires the same heads and protocol as a complete full dev run")
        identity["dev_study_sha256"] = file_hash(args.dev_run / "study.json")
    if args.smoke:
        cfg = {**cfg, "protocol": "cube-planner-study-smoke", "total_steps": 10,
               "cem_samples": 12, "cem_iterations": 2, "cem_elites": 3}
        settings.update({k: cfg[k] for k in ("protocol", "total_steps", "cem_samples", "cem_iterations", "cem_elites")})
    cases, seeds = manifest["cases"][args.split], cfg[f"{args.split}_seeds"]
    variants = {m["case_id"]: m for m in manifest["mechanism"]} if args.mechanism else {}
    if args.smoke:
        # Only the engineering check needs a case guaranteed to call CEM.
        # The scientific runs keep initially successful cases in their manifest.
        dataset = Dataset(cfg["dataset"])
        try:
            start = np.array([int(dataset.offsets[c["episode"]]) + c["start_step"] for c in cases])
            pos = dataset.f["privileged_block_0_pos"]
            eligible = np.linalg.norm(read_rows(pos, start) - read_rows(pos, start + cfg["goal_offset"]),
                                      axis=-1) > cfg["success_distance_m"]
            cases = [c for c, use in zip(cases, eligible)
                     if use and (not args.mechanism or c["case_id"] in variants)][:1]
            seeds = seeds[:1]
        finally:
            dataset.close()
    elif args.dev_prefix:
        cases = cases[:args.dev_prefix]
    if not cases:
        raise ValueError("No eligible cases for this evaluation split")
    write_json(out / "study.json", {**identity, "settings": settings, "split": args.split,
                                    "cases": cases, "seeds": seeds, "mechanism": args.mechanism,
                                    "full_dev": args.split == "dev" and not args.smoke and args.dev_prefix is None,
                                    "mechanism_missing": manifest["missing_mechanism"] if args.mechanism else []})
    dataset = Dataset(cfg["dataset"])
    rows = []
    try:
        for i, case in enumerate(cases):
            initial, target, goals = replacement_goals(dataset, case, variants.get(case["case_id"]), out)
            for label, active_target, rgb in goals:
                for seed in seeds:
                    for method in METHODS:
                        row = trial(cfg, case, label, initial, active_target, target["pos"], rgb, model,
                                    heads[method], calibration, method, seed)
                        if args.smoke and label == "original" and not row["ledger"]["model"].get("predictor_items"):
                            raise RuntimeError("Smoke did not exercise the frozen predictor")
                        append_jsonl(out / "episodes.jsonl", row)
                        rows.append(row)
            print(f"{i+1}/{len(cases)}: {case['case_id']} ({len(rows)} trials total)", flush=True)
    finally:
        dataset.close()
    originals = [r for r in rows if r["variant"] == "original"]
    summary = dict(all=summarize(originals, cfg["bootstrap_seed"]),
                   nontrivial=summarize([r for r in originals if not r["initially_successful"]], cfg["bootstrap_seed"]))
    if args.mechanism:
        reference = {(r["case_id"], r["method"], r["seed"]): r for r in originals}
        comparisons = []
        for row in rows:
            if row["variant"] == "original":
                continue
            original = reference[row["case_id"], row["method"], row["seed"]]
            common = min(cfg["action_block"], len(row["actions"]), len(original["actions"]))
            change = np.array(row["actions"][:common]) - np.array(original["actions"][:common])
            comparison = {k: row[k] for k in ("case_id", "method", "seed", "variant", "success", "original_success",
                                              "min_distance_m", "original_min_distance_m", "initially_successful")}
            comparisons.append(dict(**comparison, original_image_success=original["success"],
                                    compared_prefix_steps=common,
                                    action_prefix_rmse=float(np.sqrt(np.mean(change ** 2))) if common else None))
        write_json(out / "mechanism.json", dict(comparisons=comparisons,
                                                cases=len({r["case_id"] for r in comparisons})))
    write_json(out / "summary.json", summary)


def diagnose(cfg, args, out):
    from scipy.stats import spearmanr
    from cube_research.simulator import make_env

    if args.candidates < 2:
        raise ValueError("Need at least two candidates for a ranking check")
    manifest, model, heads, calibration, identity = load_study(cfg, args)
    write_json(out / "study.json", identity)
    dataset = Dataset(cfg["dataset"])
    try:
        for case in manifest["cases"]["dev"]:
            initial, target, rgb = dataset.case(case)
            harness = make_env(cfg, "N", cfg["environment_seed"], initial, target)
            try:
                z, goal = model.encode([harness.observe().pixels, rgb], Ledger(0))
                shape = (args.candidates, cfg["horizon"], cfg["action_block"], cfg["action_dim"])
                rng = np.random.default_rng(cfg["dev_seeds"][0])
                normalized = rng.normal(size=shape).astype(np.float32)
                normalized[0] = 0
                candidates = np.clip(normalized * model.stats["scale"] + model.stats["mean"], -1, 1).astype(np.float32)
                terminal = terminal_embedding(FrozenPredictor(model, Ledger(0)), z, candidates)
                snapshot = harness.snapshot()
                realized = []
                for commands in candidates:
                    harness.restore(snapshot)
                    for command in commands.reshape(-1, cfg["action_dim"]):
                        # Evaluator-only physics rollout: same Cube commands, no images needed.
                        harness.env.step(transform_command(command))
                    realized.append(harness.evaluate()["distance"])
                ranking = {}
                for method in METHODS:
                    scores = bind_goal(method, heads[method], goal, calibration)(terminal, goal)
                    corr = spearmanr(scores, realized).statistic if np.ptp(scores) and np.ptp(realized) else np.nan
                    ranking[method] = dict(scores=scores.tolist(), spearman=None if not np.isfinite(corr) else float(corr),
                                           chosen_distance_m=realized[int(scores.argmin())])
                append_jsonl(out / "candidates.jsonl", dict(case_id=case["case_id"], terminal_distances_m=realized,
                                                            oracle_best_distance_m=min(realized), methods=ranking))
            finally:
                harness.close()
    finally:
        dataset.close()
