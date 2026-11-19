"""Paired closed-loop evaluation and explicit missing-run accounting."""
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cube_research.model import FrozenModel
from cube_research.protocol import Ledger
from research_common import append_jsonl, file_hash, read_json, read_rows, require_complete, write_json
from .prefix_contracts import METHODS, checked_case, load_manifest, signature
from .prefix_pool import PrefixPolicy


def trial(cfg, harness, snapshot, initial_pixels, goal_rgb, model, scorer, method, seed, ledger):
    from .prefix_data import stopped
    harness.restore(snapshot)
    obs, truth = harness.observe(), harness.evaluate()
    if not np.array_equal(obs.pixels, initial_pixels):
        raise RuntimeError("Paired episode start differs after state restoration")
    distances, images, commands, evaluations = [truth["distance"]], [obs.pixels.copy()], [], [truth]
    policy = None
    if truth["distance"] > cfg["success_distance_m"]:
        policy = PrefixPolicy(model, cfg, ledger, goal_rgb, seed, method, scorer)
        for t in range(cfg["total_steps"]):
            action = policy.action(obs.pixels, cfg["total_steps"] - t)
            obs, truth = harness.step(action, ledger, "online_task")
            commands.append(np.asarray(action).tolist())
            distances.append(truth["distance"])
            images.append(obs.pixels.copy())
            evaluations.append(truth)
            if stopped(truth, cfg):
                break
    hit = np.flatnonzero(np.array(distances) <= cfg["success_distance_m"])
    return dict(success=bool(len(hit)), first_hit_step=int(hit[0]) if len(hit) else None,
                penalized_first_hit_step=int(hit[0]) if len(hit) else cfg["total_steps"] + 1,
                initially_successful=distances[0] <= cfg["success_distance_m"],
                min_distance_m=float(min(distances)), final_distance_m=float(distances[-1]),
                executed_steps=len(commands), distances_m=distances, commands=commands,
                evaluations=evaluations, pixels=np.stack(images),
                decisions=[] if policy is None else policy.decisions,
                scoring_seconds=0. if policy is None else policy.scoring_seconds,
                termination="success" if len(hit) else "terminated" if truth["terminated"] else
                "truncated" if truth["truncated"] else "budget")


def summarize(rows, cases, seeds, bootstrap_seed):
    expected = {(c["case_id"], m, s) for c in cases for m in METHODS for s in seeds}
    found = {}
    for row in rows:
        key = (row["case_id"], row["method"], row["seed"])
        if key not in expected or key in found:
            raise ValueError("Unexpected or duplicate evaluation row")
        found[key] = row
    missing = sorted(expected - found.keys())
    failed = [{k: r.get(k) for k in ("case_id", "method", "seed", "error", "reason")}
              for r in rows if r["state"] != "complete"]
    complete_cases = [c["case_id"] for c in cases if all(
        found.get((c["case_id"], m, s), {}).get("state") == "complete" for m in METHODS for s in seeds)]
    metrics = ("success", "penalized_first_hit_step", "min_distance_m", "final_distance_m",
               "executed_steps", "elapsed_seconds", "scoring_seconds")
    result = dict(expected_trials=len(expected), completed_trials=sum(r["state"] == "complete" for r in rows),
                  missing=missing, failed=failed, paired_complete_cases=complete_cases,
                  excluded_from_paired=[c["case_id"] for c in cases if c["case_id"] not in complete_cases], methods={})
    sample = (np.random.default_rng(bootstrap_seed).integers(len(complete_cases), size=(2000, len(complete_cases)))
              if complete_cases else None)
    per_case = {}
    for method in METHODS:
        valid = [r for r in rows if r["method"] == method and r["state"] == "complete"]
        n = len(cases) * len(seeds)
        successes = sum(r["success"] for r in valid)
        means = {k: float(np.mean([r[k] for r in valid])) if valid else None for k in metrics}
        result["methods"][method] = dict(expected=n, completed=len(valid),
                success_rate=successes / n if n and len(valid) == n else None,
                success_rate_bounds=[successes / n, (successes + n - len(valid)) / n] if n else None,
                completed_only_means=means)
        per_case[method] = np.array([[np.mean([found[c, method, s][k] for s in seeds]) for k in metrics]
                                    for c in complete_cases])
    for method in METHODS:
        if complete_cases:
            delta = per_case[method] - per_case["cem-mean"]
            result["methods"][method]["paired_delta_vs_cem_mean"] = dict(zip(metrics, delta.mean(0).tolist()))
            result["methods"][method]["paired_case_bootstrap_95"] = (
                dict(zip(metrics, np.quantile(delta[sample].mean(1), [.025, .975], axis=0).T.tolist()))
                if len(complete_cases) > 1 else None)
    return result


def analyze(run, out):
    study = read_json(Path(run) / "study.json")
    path = Path(run) / "episodes.jsonl"
    rows = read_rows(path) if path.exists() else []
    result = dict(all=summarize(rows, study["cases"], study["seeds"], study["bootstrap_seed"]))
    known = {}
    for row in rows:
        if "initially_successful" in row:
            previous = known.setdefault(row["case_id"], row["initially_successful"])
            if previous != row["initially_successful"]:
                raise ValueError("Paired initial success labels disagree")
    cases = [c for c in study["cases"] if known.get(c["case_id"]) is False]
    unknown = [c["case_id"] for c in study["cases"] if c["case_id"] not in known]
    result["nontrivial"] = summarize([r for r in rows if r["case_id"] in {c["case_id"] for c in cases}],
                                     cases, study["seeds"], study["bootstrap_seed"])
    result["initial_status_unknown"] = unknown
    write_json(out / "summary.json", result)
    lines = ["# Prefix-value paired evaluation", "", "| Method | Complete / expected | Success rate |", "| --- | ---: | ---: |"]
    for method, data in result["all"]["methods"].items():
        rate = "unfinished" if data["success_rate"] is None else f"{100 * data['success_rate']:.1f}%"
        lines.append(f"| {method} | {data['completed']} / {data['expected']} | {rate} |")
    lines += ["", "Unit of paired uncertainty: source case, averaged over planning seeds.",
              f"Missing: {len(result['all']['missing'])}; failed: {len(result['all']['failed'])}.",
              "These are this study's results, not the official Cube benchmark."]
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def evaluate(cfg, args, out):
    from cube_research.data import Dataset
    from cube_research.simulator import make_env
    from .prefix_training import load_scorer
    manifest = load_manifest(cfg, args.manifest)
    manifest_sha = file_hash(args.manifest / "manifest.json")
    scorer = load_scorer(cfg, args.trained, manifest_sha)
    if args.split == "test":
        if cfg.get("smoke") or args.limit is not None or args.dev_run is None:
            raise ValueError("Test requires a complete dev run and the full locked test manifest")
        require_complete(args.dev_run)
        dev = read_json(args.dev_run / "study.json")
        if (dev["split"] != "dev" or not dev["full_split"] or dev["heads_sha256"] != scorer.sha256 or
                dev["manifest_sha256"] != manifest_sha or dev["config_sha256"] != signature(cfg)):
            raise ValueError("Test must use the exact heads/protocol from full development evaluation")
    cases = manifest["cases"][args.split]
    if args.limit is not None:
        cases = cases[:args.limit]
    seeds = cfg[args.split + "_seeds"]
    dataset = Dataset(cfg["dataset"])
    try:
        if cfg.get("smoke"):
            # Engineering only: choose the first nontrivial dev case to exercise the predictor.
            cases = [c for c in cases if np.linalg.norm(dataset.f["privileged_block_0_pos"][
                int(dataset.offsets[c["episode"]]) + c["start_step"]] - checked_case(dataset, c)[1]["pos"])
                > cfg["success_distance_m"]][:1]
        if not cases:
            raise ValueError("No evaluation cases")
        study = dict(manifest_sha256=manifest_sha, heads_sha256=scorer.sha256, config_sha256=signature(cfg),
                     cases=cases, seeds=seeds, methods=METHODS, split=args.split,
                     full_split=not cfg.get("smoke") and args.limit is None,
                     bootstrap_seed=cfg["bootstrap_seed"], task="expert-segment position-only goal",
                     candidate_budget="identical CEM + mean/edit predictions + prefix predictions for every selector")
        write_json(out / "study.json", study)
        model = FrozenModel(cfg, manifest["action_stats"])
        write_json(out / "model.json", dict(device=model.device, identity=model.identity))
        failures = []
        for ci, case in enumerate(cases):
            initial, target, rgb = checked_case(dataset, case)
            harness = make_env(cfg, "N", cfg["environment_seed"], initial, target)
            try:
                write_json(out / f"{case['case_id']}-environment.json", harness.audit_model())
                snapshot, pixels = harness.snapshot(), harness.observe().pixels
                audit_ledger = Ledger(0)
                if harness.evaluate()["distance"] > cfg["success_distance_m"]:
                    harness.replay_check(np.zeros((5, 5), np.float32), audit_ledger)
                append_jsonl(out / "restore-checks.jsonl", dict(case_id=case["case_id"], ledger=asdict(audit_ledger)))
                for seed in seeds:
                    for method in METHODS:
                        started = time.perf_counter()
                        before = harness.actual_steps
                        ledger = Ledger(cfg["total_steps"])
                        row = dict(case_id=case["case_id"], episode=case["episode"], method=method, seed=seed)
                        try:
                            outcome = trial(cfg, harness, snapshot, pixels, rgb, model, scorer, method, seed, ledger)
                            name = f"{case['case_id']}-{seed}-{method}"
                            np.savez_compressed(out / (name + "-observations.npz"), pixels=outcome.pop("pixels"), goal_rgb=rgb)
                            write_json(out / (name + "-evaluation.json"), outcome.pop("evaluations"))
                            write_json(out / (name + "-decisions.json"), outcome.pop("decisions"))
                            row.update(outcome, state="complete")
                            if harness.actual_steps - before != ledger.used:
                                raise RuntimeError("Online action ledger mismatch")
                        except Exception as error:
                            row.update(state="failed", error=type(error).__name__, reason=str(error))
                            failures.append(row.copy())
                        row.update(ledger=asdict(ledger), actual_environment_steps=harness.actual_steps - before,
                                   elapsed_seconds=time.perf_counter() - started)
                        append_jsonl(out / "episodes.jsonl", row)
                print(f"{ci+1}/{len(cases)} {case['case_id']}", flush=True)
            finally:
                harness.close()
        analyze(out, out)
        if failures:
            raise RuntimeError(f"{len(failures)} failed trials; retained in episodes.jsonl and summary.json")
    finally:
        dataset.close()
