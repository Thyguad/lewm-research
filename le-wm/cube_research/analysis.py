"""Case-level paired summaries. Missing/failed planned runs remain explicit."""

from collections import defaultdict
from pathlib import Path
import numpy as np
from research_common import read_json, write_json
from .protocol import CONDITIONS, METHODS

KEY = ("case_id", "condition", "planning_seed", "strategy", "extra_budget", "method")


def summarize(roots, out, seed=99):
    expected = {}
    actual = {}
    run_status = []
    signatures = set()
    for root in map(Path, roots):
        meta = read_json(root / "metadata.json")
        signatures.add(
            (
                meta["protocol_sha256"],
                meta["manifest_sha256"],
                meta["split"],
                meta["smoke"],
                meta.get("weights_sha256"),
                meta.get("model_config_sha256"),
            )
        )
        run_status.append(dict(path=str(root), status=read_json(root / "STATUS.json")))
        for p in read_json(root / "plan.json"):
            key = tuple(p[k] for k in KEY)
            if key in expected:
                raise ValueError("Duplicate planned case/seed/method; do not double count reruns")
            expected[key] = p
        for path in (root / "recoveries").glob("*/result.json"):
            r = read_json(path)
            key = tuple(r[k] for k in KEY)
            if key in actual:
                raise ValueError("Duplicate result")
            actual[key] = r
    if len(signatures) > 1:
        raise ValueError("Do not pool protocols/splits or smoke and substantive runs")
    groups = defaultdict(list)
    for key, p in expected.items():
        r = actual.get(
            key, dict(p, status="missing", reason="No completed result for planned entry")
        )
        groups[(p["condition"], p["method"], p["strategy"], p["extra_budget"])].append(r)
    table = []
    rng = np.random.default_rng(seed)
    for (condition, method, strategy, budget), rows in sorted(groups.items()):
        cases = defaultdict(list)
        for r in rows:
            cases[r["case_id"]].append(r)
        complete_cases = {
            c: rs for c, rs in cases.items() if all(r["status"] == "complete" for r in rs)
        }
        values = np.array([np.mean([r["success"] for r in rs]) for rs in complete_cases.values()])
        lower = np.mean(
            [
                np.mean(
                    [r.get("success", False) if r["status"] == "complete" else False for r in rs]
                )
                for rs in cases.values()
            ]
        )
        ci = None
        if len(values) >= 2:
            boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
            ci = np.quantile(boot, [0.025, 0.975]).tolist()
        table.append(
            dict(
                condition=condition,
                method=method,
                strategy=strategy,
                extra_budget=budget,
                planned_cases=len(cases),
                complete_cases=len(complete_cases),
                planned_runs=len(rows),
                completed_runs=sum(r["status"] == "complete" for r in rows),
                success_rate=float(values.mean()) if len(values) else None,
                case_bootstrap_95=ci,
                success_rate_missing_as_failure=float(lower),
                mean_steps=float(
                    np.mean([r["environment_steps"] for r in rows if r["status"] == "complete"])
                )
                if any(r["status"] == "complete" for r in rows)
                else None,
                mean_cumulative_distance=float(
                    np.mean(
                        [
                            r["cumulative_distance"]
                            for r in rows
                            if r["status"] == "complete" and "cumulative_distance" in r
                        ]
                    )
                )
                if any("cumulative_distance" in r for r in rows)
                else None,
                mean_adaptation_seconds=float(
                    np.mean(
                        [
                            r["ledger"]["adaptation_seconds"]
                            for r in rows
                            if r["status"] == "complete" and "ledger" in r
                        ]
                    )
                )
                if any("ledger" in r for r in rows)
                else None,
                initially_successful_runs=sum(bool(r.get("initial_success")) for r in rows),
                excluded_from_observed_estimate=[
                    dict(
                        case_id=r["case_id"],
                        planning_seed=r["planning_seed"],
                        status=r["status"],
                        reason=r.get("reason"),
                    )
                    for r in rows
                    if r["status"] != "complete"
                ],
            )
        )
    # Pair same case, condition, planning seed, strategy and budget against Frozen.
    # Average seeds WITHIN each case, then bootstrap the case differences.
    paired = []
    for g, rows in sorted(groups.items()):
        condition, method, strategy, budget = g
        if method == "Frozen":
            continue
        by_case = defaultdict(list)
        omitted = []
        for r in rows:
            key = (r["case_id"], condition, r["planning_seed"], strategy, budget, "Frozen")
            base = actual.get(key)
            if r["status"] == "complete" and base and base["status"] == "complete":
                by_case[r["case_id"]].append(float(r["success"]) - float(base["success"]))
            else:
                omitted.append(r["case_id"])
        # Incomplete seed pairs invalidate that case, not just the failed seed.
        for cid in omitted:
            by_case.pop(cid, None)
        diffs = np.array([np.mean(v) for v in by_case.values()])
        interval = None
        if len(diffs) >= 2:
            interval = np.quantile(
                diffs[rng.integers(0, len(diffs), (2000, len(diffs)))].mean(1), [0.025, 0.975]
            ).tolist()
        paired.append(
            dict(
                condition=condition,
                method=method,
                strategy=strategy,
                extra_budget=budget,
                independent_cases=len(diffs),
                mean_success_difference=float(diffs.mean()) if len(diffs) else None,
                case_bootstrap_95=interval,
                missing_case_pairs=sorted(set(omitted)),
            )
        )
    strategy_pairs = []
    # Paired acquisition costs/benefits, including N-condition damage. Total
    # interaction ceiling is the same; passive has zero acquisition overhead.
    for g, rows in sorted(groups.items()):
        condition, method, strategy, budget = g
        if strategy == "passive":
            continue
        for reference in ("passive", "continue"):
            if reference == strategy:
                continue
            by_case = defaultdict(list)
            missing = []
            for r in rows:
                key = (
                    r["case_id"],
                    condition,
                    r["planning_seed"],
                    reference,
                    0 if reference == "passive" else budget,
                    method,
                )
                baseline = actual.get(key)
                if r["status"] == "complete" and baseline and baseline["status"] == "complete":
                    by_case[r["case_id"]].append(float(r["success"]) - float(baseline["success"]))
                else:
                    missing.append(r["case_id"])
            for cid in missing:
                by_case.pop(cid, None)
            diffs = np.array([np.mean(v) for v in by_case.values()])
            ci = None
            if len(diffs) >= 2:
                ci = np.quantile(
                    diffs[rng.integers(0, len(diffs), (2000, len(diffs)))].mean(1), [0.025, 0.975]
                ).tolist()
            strategy_pairs.append(
                dict(
                    condition=condition,
                    method=method,
                    strategy=strategy,
                    reference=reference,
                    extra_budget=budget,
                    independent_cases=len(diffs),
                    missing_case_pairs=sorted(set(missing)),
                    success_difference=float(diffs.mean()) if len(diffs) else None,
                    case_bootstrap_95=ci,
                )
            )
    diagnostic = []
    for strategy in ("passive", "continue", "fixed", "random"):
        for budget in sorted({r.get("extra_budget", 0) for r in actual.values()}):
            rs = [
                r
                for r in actual.values()
                if r["status"] == "complete"
                and r.get("diagnostic")
                and r["strategy"] == strategy
                and r["extra_budget"] == budget
            ]
            if not rs:
                continue
            cm = np.zeros((3, 3), dtype=float)
            nll = []
            brier = []
            acc = defaultdict(list)
            # Each case-condition gets unit total weight across planning seeds.
            counts = defaultdict(int)
            for r in rs:
                counts[(r["case_id"], r["condition"])] += 1
            for r in rs:
                i = CONDITIONS.index(r["condition"])
                p = np.asarray(r["diagnostic"]["probabilities"])
                j = int(p.argmax())
                w = 1 / counts[(r["case_id"], r["condition"])]
                cm[i, j] += w
                nll.append((-np.log(max(p[i], 1e-12)), w))
                brier.append((float(np.sum((p - np.eye(3)[i]) ** 2)), w))
                acc[i].append((float(i == j), w))
            ba = float(
                np.mean(
                    [
                        np.average([v for v, w in acc[i]], weights=[w for v, w in acc[i]])
                        for i in acc
                    ]
                )
            )
            diagnostic.append(
                dict(
                    strategy=strategy,
                    extra_budget=budget,
                    balanced_accuracy=ba,
                    planned_runs=sum(
                        1
                        for v in expected.values()
                        if v["method"] == "Selected"
                        and v["strategy"] == strategy
                        and v["extra_budget"] == budget
                    ),
                    completed_runs=len(rs),
                    class_coverage=len(acc),
                    confusion_case_weighted=cm.tolist(),
                    nll=float(np.average([v for v, w in nll], weights=[w for v, w in nll])),
                    brier=float(np.average([v for v, w in brier], weights=[w for v, w in brier])),
                )
            )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "summary.json",
        dict(
            cells=table,
            paired=paired,
            paired_strategies=strategy_pairs,
            diagnostic=diagnostic,
            runs=run_status,
            note="Observed estimates require all planned seeds per case; missing-as-failure is separate. No seed is an independent case.",
        ),
    )
    lines = [
        "# Cube correction cross table",
        "",
        "Protocol: expert trajectory segment; no full pick-and-place claim.",
        "",
        "| Condition | Frozen | R-I | R-D | R-U |",
        "| --- | --- | --- | --- | --- |",
    ]
    for condition in CONDITIONS:
        values = []
        for method in METHODS:
            row = next(
                (
                    t
                    for t in table
                    if (t["condition"], t["method"], t["strategy"], t["extra_budget"])
                    == (condition, method, "passive", 0)
                ),
                None,
            )
            values.append(
                "—"
                if row is None or row["success_rate"] is None
                else f"{row['success_rate']:.3f} ({row['complete_cases']}/{row['planned_cases']} cases)"
            )
        lines.append("| " + " | ".join([condition, *values]) + " |")
    lines += [
        "",
        "See summary.json for failures, paired differences, strategy/budget curves, and N-condition costs.",
    ]
    (out / "cross-table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
