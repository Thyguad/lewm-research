"""Case-level paired summaries. Missing/failed planned runs remain explicit."""

from collections import defaultdict
from pathlib import Path
import numpy as np
from research_common import read_json, write_json
from .protocol import CONDITIONS, METHODS

KEY = ("case_id", "condition", "planning_seed", "strategy", "extra_budget", "method")
GRID_ID = ('case_id', 'planning_seed', 'perturbation_id', 'condition', 'perturbation_family',
           'xyz_gain', 'pad_friction_scale')


def grid_rows(root, plan):
    """One row per sealed arm. Compare only to the same case and seed's N."""
    root = Path(root)
    expected = {(r['case_id'], r['planning_seed'], r['perturbation_id']): r for r in plan['entries']}
    if len(expected) != len(plan['entries']):
        raise ValueError('Duplicate grid entry')
    actual, audits = {}, {}
    for sub in sorted(root.glob('position-*')):
        paths = sorted((sub / 'recoveries').glob('*/result.json'))
        failures = sorted(sub.glob('failure-*.json'))
        if not paths and not failures:
            continue
        metadata = read_json(sub / 'metadata.json')
        if (metadata.get('extension_seal') != plan['seal_sha256']
                or metadata.get('experiment_protocol') != plan['experiment_protocol']
                or metadata.get('cohort_sha256') != plan['cohort_sha256']
                or metadata.get('config') != plan['config']):
            raise ValueError('Cannot pool or reuse receipts with a different sealed execution identity')
        receipt_audit = sub / 'matched-command-audit/audit.json'
        audit = read_json(receipt_audit) if receipt_audit.exists() else {}
        for p in paths + failures:
            r = read_json(p)
            key = r['case_id'], r['planning_seed'], r['perturbation_id']
            exp = expected.get(key)
            if (not exp or any(r.get(k) != exp[k] for k in GRID_ID)
                    or exp['cohort_position'] != metadata['cohort_position']):
                raise ValueError('Result does not match sealed case, seed or perturbation')
            if p in failures and key in actual:
                continue  # The inner failed result already holds the detailed receipt.
            if key in actual:
                raise ValueError('Duplicate grid result')
            trajectory = p.parent / 'evaluation.json'
            if p in failures:
                trajectory = sub / 'failed-prefix' / '-'.join(map(str, key)) / 'evaluation.json'
            truth = read_json(trajectory) if trajectory.exists() else []
            actual[key] = (r, truth)
            audits[key] = audit.get(r['perturbation_id'])

    rows = []
    for key, entry in expected.items():
        r, ts = actual.get(key, ({}, []))
        termination = r.get('termination')
        observed = r.get('status') == 'complete' and termination in ('success', 'budget')
        success = r['success'] if observed else None
        n = actual.get((*key[:2], 'N'), ({}, []))[0]
        valid_pair = observed and n.get('status') == 'complete' and n.get('termination') in ('success', 'budget')
        ledger = r.get('ledger', {})
        audit = audits.get(key)
        rows.append(dict(entry, status='success' if success else 'task_failure' if observed
                         else 'execution_error' if r.get('status') == 'failed' else 'unfinished',
                         success=success, first_success_step=r.get('first_success_step') if success else None,
                         termination=termination, reason=r.get('reason', 'No recorded execution' if not r else None),
                         initial_distance_m=ts[0]['distance'] if ts else None,
                         min_distance_m=min(t['distance'] for t in ts) if ts else None,
                         final_distance_m=ts[-1]['distance'] if ts else None,
                         success_delta_vs_n=int(success) - int(n['success']) if valid_pair else None,
                         first_success_step_delta_vs_n=(r['first_success_step'] - n['first_success_step']
                                                        if valid_pair and success and n['success'] else None),
                         unfinished_at_decision=bool(r.get('environment_steps', 0) >= plan['config']['history_steps']
                                                     and (not success or r['first_success_step'] > plan['config']['history_steps'])),
                         audit=audit, verified_task_loss=bool(valid_pair and n['success'] and not success
                                                   and audit and audit['physical_response_verified']),
                         model=ledger.get('model'), planning_seconds=ledger.get('planning_seconds'),
                         wall_seconds=r.get('wall_seconds'), online_steps=r.get('environment_steps',
                                                                              sum(ledger['phases'].values()) if ledger else None),
                         offline_snapshot_steps=ledger.get('offline_replay_steps'),
                         offline_snapshot_seconds=r.get('snapshot_replay_seconds'),
                         offline_matched_steps=audit['offline_steps'] if audit else None,
                         offline_matched_seconds=audit['offline_seconds'] if audit else None))
    return rows


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
                meta.get("cohort_sha256"),
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
                early_success_before_recovery_runs=sum(r.get("recovery_state") == "early_success" for r in rows),
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
        early = 0
        for r in rows:
            key = (r["case_id"], condition, r["planning_seed"], strategy, budget, "Frozen")
            base = actual.get(key)
            if r["status"] == "complete" and base and base["status"] == "complete":
                if r.get("recovery_state") == "early_success" or base.get("recovery_state") == "early_success":
                    early += 1
                else:
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
                early_success_seed_pairs=early,
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
            early = 0
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
                    if r.get("prefix_success") or baseline.get("prefix_success"):
                        early += 1
                    else:
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
                    early_success_seed_pairs=early,
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
