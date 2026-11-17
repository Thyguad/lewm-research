"""Offline candidate audit and aggregate plots for frozen PushT runs.

The audit command is intentionally conservative: it refuses to claim exact
state restoration from PushT's seven-value ``_set_state`` representation.
Use ``--allow-inexact-state`` only for exploratory output, and always inspect
the server replay consistency report first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def _rows(path: str | Path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def audit(args: argparse.Namespace) -> None:
    """Audit snapshots after online collection on a server.

    Full J_enc/J_task replay is delegated to the server runtime because it
    requires pygame/pymunk and the checkpoint encoder.  This command validates
    snapshot schema and joins an externally produced replay table, refusing to
    silently treat missing values as zero.
    """
    import numpy as np
    root = Path(args.run)
    if not args.allow_inexact_state:
        raise RuntimeError("PushT _set_state is not a complete physics snapshot; provide a verified replay table or pass --allow-inexact-state for exploratory output")
    replay = _rows(args.replay) if args.replay else []
    by_key = {(r["plan_id"], r["candidate_id"]): r for r in replay}
    records = []
    for path in sorted((root / "snapshots").glob("*.npz")):
        with np.load(path) as data:
            if not {"actions", "costs", "mean", "std"}.issubset(data.files):
                raise ValueError(f"invalid snapshot schema: {path}")
            costs = data["costs"]
            flat = costs.reshape(-1)
            order = np.argsort(flat)
            plan = int(path.stem.split("_")[1])
            for rank, flat_idx in enumerate(order[: args.keep]):
                candidate_id = int(flat_idx % costs.shape[-1])
                item = {"snapshot": str(path), "plan_id": plan, "candidate_id": candidate_id, "J_pred": float(flat[flat_idx]), "rank": rank}
                item.update(by_key.get((plan, candidate_id), {}))
                records.append(item)
    out = root / "audit_records.jsonl"
    out.write_text("\n".join(json.dumps(x) for x in records) + ("\n" if records else ""))
    summary = []
    for plan_id in sorted({r["plan_id"] for r in records}):
        group = [r for r in records if r["plan_id"] == plan_id and "J_task" in r and "J_enc" in r]
        if not group:
            continue
        best = min(group, key=lambda r: r["J_task"])
        selected = [r for r in group if r.get("selected") or r.get("is_selected")]
        row = {"plan_id": plan_id, "audited_count": len(group), "best_audited_candidate_id": best["candidate_id"], "best_audited_J_task": best["J_task"]}
        if selected:
            row["selection_loss_on_audited_set"] = selected[0]["J_task"] - best["J_task"]
            row["selected_candidate_id"] = selected[0]["candidate_id"]
        summary.append(row)
    (root / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(records)} audit rows to {out}; J_enc/J_task are present only when replay supplied them")


def plots(args: argparse.Namespace) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    roots = [Path(x) for x in args.run]
    root = roots[0]
    frames = []
    for item in roots:
        frame = pd.DataFrame(_rows(item / "episodes.jsonl"))
        if not frame.empty:
            frame["run"] = str(item)
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if episodes.empty:
        raise ValueError("episodes.jsonl is empty")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    grouped = episodes.groupby(["num_samples", "iterations"], dropna=False) if "num_samples" in episodes else [("run", episodes)]
    labels, rates, rate_err, times, counts = [], [], [], [], []
    for label, frame in grouped:
        labels.append(str(label)); p = frame["success"].mean(); n = len(frame); rates.append(p); rate_err.append(1.96 * np.sqrt(max(p * (1 - p), 0.0) / max(n, 1))); times.append(frame["episode_seconds"].mean()); counts.append(n)
    axes[0].errorbar(labels, rates, yerr=rate_err, fmt="o-"); axes[0].set_ylim(0, 1); axes[0].set_ylabel("success rate"); axes[0].set_xlabel("(N, I); 95% normal CI")
    axes[1].plot(labels, times, "o-"); axes[1].set_ylabel("episode seconds"); axes[1].set_xlabel("(N, I)")
    compute = [frame["candidate_predictions"].mean() for _, frame in grouped]
    axes[2].plot(compute, rates, "o-"); axes[2].set_ylabel("success rate"); axes[2].set_xlabel("approx candidate predictions")
    fig.suptitle(f"n={sum(counts)} rows; each point reports its case count")
    fig.tight_layout(); fig.savefig(root / "budget_success_time.png", dpi=160); plt.close(fig)
    audit_path = root / "audit_records.jsonl"
    if audit_path.exists():
        df = pd.DataFrame(_rows(audit_path))
        cols = [c for c in ("J_pred", "J_enc", "J_task") if c in df]
        if len(cols) >= 2:
            ax = df.plot.scatter(x=cols[0], y=cols[1]); ax.get_figure().tight_layout(); ax.get_figure().savefig(root / "audit_cost_scatter.png", dpi=160); plt.close(ax.get_figure())
    cem_path = root / "cem_iterations.jsonl"
    if cem_path.exists():
        cem = pd.DataFrame(_rows(cem_path))
        if not cem.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            if "iteration_min_cost" in cem:
                ax.plot(cem["iteration"], cem["iteration_min_cost"].apply(lambda x: x[0] if isinstance(x, list) else x), label="round minimum")
            if "elite_mean_cost" in cem:
                ax.plot(cem["iteration"], cem["elite_mean_cost"].apply(lambda x: x[0] if isinstance(x, list) else x), label="elite mean")
            returns = cem[cem.get("event", "") == "return_mean"] if "event" in cem else pd.DataFrame()
            if not returns.empty:
                ax.axhline(float(returns.iloc[-1]["returned_mean_cost"][0]), linestyle="--", label="returned mean")
            ax.set_xlabel("CEM iteration"); ax.set_ylabel("cost (smaller is better)"); ax.legend(); fig.tight_layout(); fig.savefig(root / "cem_cost_by_iteration.png", dpi=160); plt.close(fig)
    print(f"wrote plots under {root}")


def parser():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("--run", required=True); a.add_argument("--replay"); a.add_argument("--keep", type=int, default=10); a.add_argument("--allow-inexact-state", action="store_true")
    g = sub.add_parser("plot"); g.add_argument("--run", required=True, nargs="+")
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    (audit if args.command == "audit" else plots)(args)
