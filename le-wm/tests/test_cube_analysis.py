from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cube_research.analysis import summarize
from research_common import write_json, read_json


def test_failed_and_missing_runs_not_dropped_and_seeds_clustered(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    write_json(
        root / "metadata.json",
        dict(protocol_sha256="a", manifest_sha256="b", split="dev", smoke=False),
    )
    write_json(root / "STATUS.json", dict(state="incomplete"))
    plan = []
    for c in ("a", "b"):
        for seed in (1, 2):
            for method in ("Frozen", "R-I"):
                p = dict(
                    case_id=c,
                    planning_seed=seed,
                    condition="I",
                    strategy="passive",
                    extra_budget=0,
                    method=method,
                )
                plan.append(p)
                if c == "b" and method == "R-I":
                    continue
                path = root / "recoveries" / str(len(plan))
                path.mkdir(parents=True)
                write_json(
                    path / "result.json",
                    dict(p, status="complete", success=method == "R-I", environment_steps=10),
                )
    write_json(root / "plan.json", plan)
    summarize([root], tmp_path / "summary")
    s = read_json(tmp_path / "summary/summary.json")
    r = next(r for r in s["cells"] if r["method"] == "R-I")
    assert r["planned_cases"] == 2 and r["complete_cases"] == 1
    assert r["success_rate_missing_as_failure"] == 0.5
    assert len(r["excluded_from_observed_estimate"]) == 2
    assert s["paired"][0]["independent_cases"] == 1
    assert s["paired"][0]["missing_case_pairs"] == ["b"]


def test_empty_cross_table_has_no_fabricated_values(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    write_json(
        r / "metadata.json",
        dict(protocol_sha256="a", manifest_sha256="b", split="dev", smoke=False),
    )
    write_json(r / "STATUS.json", dict(state="failed"))
    write_json(r / "plan.json", [])
    summarize([r], tmp_path / "s")
    assert "| N | — | — | — | — |" in (tmp_path / "s/cross-table.md").read_text()
