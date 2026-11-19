#!/usr/bin/env python3
"""Independent Cube E0/E1/E2 CLI. Import-light help; never auto-download."""

from __future__ import annotations
import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(Path(__file__).parent / "config/research/cube.yaml"))
    p.add_argument("--dataset")
    p.add_argument("--weights")
    p.add_argument("--device")
    p.add_argument("--render-backend", choices=["auto", "egl", "osmesa", "glfw"])
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("doctor")
    q.add_argument("--out")
    q = sub.add_parser("cases")
    q.add_argument("--out", required=True)
    q = sub.add_parser("lock")
    q.add_argument("--cases", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--diagnostic")
    q = sub.add_parser("run")
    q.add_argument("--cases", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--stage", choices=["e0", "e1", "collect", "evaluate"], required=True)
    q.add_argument("--split", choices=["dev", "train", "test"], default="dev")
    q.add_argument("--limit", type=int, default=3)
    q.add_argument("--case-id")
    q.add_argument("--cohort", help="sealed development cohort from screen")
    q.add_argument("--cohort-position", type=int, help="1-based position in the sealed cohort (E0 only)")
    q.add_argument("--paired-e0", help="completed E0 with intervention audit; required for cohort E1")
    q.add_argument("--lock")
    q.add_argument("--diagnostic")
    q.add_argument("--smoke", action="store_true")
    q = sub.add_parser("screen")
    q.add_argument("--cases", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--batch", type=int, default=0)
    q = sub.add_parser("perturb-e0", help="sealed two-case perturbation grid; Frozen only")
    q.add_argument("--cases", required=True)
    q.add_argument("--cohort", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("perturb-e1", help="fixed step-25 recovery comparison against a sealed E0 failure")
    q.add_argument("--cases", required=True)
    q.add_argument("--cohort", required=True)
    q.add_argument("--paired-e0", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("audit")
    q.add_argument("--cases", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--limit", type=int, default=3)
    q.add_argument("--steps", type=int, default=20)
    q = sub.add_parser("train")
    q.add_argument("--cases", required=True)
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q = sub.add_parser("analyze")
    q.add_argument("--runs", nargs="+", required=True)
    q.add_argument("--out", required=True)
    return p


def main():
    args = parser().parse_args()
    from cube_research.protocol import read_config, validate_config

    cfg = read_config(args.config)
    for key in ("dataset", "weights", "device", "render_backend"):
        if getattr(args, key, None) is not None:
            cfg[key] = getattr(args, key)
    if cfg["dataset"]:
        cfg["dataset"] = str(Path(cfg["dataset"]).expanduser().resolve())
    if cfg["weights"]:
        cfg["weights"] = str(Path(cfg["weights"]).expanduser().resolve())
    rules = None
    if args.command == "screen" or getattr(args, "cohort", None):
        from cube_research.cohort import profile

        cfg, rules = profile(cfg)
    validate_config(cfg)
    if cfg["render_backend"] != "auto":
        import os

        os.environ["MUJOCO_GL"] = cfg["render_backend"]
    if args.command == "doctor":
        import importlib.util, json
        from research_common import write_json
        from cube_research.data import local_inputs

        packages = {
            m: importlib.util.find_spec(m) is not None
            for m in (
                "torch",
                "torchvision",
                "mujoco",
                "dm_control",
                "ogbench",
                "h5py",
                "hdf5plugin",
                "hydra",
                "stable_pretraining",
            )
        }
        try:
            local_inputs(cfg)
            inputs = "present"
        except (FileNotFoundError, ValueError) as exc:
            inputs = str(exc)
        result = dict(
            packages=packages,
            inputs=inputs,
            config=cfg,
            status="ready_for_smoke"
            if all(packages.values()) and inputs == "present"
            else "blocked",
            note="Presence check only, no weights loaded, rendering or device validated",
        )
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, indent=2))
        return int(result["status"] == "blocked")
    if args.command == "cases":
        from cube_research.data import make_manifest

        make_manifest(cfg, args.out)
    elif args.command == "lock":
        from cube_research.experiments import lock

        lock(cfg, args.cases, args.out, args.diagnostic)
    elif args.command == "screen":
        from cube_research.cohort import screen

        return int(not screen(cfg, rules, args.cases, args.out, args.batch))
    elif args.command == "perturb-e0":
        from cube_research.perturbations import run_grid

        return int(not run_grid(cfg, args.cases, args.cohort, args.out))
    elif args.command == "perturb-e1":
        from cube_research.perturbation_recovery import run_recovery

        return int(not run_recovery(cfg, args.cases, args.cohort, args.paired_e0, args.out))
    elif args.command == "run":
        from cube_research.experiments import run

        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.cohort and args.stage in ("e0", "e1"):
            args.limit = 1
        if args.cohort_position is not None and not (args.cohort and args.stage == "e0"):
            raise ValueError("--cohort-position is only for cohort E0")
        if args.paired_e0 and not (args.cohort and args.stage == "e1"):
            raise ValueError("--paired-e0 is only for cohort E1")
        if args.smoke:
            if args.split != "dev" or args.stage != "e1":
                raise ValueError("Smoke uses dev E1 only")
            cfg.update(
                total_steps=7,
                history_steps=5,
                extra_budgets=[0],
                horizon=1,
                execute_blocks=1,
                cem_samples=4,
                cem_elites=2,
                cem_iterations=1,
                adapt_steps=1,
                planning_seeds=[42],
            )
            validate_config(cfg)
            args.limit = 1
        return int(
            not run(
                cfg,
                args.cases,
                args.out,
                args.stage,
                args.split,
                args.limit,
                args.lock,
                args.diagnostic,
                args.smoke,
                args.case_id,
                args.cohort,
                args.paired_e0,
                cohort_position=args.cohort_position,
            )
        )
    elif args.command == "audit":
        from cube_research.audit import audit

        return int(not audit(cfg, args.cases, args.out, args.limit, args.steps))
    elif args.command == "train":
        from cube_research.experiments import train

        train(cfg, args.cases, args.runs, args.out)
    elif args.command == "analyze":
        from cube_research.analysis import summarize

        summarize(args.runs, args.out, cfg["bootstrap_seed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
