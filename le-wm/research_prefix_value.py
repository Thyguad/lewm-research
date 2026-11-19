"""Execution-supervised five-step action values on a frozen Cube world model."""
import argparse
import os
from pathlib import Path

from research_common import file_hash, finish_run, new_run, runtime_metadata, write_json

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/research/ideas/prefix-value.yaml")
    p.add_argument("--inputs", type=Path, help="Local data/weights/device/render YAML only")
    for key in ("dataset", "weights", "device"):
        p.add_argument("--" + key)
    p.add_argument("--render-backend", choices=("auto", "glfw", "egl", "osmesa"))
    p.add_argument("--smoke", action="store_true", help="Separate short engineering protocol, never formal evidence")
    sub = p.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Reuse idea1 split; separate training/calibration trajectories")
    prepare.add_argument("--goal-cache", type=Path, required=True)
    collect = sub.add_parser("collect", help="Execute prefix + fixed L2 continuation branches")
    collect.add_argument("--split", choices=("train", "calibration"), required=True)
    collect.add_argument("--round", type=int, choices=(0, 1), default=0)
    collect.add_argument("--trained", type=Path)
    collect.add_argument("--offset", type=int, default=0)
    collect.add_argument("--limit", type=int)
    train = sub.add_parser("train", help="Fit both heads; checkpoint by calibration selection regret")
    train.add_argument("--train-data", nargs="+", type=Path, required=True)
    train.add_argument("--calibration-data", nargs="+", type=Path, required=True)
    evaluate = sub.add_parser("evaluate", help="Four methods, paired cases and planning seeds")
    evaluate.add_argument("--trained", type=Path, required=True)
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--dev-run", type=Path)
    analyze = sub.add_parser("analyze", help="Summarize completed or partial evaluation, no model loading")
    analyze.add_argument("--run", type=Path, required=True)
    for name, command in sub.choices.items():
        command.add_argument("--out", type=Path, required=True, help="A new directory; never overwrite results")
        if name in ("collect", "train", "evaluate"):
            command.add_argument("--manifest", type=Path, required=True)
    return p


def read_config(args):
    import yaml
    from planning_ideas.prefix_contracts import validate_config
    from research_device import validate_device_name
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    local = yaml.safe_load(args.inputs.read_text(encoding="utf-8")) if args.inputs else {}
    for key in ("dataset", "weights", "device", "render_backend"):
        override = getattr(args, key)
        if override is not None:
            cfg[key] = override
        elif key in local:
            cfg[key] = local[key]
    cfg["smoke"] = args.smoke
    if args.smoke:
        cfg.update(total_steps=10, cem_samples=12, cem_iterations=2, cem_elites=3,
                   train_states=4, calibration_states=4, warmup_steps=[0, 5], dev_seeds=[42])
        cfg["pool"].update(size=8, archive_per_iteration=6, edits=2, branches=3)
        cfg["training"].update(steps=20, evaluate_every=10, batch_states=2)
    validate_config(cfg)
    validate_device_name(cfg["device"])
    if cfg["render_backend"] not in ("auto", "glfw", "egl", "osmesa"):
        raise ValueError("Invalid rendering backend")
    if getattr(args, "offset", 0) < 0 or (getattr(args, "limit", None) is not None and args.limit <= 0):
        raise ValueError("Offset must be nonnegative; limit must be positive")
    return cfg


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = read_config(args)
    if args.command in ("prepare", "collect", "evaluate"):
        from cube_research.data import local_inputs
        local_inputs(cfg)
    if cfg["render_backend"] != "auto":
        os.environ["MUJOCO_GL"] = cfg["render_backend"]
    out = new_run(args.out)
    try:
        sources = [Path(__file__), *(ROOT / "planning_ideas").glob("*.py"),
                   *(ROOT / "cube_research").glob("*.py"), ROOT / "research_runtime.py", ROOT / "research_common.py"]
        write_json(out / "run.json", dict(command=args.command, config=cfg, arguments=vars(args),
                   runtime=runtime_metadata(ROOT.parent),
                   source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources}))
        if args.command == "prepare":
            from planning_ideas.prefix_contracts import prepare
            prepare(cfg, args.goal_cache, out)
        elif args.command == "collect":
            from planning_ideas.prefix_data import collect
            collect(cfg, args, out)
        elif args.command == "train":
            from planning_ideas.prefix_training import train
            train(cfg, args, out)
        elif args.command == "evaluate":
            from planning_ideas.prefix_evaluation import evaluate
            evaluate(cfg, args, out)
        else:
            from planning_ideas.prefix_evaluation import analyze
            analyze(args.run, out)
        finish_run(out, "complete")
    except (Exception, KeyboardInterrupt) as error:
        finish_run(out, "failed", error=str(error), error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
