"""CLI for independent planning-method experiments with frozen LeWM."""

import argparse
import os
from pathlib import Path

from cube_research.data import local_inputs
from research_common import file_hash, finish_run, new_run, runtime_metadata, write_json

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/research/ideas/goal-set.yaml")
    p.add_argument("--inputs", type=Path, help="YAML supplying local dataset, weights, device and render backend only")
    p.add_argument("--dataset")
    p.add_argument("--weights")
    p.add_argument("--device")
    p.add_argument("--render-backend", choices=("auto", "glfw", "egl", "osmesa"))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Create episode manifest, visualize pairs and encode train/dev RGB")
    fit = sub.add_parser("train", help="Fit all heads from cached features on CPU")
    fit.add_argument("--cache", required=True, type=Path)
    evaluate = sub.add_parser("evaluate", help="Run paired Cube closed-loop trials")
    for name in ("cache", "trained"):
        evaluate.add_argument(f"--{name}", required=True, type=Path)
    evaluate.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--dev-run", type=Path, help="Required for test; completed full dev run of these heads")
    evaluate.add_argument("--mechanism", action="store_true", help="Dev-only replacement goals from independent donor trajectories")
    evaluate.add_argument("--smoke", action="store_true", help="One dev case/seed, reduced CEM and 10 control steps")
    evaluate.add_argument("--dev-prefix", type=int, help="First N dev cases, all seeds/methods, full CEM; not valid for test lock")
    diagnose = sub.add_parser("diagnose", help="Dev-only scoring/physics agreement on shared fixed action candidates")
    for name in ("cache", "trained"):
        diagnose.add_argument(f"--{name}", required=True, type=Path)
    diagnose.add_argument("--candidates", type=int, default=32)
    for command in sub.choices.values():
        command.add_argument("--out", type=Path, required=True, help="New output directory")
    return p


def read_config(args):
    import yaml
    from research_device import validate_device_name

    cfg = yaml.safe_load(args.config.read_text())
    local = yaml.safe_load(args.inputs.read_text()) if args.inputs else {}
    for name in ("dataset", "weights", "device", "render_backend"):
        override = getattr(args, name)
        if override is not None:
            cfg[name] = override
        elif name in local:
            cfg[name] = local[name]
    validate_device_name(cfg["device"])
    if cfg["render_backend"] not in ("auto", "glfw", "egl", "osmesa"):
        raise ValueError("Invalid render backend")
    protocol = dict(protocol="cube-planner-study-v1", sensors="rgb", image_size=224,
                    action_dim=5, action_block=5, horizon=5, execute_blocks=1, goal_offset=25,
                    total_steps=50, success_distance_m=.04, cem_samples=300, cem_iterations=10, cem_elites=30)
    if any(cfg.get(k) != v for k, v in protocol.items()):
        raise ValueError("cube-planner-study-v1 has fixed task/interaction/CEM parameters")
    for key in ("dev_seeds", "test_seeds"):
        seeds = cfg[key]
        if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"Invalid planning seeds: {key}")
    pair, train = cfg["pairing"], cfg["training"]
    for source, keys in ((cfg, ("score_batch_size", "train_episodes", "dev_episodes", "test_episodes",
                                "donor_episodes", "dev_cases", "mechanism_cases")),
                         (pair, ("frame_stride", "max_per_label")),
                         (train, ("rank", "hidden", "steps", "cpu_threads", "batch_size"))):
        if any(type(source[key]) is not int or source[key] <= 0 for key in keys):
            raise ValueError(f"Expected positive integers for {keys}")
    if (not 0 < pair["positive_m"] < pair["negative_m"] < pair["hard_max_m"]
            or not 0 <= train["radius_quantile"] <= 1 or train["batch_size"] % 2):
        raise ValueError("Invalid pair thresholds, radius quantile or unbalanced minibatch size")
    return cfg


def main(argv=None):
    args = parser().parse_args(argv)
    cfg = read_config(args)
    local_inputs(cfg)
    if cfg["render_backend"] != "auto":
        os.environ["MUJOCO_GL"] = cfg["render_backend"]
    out = new_run(args.out)
    try:
        from planning_ideas.goal_training import fit, prepare
        sources = [Path(__file__), *(ROOT / "planning_ideas").glob("*.py"),
                   *(ROOT / "cube_research").glob("*.py"), ROOT / "research_runtime.py"]
        write_json(out / "run.json", dict(command=args.command, config=cfg,
                                          arguments=vars(args), runtime=runtime_metadata(ROOT.parent),
                                          source_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sources}))
        if args.command == "prepare":
            prepare(cfg, out)
        elif args.command == "train":
            fit(cfg, args.cache, out)
        else:
            from planning_ideas.goal_eval import diagnose, evaluate
            (evaluate if args.command == "evaluate" else diagnose)(cfg, args, out)
        finish_run(out, "complete")
    except (Exception, KeyboardInterrupt) as error:
        finish_run(out, "failed", error=str(error), error_type=type(error).__name__)
        raise


if __name__ == "__main__":
    main()
