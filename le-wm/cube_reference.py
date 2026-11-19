"""Normal-condition Cube reference evaluation; no historical-case prerequisites."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from research_common import write_json
from research_device import prepare_headless


def parser():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["manifest", "evaluate-50"])
    p.add_argument("--config", type=Path, default=root / "le-wm/config/research/cube-local.yaml")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--manifest", type=Path, help="fixed cases for evaluate-50")
    p.add_argument("--check-only", action="store_true",
                   help="validate inputs and record identity without environment rollouts")
    p.add_argument("--device", default=None, help="override config device")
    p.add_argument("--render-backend", choices=["auto", "egl", "osmesa", "glfw"])
    p.add_argument("--protocol", choices=["paper-300x10-top30", "public-300x30-top30"],
                   default="paper-300x10-top30")
    return p


def check_manifest(path, expected):
    if json.loads(Path(path).read_text()) != expected:
        raise ValueError("Manifest does not match this dataset and reference sampling")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.mode == "evaluate-50" and not args.manifest:
        p.error("evaluate-50 requires --manifest")
    if args.check_only and args.mode != "evaluate-50":
        p.error("--check-only is only for evaluate-50")

    cfg = yaml.safe_load(args.config.read_text())
    if args.device is not None:
        cfg["device"] = args.device
    backend = args.render_backend or cfg.get("render_backend", "auto")
    if backend not in ("auto", "egl", "osmesa", "glfw"):
        raise ValueError(f"Unsupported render backend: {backend}")
    if backend != "auto":
        os.environ["MUJOCO_GL"] = backend
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw" if sys.platform == "darwin" else "egl")
    prepare_headless()
    os.environ["HF_HUB_OFFLINE"] = "1"

    from cube_research.data import local_inputs
    from cube_research.model import FrozenModel
    from cube_research.reference import (
        CEM, EVAL, PLAN, PROTOCOLS, fit_actions, fixed_manifest,
        open_dataset, run_policy, runtime_identity,
    )

    local_inputs(cfg)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / "STATUS.json"
    write_json(status, {"status": "running", "mode": args.mode})
    data = None
    try:
        data = open_dataset(cfg["dataset"], out / "local-cache")
        manifest = fixed_manifest(data)
        write_json(out / "reference-50.json", manifest)
        if args.mode == "evaluate-50":
            check_manifest(args.manifest, manifest)
            scaler = fit_actions(data)
            stats = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                         rows=int(scaler.n_samples_seen_))
            bridge = FrozenModel(cfg, stats)
            write_json(out / "runtime.json", runtime_identity(bridge.device, bridge))
            write_json(out / "protocol.json", dict(
                name=args.protocol, condition="N", evaluation=EVAL,
                planning=vars(PLAN),
                solver=dict(CEM, n_steps=PROTOCOLS[args.protocol], device=bridge.device),
                scaler=stats, mujoco_gl=os.environ["MUJOCO_GL"],
            ))
            cases = manifest["cases"]
            write_json(out / "planned-evaluation.json",
                       dict(cases=cases, policies=["cem", "zero", "random"]))
            if not args.check_only:
                run_policy(data, cases, out / "cem", "cem", bridge.base, scaler,
                           bridge.device, args.protocol)
                for name in ("zero", "random"):
                    run_policy(data, cases, out / name, name)
    except BaseException as exc:
        write_json(status, dict(status="unfinished", mode=args.mode, error=repr(exc)), overwrite=True)
        raise
    else:
        write_json(status, dict(
            status="ready-no-rollouts" if args.check_only else "completed", mode=args.mode,
        ), overwrite=True)
    finally:
        if data is not None and data.h5_file is not None:
            data.h5_file.close()


if __name__ == "__main__":
    main()
