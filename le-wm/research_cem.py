"""Frozen PushT budget experiments and CEM diagnostics.

Examples are documented in README_frozen_diagnostics.md.  Commands in this
file require the server environment at runtime; importing and ``--help`` do
not load a model or create an environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from research_common import (
    Case,
    freeze_eval,
    read_cases,
    runtime_metadata,
    source_identity,
    stable_case_id,
    sync_device,
    write_cases,
    write_json,
)


def _load_cfg(args: argparse.Namespace) -> dict[str, Any]:
    import yaml

    cfg = {
        "horizon": args.horizon,
        "receding_horizon": args.receding_horizon,
        "action_block": args.action_block,
        "eval_budget": args.eval_budget,
        "goal_offset": args.goal_offset,
        "num_samples": args.num_samples,
        "iterations": args.iterations,
        "topk": args.topk,
        "warm_start": args.warm_start,
        "seed": args.planning_seed,
        "device": args.device,
        "policy": args.policy,
        "cache_dir": args.cache_dir,
        "dataset": args.dataset,
    }
    if args.config:
        with open(args.config) as f:
            loaded = yaml.safe_load(f) or {}
        cfg.update(loaded)
    return cfg


def generate_cases(args: argparse.Namespace) -> None:
    """Sample and persist valid dataset rows once for all budget conditions."""
    import numpy as np
    import stable_worldmodel as swm

    dataset = swm.data.HDF5Dataset(args.dataset, cache_dir=args.cache_dir)
    ep_col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = dataset.get_col_data(ep_col)
    steps = dataset.get_col_data("step_idx")
    by_ep: dict[int, int] = {}
    for ep, step in zip(episodes, steps):
        by_ep[int(ep)] = max(by_ep.get(int(ep), -1), int(step))
    valid = [(ep, step) for ep, last in by_ep.items() for step in range(last - args.goal_offset)]
    if len(valid) < args.num_cases:
        raise ValueError(f"Only {len(valid)} valid starts, requested {args.num_cases}")
    rng = np.random.default_rng(args.case_seed)
    selected = rng.choice(len(valid), size=args.num_cases, replace=False)
    cases = [
        Case(stable_case_id(int(valid[i][0]), int(valid[i][1]), args.goal_offset), int(valid[i][0]), int(valid[i][1]), args.goal_offset)
        for i in np.sort(selected)
    ]
    write_cases(args.output, cases, metadata={"case_seed": args.case_seed, "dataset": args.dataset})
    print(f"wrote {len(cases)} cases to {args.output}")


def _transform(cfg: Any):
    import stable_pretraining as spt
    import torch
    from torchvision.transforms import v2 as transforms

    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.img_size),
    ])


class LoggingCEMSolver:
    """CEM compatible with stable-worldmodel's solver protocol.

    The implementation intentionally mirrors the reference solver, including
    ``std`` for the distribution scale and returning the final updated mean.
    Diagnostic RNG is NumPy-only and never touches ``torch_gen``.
    """

    def __init__(self, model, *, num_samples, n_steps, topk, device, seed, log_path=None, snapshot_dir=None, formal_timing=True, diagnostic_rng_seed=0):
        import numpy as np
        import torch

        self.model, self.num_samples, self.n_steps, self.topk = model, int(num_samples), int(n_steps), int(topk)
        self.device, self.seed = device, int(seed)
        self.torch_gen = torch.Generator(device=device).manual_seed(self.seed)
        self.log_path = Path(log_path) if log_path else None
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.formal_timing = formal_timing
        self.diagnostic_rng = np.random.default_rng(diagnostic_rng_seed)
        self.plan_id = 0
        self._n_envs = 1

    def configure(self, *, action_space, n_envs, config):
        self._action_space, self._n_envs, self._config = action_space, n_envs, config
        self._action_dim = int(np.prod(action_space.shape[1:]))

    @property
    def n_envs(self): return self._n_envs
    @property
    def action_dim(self): return self._action_dim * self._config.action_block
    @property
    def horizon(self): return self._config.horizon
    def __call__(self, *a, **kw): return self.solve(*a, **kw)

    def _write(self, row):
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(json.dumps(row, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")

    def solve(self, info_dict: dict, init_action=None):
        import numpy as np
        import torch

        if self.topk < 2 or self.topk > self.num_samples:
            raise ValueError("topk must satisfy 2 <= topk <= num_samples")
        sync_device(self.device)
        formal_start = time.perf_counter()
        mean = torch.zeros((self.n_envs, self.horizon, self.action_dim), device=self.device) if init_action is None else init_action.to(self.device)
        var = torch.ones_like(mean)
        plan_id = self.plan_id
        self.plan_id += 1
        all_logs = []
        search_seconds = 0.0
        for iteration in range(self.n_steps):
            candidates = torch.randn((self.n_envs, self.num_samples, self.horizon, self.action_dim), generator=self.torch_gen, device=self.device) * var[:, None] + mean[:, None]
            candidates[:, 0] = mean
            expanded = {}
            for key, value in info_dict.items():
                if torch.is_tensor(value):
                    expanded[key] = value.unsqueeze(1).expand(self.n_envs, self.num_samples, *value.shape[1:])
                elif isinstance(value, np.ndarray):
                    expanded[key] = np.repeat(value[:, None, ...], self.num_samples, axis=1)
                else:
                    expanded[key] = value
            score_start = time.perf_counter()
            costs = self.model.get_cost(expanded, candidates)
            sync_device(self.device)
            score_elapsed = time.perf_counter() - score_start
            search_seconds += score_elapsed
            vals, inds = torch.topk(costs, self.topk, dim=1, largest=False)
            batch = torch.arange(self.n_envs, device=self.device)[:, None]
            elites = candidates[batch, inds]
            mean, var = elites.mean(1), elites.std(1)
            row = {"case_id": getattr(self, "case_id", None), "plan_id": plan_id, "environment_step": getattr(self, "environment_step", None), "iteration": iteration, "candidate_count": self.num_samples, "horizon": self.horizon, "topk": self.topk, "elite_mean_cost": vals.mean(1).detach().cpu().tolist(), "iteration_min_cost": costs.min(1).values.detach().cpu().tolist(), "distribution_std_mean": var.mean().item(), "score_seconds": score_elapsed}
            all_logs.append(row)
            self._write(row)
            if self.snapshot_dir:
                self.snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot = {"actions": candidates.detach().cpu().numpy(), "costs": costs.detach().cpu().numpy(), "elite_indices": inds.detach().cpu().numpy(), "mean": mean.detach().cpu().numpy(), "std": var.detach().cpu().numpy()}
                for key in ("state", "goal_state", "pixels", "goal"):
                    value = info_dict.get(key)
                    if torch.is_tensor(value):
                        snapshot[key] = value.detach().cpu().numpy()
                    elif isinstance(value, np.ndarray):
                        snapshot[key] = value
                if hasattr(self, "case_id"):
                    snapshot["case_id"] = np.asarray([self.case_id])
                np.savez_compressed(self.snapshot_dir / f"plan_{plan_id:06d}_iter_{iteration:03d}.npz", **snapshot)
        # The returned action is the updated mean. Score it separately.
        final_info = {}
        for key, value in info_dict.items():
            if torch.is_tensor(value):
                final_info[key] = value.unsqueeze(1)
            elif isinstance(value, np.ndarray):
                final_info[key] = value[:, None, ...]
            else:
                final_info[key] = value
        extra_start = time.perf_counter()
        final_score = self.model.get_cost(final_info, mean[:, None])[:, 0]
        sync_device(self.device)
        extra_score_seconds = time.perf_counter() - extra_start
        formal_elapsed = search_seconds
        self._write({"case_id": getattr(self, "case_id", None), "plan_id": plan_id, "environment_step": getattr(self, "environment_step", None), "event": "return_mean", "returned_mean_cost": final_score.detach().cpu().tolist(), "extra_score_seconds": extra_score_seconds, "formal_solve_seconds": formal_elapsed, "wall_seconds_including_diagnostics": time.perf_counter() - formal_start})
        return {"actions": mean.detach().cpu(), "mean": [mean.detach().cpu()], "var": [var.detach().cpu()], "costs": [r["elite_mean_cost"] for r in all_logs[-1:]], "diagnostic": all_logs}


class InstrumentedPolicy:
    """Small adapter that exposes the environment-step context to the solver."""

    def __init__(self, policy, solver):
        self._policy, self._solver, self.environment_step = policy, solver, 0

    def set_env(self, env):
        self._policy.set_env(env)

    def get_action(self, info):
        self._solver.environment_step = self.environment_step
        action = self._policy.get_action(info)
        self.environment_step += 1
        return action


def run_experiment(args: argparse.Namespace) -> None:
    import numpy as np
    import stable_worldmodel as swm
    import torch
    from sklearn import preprocessing

    cases, case_meta = read_cases(args.cases)
    cfg = _load_cfg(args)
    if not cases:
        raise ValueError("case manifest is empty")
    device = torch.device(cfg["device"])
    dataset = swm.data.HDF5Dataset(cfg["dataset"], cache_dir=cfg["cache_dir"])
    model = freeze_eval(swm.wm.utils.load_pretrained(cfg["policy"], cache_dir=cfg["cache_dir"]))
    if next(model.parameters()).device != device:
        model.to(device)
    model.interpolate_pos_encoding = True
    print("loaded_model", source_identity(model))
    process = {}
    for col in ("action", "proprio", "state"):
        if col in dataset.column_names:
            scaler = preprocessing.StandardScaler()
            values = dataset.get_col_data(col)
            scaler.fit(values[~np.isnan(values).any(axis=1)])
            process[col] = scaler
            if col != "action":
                process[f"goal_{col}"] = scaler
    transform = {"pixels": _transform(type("C", (), {"img_size": args.img_size})()), "goal": _transform(type("C", (), {"img_size": args.img_size})())}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_metadata.json", {**runtime_metadata(Path.cwd()), "config": cfg, "cases": str(args.cases), "case_metadata": case_meta, "model": source_identity(model), "model_frozen": all(not p.requires_grad for p in model.parameters())})
    config = swm.PlanConfig(horizon=cfg["horizon"], receding_horizon=cfg["receding_horizon"], action_block=cfg["action_block"], warm_start=cfg["warm_start"])
    rows = []
    for case in cases:
        solver = LoggingCEMSolver(model, num_samples=cfg["num_samples"], n_steps=cfg["iterations"], topk=cfg["topk"], device=device, seed=cfg["seed"], log_path=output / "cem_iterations.jsonl", snapshot_dir=output / "snapshots" if args.save_snapshots else None)
        solver.case_id = case.case_id
        policy = InstrumentedPolicy(swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform), solver)
        world = swm.World(env_name="swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=2 * cfg["eval_budget"], verbose=0)
        t0 = time.perf_counter()
        metrics = world.evaluate_from_dataset(dataset, [case.episode], [case.start_step], case.goal_offset, cfg["eval_budget"], callables=[{"method": "_set_state", "args": {"state": {"value": "state"}}}, {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}}], save_video=False)
        elapsed = time.perf_counter() - t0
        rows.append({"case_id": case.case_id, "episode": case.episode, "start_step": case.start_step, "goal_offset": case.goal_offset, "success": bool(metrics["episode_successes"][0]), "environment_steps": cfg["eval_budget"], "planning_calls": solver.plan_id, "episode_seconds": elapsed, "candidate_predictions": solver.plan_id * cfg["iterations"] * cfg["num_samples"] * cfg["horizon"], "planning_seed": cfg["seed"], "num_samples": cfg["num_samples"], "iterations": cfg["iterations"]})
        world.close()
    (output / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} episode rows to {output}")


def preflight(args: argparse.Namespace) -> None:
    """Load only the checkpoint and report the actual cost module identity."""
    import torch
    import stable_worldmodel as swm

    model = freeze_eval(swm.wm.utils.load_pretrained(args.policy, cache_dir=args.cache_dir))
    model.to(torch.device(args.device))
    print(json.dumps({"model": source_identity(model), "device": str(args.device), "frozen": all(not p.requires_grad for p in model.parameters()), "parameter_count": sum(p.numel() for p in model.parameters())}, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate-cases")
    g.add_argument("--dataset", required=True); g.add_argument("--cache-dir", default=None); g.add_argument("--output", required=True); g.add_argument("--num-cases", type=int, default=20); g.add_argument("--goal-offset", type=int, default=25); g.add_argument("--case-seed", type=int, default=123)
    r = sub.add_parser("run")
    r.add_argument("--cases", required=True); r.add_argument("--policy", required=True); r.add_argument("--dataset", default="pusht_expert_train"); r.add_argument("--cache-dir", default=None); r.add_argument("--output", required=True); r.add_argument("--device", default="cuda"); r.add_argument("--horizon", type=int, default=5); r.add_argument("--receding-horizon", type=int, default=5); r.add_argument("--action-block", type=int, default=5); r.add_argument("--eval-budget", type=int, default=50); r.add_argument("--goal-offset", type=int, default=25); r.add_argument("--num-samples", type=int, default=100); r.add_argument("--iterations", type=int, default=10); r.add_argument("--topk", type=int, default=10); r.add_argument("--planning-seed", type=int, default=42); r.add_argument("--img-size", type=int, default=224); r.add_argument("--warm-start", action=argparse.BooleanOptionalAction, default=True); r.add_argument("--save-snapshots", action="store_true"); r.add_argument("--config")
    f = sub.add_parser("preflight"); f.add_argument("--policy", required=True); f.add_argument("--cache-dir", default=None); f.add_argument("--device", default="cuda")
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    {"generate-cases": generate_cases, "run": run_experiment, "preflight": preflight}[a.command](a)
