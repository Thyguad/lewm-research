"""Train on executed branch outcomes; choose checkpoints on held-out trajectory regret."""
import copy
import time
from pathlib import Path

import numpy as np

from research_common import append_jsonl, file_hash, read_json, read_rows, require_complete, write_json
from .prefix_contracts import SCHEMA, data_signature, features, load_manifest, signature

HEADS = ("value-no-future", "value")


def load_groups(cfg, manifest, manifest_path, paths, split):
    expected_manifest = file_hash(manifest_path / "manifest.json")
    expected_cases = {c["case_id"]: c for c in manifest["cases"][split]}
    groups, seen, sources, skipped = [], set(), [], []
    for path in paths:
        path = Path(path)
        require_complete(path)
        info = read_json(path / "collection.json")
        run = read_json(path / "run.json")
        if info["manifest_sha256"] != expected_manifest or info["split"] != split or data_signature(run["config"]) != data_signature(cfg):
            raise ValueError("Collection identity/split/config mismatch")
        if split == "calibration" and info["round"] != 0:
            raise ValueError("Calibration must remain independent of learned collection")
        rows = read_rows(path / "states.jsonl")
        ids = [r["state_id"] for r in rows]
        if ids != info["requested_states"] or len(ids) != len(set(ids)):
            raise ValueError("Incomplete or duplicate state records in collection")
        for row in rows:
            if row["case_id"] not in expected_cases or row["episode"] != expected_cases[row["case_id"]]["episode"]:
                raise ValueError("Collection case is outside its declared split")
            if row["round"] != info["round"] or row["warmup_steps"] != expected_cases[row["case_id"]]["warmup_steps"]:
                raise ValueError("Changed state generation identity")
            if row["state_id"] in seen:
                raise ValueError("Duplicate state across collection chunks")
            seen.add(row["state_id"])
            if row["state"] == "skipped":
                skipped.append(dict(state_id=row["state_id"], reason=row["reason"]))
                continue
            if row["state"] != "complete" or not row["budget"]["accounted"]:
                raise ValueError("Failed/unaccounted state cannot silently enter training")
            folder = path / "states" / row["state_id"]
            for name in ("inputs", "labels"):
                ext = "npz" if name == "inputs" else "json"
                if file_hash(folder / f"{name}.{ext}") != row[f"{name}_sha256"]:
                    raise ValueError(f"Modified {name} file")
            with np.load(folder / "inputs.npz", allow_pickle=False) as data:
                inputs = {k: data[k].copy() for k in data.files}
            # Validate both dimensional contracts and the no-privilege allowlist.
            features(inputs, manifest["action_stats"], future=True)
            labels = np.asarray(read_json(folder / "labels.json")["utility"], np.float32)
            if labels.shape != (len(inputs["commands"]),) or not np.isfinite(labels).all() or (labels < 0).any() or (labels > 1).any():
                raise ValueError("Invalid utility labels")
            groups.append(dict(inputs=inputs, labels=labels, episode=row["episode"],
                               state_id=row["state_id"], round=row["round"]))
        sources.append(dict(path=str(path), collection_sha256=file_hash(path / "collection.json"),
                            states_sha256=file_hash(path / "states.jsonl"),
                            actual_environment_steps=sum(r["budget"]["actual_environment_steps"] for r in rows)))
    if not groups:
        raise ValueError(f"No usable {split} groups")
    return groups, sources, skipped


def grouped_metrics(scores, groups, tolerance):
    """Average states within source trajectories, then trajectories equally."""
    episodes = {}
    for score, group in zip(scores, groups):
        target = group["labels"]
        difference = target[:, None] - target[None, :]
        pairs = difference > tolerance
        agreement = (score[:, None] > score[None, :]).astype(float)
        agreement += .5 * (score[:, None] == score[None, :])
        result = dict(regret=float(target.max() - target[int(np.argmax(score))]),
                      mse=float(np.mean((score - target) ** 2)),
                      pair_accuracy=float(agreement[pairs].mean()) if pairs.any() else None,
                      oracle_utility=float(target.max()), chosen_utility=float(target[int(np.argmax(score))]),
                      informative_state_fraction=float(pairs.any()),
                      successful_branch_fraction=float(np.mean(target >= .9)))
        episodes.setdefault(group["episode"], []).append(result)
    out = {}
    for key in ("regret", "mse", "pair_accuracy", "oracle_utility", "chosen_utility",
                "informative_state_fraction", "successful_branch_fraction"):
        values = [[row[key] for row in rows if row[key] is not None] for rows in episodes.values()]
        means = [float(np.mean(v)) for v in values if v]
        out[key] = float(np.mean(means)) if means else None
    out.update(states=len(groups), trajectories=len(episodes))
    return out


def train(cfg, args, out):
    import torch
    from .prefix_value import ValueHead, objective
    spec = cfg["training"]
    torch.set_num_threads(spec["cpu_threads"])
    manifest = load_manifest(cfg, args.manifest, check_assets=False)
    train_groups, sources, train_skips = load_groups(cfg, manifest, args.manifest, args.train_data, "train")
    val_groups, val_sources, val_skips = load_groups(cfg, manifest, args.manifest, args.calibration_data, "calibration")
    if {g["episode"] for g in train_groups} & {g["episode"] for g in val_groups}:
        raise ValueError("Training/calibration trajectory leakage")
    rounds = {g["round"] for g in train_groups}
    if not rounds <= {0, 1} or 0 not in rounds:
        raise ValueError("Training needs base data; refresh is optional and limited to one round")
    checkpoint = dict(schema=SCHEMA, manifest_sha256=file_hash(args.manifest / "manifest.json"),
                      config_sha256=signature(cfg), action_stats=manifest["action_stats"],
                      round=max(rounds), smoke=cfg.get("smoke", False), hidden=spec["hidden"], heads={})
    report = dict(train_sources=sources, calibration_sources=val_sources,
                  skipped=dict(train=train_skips, calibration=val_skips),
                  training_states=len(train_groups), calibration_states=len(val_groups),
                  training_trajectories=len({g["episode"] for g in train_groups}),
                  calibration_trajectories=len({g["episode"] for g in val_groups}),
                  training_labels=sum(len(g["labels"]) for g in train_groups),
                  collection_rounds=sorted(rounds),
                  cost_scope="Listed complete source runs, including their skips/replay/warmup; discarded failed runs remain separately logged",
                  backbone_updates=0, head_device="cpu", head_dtype="float32", methods={})
    # Uniform source-trajectory sampling prevents multiple states/pairs from one
    # trajectory dominating the optimizer. Identical draws for both ablations.
    by_episode = {}
    for i, group in enumerate(train_groups):
        by_episode.setdefault(group["episode"], []).append(i)
    episodes = sorted(by_episode)
    for method in HEADS:
        started = time.perf_counter()
        torch.manual_seed(spec["seed"])
        rng = np.random.default_rng(spec["seed"])
        xs = [features(g["inputs"], manifest["action_stats"], future=method == "value") for g in train_groups]
        all_x = np.concatenate(xs)
        mean, scale = all_x.mean(0), np.maximum(all_x.std(0), 1e-4)
        head = ValueHead(mean, scale, spec["hidden"])
        optimizer = torch.optim.AdamW(head.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
        x = [torch.from_numpy(a) for a in xs]
        y = [torch.from_numpy(g["labels"]) for g in train_groups]
        vx = [torch.from_numpy(features(g["inputs"], manifest["action_stats"], future=method == "value")) for g in val_groups]
        best, best_state, best_step = None, None, None
        for step in range(spec["steps"] + 1):
            loss_value = None
            if step:
                head.train()
                selected = [int(rng.choice(by_episode[int(rng.choice(episodes))])) for _ in range(spec["batch_states"])]
                losses = [objective(head(x[i]), y[i], spec)[0] for i in selected]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{method} nonfinite loss at {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                loss_value = float(loss.detach())
            if step % spec["evaluate_every"] == 0 or step == spec["steps"]:
                head.eval()
                with torch.inference_mode():
                    val = grouped_metrics([head(a).numpy() for a in vx], val_groups, spec["tie_tolerance"])
                    fit = grouped_metrics([head(a).numpy() for a in x], train_groups, spec["tie_tolerance"])
                append_jsonl(out / "curves.jsonl", dict(method=method, step=step, minibatch_loss=loss_value,
                                                         train=fit, calibration=val))
                key = (val["regret"], val["mse"])
                if best is None or key < best:
                    best, best_state, best_step = key, copy.deepcopy(head.state_dict()), step
                print(f"{method} {step}/{spec['steps']}: calibration regret={val['regret']:.4f}", flush=True)
        checkpoint["heads"][method] = dict(state=best_state, selected_step=best_step)
        report["methods"][method] = dict(parameters=sum(p.numel() for p in head.parameters()),
                                        updates=spec["steps"], selected_step=best_step,
                                        calibration_regret=best[0], calibration_mse=best[1],
                                        training_seconds=time.perf_counter() - started)
    torch.save(checkpoint, out / "heads.pt")
    write_json(out / "training.json", report)


class Scorer:
    def __init__(self, checkpoint, sha256):
        import torch
        from .prefix_value import ValueHead
        self.stats, self.round, self.sha256 = checkpoint["action_stats"], checkpoint["round"], sha256
        self.heads = {}
        for method in HEADS:
            state = checkpoint["heads"][method]["state"]
            head = ValueHead(state["mean"], state["scale"], checkpoint["hidden"])
            head.load_state_dict(state)
            self.heads[method] = head.eval().requires_grad_(False)

    def __call__(self, inputs, method):
        import torch
        x = features(inputs, self.stats, future=method == "value")
        with torch.inference_mode():
            return self.heads[method](torch.from_numpy(x)).numpy()


def load_scorer(cfg, trained, manifest_sha):
    import torch
    require_complete(trained)
    torch.set_num_threads(cfg["training"]["cpu_threads"])
    path = Path(trained) / "heads.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or checkpoint["manifest_sha256"] != manifest_sha or checkpoint["config_sha256"] != signature(cfg):
        raise ValueError("Scorer checkpoint does not match manifest/protocol")
    return Scorer(checkpoint, file_hash(path))
