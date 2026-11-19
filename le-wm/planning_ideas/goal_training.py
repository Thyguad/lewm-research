"""Create a split manifest, cache frozen features, and fit terminal scorers."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from cube_research.data import Dataset
from cube_research.model import FrozenModel
from cube_research.protocol import Ledger, digest
from research_common import file_hash, read_json, require_complete, write_json
from .goal_pairs import (GRIPPER, JOINTS, POSITION, frame_rows, labels, matched_pairs,
                         mechanism_cases, read_rows, robot_distance, split_episodes)


def input_identity(cfg):
    path = Path(cfg["dataset"]).resolve()
    st = path.stat()
    weights = Path(cfg["weights"])
    # Avoid scanning 95 GB at each command. Used pixels and labels are also
    # bound to the experiment by the hashes of the resulting feature caches.
    return dict(dataset_path=str(path), dataset_bytes=st.st_size, dataset_mtime_ns=st.st_mtime_ns,
                weights_sha256=file_hash(weights), model_config_sha256=file_hash(weights.parent / "config.json"))


def data_config(cfg):
    keys = ("split_seed", "goal_offset", "train_episodes", "dev_episodes", "test_episodes",
            "donor_episodes", "dev_cases", "mechanism_cases", "success_distance_m", "pairing")
    return {k: cfg[k] for k in keys}


def load_manifest(cfg, cache):
    require_complete(cache)
    manifest = read_json(cache / "manifest.json")
    if manifest["schema"] != "cube-goal-set-cache-v1":
        raise ValueError("Unsupported feature cache")
    if manifest["inputs"] != input_identity(cfg) or manifest["data_config"] != data_config(cfg):
        raise ValueError("Data/model or split/pair config differs from prepared manifest")
    return manifest


def write_examples(dataset, path, triplets, seed):
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(seed)
    sample = triplets[rng.choice(len(triplets), min(6, len(triplets)), replace=False)]
    images = read_rows(dataset.f["pixels"], sample[:, [2, 0, 1]].ravel())
    w, h, header = 160, 160, 22
    canvas = Image.new("RGB", (3 * w, len(sample) * (h + header)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (pix, row) in enumerate(zip(images, sample[:, [2, 0, 1]].ravel())):
        y, x = (i // 3) * (h + header), (i % 3) * w
        draw.text((x + 4, y + 3), f"{('goal', 'positive', 'negative')[i % 3]}: {row}", fill="black")
        canvas.paste(Image.fromarray(pix).resize((w, h)), (x, y + header))
    canvas.save(path)


def prepare(cfg, out):
    dataset = Dataset(cfg["dataset"])
    try:
        splits, cases = split_episodes(dataset.lengths, cfg)
        variants, missing = mechanism_cases(dataset, cases["dev"], splits["donor"], cfg)
        action_stats = dataset.stats()
        model = FrozenModel(cfg, action_stats)
        ledger = Ledger(0)
        pair_stats = {}
        for index, split in enumerate(("train", "dev")):
            spec = cfg["pairing"]
            rows, episodes = frame_rows(dataset, splits[split], spec["frame_stride"])
            pos, joint, grip = labels(dataset, rows)
            pairs, hard_count = matched_pairs(pos, joint, grip, spec,
                                              np.random.default_rng(cfg["split_seed"] + index + 1))
            if not len(pairs):
                raise ValueError(f"No target-matched pairs in {split}; inspect data or matching thresholds")
            selected, pair_index = np.unique(pairs, return_inverse=True)
            pair_index = pair_index.reshape(pairs.shape)
            zs = []
            # Read each storage chunk once, keeping the full set of RGB frames
            # out of RAM. An encoding batch may span several storage chunks.
            batch = cfg["score_batch_size"]
            for start in range(0, len(selected), batch):
                frames = read_rows(dataset.f["pixels"], rows[selected[start:start + batch]])
                zs.append(model.encode(frames, ledger))
            np.savez(out / f"{split}.npz", embeddings=np.concatenate(zs), pairs=pair_index,
                     rows=rows[selected], episodes=episodes[selected], positions=pos[selected],
                     joints=joint[selected], gripper=grip[selected])
            write_examples(dataset, out / f"{split}-examples.png", rows[pairs], cfg["split_seed"])
            def stats(a):
                return dict(zip(("min", "median", "max"), np.quantile(a, [0, .5, 1]).tolist()))
            p, n, g = pairs.T
            arm_p = [robot_distance(joint[i:i+1], grip[i:i+1], joint[j], grip[j], spec)[0]
                     for i, j in zip(p, g)]
            arm_n = [robot_distance(joint[i:i+1], grip[i:i+1], joint[j], grip[j], spec)[0]
                     for i, j in zip(n, g)]
            pair_stats[split] = dict(
                positives=len(p), negatives=len(n), hard_negatives=hard_count, frames=len(selected),
                unique_goals=len(np.unique(g)), sampled_trajectories=len(splits[split]),
                paired_trajectories=len(np.unique(episodes[selected])), augmented_frames=0,
                positive_distance_m=stats(np.linalg.norm(pos[p] - pos[g], axis=-1)),
                negative_distance_m=stats(np.linalg.norm(pos[n] - pos[g], axis=-1)),
                positive_robot_distance=stats(arm_p), negative_robot_distance=stats(arm_n),
            )
            print(f"{split}: {len(p)} matched bundles / {len(selected)} frozen RGB embeddings", flush=True)
        manifest = dict(schema="cube-goal-set-cache-v1", inputs=input_identity(cfg),
                        data_config=data_config(cfg), action_stats=action_stats, splits=splits,
                        cases=cases, mechanism=variants, missing_mechanism=missing,
                        labels=[POSITION, JOINTS, GRIPPER], latent_dim=model.dim,
                        feature_hashes={s: file_hash(out / f"{s}.npz") for s in ("train", "dev")},
                        model=model.identity, device=model.device, encoding=asdict(ledger), pair_stats=pair_stats)
        write_json(out / "manifest.json", manifest)
    finally:
        dataset.close()


def fit(cfg, cache, out):
    import torch
    from .goal_set import METHODS, build_head, costs, training_loss

    spec = cfg["training"]
    torch.set_num_threads(spec["cpu_threads"])
    manifest = load_manifest(cfg, cache)

    def tensors(split):
        path = cache / f"{split}.npz"
        if file_hash(path) != manifest["feature_hashes"][split]:
            raise ValueError(f"Modified {split} feature cache")
        with np.load(path) as data:
            return tuple(torch.as_tensor(data[k], dtype=t) for k, t in (
                ("embeddings", torch.float32), ("positions", torch.float32), ("pairs", torch.long)))

    z, pos, pair = tensors("train")
    dev, _, dev_pair = tensors("dev")
    p, n, g = pair.T
    pos_norm = (z[p] - z[g]).norm(dim=-1)
    neg_norm2 = (z[n] - z[g]).square().sum(dim=-1)
    mu, sigma = pos.mean(0), pos.std(0, unbiased=False).clamp_min(1e-4)
    calibration = dict(radius=float(torch.quantile(pos_norm, spec["radius_quantile"])),
                       scale2=max(1e-8, float(neg_norm2.median())),
                       position_axis_scale=sigma.tolist(), position_distance_unit=cfg["pairing"]["negative_m"])
    checkpoint = dict(schema="cube-goal-set-heads-v1", dim=manifest["latent_dim"], calibration=calibration, training=spec,
                      manifest_sha256=file_hash(cache / "manifest.json"), states={})
    report = {}
    for method in METHODS:
        torch.manual_seed(spec["seed"])
        rng = torch.Generator().manual_seed(spec["seed"])
        head = build_head(method, checkpoint["dim"], spec)
        if head is not None:
            optimizer = torch.optim.AdamW(head.parameters(), lr=spec["learning_rate"])
            for step in range(spec["steps"]):
                batch = pair[torch.randint(len(pair), (spec["batch_size"] // 2,), generator=rng)]
                pp, nn, gg = batch.T
                aa, bb = torch.cat((pp, nn)), gg.repeat(2)
                mask = torch.arange(len(aa)) < len(pp)
                loss = training_loss(method, head, z[aa], z[bb], (pos[aa], pos[bb], mu, sigma),
                                     mask, calibration, spec)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{method} non-finite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            checkpoint["states"][method] = head.eval().state_dict()
        with torch.inference_mode():
            pp, nn, gg = dev_pair.T
            p_cost = costs(method, head, dev[pp], dev[gg], calibration)
            n_cost = costs(method, head, dev[nn], dev[gg], calibration)
            report[method] = dict(parameters=0 if head is None else sum(x.numel() for x in head.parameters()),
                                  matched_dev_accuracy=float(((p_cost < n_cost).float() +
                                                               .5 * (p_cost == n_cost).float()).mean()),
                                  mean_positive=float(p_cost.mean()), mean_negative=float(n_cost.mean()),
                                  near_zero_negative_fraction=float((n_cost <= 1e-6).float().mean()),
                                  updates=0 if head is None else spec["steps"])
        print(f"{method}: dev matched ranking {report[method]['matched_dev_accuracy']:.3f}", flush=True)
    torch.save(checkpoint, out / "heads.pt")
    write_json(out / "training.json", dict(methods=report, calibration=calibration,
                                          coordinate_mean=mu.tolist(), coordinate_scale=sigma.tolist(),
                                          backbone_updates=0, labels=manifest["labels"]))
