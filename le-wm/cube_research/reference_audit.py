"""Offline score parity and reconstruction of the saved legacy diagnostic."""

import json

import numpy as np
import torch
from torchvision import tv_tensors

from research_common import write_json
from research_runtime import image_transform
from .protocol import Ledger
from .reference import (OFFSET, PLAN, action_blocks, expanded_inputs, make_policy,
                        policy_inputs)


def window(data, case):
    start = case['start_step']
    return data.load_chunk([case['episode']], [start], [start + OFFSET + 1])[0]


def research_rollout(model, pixels, goal, raw):
    ledger = Ledger(1)
    z0, zg = model.encode([pixels, goal], ledger)
    z = np.repeat(z0[None, None], len(raw), axis=0)
    for h in range(PLAN.horizon):
        pred = model.predict(z[:, -3:], raw[:, max(0, h - 2):h + 1], ledger)
        z = np.concatenate((z, pred[:, None]), axis=1)
    return z, zg, np.square(z[:, -1] - zg).sum(-1)


def legacy_candidate(model, prior, case, goal):
    cid = case['case_id']
    for path in (prior / 'e0-dev/histories').glob('*/metadata.json'):
        meta = json.loads(path.read_text())
        if meta['case']['case_id'] == cid and meta['condition'] == 'N':
            break
    else:
        raise ValueError(f'No legacy N history for {cid}')
    with np.load(path.parent / 'allowed.npz') as log:
        pixels, first_block = log['pixels'][0], log['commands'][:5]
    rng = np.random.default_rng(42)
    mean, std = np.zeros((5, 5, 5)), np.ones((5, 5, 5))
    for _ in range(2):
        sample = np.clip(rng.normal(mean, std, (30, 5, 5, 5)), -1, 1).astype(np.float32)
        _, _, costs = research_rollout(model, pixels, goal, sample)
        elite = sample[np.argsort(costs)[:5]]
        mean, std = elite.mean(0), np.maximum(elite.std(0), .03)
    idx = costs.argmin()
    historical = json.loads((prior / 'planning-diagnosis/report.json').read_text())[cid]
    return sample[idx].reshape(25, 5), dict(
        first_logged_block_max_abs=float(np.max(np.abs(sample[idx, 0] - first_block))),
        original_cost=historical['predicted_goal_cost']['cem'], reproduced_cost=float(costs[idx]),
        source=str(path.parent / 'allowed.npz'),
        note='Legacy offline first plan reconstructed; only its first block was executed as that plan')


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@torch.inference_mode()
def score(bridge, scaler, pixels, goal, raw, normalized=None):
    policy = make_policy(None, scaler)
    info = policy._prepare_info(policy_inputs({'pixels': pixels[None, None], 'goal': goal[None, None]}))
    tx = image_transform(224)(tv_tensors.Image(np.stack([pixels, goal]).transpose(0, 3, 1, 2)))
    blocks = action_blocks(raw, scaler) if normalized is None else normalized
    grouped = raw.reshape(-1, 5, 5, 5)
    research_blocks = ((grouped - bridge.stats['mean']) / bridge.stats['scale']).astype('float32')
    model_info = expanded_inputs(info, len(raw), bridge.device)
    costs = bridge.base.get_cost(model_info, torch.as_tensor(blocks[None], dtype=torch.float32,
                                                             device=bridge.device))[0].cpu().numpy()
    roll = model_info['predicted_emb'][0].cpu().numpy()
    zg = model_info['goal_emb'][0, -1].cpu().numpy()
    rr, rg, rc = research_rollout(bridge, pixels, goal, grouped)
    return dict(pixel_max_abs=[max_abs(info[k][0, 0].numpy(), tx[i].numpy())
                               for i, k in enumerate(('pixels', 'goal'))],
                normalized_max_abs=max_abs(blocks, research_blocks.reshape(blocks.shape)),
                initial_emb_max_abs=max_abs(roll[:, 0], rr[:, 0]), goal_emb_max_abs=max_abs(zg, rg),
                per_block_max_abs=[max_abs(roll[:, i], rr[:, i]) for i in range(1, 6)],
                cost_max_abs=max_abs(costs, rc), reference_cost=costs, research_cost=rc,
                reference_order=np.argsort(costs), research_order=np.argsort(rc)), (roll, zg, costs)


def parity_pass(row):
    return (max(row['pixel_max_abs']) == 0 and row['normalized_max_abs'] <= 5e-6
            and max(row['initial_emb_max_abs'], row['goal_emb_max_abs'], *row['per_block_max_abs']) <= 1e-3
            and np.allclose(row['reference_cost'], row['research_cost'], atol=.05, rtol=1e-4)
            and np.array_equal(row['reference_order'], row['research_order']))


def fixed_audit(data, cases, bridge, scaler, out, prior, manifest):
    stats = json.loads((prior / 'cases.json').read_text())['stats']
    report = dict(passed=False, research_scaler_difference=dict(mean=max_abs(stats['mean'], scaler.mean_),
                  scale=max_abs(stats['scale'], scaler.scale_)), cases={}, manifest_preview=[])
    # Full reference manifest is generated first. These previews are offline;
    # the earlier two regressions are a separate, deliberately selected set.
    for c in manifest['cases'][:2]:
        ep = window(data, c)
        report['manifest_preview'].append(dict(case=c, action_shape=list(ep['action'].shape),
                                               pixels_shape=list(ep['pixels'].shape)))
    extras = []
    for c in cases:
        ep = window(data, c)
        if ep['action'].shape != (26, 5) or ep['qpos'].dtype != torch.float64:
            raise ValueError('Unexpected action shape or simulator precision')
        pixels, goal = ep['pixels'][[0, -1]].permute(0, 2, 3, 1).numpy()
        legacy, legacy_info = legacy_candidate(bridge, prior, c, goal)
        expert = ep['action'][:25].numpy()
        raw = np.stack([expert, np.zeros_like(expert),
                        np.random.default_rng(42).uniform(-1, 1, (25, 5)).astype('float32'), legacy])
        row, arrays = score(bridge, scaler, pixels, goal, raw)
        report['cases'][c['case_id']] = dict(candidate_names=['expert', 'zero', 'random_uniform', 'legacy_cem'],
                                              shapes=dict(raw=raw.shape, blocks=(len(raw), 5, 25)),
                                              action_dtype=str(raw.dtype), qpos_dtype=str(ep['qpos'].dtype),
                                              legacy=legacy_info, same_device=row, passed=parity_pass(row))
        np.savez_compressed(out / f"fixed-actions-{c['case_id']}.npz", raw=raw,
                            normalized=action_blocks(raw, scaler))
        extras.append((pixels, goal, raw, arrays, row))
    device = bridge.device
    other = 'cpu' if device == 'mps' else ('mps' if device == 'cpu' and torch.backends.mps.is_available() else None)
    if other:
        bridge.base.to(other)
        bridge.device = other
        try:
            for c, (pixels, goal, raw, arrays, row) in zip(cases, extras):
                compared, other_arrays = score(bridge, scaler, pixels, goal, raw)
                report['cases'][c['case_id']]['cross_device'] = dict(
                    devices=[device, other], other_device_parity=compared,
                    other_device_passed=parity_pass(compared),
                    rollout_max_abs=max_abs(arrays[0], other_arrays[0]),
                    goal_max_abs=max_abs(arrays[1], other_arrays[1]),
                    cost_max_abs=max_abs(arrays[2], other_arrays[2]),
                    same_rank=np.array_equal(row['reference_order'], compared['reference_order']))
        finally:
            bridge.base.to(device)
            bridge.device = device
    report['passed'] = all(r['passed'] and r.get('cross_device', {}).get('other_device_passed', True)
                           for r in report['cases'].values())
    write_json(out / 'fixed-score.json', report)
    return report['passed']


def post_score(data, cases, bridge, scaler, out):
    # The CEM-selected actions are scored next to the expert only offline.
    # Exact normalized arrays are reused on the reference side.
    plan = json.loads((out / 'cem/plan-00.json').read_text())
    result = {}
    from .reference import command_rows

    for i, c in enumerate(cases):
        ep = window(data, c)
        pixels, goal = ep['pixels'][[0, -1]].permute(0, 2, 3, 1).numpy()
        original = np.load(out / f"fixed-actions-{c['case_id']}.npz")
        with original:
            normalized = np.concatenate((original['normalized'],
                                         np.asarray([plan['sampled'][i]['best']['normalized'], plan['mean'][i]],
                                                    dtype=np.float32)))
            raw = np.concatenate((original['raw'], command_rows(normalized[-2:], scaler)))
        row, _ = score(bridge, scaler, pixels, goal, raw, normalized)
        result[c['case_id']] = dict(candidate_names=['expert', 'zero', 'random_uniform', 'legacy_cem',
                                                     'best_reference_sample', 'reference_final_mean'],
                                     same_device=row, passed=parity_pass(row),
                                     best_cost_from_solver=plan['sampled'][i]['best']['cost'],
                                     mean_cost_from_online_rescore=plan['final_mean_cost'][i])
    write_json(out / 'first-plan-score.json', result)
    return all(r['passed'] for r in result.values())
