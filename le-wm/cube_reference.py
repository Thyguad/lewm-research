"""Run the local, normal-condition LeWM Cube reference evaluator."""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from research_common import write_json
from research_device import prepare_headless


AUDIT_IDS = ('e22d467bfaaa9357', 'e1db09ce9f216bf6')


def verified_scores(audit):
    fixed = json.loads((audit / 'fixed-score.json').read_text())
    first_plan = json.loads((audit / 'first-plan-score.json').read_text())
    return (fixed['passed'] and set(fixed['cases']) == set(first_plan) == set(AUDIT_IDS)
            and all(r['passed'] for r in first_plan.values()))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['manifest', 'audit-two', 'evaluate-50'])
    parser.add_argument('--config', type=Path, default=root / 'le-wm/config/research/cube-local.yaml')
    parser.add_argument('--prior', type=Path, default=root / 'runs/cube')
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    parser.add_argument('--manifest', type=Path, help='fixed cases for evaluate-50')
    parser.add_argument('--verified-audit', type=Path, help='successful audit-two output for evaluate-50')
    parser.add_argument('--check-only', action='store_true', help='validate evaluate-50 without constructing environments')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--protocol', choices=['paper-300x10-top30', 'public-300x30-top30'],
                        default='paper-300x10-top30')
    args = parser.parse_args()
    if args.mode == 'evaluate-50' and not (args.manifest and args.verified_audit):
        parser.error('evaluate-50 requires --manifest and --verified-audit')
    if args.check_only and args.mode != 'evaluate-50':
        parser.error('--check-only is only for evaluate-50')
    if args.mode == 'audit-two' and args.protocol != 'paper-300x10-top30':
        parser.error('audit-two only runs the paper setting')
    prepare_headless()
    os.environ.setdefault('MUJOCO_GL', 'glfw' if sys.platform == 'darwin' else 'egl')
    os.environ['HF_HUB_OFFLINE'] = '1'

    from cube_research.data import local_inputs
    from cube_research.model import FrozenModel
    from cube_research.reference import (CEM, EVAL, PLAN, PROTOCOLS, fit_actions, fixed_manifest,
                                         open_dataset, run_policy, runtime_identity)

    cfg = yaml.safe_load(args.config.read_text())
    local_inputs(cfg)
    cfg['device'] = args.device
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'STATUS.json'
    write_json(status, {'status': 'running', 'mode': args.mode})
    data = None
    try:
        data = open_dataset(cfg['dataset'], out / 'local-cache')
        manifest = fixed_manifest(data)
        write_json(out / 'reference-50.json', manifest)
        if args.mode != 'manifest':
            scaler = fit_actions(data)
            archived = json.loads((args.prior / 'cases.json').read_text()) if args.mode == 'audit-two' else None
            stats = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), rows=int(scaler.n_samples_seen_))
            bridge = FrozenModel(cfg, archived['stats'] if archived else stats)
            identity = runtime_identity(bridge.device, bridge)
            write_json(out / 'runtime.json', identity)
            write_json(out / 'protocol.json', dict(name=args.protocol, condition='N', evaluation=EVAL,
                       planning=vars(PLAN), solver=dict(CEM, n_steps=PROTOCOLS[args.protocol], device=bridge.device),
                       scaler=stats, mujoco_gl=os.environ['MUJOCO_GL']))
            if args.mode == 'audit-two':
                from cube_research.reference_audit import fixed_audit, post_score

                by_id = {c['case_id']: c for c in archived['cases']}
                cases = [by_id[k] for k in AUDIT_IDS]
                write_json(out / 'audit-cases.json', cases)
                if not fixed_audit(data, cases, bridge, scaler, out, args.prior, manifest):
                    raise RuntimeError('Fixed-candidate score parity failed; no online run started')
                print('Fixed-candidate parity passed; running two N regression cases', flush=True)
                run_policy(data, cases, out / 'cem', 'cem', bridge.base, scaler, bridge.device, args.protocol)
                if not post_score(data, cases, bridge, scaler, out):
                    raise RuntimeError('First-plan score parity failed; investigate before evaluate-50')
            else:
                audit = args.verified_audit
                if (json.loads(args.manifest.read_text()) != manifest
                        or json.loads((audit / 'reference-50.json').read_text()) != manifest):
                    raise ValueError('Manifest does not match this dataset and reference sampling')
                verified = json.loads((audit / 'runtime.json').read_text())
                if (json.loads((audit / 'STATUS.json').read_text()) != {'status': 'completed', 'mode': 'audit-two'}
                        or not verified_scores(audit)
                        or json.loads((audit / 'protocol.json').read_text())['scaler'] != stats
                        or any(verified[k] != identity[k] for k in ('sources', 'packages', 'execution'))
                        or any(verified['model'][k] != identity['model'][k]
                               for k in ('weights_sha256', 'config_sha256'))):
                    raise ValueError('A passed audit-two with matching runtime and checkpoint is required')
                cases = manifest['cases']
                write_json(out / 'planned-evaluation.json', dict(cases=cases, policies=['cem', 'zero', 'random']))
                if not args.check_only:
                    run_policy(data, cases, out / 'cem', 'cem', bridge.base, scaler, bridge.device, args.protocol)
                    for name in ('zero', 'random'):
                        run_policy(data, cases, out / name, name)
    except BaseException as exc:
        write_json(status, dict(status='unfinished', mode=args.mode, error=repr(exc)), overwrite=True)
        raise
    else:
        write_json(status, dict(status='ready-no-rollouts' if args.check_only else 'completed', mode=args.mode),
                   overwrite=True)
    finally:
        if data is not None and data.h5_file is not None:
            data.h5_file.close()


if __name__ == '__main__':
    main()
