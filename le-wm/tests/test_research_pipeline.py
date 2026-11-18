"""Small CPU integration: local synthetic checkpoint/data, no network/training.

Exercises collection, diagnostic parity, replay, replan and output contracts.
These fixtures are not pretrained LeWM and produce no research benchmark result.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_cem import DEFAULTS, parser as collect_parser, run_experiment
from research_common import Case, stable_case_id, write_cases, write_json, read_json, require_complete
from research_diag import parser as diag_parser, audit, check_pair, replan, plots, diagnostic_plots
from research_runtime import make_world, base_env, render_frame, advance_world
from test_research_lewm_devices import model


class PipelineTest(unittest.TestCase):
    def test_cpu_end_to_end_without_pretrained_assets(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / 'cache'
            (cache / 'datasets').mkdir(parents=True)
            ckpt = cache / 'checkpoints/tiny'
            ckpt.mkdir(parents=True)
            torch.save(model().state_dict(), ckpt / 'weights.pt')
            write_json(ckpt / 'config.json', {
                '_target_': 'stable_worldmodel.wm.lewm.LeWM',
                'encoder': {'_target_': 'test_research_lewm_devices.TinyEncoder'},
                'predictor': {'_target_': 'stable_worldmodel.wm.lewm.module.Predictor',
                              'num_frames': 3, 'depth': 1, 'heads': 2, 'mlp_dim': 16,
                              'input_dim': 8, 'hidden_dim': 8, 'dim_head': 4},
                'action_encoder': {'_target_': 'stable_worldmodel.wm.lewm.module.Embedder',
                                   'input_dim': 10, 'smoothed_dim': 8, 'emb_dim': 8},
            })
            world = make_world(DEFAULTS)
            try:
                world.reset(seed=[321], options=[{}])
                frames, states, actions = [], [], []
                for i in range(12):
                    frames.append(render_frame(world, 224))
                    states.append(base_env(world)._get_obs())
                    action = np.array([0.015 * np.cos(i), 0.015 * np.sin(i)], dtype=np.float32)
                    actions.append(action)
                    advance_world(world, action)
            finally:
                world.close()
            states = np.stack(states)
            with h5py.File(cache / 'datasets/tiny.h5', 'w') as f:
                for name, value in {'ep_len': [12], 'ep_offset': [0], 'pixels': np.stack(frames),
                                    'state': states, 'proprio': states[:, [0, 1, 5, 6]],
                                    'action': np.stack(actions), 'seed': np.full(12, 321)}.items():
                    f.create_dataset(name, data=value)
            case_id = stable_case_id(0, 0, 5)
            manifest = root / 'cases.json'
            write_cases(manifest, [Case(case_id, 0, 0, 5)], metadata={'dataset': 'tiny'})
            shared = ['run', '--cases', str(manifest), '--dataset', 'tiny', '--policy', 'tiny',
                      '--cache-dir', str(cache), '--device', 'cpu', '--num-samples', '6',
                      '--iterations', '1', '--topk', '2', '--horizon', '2', '--receding-horizon', '1',
                      '--eval-budget', '10', '--goal-offset', '5', '--candidate-batch-size', '2']
            formal, diagnostic = root / 'formal', root / 'diagnostic'
            run_experiment(collect_parser().parse_args(shared + ['--output', str(formal)]))
            run_experiment(collect_parser().parse_args(shared + ['--output', str(diagnostic),
                           '--save-snapshots', '--max-snapshot-plans', '2', '--keep-best', '1', '--keep-random', '1']))
            parse = diag_parser().parse_args
            check_pair(parse(['check-pair', '--formal', str(formal), '--diagnostic', str(diagnostic)]))
            audit_root = root / 'audit'
            audit(parse(['audit', '--run', str(diagnostic), '--output', str(audit_root)]))
            require_complete(audit_root)
            self.assertEqual(read_json(audit_root / 'STATUS.json')['candidates'], 6)
            self.assertEqual(len(read_json(audit_root / 'audit_summary.json')), 2)
            snap = diagnostic / 'snapshots' / case_id / 'plan_000001/snapshot.json'
            repeated = root / 'replan'
            replan(parse(['replan', '--run', str(diagnostic), '--snapshot', str(snap), '--output', str(repeated),
                          '--num-samples', '6', '--iterations', '1', '--topk', '2']))
            new_snap = repeated / 'snapshots' / case_id / 'plan_000001/snapshot.json'
            self.assertEqual(read_json(new_snap)['candidate_batch_size'], 2)
            with np.load(snap.parent / 'arrays.npz') as a, np.load(new_snap.parent / 'arrays.npz') as b:
                np.testing.assert_array_equal(a['returned_actions'], b['returned_actions'])
            plots(parse(['plot', '--run', str(formal), '--output', str(root / 'budget_figures')]))
            diagnostic_plots(parse(['plot-diagnostics', '--run', str(diagnostic), '--audit', str(audit_root),
                                   '--output', str(root / 'diagnostic_figures')]))
            self.assertTrue((root / 'budget_figures/budget_success_time.png').is_file())
            self.assertEqual(len(list((root / 'diagnostic_figures').glob('*.png'))), 4)


if __name__ == '__main__':
    unittest.main()
