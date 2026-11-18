"""Lightweight configuration/result-contract tests: no model or simulator imports."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_cem import _load_cfg, parser
from research_common import (
    Case, file_hash, finish_run, new_run, read_cases, stable_case_id,
    require_complete, write_cases, write_json,
)
from research_diag import budget_rows, success_interval


class Contracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_cli_overrides_yaml_without_resetting_unspecified_fields(self):
        cfg = self.root / 'config.yaml'
        cfg.write_text('num_samples: 80\niterations: 7\nwarm_start: true\n')
        args = parser().parse_args(['run', '--cases', 'c.json', '--output', 'run', '--config', str(cfg),
                                    '--num-samples', '30', '--no-warm-start'])
        values = _load_cfg(args)
        self.assertEqual((values['num_samples'], values['iterations'], values['warm_start']), (30, 7, False))
        self.assertFalse(values['diagnostics'])

    def test_invalid_config_rejected(self):
        for content in ('unknown: 1', 'topk: 1', 'horizon: true', 'receding_horizon: 6',
                        'horizon: 20', 'diagnostics: "false"', 'keep_random: 1.5',
                        'candidate_batch_size: -1', 'candidate_batch_size: true', 'device: mps:1'):
            with self.subTest(content=content):
                cfg = self.root / 'bad.yaml'
                cfg.write_text(content)
                args = parser().parse_args(['run', '--cases', 'c', '--output', 'r', '--config', str(cfg)])
                with self.assertRaises(ValueError):
                    _load_cfg(args)

    def test_mac_profile_preserves_task_and_uses_bounded_batches(self):
        cfg_path = Path(__file__).resolve().parents[1] / 'config/research/pusht_mac.yaml'
        args = parser().parse_args(['run', '--cases', 'c', '--output', 'r', '--config', str(cfg_path)])
        cfg = _load_cfg(args)
        self.assertEqual(cfg['device'], 'mps')
        self.assertEqual(cfg['candidate_batch_size'], 16)
        self.assertEqual((cfg['horizon'], cfg['receding_horizon'], cfg['action_block'], cfg['eval_budget']), (5, 5, 5, 50))
        args = parser().parse_args(['run', '--cases', 'c', '--output', 'r', '--config', str(cfg_path), '--device', 'cpu'])
        self.assertEqual(_load_cfg(args)['device'], 'cpu')

    def test_snapshots_enable_diagnostics(self):
        args = parser().parse_args(['run', '--cases', 'c', '--output', 'r', '--save-snapshots'])
        self.assertTrue(_load_cfg(args)['diagnostics'])

    def test_case_identity_and_duplicates(self):
        case = Case(stable_case_id(3, 2, 25), 3, 2, 25)
        path = self.root / 'cases.json'
        write_cases(path, [case])
        self.assertEqual(read_cases(path)[0], [case])
        data = json.loads(path.read_text())
        data['cases'].append(data['cases'][0])
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            read_cases(path)
        data['cases'] = [{**data['cases'][0], 'case_id': '../../unsafe'}]
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            read_cases(path)

    def test_partial_runs_and_overwrite_refused(self):
        root = new_run(self.root / 'run')
        with self.assertRaises(ValueError):
            require_complete(root)
        finish_run(root, 'failed', error='interrupted')
        with self.assertRaises(ValueError):
            require_complete(root)
        with self.assertRaises(FileExistsError):
            new_run(root)
        finish_run(root, 'complete')
        require_complete(root)
        with self.assertRaises(FileExistsError):
            write_json(root / 'STATUS.json', {})

    def test_ci_extremes_are_nonzero_and_clustered(self):
        for single in (True, False):
            _, lo, hi, _ = success_interval([0] * 20, single)
            self.assertEqual(lo, 0)
            self.assertGreater(hi, 0)
            _, lo, hi, _ = success_interval([1] * 20, single)
            self.assertLess(lo, 1)
            self.assertEqual(hi, 1)
        self.assertIn('clusters', success_interval([0.5] * 20, False)[3])

    def fixture_run(self, name, n=30, seed=42, diagnostics=False, model_hash='weights'):
        root = new_run(self.root / name)
        write_json(root / 'processors.json', {'action': {'mean': [0, 0], 'scale': [1, 1]}})
        cfg = _load_cfg(parser().parse_args(['run', '--cases', 'c', '--output', 'r']))
        cfg.update(num_samples=n, planning_seed=seed, diagnostics=diagnostics)
        cases = [{'case_id': 'a'}, {'case_id': 'b'}]
        meta = {
            'schema': 'lewm-run-v2', 'run_id': name, 'config': cfg, 'cases': cases,
            'model': {'type': 'test', 'weights_sha256': model_hash, 'config_sha256': 'config'},
            'processors_sha256': file_hash(root / 'processors.json'),
            'sources': {}, 'research_sources': {}, 'packages': {}, 'dataset_sha256': 'dataset',
            'protocol': 'test', 'execution': {}, 'python': '3.11', 'platform': 'test',
        }
        write_json(root / 'run_metadata.json', meta)
        rows = [{'run_id': name, 'case_id': c['case_id'], 'num_samples': n, 'iterations': 10,
                 'planning_seed': seed, 'diagnostics': diagnostics, 'success': i == 0}
                for i, c in enumerate(cases)]
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        finish_run(root, 'complete')
        return root

    def test_budget_pool_requires_matched_cases_seeds_and_model(self):
        a = self.fixture_run('a')
        b = self.fixture_run('b', n=100)
        self.assertEqual(len(budget_rows([a, b])), 2)
        for other in (self.fixture_run('seed', n=100, seed=43),
                      self.fixture_run('model', n=100, model_hash='other'),
                      self.fixture_run('diag', diagnostics=True)):
            with self.subTest(other=other):
                with self.assertRaises(ValueError):
                    budget_rows([a, other])
        with self.assertRaises(ValueError):
            budget_rows([a, a])

    def test_budget_pool_rejects_missing_and_duplicate_rows(self):
        a = self.fixture_run('a')
        path = a / 'episodes.jsonl'
        first = path.read_text().splitlines()[0]
        for content in (first, first + '\n' + first):
            path.write_text(content + '\n')
            with self.assertRaises(ValueError):
                budget_rows([a])

    def test_budget_pool_rejects_mixed_batches_and_backends(self):
        a = self.fixture_run('a')
        for name in ('batch', 'backend'):
            b = self.fixture_run(name, n=100)
            path = b / 'run_metadata.json'
            meta = json.loads(path.read_text())
            if name == 'batch':
                meta['config']['candidate_batch_size'] = 16
            else:
                meta['execution']['device'] = 'mps'
            path.write_text(json.dumps(meta))
            with self.subTest(name=name), self.assertRaises(ValueError):
                budget_rows([a, b])

    def test_imports_stay_lightweight(self):
        for name in ('torch', 'numpy', 'gymnasium', 'stable_worldmodel', 'matplotlib'):
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
