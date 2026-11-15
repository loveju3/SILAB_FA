"""Experiment comparison uses synthetic aggregate reports, never holdout rows."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helper_tool import comparison


class ComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def report(self, name, metadata, summary=None):
        folder = self.root / name
        folder.mkdir()
        (folder / 'run.json').write_text(json.dumps(metadata), encoding='utf-8')
        if summary is not None:
            (folder / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
        (folder / 'predictions.jsonl').write_text('must never read prediction rows')
        (folder / 'errors.jsonl').write_text('must never read error rows')
        return folder

    def test_reads_only_whitelisted_aggregates_and_preserves_missing_values(self):
        self.report('20261115-080000-aaaaaaaa',
                    {'filename': 'synthetic.jsonl', 'dataset': 'synthetic-history.jsonl', 'count': 3,
                     'method': 'bge-finetuned', 'limit': '2gb', 'model_path': '/models/run/finetuned',
                     'dataset_sha256': 'a' * 64, 'timing': {'total_seconds': 2.3, 'prediction_seconds': 2}},
                    {'total': .7, 'metrics': {'category': .5, 'device': 1, 'urgency': .7, 'duplicate': .8},
                     'invalid_predictions': {'must-not-return': 'row details'}, 'confusion_matrix': ['must-not-return']})
        self.report('20261115-080001-bbbbbbbb', {'label': 'unscored', 'dataset': 'tickets.jsonl', 'count': 2})
        load = comparison.load_json
        def read(path):
            self.assertIn(path.name, ('run.json', 'summary.json'))
            return load(path)
        with patch.object(comparison, 'load_json', side_effect=read):
            result = comparison.list_experiments(self.root)
        self.assertEqual(result['skipped'], 0)
        unscored, scored = result['experiments']
        self.assertIsNone(unscored['total'])
        self.assertIsNone(unscored['timing']['prediction_seconds'])
        self.assertIsNone(unscored['metrics']['category'])
        self.assertEqual(scored['total'], .7)
        self.assertEqual(scored['model_path'], '/models/run/finetuned')
        self.assertEqual(scored['timing']['prediction_seconds'], 2)
        self.assertNotIn('must-not-return', json.dumps(result))

    def test_legacy_reports_corrupt_reports_and_invalid_metrics(self):
        self.report('20261115-080000-aaaaaaaa', {'label': 'old'},
                    {'gold_file': 'C:\\local\\synthetic.jsonl', 'gold_count': 5, 'total': 0,
                     'metrics': {'category': float('nan'), 'device': 2, 'urgency': -1, 'duplicate': True}})
        broken = self.report('20261115-080001-bbbbbbbb', {})
        (broken / 'summary.json').write_text('{incomplete')
        self.report('not-a-run', {'label': 'ignore'})
        result = comparison.list_experiments(self.root)
        self.assertEqual(result['skipped'], 1)
        self.assertEqual(len(result['experiments']), 1)
        row = result['experiments'][0]
        self.assertEqual(row['dataset'], 'synthetic.jsonl')
        self.assertEqual(row['count'], 5)
        self.assertEqual(row['total'], 0)
        self.assertIsNone(row['dataset_sha256'])
        self.assertTrue(all(value is None for value in row['metrics'].values()))


if __name__ == '__main__':
    unittest.main()
