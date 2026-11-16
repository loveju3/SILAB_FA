"""Training API contracts; synthetic uploads only, no model training in job tests."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import sys
import unittest
from unittest.mock import patch

from helper_tool import training_jobs as jobs
from Solution.training.settings import FineTuneSettings, LinearSettings


class TrainingJobsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registry = self.root / 'registry'
        patcher = patch.object(jobs, 'REGISTRY', self.registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.payload = {'kind': 'tfidf', 'source': 'upload', 'history_name': 'new-history.jsonl',
                        'history_only': True, 'output_dir': str(self.root / 'output'),
                        'history_content': json.dumps({'ticket_id': 'synthetic', 'created_at': '2026-01-01T00:00:00Z',
                                                      'reporter': '甲', 'raw_text': '機器異音', 'category': 'noise'})}

    def test_settings_reject_invalid_or_excessive_values(self):
        self.assertEqual(FineTuneSettings.from_dict().encoder_lr, 1e-5)
        self.assertEqual(FineTuneSettings.from_dict({'epochs': 100}).epochs, 100)
        for values in ({'epochs': True}, {'epochs': 0}, {'epochs': -1}, {'epochs': 1.5}, {'encoder_lr': .1}, {'dropout': float('nan')},
                       {'batch_size': 1.2}, {'max_length': True}, {'weight_decay': 0}, {'unknown': 1}, []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                FineTuneSettings.from_dict(values)

    def test_linear_settings_defaults_match_previous_recipe_and_validate(self):
        defaults = LinearSettings.from_dict()
        self.assertEqual((defaults.c, defaults.class_weight, defaults.analyzer, defaults.ngram_min, defaults.ngram_max,
                          defaults.lr_max_iter, defaults.svc_max_iter, defaults.min_df, defaults.max_features,
                          defaults.sublinear_tf), (1.0, 'balanced', 'char', 2, 4, 2000, 5000, 1, 0, True))
        self.assertEqual(set(LinearSettings.from_dict({}, 'bge-frozen').to_dict('bge-frozen')), set(LinearSettings.HEAD_FIELDS))
        self.assertEqual(LinearSettings.from_dict({'c': .1, 'ngram_max': 5}).ngram_max, 5)
        for values in ({'c': True}, {'c': float('inf')}, {'class_weight': 'auto'}, {'lr_max_iter': 10.5},
                       {'max_features': 50}, {'sublinear_tf': 1}, {'analyzer': 'word'}, []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LinearSettings.from_dict(values)

    def test_deleted_training_folder_does_not_break_job_list(self):
        with patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic failure')):
            for _ in range(2):
                with self.assertRaises(OSError):
                    jobs.start_job(self.payload)
        first, second = jobs.list_jobs()
        jobs.shutil.rmtree(first['directory'])
        self.assertEqual([job['job_id'] for job in jobs.list_jobs()], [second['job_id']])
        with self.assertRaisesRegex(FileNotFoundError, '已不存在'):
            jobs.get_job(first['job_id'])

    def test_linear_settings_are_saved_in_job_config(self):
        with patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic failure')):
            with self.assertRaises(OSError):
                jobs.start_job({**self.payload, 'linear_settings': {'c': .5, 'ngram_max': 3}})
        config = jobs.list_jobs()[0]['config']
        self.assertEqual((config['linear_settings']['c'], config['linear_settings']['ngram_max']), (.5, 3))
        self.assertEqual(config['linear_settings']['class_weight'], 'balanced')

    def test_invalid_uploads_publish_no_training_job(self):
        for change in ({'history_only': False}, {'history_name': 'dev.jsonl'}, {'source': 'dev'},
                       {'history_content': '{broken'}, {'history_content': '{}'},
                       {'history_content': self.payload['history_content'].replace('noise', 'invalid')},
                       {'kind': 'shell'}, {'settings': {'encoder_lr': .1}},
                       {'linear_settings': {'c': 0}}, {'linear_settings': {'ngram_min': 5, 'ngram_max': 2}},
                       {'linear_settings': {'unknown': 1}},
                       {'kind': 'bge-frozen', 'linear_settings': {'analyzer': 'char'}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                jobs.start_job({**self.payload, **change})
        self.assertEqual(jobs.list_jobs(), [])
        self.assertEqual(list((self.root / 'output').glob('*')), [])

    def test_cross_process_lock_rejects_second_training(self):
        self.registry.mkdir()
        fd = os.open(self.registry / '.training.lock', os.O_CREAT | os.O_RDWR, 0o600)
        try:
            jobs.acquire_lock(fd)
            with self.assertRaisesRegex(ValueError, '已有訓練'):
                jobs.start_job(self.payload)
        finally:
            os.close(fd)
        self.assertEqual(jobs.list_jobs(), [])

    def test_failed_process_and_records_survive_manager_reload(self):
        with patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic launch failure')):
            with self.assertRaisesRegex(OSError, 'synthetic'):
                jobs.start_job(self.payload)
        entries = jobs.list_jobs()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry['state'], 'failed')
        directory = Path(entry['directory'])
        self.assertEqual((directory / '.gitignore').read_text(), '*\n')
        self.assertEqual(entry['config']['source_name'], 'new-history.jsonl')
        self.assertEqual(json.loads((directory / 'data/history.jsonl').read_text())['category'], 'noise')
        self.assertEqual(jobs.get_job(entry['job_id'])['state'], 'failed')
        with self.assertRaisesRegex(ValueError, 'ID'):
            jobs.get_job('../../config')

    def test_abrupt_worker_exit_is_visible_and_retains_log(self):
        real_popen = subprocess.Popen
        with patch.object(jobs.subprocess, 'Popen', side_effect=lambda command, **kwargs: real_popen(
                [sys.executable, '-c', 'raise SystemExit(7)'], **kwargs)):
            entry = jobs.start_job(self.payload)
        for _ in range(150):
            entry = jobs.get_job(entry['job_id'])
            if entry['state'] == 'failed':
                break
            time.sleep(.01)
        self.assertEqual(entry['state'], 'failed')
        self.assertIn('7', entry['message'])
        self.assertTrue((Path(entry['directory']) / 'worker.log').exists())

    def test_backend_validation_and_cloud_consent(self):
        for changes in ({'backend': 'invalid'}, {'backend': 'gpu'}, {'backend': 'modal'},
                        {'kind': 'bge-finetune', 'backend': 'modal', 'cloud_upload_confirmed': False}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                jobs.start_job({**self.payload, **changes})
        self.assertEqual(jobs.list_jobs(), [])

    def test_modal_resources_are_frozen_in_job_config(self):
        resources = jobs.ModalSettings('test-training', 'L4', 1, 4096, 600)
        # The real encoder is an optional download; only its presence is checked here.
        manifest = self.root / 'encoder_manifest.json'
        manifest.write_text('{}', encoding='utf-8')
        with patch.object(jobs.ModalSettings, 'load', return_value=resources), \
             patch.object(jobs, 'BGE_ENCODER_MANIFEST', manifest), \
             patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic failure')):
            with self.assertRaises(OSError):
                jobs.start_job({**self.payload, 'kind': 'bge-finetune', 'backend': 'modal',
                                'cloud_upload_confirmed': True})
        job = jobs.list_jobs()[0]
        self.assertEqual(job['config']['modal_settings']['gpu'], 'L4')
        self.assertEqual(job['config']['modal_settings']['timeout_seconds'], 600)
        self.assertTrue((Path(job['directory']) / 'source/Solution/config/modal_finetune.json').is_file())

    def test_batch_events_do_not_evict_epoch_metrics(self):
        with patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic failure')):
            with self.assertRaises(OSError):
                jobs.start_job(self.payload)
        job = jobs.list_jobs()[0]
        folder = Path(job['directory'])
        events = [{'event': 'epoch', 'epoch': 1, 'validation': {'six_class_macro_f1': .5}}]
        events += [{'event': 'batch', 'completed': n} for n in range(100)]
        (folder / 'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events) + '\n', encoding='utf-8')
        loaded = jobs.get_job(job['job_id'])
        self.assertEqual([e['event'] for e in loaded['events']], ['epoch'])


if __name__ == '__main__':
    unittest.main()
