"""Offline Modal transport tests with synthetic models; never connect to Modal."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import hashlib
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from Solution.modal_training import restore_archive, run_modal, remote_python_version, remote_train
import sys
import os
from Solution.training.settings import FineTuneSettings, ModalSettings
from dataclasses import asdict


class ModalArchiveTest(unittest.TestCase):
    def test_remote_generator_packages_epoch_dated_model_and_restores_complete_bundle(self):
        content = b'synthetic checkpoint bytes'
        def comparison(history, output, settings, notify, device):
            self.assertEqual(device, 'cuda')
            output.mkdir()
            old_file = output / 'frozen/base/model.safetensors'
            old_file.parent.mkdir(parents=True)
            old_file.write_bytes(content)
            os.utime(old_file, (0, 0))
            results = []
            for name in ('bge-lr', 'bge-svc', 'frozen-neural', 'finetuned'):
                model = output / name
                model.mkdir()
                (model / 'model.bin').write_bytes(content)
                results.append({'variant': name, 'model_path': str(model)})
            (output / 'comparison.json').write_text(json.dumps({'results': results}), encoding='utf-8')
            notify('epoch', variant='finetuned', epoch=1)
        modules = {'training': SimpleNamespace(),
                   'training.finetuning': SimpleNamespace(run_comparison=comparison),
                   'training.settings': SimpleNamespace(FineTuneSettings=FineTuneSettings)}
        with patch.dict('sys.modules', modules), patch.object(sys, 'path', list(sys.path)):
            messages = list(remote_train('synthetic history', {}, 'synthetic-run'))
        data = b''.join(message[1] for message in messages if message[0] == 'artifact')
        self.assertEqual(messages[-1], ('complete', hashlib.sha256(data).hexdigest()))
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            info = bundle.getinfo('models/frozen/base/model.safetensors')
            self.assertEqual(info.date_time[0], 1980)
            self.assertEqual(bundle.read(info), content)
        with tempfile.TemporaryDirectory() as temporary:
            restore_archive(io.BytesIO(data), Path(temporary))
            self.assertEqual((Path(temporary) / 'models/finetuned/model.bin').read_bytes(), content)

    def archive(self, files):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as bundle:
            for name, value in files.items():
                bundle.writestr(name, value)
        stream.seek(0)
        return stream

    def test_models_relocated_to_local_run(self):
        results = [{'variant': name, 'model_path': '/tmp/run/models/' + name} for name in ('a', 'b', 'c', 'd')]
        files = {'models/comparison.json': json.dumps({'results': results})}
        files.update({'models/' + row['variant']: 'synthetic model' for row in results})
        with tempfile.TemporaryDirectory() as temporary:
            restore_archive(self.archive(files), Path(temporary))
            report = json.loads((Path(temporary) / 'models/comparison.json').read_text(encoding='utf-8'))
            self.assertEqual(report['training_backend'], 'modal')
            self.assertTrue(all(Path(row['model_path']).is_file() for row in report['results']))

    def test_traversal_and_symlinks_rejected(self):
        for name in ('models/../../escape', '/tmp/escape', 'data/history.jsonl', 'models/..\\escape'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, '路徑'):
                    restore_archive(self.archive({name: 'bad'}), Path(temporary))
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as bundle:
            info = zipfile.ZipInfo('models/link')
            info.external_attr = 0o120777 << 16
            bundle.writestr(info, '/tmp/outside')
        stream.seek(0)
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
            restore_archive(stream, Path(temporary))

    def test_modal_stream_upload_scope_and_local_model_restore(self):
        results = [{'variant': name, 'model_path': '/tmp/run/models/' + name} for name in ('a', 'b', 'c', 'd')]
        files = {'models/comparison.json': json.dumps({'results': results})}
        files.update({'models/' + row['variant']: 'synthetic model' for row in results})
        archive = self.archive(files).getvalue()
        mounts, calls, notifications, options, images = [], [], [], [], []
        class Image:
            def pip_install(self, *args): return self
            def env(self, values): return self
            def add_local_dir(self, source, destination, **kwargs):
                mounts.append((Path(source), destination))
                return self
        class Remote:
            def remote_gen(self, content, settings, run_id):
                calls.append((content, settings, run_id))
                yield ('event', 'stage', {'message': 'synthetic training'})
                yield ('artifact', archive[:100])
                yield ('artifact', archive[100:])
                yield ('complete', hashlib.sha256(archive).hexdigest())
        class App:
            def function(self, **kwargs):
                options.append(kwargs)
                return lambda function: Remote()
            def run(self): return nullcontext()
        def create_image(**kwargs):
            images.append(kwargs)
            return Image()
        modal = SimpleNamespace(__version__='synthetic-sdk', Image=SimpleNamespace(debian_slim=create_image), App=lambda name: App())
        with tempfile.TemporaryDirectory() as temporary, patch.dict('sys.modules', {'modal': modal}):
            directory = Path(temporary)
            (directory / 'source/Solution').mkdir(parents=True)
            for name in ('requirements-ml.txt', 'requirements-embedding.txt'):
                (directory / 'source/Solution' / name).write_text('numpy==2.3.4\n')
            (directory / 'data').mkdir()
            (directory / 'data/history.jsonl').write_text('selected synthetic history', encoding='utf-8')
            resources = {'app_name': 'test-training', 'gpu': 'L4', 'cpu': 1, 'memory_mb': 4096, 'timeout_seconds': 600}
            run_modal(directory, FineTuneSettings(), lambda event, **values: notifications.append((event, values)), resources)
            self.assertEqual({key: options[0][key] for key in ('gpu', 'cpu', 'memory', 'timeout')},
                             {'gpu': 'L4', 'cpu': 1, 'memory': 4096, 'timeout': 600})
            runtime = json.loads((directory / 'modal-runtime.json').read_text(encoding='utf-8'))
            self.assertEqual(runtime['resources'], resources)
            self.assertEqual(images[0]['python_version'], f'{sys.version_info.major}.{sys.version_info.minor}')
            self.assertEqual(runtime['python_version'], images[0]['python_version'])
            self.assertEqual(calls[0][0], 'selected synthetic history')
            self.assertEqual([destination for _, destination in mounts],
                             ['/workspace', '/workspace/Solution/models/bge-small-zh-v1.5'])
            self.assertEqual(mounts[0][0], directory / 'source')
            self.assertTrue((directory / 'models/a').is_file())
            self.assertFalse((directory / 'modal-models.zip.part').exists())
            self.assertTrue(any(values.get('message') == 'synthetic training' for _, values in notifications))


class ModalSettingsTest(unittest.TestCase):
    def test_serialized_image_matches_supported_client_minor_versions(self):
        for minor in (12, 13):
            with self.subTest(minor=minor), patch('Solution.modal_training.sys.version_info', SimpleNamespace(major=3, minor=minor)):
                self.assertEqual(remote_python_version(), f'3.{minor}')
        with patch('Solution.modal_training.sys.version_info', SimpleNamespace(major=3, minor=14)), self.assertRaises(ValueError):
            remote_python_version()

    def test_project_file_and_utf8_bom(self):
        settings = ModalSettings.load()
        self.assertEqual(settings.gpu, 'T4')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'modal.json'
            path.write_text(json.dumps(asdict(settings)), encoding='utf-8-sig')
            self.assertEqual(ModalSettings.load(path), settings)

    def test_invalid_settings_and_credentials_are_rejected(self):
        original = asdict(ModalSettings.load())
        for change in ({'token': 'never-store-here'}, {'gpu': 'invalid'}, {'cpu': True},
                       {'memory_mb': -1}, {'timeout_seconds': 0}, {'app_name': '../escape'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ModalSettings.from_dict({**original, **change})
        with self.assertRaises(ValueError):
            ModalSettings.from_dict({})


if __name__ == '__main__':
    unittest.main()
