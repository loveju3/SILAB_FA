import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from unittest.mock import patch
import urllib.error

from agent import setup as agent_setup
from agent.setup import RUNTIME_TAG, extract_runtime, runtime_asset


class SetupTests(unittest.TestCase):
    def test_selects_native_release_and_pinned_hash(self):
        for system, machine, suffix in (
            ('win32', 'AMD64', 'win-cpu-x64.zip'),
            ('darwin', 'arm64', 'macos-arm64.tar.gz'),
            ('linux', 'x86_64', 'ubuntu-x64.tar.gz'),
            ('linux', 'aarch64', 'ubuntu-arm64.tar.gz'),
        ):
            url, digest = runtime_asset(system, machine)
            self.assertTrue(url.endswith(suffix))
            self.assertIn('/' + RUNTIME_TAG + '/', url)
            self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            runtime_asset('darwin', 'x86_64')

    def test_runtime_download_uses_curl_not_python_certificates(self):
        with patch.object(agent_setup.shutil, 'which', return_value='/usr/bin/curl'), \
                patch.object(agent_setup.subprocess, 'run') as run, \
                patch.object(agent_setup.urllib.request, 'urlopen') as urlopen:
            agent_setup.fetch('https://example.invalid/runtime.tar.gz', Path('runtime.part'))
        urlopen.assert_not_called()
        self.assertEqual(run.call_args.args[0][0], '/usr/bin/curl')

    def test_certificate_failure_without_curl_explains_the_fix(self):
        error = urllib.error.URLError('[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed')
        with patch.object(agent_setup.shutil, 'which', return_value=None), \
                patch.object(agent_setup.urllib.request, 'urlopen', side_effect=error):
            with self.assertRaisesRegex(ValueError, 'Install Certificates'):
                agent_setup.fetch('https://example.invalid/runtime.tar.gz', Path('unused.part'))

    def test_macos_archive_flattens_release_folder_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'runtime.tar.gz'
            with tarfile.open(archive, 'w:gz') as source:
                for name, content in [('llama-server', b'binary'), ('libllama.dylib', b'library')]:
                    entry = tarfile.TarInfo(f'llama-{RUNTIME_TAG}/{name}')
                    entry.size, entry.mode = len(content), 0o755
                    source.addfile(entry, io.BytesIO(content))
            extract_runtime(archive, root / 'runtime')
            self.assertEqual((root / 'runtime/llama-server').read_bytes(), b'binary')
            self.assertTrue((root / 'runtime/libllama.dylib').is_file())

    def test_rejects_archive_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'runtime.zip'
            with zipfile.ZipFile(archive, 'w') as source:
                source.writestr('../escape', 'bad')
            with self.assertRaises(ValueError):
                extract_runtime(archive, root / 'runtime')
            archive = root / 'runtime.tar.gz'
            with tarfile.open(archive, 'w:gz') as source:
                entry = tarfile.TarInfo('../escape')
                source.addfile(entry)
            with self.assertRaises(tarfile.FilterError):
                extract_runtime(archive, root / 'runtime')
            self.assertFalse((root / 'escape').exists())

    def test_windows_setup_keeps_identical_runtime_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'runtime.zip'
            with zipfile.ZipFile(archive, 'w') as source:
                source.writestr('llama-server.exe', b'binary')
            runtime = root / 'runtime'
            extract_runtime(archive, runtime)
            binary = runtime / 'llama-server.exe'
            before = binary.stat().st_mtime_ns
            extract_runtime(archive, runtime)
            self.assertEqual(binary.stat().st_mtime_ns, before)
            binary.write_bytes(b'broken')
            extract_runtime(archive, runtime)
            self.assertEqual(binary.read_bytes(), b'binary')


if __name__ == '__main__':
    unittest.main()
