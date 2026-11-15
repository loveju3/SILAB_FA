import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from agent.setup import RUNTIME_TAG, extract_runtime, runtime_asset


class SetupTests(unittest.TestCase):
    def test_selects_native_release_and_pinned_hash(self):
        for system, machine, suffix in (
            ('win32', 'AMD64', 'win-cpu-x64.zip'),
            ('darwin', 'arm64', 'macos-arm64.tar.gz'),
        ):
            url, digest = runtime_asset(system, machine)
            self.assertTrue(url.endswith(suffix))
            self.assertIn('/' + RUNTIME_TAG + '/', url)
            self.assertEqual(len(digest), 64)
        with self.assertRaises(ValueError):
            runtime_asset('linux', 'x86_64')

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
