"""install.py: stage order, fail-fast, model ZIP verification and venv relaunch."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import install  # noqa: E402
import _venv  # noqa: E402


class InstallFlowTests(unittest.TestCase):
    def invoke(self, argv, run_effect=None, missing=()):
        output = io.StringIO()
        with patch.object(install, "check_platform"), \
                patch.object(install, "create_venv") as venv, \
                patch.object(install, "install_packages") as packages, \
                patch.object(install, "prepare_model") as model, \
                patch.object(install, "model_ready", return_value=True), \
                patch.object(install, "check_data", return_value=list(missing)), \
                patch.object(install, "check_agent_runtime"), \
                patch.object(install, "run", side_effect=run_effect) as run, \
                patch.object(install.VENV_PYTHON.__class__, "is_file", return_value=True), \
                contextlib.redirect_stdout(output):
            result = install.main(argv)
        return result, output.getvalue(), {"venv": venv, "packages": packages, "model": model, "run": run}

    def test_default_runs_every_stage_then_passes(self):
        result, output, calls = self.invoke([])
        self.assertEqual(result, 0)
        self.assertIn("安裝結果：PASS", output)
        for name in ("venv", "packages", "model"):
            calls[name].assert_called_once()
        self.assertIn("--self-test", [str(part) for part in calls["run"].call_args.args[0]])
        self.assertIn("run_demo.py", output)

    def test_check_only_does_not_install_or_download(self):
        result, _, calls = self.invoke(["--check"])
        self.assertEqual(result, 0)
        for name in ("venv", "packages", "model"):
            calls[name].assert_not_called()

    def test_failed_self_test_stops_without_pass(self):
        failure = subprocess.CalledProcessError(1, ["python"])
        result, output, _ = self.invoke([], run_effect=failure)
        self.assertEqual(result, 1)
        self.assertIn("安裝結果：FAIL（階段：驗證套件與模型推論）", output)
        self.assertNotIn("安裝結果：PASS", output)

    def test_missing_public_tables_fail_but_missing_tickets_only_warn(self):
        result, output, _ = self.invoke([], missing=["devices.json"])
        self.assertEqual(result, 1)
        self.assertIn("devices.json", output)
        result, output, _ = self.invoke([], missing=["dev.jsonl"])
        self.assertEqual(result, 0)
        self.assertIn("尚缺 dev.jsonl", output)

    def test_with_agent_runs_agent_setup(self):
        result, _, calls = self.invoke(["--with-agent"])
        self.assertEqual(result, 0)
        self.assertTrue(str(calls["run"].call_args.args[0][1]).endswith("agent/setup.py"))

    def test_unsupported_platform_stops_before_any_command(self):
        with patch.object(install, "check_platform", side_effect=install.SetupError("unsupported")), \
                patch.object(install, "run") as run, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(install.main([]), 1)
        run.assert_not_called()
        self.assertIn("unsupported", output.getvalue())

    def test_nested_requirements_are_included(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.txt").write_text("numpy==2.3.4\n", encoding="utf-8")
            (root / "req.txt").write_text("# c\n-r base.txt\ntorch==2.9.0\n", encoding="utf-8")
            self.assertEqual(list(install.pinned(root / "req.txt")), [("numpy", "2.3.4"), ("torch", "2.9.0")])

    def test_wrong_package_version_fails_self_test(self):
        with patch("importlib.metadata.version", return_value="0.0.0"):
            with self.assertRaisesRegex(install.SetupError, "預期"):
                install.self_test()


class DownloadTests(unittest.TestCase):
    """Downloads prefer curl (OS certificate store) over urllib."""

    def test_uses_system_curl_when_available(self):
        with patch.object(install.shutil, "which", return_value="/usr/bin/curl"), \
                patch.object(install.subprocess, "run") as run, \
                patch.object(install.urllib.request, "urlopen") as urlopen:
            install.fetch("https://example.invalid/model.zip", "model.zip.part")
        urlopen.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/usr/bin/curl")
        self.assertIn("-f", command[1])
        self.assertEqual(command[-3:], ["-o", "model.zip.part", "https://example.invalid/model.zip"])

    def test_urllib_certificate_failure_explains_the_fix(self):
        error = install.urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        with patch.object(install.shutil, "which", return_value=None), \
                patch.object(install.urllib.request, "urlopen", side_effect=error):
            with self.assertRaisesRegex(OSError, "Install Certificates"):
                install.fetch("https://example.invalid/model.zip", "unused.part")

    def test_failed_download_becomes_setup_error_with_manual_route(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(install, "ROOT", Path(temporary)), \
                patch.object(install, "fetch", side_effect=subprocess.CalledProcessError(22, ["curl"])), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(install.SetupError, "--model-zip"):
                install.download_model_zip()


class ModelZipTests(unittest.TestCase):
    def archive(self, root, corrupt=False, escape=False):
        files = {"encoder/" + name: b"fixture-" + name.encode() for name in (
            "config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json",
            "special_tokens_map.json", "vocab.txt")}
        files["encoder/encoder_manifest.json"] = json.dumps({"files_sha256": {
            Path(name).name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}).encode()
        files["heads.safetensors"] = b"heads"
        files["model.json"] = json.dumps({"format": "bge-neural-v1",
                                          "heads_sha256": hashlib.sha256(b"heads").hexdigest()}).encode()
        manifest = {"model_files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                                    for name, content in files.items()}}
        if escape:
            manifest["model_files"]["../evil"] = {"bytes": 1, "sha256": "0"}
        path = root / "model.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("MANIFEST.json", json.dumps(manifest))
            for name, content in files.items():
                archive.writestr("finetuned/" + name, b"tampered" if corrupt and name == "heads.safetensors" else content)
        return path

    def install_into(self, root, **kwargs):
        target = root / "repo/transfer/finetuned"
        with patch.object(install, "MODEL", target), contextlib.redirect_stdout(io.StringIO()):
            install.prepare_model(str(self.archive(root, **kwargs)))
        return target

    def test_valid_zip_is_extracted_and_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = self.install_into(Path(temporary))
            install.validate_model(target)
            self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["finetuned"])

    def test_corrupt_or_escaping_zip_leaves_nothing_behind(self):
        for options in ({"corrupt": True}, {"escape": True}):
            with self.subTest(**options), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaises(install.SetupError):
                    self.install_into(root, **options)
                self.assertFalse((root / "repo/transfer/finetuned").exists())
                self.assertEqual(list((root / "repo/transfer").iterdir()), [])

    def test_existing_valid_model_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.install_into(root)
            before = (target / "model.json").read_bytes()
            self.install_into(root)
            self.assertEqual((target / "model.json").read_bytes(), before)


class VenvRelaunchTests(unittest.TestCase):
    def test_noop_without_venv_or_with_guard(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(_venv, "VENV", Path(temporary) / ".venv"), patch("os.execv") as execv:
            _venv.ensure()
        execv.assert_not_called()
        with patch.dict(os.environ, {_venv.GUARD: "1"}), patch("os.execv") as execv:
            _venv.ensure()
        execv.assert_not_called()

    @unittest.skipIf(os.name == "nt", "POSIX exec path")
    def test_relaunches_with_venv_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            venv = Path(temporary) / ".venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin/python").write_text("")
            with patch.object(_venv, "VENV", venv), patch.dict(os.environ, {}, clear=False), \
                    patch("os.execv") as execv:
                os.environ.pop(_venv.GUARD, None)
                _venv.ensure()
            execv.assert_called_once()
            self.assertEqual(execv.call_args.args[0], str(venv / "bin/python"))


if __name__ == "__main__":
    unittest.main()
