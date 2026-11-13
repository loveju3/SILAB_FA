"""End-to-end local prediction/scoring and container command contracts."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from helper_tool import experiments


class ExperimentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.output, self.reports = self.root / "output", self.root / "reports"

    def payload(self, labelled=True, **changes):
        row = {"ticket_id": "T1", "created_at": "2026-01-01T00:00:00Z", "reporter": "甲",
               "raw_text": "空壓機很吵"}
        if labelled:
            row.update(device_id="D01", category="noise", urgency="normal", duplicate_of=None)
        return {"dataset": "custom", "dataset_name": "我的測試.jsonl", "dataset_content": json.dumps(row),
                "output_dir": str(self.output), **changes}

    def test_actual_solution_produces_named_predictions_score_and_all_records(self):
        result = experiments.run_experiment(self.payload(), self.reports)
        info = result["execution"]
        self.assertRegex(info["filename"], r"^\d{8}-\d{6}-\d{6}_我的測試_solution-a_native_0\.\d{6}\.jsonl$")
        output = Path(info["output_path"])
        self.assertEqual(output.parent, self.output)
        prediction = json.loads(output.read_text())
        self.assertEqual((prediction["device_id"], prediction["category"]), ("D01", "noise"))
        self.assertEqual(result["errors"], [])
        self.assertEqual(len(result["records"]), 1)
        self.assertEqual(result["records"][0]["source_file"], "我的測試.jsonl")
        self.assertEqual(len(result["downloads"]), 6)
        self.assertEqual((self.reports / result["run_id"] / "predictions.jsonl").read_bytes(), output.read_bytes())
        self.assertEqual(result["summary"]["prediction_file"], str(output))

    def test_baseline_errors_remain_available(self):
        payload = self.payload(method="baseline")
        row = json.loads(payload["dataset_content"])
        row["raw_text"] = "貨梯登不進去"
        row.update(device_id="D07", category="software")
        payload["dataset_content"] = json.dumps(row)
        result = experiments.run_experiment(payload, self.reports)
        self.assertIn("category", result["errors"][0]["wrong_fields"])
        self.assertIn("_baseline_native_", result["label"])
        self.assertEqual(result["records"][0], result["errors"][0])

    def test_unlabelled_dataset_produces_predictions_without_fabricated_score(self):
        result = experiments.run_experiment(self.payload(labelled=False), self.reports)
        self.assertIsNone(result["summary"])
        self.assertEqual(result["errors"], [])
        self.assertIsNone(result["records"][0]["gold"])
        self.assertTrue(result["label"].endswith("_未評分.jsonl"))
        self.assertEqual(set(result["downloads"]), {"predictions.jsonl"})

    def test_total_and_prediction_timing_are_returned_and_saved(self):
        def execute(method, limit, folder):
            (folder / "predictions.jsonl").write_text(json.dumps({
                "ticket_id": "T1", "device_id": "D01", "category": "noise", "urgency": "normal",
                "duplicate_of": None, "llm_used": False, "suggestion": "檢查聲音來源"}), encoding="utf-8")
            return {"method": method, "limit": limit, "wall_seconds": 2.5}

        for labelled in (True, False):
            with self.subTest(labelled=labelled), patch.object(experiments, "execute", side_effect=execute), \
                 patch.object(experiments, "perf_counter", side_effect=[100.0, 110.0]):
                result = experiments.run_experiment(self.payload(labelled=labelled), self.reports)
            self.assertEqual(result["timing"], {"total_seconds": 10.0, "prediction_seconds": 2.5})
            saved = json.loads((self.reports / result["run_id"] / "run.json").read_text())
            self.assertEqual(saved["timing"], result["timing"])

    def test_invalid_input_and_options_publish_nothing(self):
        payloads = [self.payload(method="arbitrary.py"), self.payload(limit="pretend-2gb"),
                    self.payload(method=[]), self.payload(limit={}),
                    self.payload(dataset="../dev"), self.payload(output_dir=""),
                    self.payload(dataset_content=""), self.payload(dataset_content="[]"),
                    self.payload(dataset_content='{"ticket_id":"x","category":"noise"}'),
                    self.payload(dataset_content='{"ticket_id":"x"}')]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                experiments.run_experiment(payload, self.reports)
        self.assertFalse(self.reports.exists())
        self.assertFalse(self.output.exists())

    def test_name_collision_does_not_overwrite_previous_output(self):
        with patch.object(experiments, "datetime") as clock:
            clock.now.return_value = datetime(2026, 11, 13, 12, 0, 0, 123456)
            first = experiments.run_experiment(self.payload(), self.reports)
            output = Path(first["execution"]["output_path"])
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                experiments.run_experiment(self.payload(), self.reports)
            self.assertEqual(output.read_bytes(), before)
            self.assertEqual(len(list(self.reports.iterdir())), 1)

    def test_timeout_does_not_publish_output(self):
        with patch.object(experiments.subprocess, "run", side_effect=subprocess.TimeoutExpired("predict", 480)):
            with self.assertRaisesRegex(ValueError, "480"):
                experiments.run_experiment(self.payload(), self.reports)
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertFalse(self.reports.exists())

    def test_container_limits_are_enforced_and_checked(self):
        def run(command, **kwargs):
            if command[:2] == ["docker", "run"]:
                (self.root / "limits.json").write_text(json.dumps({
                    "memory.max": "2147483648", "memory.swap.max": "0", "exit_code": 0}))
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.object(experiments.shutil, "which", return_value="/usr/bin/docker"), \
             patch.object(experiments.subprocess, "run", side_effect=run) as process:
            result = experiments.execute("solution-a", "2gb", self.root)
        command = process.call_args_list[0].args[0]
        for flag in ("--memory=2g", "--memory-swap=2g", "--network=none", "--cpus=2", "--pull=never"):
            self.assertIn(flag, command)
        self.assertIn("limits", result)
        self.assertEqual(process.call_args_list[-1].args[0][:3], ["docker", "rm", "-f"])

    def test_container_false_limit_and_failure_are_not_success(self):
        (self.root / "limits.json").write_text(json.dumps({"memory.max": "max", "memory.swap.max": "max", "exit_code": 0}))
        with patch.object(experiments.shutil, "which", return_value="docker"), \
             patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(ValueError, "未驗證"):
                experiments.execute("solution-a", "2gb", self.root)
        with patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 2, "", "failure")):
            with self.assertRaisesRegex(ValueError, "failure"):
                experiments.execute("solution-a", "native", self.root)

    def test_container_timeout_still_cleans_up_its_own_container(self):
        with patch.object(experiments.shutil, "which", return_value="docker"), \
             patch.object(experiments.subprocess, "run", side_effect=[
                 subprocess.TimeoutExpired("docker", 480), subprocess.CompletedProcess([], 0)]) as process:
            with self.assertRaisesRegex(ValueError, "480"):
                experiments.execute("solution-a", "2gb", self.root)
        self.assertEqual(process.call_args_list[-1].args[0][:3], ["docker", "rm", "-f"])

    def test_ml_native_method_is_forwarded_to_prediction_cli(self):
        with patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as process:
            experiments.execute("tfidf-lr", "native", self.root)
        command = process.call_args.args[0]
        self.assertEqual(command[command.index("--method") + 1], "tfidf-lr")
        self.assertIn("--metrics", command)
        self.assertNotIn("shell", process.call_args.kwargs)

    def test_ml_container_uses_ml_image_and_frozen_method(self):
        (self.root / "limits.json").write_text(json.dumps({
            "memory.max": "2147483648", "memory.swap.max": "0", "exit_code": 0}))
        with patch.object(experiments.shutil, "which", return_value="docker"), \
             patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as process:
            experiments.execute("tfidf-svc", "2gb", self.root)
        command = process.call_args_list[0].args[0]
        self.assertIn("silab-fa:scheme-b", command)
        self.assertEqual(command[command.index("--method") + 1], "tfidf-svc")

    def test_embedding_container_selects_c_image_and_method(self):
        (self.root / "limits.json").write_text(json.dumps({
            "memory.max": "2147483648", "memory.swap.max": "0", "exit_code": 0}))
        with patch.object(experiments.shutil, "which", return_value="docker"), \
             patch.object(experiments.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")) as process:
            experiments.execute("bge-lr", "2gb", self.root)
        command = process.call_args_list[0].args[0]
        self.assertIn("silab-fa:scheme-c", command)
        self.assertEqual(command[command.index("--method") + 1], "bge-lr")

    def test_file_label_cannot_escape_output_directory(self):
        payload = self.payload(dataset_name="../../不可執行;$(whoami).jsonl")
        result = experiments.run_experiment(payload, self.reports)
        output = Path(result["execution"]["output_path"])
        self.assertEqual(output.parent, self.output)
        self.assertNotIn("$", output.name)
        self.assertNotIn(";", output.name)


class SavedModelPathTest(unittest.TestCase):
    def test_saved_model_path_is_forwarded_without_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = folder / 'saved model' / 'finetuned'
            with patch.object(experiments.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as runner:
                result = experiments.execute('bge-finetuned', 'native', folder, model)
            command = runner.call_args.args[0]
            self.assertEqual(command[command.index('--model-path') + 1], str(model))
            self.assertEqual(result['model_path'], str(model))
            self.assertNotIn('shell', runner.call_args.kwargs)

    def test_container_mounts_saved_checkpoint_readonly(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = folder / 'saved model' / 'finetuned'
            (folder / 'limits.json').write_text(json.dumps({'memory.max':'2147483648','memory.swap.max':'0','exit_code':0}))
            with patch.object(experiments.shutil, 'which', return_value='docker'), \
                 patch.object(experiments.subprocess, 'run', return_value=subprocess.CompletedProcess([],0,'','')) as runner:
                experiments.execute('bge-finetuned', '2gb', folder, model)
            command = runner.call_args_list[0].args[0]
            self.assertIn(f'type=bind,src={model.parent},dst=/models,readonly',command)
            self.assertEqual(command[command.index('--model-path') + 1], '/models/finetuned')


if __name__ == "__main__":
    unittest.main()
