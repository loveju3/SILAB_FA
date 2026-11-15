"""UI backend regression tests, without requiring a listening socket."""
import json
import os
import socket
import threading
from urllib.request import urlopen
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helper_tool import ui_server


class UITest(unittest.TestCase):
    def test_idle_preconnection_does_not_block_homepage(self):
        server = ui_server.LocalHTTPServer(("127.0.0.1", 0), ui_server.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with socket.create_connection(server.server_address, timeout=2):
                with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=3) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b"SILAB", response.read())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name)
        mock = patch.object(ui_server, "RUNS", self.runs)
        mock.start()
        self.addCleanup(mock.stop)

    @unittest.skipUnless(os.environ.get("SILAB_VALIDATE_DEV") == "1", "dev 僅供固定版本驗證，需明確啟用")
    def test_baseline_generates_ui_data_and_downloads(self):
        result = ui_server.evaluate_upload({"baseline": True})
        self.assertAlmostEqual(result["summary"]["total"], .441574, places=6)
        self.assertEqual(len(result["errors"]), 234)
        self.assertEqual(result["label"], "Baseline · dev")
        self.assertEqual(len(result["downloads"]), 5)
        run = self.runs / result["run_id"]
        self.assertTrue((run / "errors.csv").exists())
        self.assertEqual(json.loads((run / "summary.json").read_text())["prediction_file"], "Baseline · dev")
        self.assertNotIn("silab-ui-", result["errors"][0]["source_file"])

    def test_custom_upload_uses_display_names_and_unique_runs(self):
        gold = {"ticket_id": "T1", "raw_text": "空壓機□異常", "device_id": "D01",
                "category": "noise", "urgency": "normal", "duplicate_of": None}
        prediction = {**gold, "category": "other", "llm_used": False}
        payload = {"gold": "custom", "gold_content": json.dumps(gold), "gold_name": "答案.jsonl",
                   "prediction_content": json.dumps(prediction), "prediction_name": "實驗.jsonl"}
        result = ui_server.evaluate_upload(payload)
        self.assertEqual(result["summary"]["prediction_file"], "實驗.jsonl")
        self.assertEqual(result["errors"][0]["source_file"], "答案.jsonl")
        self.assertEqual(result["errors"][0]["wrong_fields"], ["category"])
        self.assertGreaterEqual(result["timing"]["total_seconds"], 0)
        self.assertIsNone(result["timing"]["prediction_seconds"])
        saved = json.loads((self.runs / result["run_id"] / "run.json").read_text())
        self.assertEqual(saved["timing"], result["timing"])
        self.assertEqual(len(saved['dataset_sha256']), 64)
        self.assertEqual(saved['count'], 1)
        self.assertNotEqual(result["run_id"], ui_server.evaluate_upload(payload)["run_id"])

    def test_baseline_custom_data_includes_prediction_duration(self):
        row = {"ticket_id": "S1", "raw_text": "空壓機很吵", "device_id": "D01",
               "category": "noise", "urgency": "normal", "duplicate_of": None}
        result = ui_server.evaluate_upload({"gold": "custom", "gold_name": "合成測試.jsonl",
                                            "gold_content": json.dumps(row), "baseline": True})
        self.assertGreaterEqual(result["timing"]["prediction_seconds"], 0)
        self.assertGreaterEqual(result["timing"]["total_seconds"], result["timing"]["prediction_seconds"])

    def test_missing_and_bad_uploads_report_errors(self):
        for payload in ({}, {"gold": "custom"}, {"gold": "unknown"}, {"gold": "history", "prediction_content": "not json"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ui_server.evaluate_upload(payload)
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_experiment_history_endpoint_returns_aggregate_metadata(self):
        folder = self.runs / '20261115-080000-aaaaaaaa'
        folder.mkdir()
        (folder / 'run.json').write_text(json.dumps({'label': 'synthetic experiment'}), encoding='utf-8')
        (folder / 'summary.json').write_text(json.dumps({'total': .5, 'metrics': {'category': .4},
                                                        'raw_text': 'must-not-return'}), encoding='utf-8')
        server = ui_server.LocalHTTPServer(('127.0.0.1', 0), ui_server.Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with urlopen(f'http://127.0.0.1:{server.server_port}/api/experiments', timeout=3) as response:
                result = json.load(response)
            self.assertEqual(result['experiments'][0]['total'], .5)
            self.assertNotIn('raw_text', result['experiments'][0])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)

    def test_unlabelled_answers_do_not_produce_reports(self):
        with self.assertRaisesRegex(ValueError, "缺少標籤"):
            ui_server.evaluate_upload({"gold": "custom", "gold_content": '{"ticket_id":"T1"}',
                                       "prediction_content": '{"ticket_id":"T1"}'})
        self.assertEqual(list(self.runs.iterdir()), [])

    def test_baseline_custom_gold_requires_raw_text(self):
        with self.assertRaisesRegex(ValueError, "raw_text"):
            ui_server.evaluate_upload({"gold": "custom", "gold_content": '{"ticket_id":"T1"}',
                                       "baseline": True})
        self.assertEqual(list(self.runs.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
