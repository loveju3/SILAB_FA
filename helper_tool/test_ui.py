"""UI backend regression tests, without requiring a listening socket."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from helper_tool import ui_server


class UITest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runs = Path(self.temp.name)
        mock = patch.object(ui_server, "RUNS", self.runs)
        mock.start()
        self.addCleanup(mock.stop)

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
        self.assertNotEqual(result["run_id"], ui_server.evaluate_upload(payload)["run_id"])

    def test_missing_and_bad_uploads_report_errors(self):
        for payload in ({}, {"gold": "custom"}, {"gold": "unknown"}, {"prediction_content": "not json"}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ui_server.evaluate_upload(payload)
        self.assertEqual(list(self.runs.iterdir()), [])

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
