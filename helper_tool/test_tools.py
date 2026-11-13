"""Regression tests for official score parity and actionable diagnostics."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from helper_tool.common import ROOT, read_records
from helper_tool.evaluate import analyze, write_reports
from helper_tool.lookup import build_index, display


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def save(self, name, rows):
        path = self.directory / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        return path

    def gold(self, ticket_id="T1", **changes):
        return {"ticket_id": ticket_id, "raw_text": "設備□異常\n請查看", "reporter": "甲",
                "created_at": "2026-01-01T00:00:00Z", "device_id": "D01",
                "category": "noise", "urgency": "normal", "duplicate_of": None, **changes}

    def score(self, gold, predictions):
        summary, errors, output = analyze(self.save("pred.jsonl", predictions), self.save("gold.jsonl", gold))
        weighted = sum(summary["metrics"][k] * v for k, v in summary["weights"].items())
        self.assertAlmostEqual(summary["total"], weighted, places=14)
        return summary, errors, output

    @unittest.skipUnless(os.environ.get("SILAB_VALIDATE_DEV") == "1", "dev 僅供固定版本驗證，需明確啟用")
    def test_real_baseline_matches_official_and_exports(self):
        prediction = self.directory / "baseline.jsonl"
        subprocess.run([sys.executable, str(ROOT / "baseline/predict.py"), str(ROOT / "data/dev.jsonl"),
                        str(prediction)], check=True, capture_output=True)
        summary, errors, output = analyze(prediction, ROOT / "data/dev.jsonl")
        self.assertAlmostEqual(summary["total"], .442, places=3)
        self.assertAlmostEqual(summary["total"], sum(summary["metrics"][k] * v for k, v in summary["weights"].items()))
        self.assertEqual(summary["gold_count"], 300)
        self.assertEqual(summary["duplicate_counts"], {"tp": 0, "fp": 0, "fn": 85})
        report_dir = self.directory / "report"
        write_reports(report_dir, summary, errors, output)
        self.assertEqual(len(list(report_dir.iterdir())), 5)
        self.assertEqual(len(list(read_records(report_dir / "errors.jsonl"))), len(errors))

    def test_perfect_all_classes(self):
        gold = [self.gold(f"T{i}", category=category, urgency=("low", "normal", "high")[i % 3],
                          duplicate_of="T0" if i else None)
                for i, category in enumerate(("noise", "temperature", "power", "software", "other"))]
        summary, errors, _ = self.score(gold, [{**g, "llm_used": False} for g in gold])
        self.assertEqual(summary["total"], 1)
        self.assertEqual(errors, [])

    def test_abstention_category_exclusion_and_device_credit(self):
        gold = [self.gold(device_id=None, category=None)]
        summary, errors, _ = self.score(gold, [{**gold[0], "category": "other", "llm_used": False}])
        self.assertEqual(summary["metrics"]["device"], 0)
        self.assertEqual(errors[0]["wrong_fields"], ["device_id"])
        self.assertEqual(errors[0]["differences"]["device_id"]["reason"], "abstention_requires_both_null")
        summary, errors, _ = self.score(gold, [{**gold[0], "llm_used": False}])
        self.assertEqual(summary["metrics"]["device"], 1)
        self.assertEqual(errors, [])

    def test_duplicate_wrong_target_false_positive_and_false_negative(self):
        gold = [self.gold("T1", duplicate_of="T0"), self.gold("T2"), self.gold("T3", duplicate_of="T2")]
        predictions = [{**g, "llm_used": False, "duplicate_of": duplicate}
                       for g, duplicate in zip(gold, ("T9", "T1", None))]
        summary, errors, _ = self.score(gold, predictions)
        self.assertEqual(summary["duplicate_counts"], {"tp": 0, "fp": 2, "fn": 2})
        self.assertEqual([e["differences"]["duplicate_of"]["reason"] for e in errors],
                         ["wrong_target", "false_positive", "false_negative"])

    def test_missing_invalid_extra_and_suggestion(self):
        gold = [self.gold("T1"), self.gold("T2", device_id=None, category=None)]
        predictions = [{**gold[0], "llm_used": "false"}, {**self.gold("EXTRA"), "llm_used": False}]
        summary, errors, _ = self.score(gold, predictions)
        self.assertEqual(summary["missing_ids"], ["T2"])
        self.assertEqual(summary["extra_ids"], ["EXTRA"])
        self.assertEqual(summary["metrics"]["device"], 0)
        self.assertEqual(summary["suggestion_warning_ids"], ["T1", "EXTRA"])
        self.assertTrue(all(len(e["wrong_fields"]) == 4 for e in errors))

    def test_missing_required_field_is_invalid(self):
        gold = self.gold()
        prediction = {**gold, "llm_used": False}
        del prediction["category"]
        summary, errors, _ = self.score([gold], [prediction])
        self.assertEqual(summary["total"], 0)
        self.assertEqual(errors[0]["validation_errors"], ["缺少必要欄位：category"])

    def test_empty_prediction_and_unlabelled_gold(self):
        summary, _, _ = self.score([self.gold()], [])
        self.assertEqual(summary["total"], 0)
        with self.assertRaisesRegex(ValueError, "缺少標籤"):
            self.score([{"ticket_id": "T1", "raw_text": "無答案"}], [])

    def test_duplicate_ids_and_malformed_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "ticket_id 重複"):
            list(read_records(self.save("duplicate.jsonl", [self.gold(), self.gold()])))
        path = self.directory / "bad.jsonl"
        path.write_text('{"ticket_id":', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bad.jsonl:1: JSON"):
            list(read_records(path))

    def test_lookup_preserves_source_and_multiple_matches(self):
        a = self.save("a.jsonl", [self.gold()])
        b = self.save("b.jsonl", [{**self.gold(), "llm_used": False}])
        index = build_index([a, b])
        self.assertEqual(len(index["T1"]), 2)
        self.assertEqual(index["T1"][0]["source_line"], 1)
        self.assertEqual(index["T1"][0]["record"]["raw_text"], "設備□異常\n請查看")
        out = io.StringIO()
        with redirect_stdout(out):
            found = display(["T1", "unknown"], index, as_json=True)
        self.assertFalse(found)
        self.assertEqual(json.loads(out.getvalue())["unknown"], [])

    def test_cli_from_other_directory(self):
        source = self.save("tickets.jsonl", [self.gold("T1")])
        result = subprocess.run([sys.executable, str(ROOT / "helper_tool/lookup.py"), "T1", "--files", str(source), "--json"],
                                cwd=self.directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["T1"][0]["record"], self.gold("T1"))


if __name__ == "__main__":
    unittest.main()
