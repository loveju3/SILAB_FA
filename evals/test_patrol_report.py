"""Source provenance tests with synthetic inputs only; no LLM or dev access."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import run_patrol


class PatrolReportTest(unittest.TestCase):
    def test_multiline_quote_is_valid_but_fabricated_quote_is_not(self):
        index = {f"FAKE{i}": {"raw_text": "D01 異音\n請安排檢查"} for i in range(10)}
        output = [{"ticket_id": key, "reason": "原文提到「D01 異音\n請安排檢查」", "priority": "medium"}
                  for key in index]
        self.assertTrue(run_patrol.contract_valid(output, index))
        output[-1]["reason"] = "原文提到「未出現在原文中的文字」"
        self.assertFalse(run_patrol.contract_valid(output, index))

    def test_original_line_and_record_order_survive_blanks_and_time_order(self):
        rows = [{"ticket_id": "FAKE_B", "created_at": "2026-10-02T00:00:00Z", "reporter": "甲",
                 "raw_text": "原文\n保留換行", "category": "must-not-leak"},
                {"ticket_id": "FAKE_A", "created_at": "2026-10-01T00:00:00Z", "reporter": "乙", "raw_text": "第二筆"}]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            path = root / "data/tickets.jsonl"
            content = "\n" + json.dumps(rows[0]) + "\n\n" + json.dumps(rows[1]) + "\n"
            path.write_text(content, encoding="utf-8")
            with patch.object(run_patrol, "ROOT", root):
                actual, positions, info = run_patrol.load_source("tickets")
            selected = run_patrol.selected_records([{"ticket_id": "FAKE_A", "priority": "high", "reason": "測試"},
                                                    {"ticket_id": "FAKE_B", "priority": "medium", "reason": "測試"}], positions)
            self.assertEqual((selected[0]["rank"], selected[0]["record_index"], selected[0]["source_line"]), (1, 2, 4))
            self.assertEqual((selected[1]["record_index"], selected[1]["source_line"]), (1, 2))
            self.assertEqual(selected[1]["record"]["raw_text"], "原文\n保留換行")
            self.assertNotIn("category", actual[0])
            self.assertEqual(info["count"], 2)
            self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            path.write_text("changed after run", encoding="utf-8")
            self.assertEqual(selected[1]["record"]["raw_text"], "原文\n保留換行")

    def test_synthetic_source_is_explicit_and_not_a_real_ticket_mapping(self):
        rows, positions, info = run_patrol.load_source("synthetic")
        self.assertEqual((info["dataset"], info["count"]), ("synthetic", 12))
        self.assertTrue(all(row["ticket_id"].startswith("SYN") for row in rows))
        self.assertEqual(len(positions), 12)

    def test_unknown_source_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            run_patrol.load_source("dev")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            row = {"ticket_id": "FAKE_A", "created_at": "2026-10-01T00:00:00Z", "reporter": "甲", "raw_text": "測試"}
            (root / "data/tickets.jsonl").write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
            with patch.object(run_patrol, "ROOT", root), self.assertRaises(ValueError):
                run_patrol.load_source("tickets")


if __name__ == "__main__":
    unittest.main()
