"""Evaluation contracts using synthetic fixtures only; no real history/dev or LLM."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evals import run
from evals.history_suite import build_history_suite


class EvalContracts(unittest.TestCase):
    def test_null_not_found_is_distinct_from_process_failure(self):
        case = {"id": "public-unknown", "answer": None, "expected_evidence": []}
        self.assertTrue(run.grade_answer(case, {"answer": None, "evidence": []}, {"S1"}))
        for answer in (None, [], {}, {"answer": None},
                       {"answer": None, "evidence": ["S1"]}):
            self.assertFalse(run.grade_answer(case, answer, {"S1"}))

    def test_refusal_must_be_explicit_and_use_no_data_tools(self):
        case = {"id": "scope", "answer": {"status": "out_of_scope"},
                "evidence_required": False, "expected_evidence": [], "allowed_tools": []}
        answer = {"answer": case["answer"], "evidence": [], "tools_used": []}
        self.assertTrue(run.grade_answer(case, answer, {"S1"}))
        for bad in ({**answer, "tools_used": ["list_devices"]},
                    {**answer, "answer": None}, {**answer, "evidence": ["S1"]},
                    {"answer": case["answer"], "evidence": []}):
            self.assertFalse(run.grade_answer(case, bad, {"S1"}))

    def test_exact_citations_prevent_correct_answer_from_wrong_records(self):
        case = {"id": "date", "answer": 2, "expected_evidence": ["S1", "S2"]}
        self.assertTrue(run.grade_answer(case, {"answer": 2, "evidence": ["S2", "S1"]}, {"S1", "S2", "S3"}))
        for value, evidence in ((True, ["S1", "S2"]), (2, ["S1", "S3"]),
                                (2, ["S1", "S1"]), (2, []), (2, ["UNKNOWN"])):
            self.assertFalse(run.grade_answer(case, {"answer": value, "evidence": evidence}, {"S1", "S2", "S3"}))

    def test_fixture_follows_answer_format_and_snapshot(self):
        from agent.tools import SNAPSHOT
        fixture = json.loads((run.ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["snapshot"], SNAPSHOT)
        ids = {t["ticket_id"] for t in fixture["tickets"]}
        self.assertEqual(set(fixture["semantic_devices"]), ids)
        case_ids = {c["id"] for c in fixture["cases"]}
        for case in fixture["cases"]:
            # HW_RULES: answer is a bare value; no invented {"status": ...} objects.
            self.assertNotIsInstance(case["answer"], dict, case["id"])
            self.assertIn(case.get("split", "dev"), ("dev", "holdout"), case["id"])
            self.assertLessEqual(set(case.get("expected_evidence", [])), ids, case["id"])
            self.assertLessEqual(len(case.get("expected_evidence", [])), 20, case["id"])
            if "paired_with" in case:
                self.assertIn(case["paired_with"], case_ids, case["id"])
        self.assertTrue(any(c.get("split") == "holdout" for c in fixture["cases"]))

    def test_split_selection_and_category_summary(self):
        cases = [{"id": "a"}, {"id": "b", "split": "dev"}, {"id": "c", "split": "holdout"}]
        self.assertEqual([c["id"] for c in run.select_cases(cases)], ["a", "b"])
        self.assertEqual([c["id"] for c in run.select_cases(cases, split="holdout")], ["c"])
        self.assertEqual(len(run.select_cases(cases, split="all")), 3)
        self.assertEqual([c["id"] for c in run.select_cases(cases, "c")], ["c"])  # explicit --case
        rows = [{"category": "x", "passed": True}, {"category": "x", "passed": False}, {"passed": True}]
        self.assertEqual(run.summarize(rows), {"x": {"passed": 1, "total": 2},
                                               "uncategorized": {"passed": 1, "total": 1}})

    def test_expected_tools_is_diagnostic_only(self):
        case = {"id": "m", "answer": 1, "evidence_required": False, "expected_tools": ["a", "b"]}
        answer = {"answer": 1, "evidence": [], "tools_used": ["a"]}
        self.assertFalse(run.tools_matched(case, answer))
        self.assertTrue(run.grade_answer(case, answer, set()))
        self.assertTrue(run.tools_matched(case, {**answer, "tools_used": ["b", "a"]}))
        self.assertIsNone(run.tools_matched({"id": "n"}, answer))

    def test_history_keeps_original_public_cases_and_has_reasons(self):
        fixture = json.loads((run.ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))
        self.assertEqual(len({c["id"] for c in fixture["cases"]}), len(fixture["cases"]))
        self.assertTrue(all(c.get("rationale") for c in fixture["cases"]))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "evals").mkdir()
            # Shuffle to ensure history selection is by ID rather than fixture position.
            (root / "evals/fixtures.json").write_text(json.dumps({**fixture, "cases": fixture["cases"][::-1]}), encoding="utf-8")
            row = {"ticket_id": "FAKE", "created_at": "2026-01-01T00:00:00Z",
                   "reporter": "合成人", "raw_text": "空壓機", "answer_label": "ignored"}
            (root / "data/history.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = build_history_suite(root)
        self.assertEqual(len(result["cases"]), 12)
        self.assertTrue(all(c.get("rationale") for c in result["cases"]))
        self.assertEqual(len([c for c in result["cases"] if c["id"].startswith("history-")]), 6)
        self.assertNotIn("answer_label", result["tickets"][0])

    def test_reason_survives_runner_events_and_saved_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "evals").mkdir()
            case = {"id": "public-unknown", "question": "合成查詢", "answer": None,
                    "rationale": "驗證查無資料不虛構答案", "expected_evidence": []}
            (root / "evals/fixtures.json").write_text(json.dumps({"provenance": "synthetic", "tickets": [], "cases": [case]}), encoding="utf-8")
            output = root / "report.json"
            stdout = io.StringIO()
            result = subprocess.CompletedProcess([], 0, json.dumps({"answer": None, "evidence": []}), "")
            with patch.object(run, "ROOT", root), patch.object(run, "digest", return_value="fake-hash"), \
                 patch.object(run.subprocess, "check_output", return_value="fake-revision"), \
                 patch.object(run.subprocess, "run", return_value=result), \
                 patch.object(sys, "argv", ["evals/run.py", "--events", "--output", str(output)]), redirect_stdout(stdout):
                self.assertEqual(run.main(), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            events = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(report["cases"][0]["rationale"], case["rationale"])
            self.assertEqual(events[1]["case"]["rationale"], case["rationale"])
            self.assertEqual(report["total"], 1)


if __name__ == "__main__":
    unittest.main()
