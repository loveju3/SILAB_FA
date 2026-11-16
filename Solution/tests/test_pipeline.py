"""Observable contracts: causal context, sorting, rules, label isolation and CLI."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

from triage.classification import RuleClassifier
from triage.contracts import validate_predictions
from triage.devices import DeviceResolver
from triage.io import ROOT, load_json, read_tickets
from triage.pipeline import predict
from triage.rules import urgency
from triage.text import error_codes, normalize, urgency_text


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        codes = load_json(ROOT / "data/error_codes.json")
        self.resolver = DeviceResolver(load_json(ROOT / "data/devices.json"), codes,
                                       load_json(SOLUTION / "config/device_aliases.json"))
        self.classifier = RuleClassifier(load_json(SOLUTION / "config/category_rules.json"), codes)

    def row(self, tid, text, minute=0, **extra):
        return {"ticket_id": tid, "created_at": f"2026-01-01T00:{minute:02d}:00Z",
                "reporter": "甲", "raw_text": text, **extra}

    def run_rows(self, rows):
        path = self.path / "input.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        tickets = read_tickets(path)
        predictions, timings = predict(tickets, self.resolver, self.classifier)
        self.assertEqual(len(timings), len(rows))
        validate_predictions(tickets, predictions, set(self.resolver.devices))
        return predictions

    def test_sorted_recent_duplicate_and_output_order(self):
        result = self.run_rows([self.row("T3", "空壓機很吵", 3), self.row("T1", "空壓機很吵", 1),
                                self.row("T2", "空壓機很吵", 2)])
        self.assertEqual([p.ticket_id for p in result], ["T3", "T1", "T2"])
        self.assertEqual([p.duplicate_of for p in result], ["T2", None, "T1"])

    def test_equal_timestamp_uses_input_order(self):
        a, b = self.run_rows([self.row("Z", "空壓機很吵"), self.row("A", "空壓機很吵")])
        self.assertIsNone(a.duplicate_of)
        self.assertEqual(b.duplicate_of, "Z")

    def test_reporter_and_category_are_separate_keys(self):
        result = self.run_rows([self.row("1", "空壓機很吵"), self.row("2", "空壓機不冷", 1),
                                self.row("3", "空壓機很吵", 2, reporter="乙"),
                                self.row("4", "空壓機很吵", 3)])
        self.assertEqual([r.duplicate_of for r in result], [None, None, None, "1"])

    def test_context_changes_category_and_resets_between_runs(self):
        result = self.run_rows([self.row("1", "冰水主機燙手"), self.row("2", "那台一開就跳", 1)])
        self.assertEqual((result[1].device_id, result[1].category), ("D02", "power"))
        fresh = self.run_rows([self.row("3", "那台一開就跳")])[0]
        self.assertIsNone(fresh.device_id)
        self.assertEqual(fresh.category, "power")

    def test_no_future_context_and_conflicting_location(self):
        result = self.run_rows([self.row("1", "那台很吵"), self.row("2", "UPS很吵", 1),
                                self.row("3", "頂樓那台很吵", 2)])
        self.assertIsNone(result[0].device_id)
        self.assertIsNone(result[2].device_id)

    def test_shared_location_and_error_code(self):
        a, b, c = self.run_rows([self.row("1", "頂樓很吵"), self.row("2", "e_03", 1),
                                  self.row("3", "空压机 E99 很吵", 2)])
        self.assertIsNone(a.device_id)
        self.assertEqual((b.device_id, b.category), ("D02", "temperature"))
        self.assertEqual((c.device_id, c.category), ("D01", "noise"))
        self.assertEqual(error_codes("E-03 e03 E 03 E_03 E-3 E999"), ["E-03"])

    def test_other_null_and_independent_device_category(self):
        a, b, c, d = self.run_rows([self.row("1", "有狀況"), self.row("2", "貨梯把手斷掉", 1),
                                     self.row("3", "很吵", 2), self.row("4", "空壓機異常", 3)])
        self.assertEqual((a.device_id, a.category), (None, None))
        self.assertEqual(b.category, "other")
        self.assertEqual((c.device_id, c.category), (None, "noise"))
        self.assertEqual((d.device_id, d.category), ("D01", None))
        self.assertTrue(all(p.duplicate_of is None for p in (a, c, d)))

    def test_labels_cannot_change_predictions(self):
        plain = [self.row("1", "空壓機很吵"), self.row("2", "那台一開就跳", 1)]
        poison = [{**r, "device_id": "D12", "category": "temperature", "urgency": "low",
                   "duplicate_of": "FUTURE", "status": "closed"} for r in plain]
        self.assertEqual(self.run_rows(plain), self.run_rows(poison))

    def test_urgency_order_literal_and_conservative_tolerance(self):
        cases = [("好像有火花", False, "high"), ("沒有漏電", False, "high"),
                 ("停機？", True, "high"), ("停機？", False, "normal"),
                 ("跳機", True, "normal"), ("asap help error GG", True, "normal"),
                 ("又來来了", False, "high"), ("??來了", False, "high"),
                 ("已經報修", False, "high"), ("可能是", False, "low"),
                 ("應該", False, "normal"), ("E-07", True, "normal")]
        for text, critical, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(urgency(urgency_text(text), critical), expected)

    def test_original_text_and_signature(self):
        text = "空压机□很吵\n--\n從我的 iPhone 傳送"
        result = self.run_rows([self.row("1", text)])[0]
        self.assertIn("空压机□很吵", result.suggestion)
        self.assertLessEqual(len(result.suggestion), 40)
        self.assertEqual(normalize("全形Ｅ－０３"), "全形e-03")

    def test_invalid_inputs_and_empty_file(self):
        for rows in ([self.row("1", "a"), self.row("1", "b")],
                     [self.row("1", "a", created_at="yesterday")],
                     [self.row("1", "a", created_at="2026-01-01T00:00:00")],
                     [self.row("1", None)], [{"ticket_id": "1"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.run_rows(rows)
        self.assertEqual(self.run_rows([]), [])

    def test_cli_from_other_directory_and_failure_preserves_output(self):
        input_path, output, metrics = [self.path / n for n in ("input.jsonl", "output.jsonl", "metrics.json")]
        input_path.write_text(json.dumps(self.row("1", "空壓機很吵")), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "predict.py"), str(input_path), str(output), "--method", "rules",
               "--metrics", str(metrics)]
        result = subprocess.run(cmd, cwd=self.path, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(json.loads(output.read_text())["category"], "noise")
        self.assertEqual(json.loads(metrics.read_text())["latency"]["count"], 1)
        before = output.read_bytes()
        input_path.write_text("{bad", encoding="utf-8")
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(output.read_bytes(), before)


class MethodSelectionTest(unittest.TestCase):
    """Two-argument default, explicit methods and the 2 GB mode."""

    def choose(self, *argv, installed=True):
        import argparse
        import importlib.util
        from unittest import mock
        spec = importlib.util.spec_from_file_location("solution_predict_cli", SOLUTION / "predict.py")
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = argparse.Namespace(method=None, model_path=None, mem_limit=None)
        for key, value in zip(argv[::2], argv[1::2]):
            setattr(args, key, value)
        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(cli, "apply_memory_limit") as limit:
            model = Path(folder) / "finetuned"
            if installed:
                model.mkdir()
                (model / "model.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(cli, "FINETUNED", model):
                method, path = cli.choose_method(args)
            return method, path == model, limit.called

    def test_default_prefers_installed_finetuned_model(self):
        self.assertEqual(self.choose(), ("bge-finetuned", True, False))

    def test_default_falls_back_to_rules_without_model(self):
        method, _, _ = self.choose(installed=False)
        self.assertEqual(method, "rules")

    def test_explicit_method_and_mem_limit(self):
        self.assertEqual(self.choose("method", "rules")[0], "rules")
        self.assertEqual(self.choose("method", "bge-finetuned")[:2], ("bge-finetuned", True))
        # 2 GB mode keeps the default model choice and only adds the memory cap.
        self.assertEqual(self.choose("mem_limit", "2g"), ("bge-finetuned", True, True))
        self.assertEqual(self.choose("mem_limit", "2g", installed=False)[::2], ("rules", True))
        self.assertEqual(self.choose("mem_limit", "2g", "method", "rules"), ("rules", False, True))


if __name__ == "__main__":
    unittest.main()
