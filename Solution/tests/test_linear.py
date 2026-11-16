"""Learned-classifier contracts using synthetic examples only, never dev."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

from triage.contracts import Ticket
from triage.devices import DeviceResolver
from triage.io import ROOT, load_json
from training.data import history_split, load_history


class HistorySplitTest(unittest.TestCase):
    def setUp(self):
        self.resolver = DeviceResolver(load_json(ROOT / "data/devices.json"),
                                       load_json(ROOT / "data/error_codes.json"),
                                       load_json(SOLUTION / "config/device_aliases.json"))

    def test_time_order_and_device_masked_templates_do_not_cross_split(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        texts = ["空壓機軸承發出振動", "冰水主機冷卻水很熱", "貨梯把手斷掉",
                 "UPS軸承發出振動", "監控主機連線畫面斷線"]
        tickets = [Ticket(str(i), start + timedelta(minutes=i), "甲", t) for i, t in enumerate(texts)]
        train, validation, purged = history_split(tickets, self.resolver, fraction=.6)
        self.assertEqual(validation, [3, 4])
        self.assertIn(0, purged)
        self.assertNotIn(0, train)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertTrue(all(tickets[i].created_at < tickets[validation[0]].created_at for i in train))

    def test_training_rejects_dev_path_without_opening_it(self):
        with self.assertRaisesRegex(ValueError, "history"):
            load_history(Path("/nonexistent/dev.jsonl"))


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "ML 測試需 requirements-ml.txt")
class LinearTest(unittest.TestCase):
    def setUp(self):
        from training.linear import fit_bundle
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.codes = load_json(ROOT / "data/error_codes.json")
        self.texts = ["軸承異音很吵", "馬達震動噪音", "外殼燙手過熱", "出風口熱風",
                      "啟動電源跳電", "供電電壓不足", "通訊連線斷線", "系統畫面當機",
                      "把手斷掉", "水管滴水", "有狀況", "需要查看"]
        self.labels = ["noise", "noise", "temperature", "temperature", "power", "power",
                       "software", "software", "other", "other", None, None]
        self.fit = fit_bundle

    def test_lr_and_svc_fit_only_training_vocabulary_and_keep_null_separate(self):
        for method in ("tfidf-lr", "tfidf-svc"):
            with self.subTest(method=method):
                bundle = self.fit(self.texts, self.labels, method, True)
                self.assertNotIn("證專", bundle["vectorizer"].vocabulary_)
                bundle["vectorizer"].transform(["驗證專用文字"])
                self.assertNotIn("證專", bundle["vectorizer"].vocabulary_)
                self.assertEqual(set(bundle["category_model"].classes_), set(self.labels) - {None})
                self.assertEqual(set(bundle["information_model"].classes_), {"known", "unknown"})

    def test_settings_reach_vectorizer_and_heads(self):
        from training.settings import LinearSettings
        default = self.fit(self.texts, self.labels, "tfidf-lr", False)
        self.assertEqual((default["vectorizer"].ngram_range, default["category_model"].C,
                          default["category_model"].class_weight), ((2, 4), 1.0, "balanced"))
        settings = LinearSettings(c=.25, class_weight="none", information_c=3.0, svc_max_iter=800,
                                  analyzer="char_wb", ngram_min=1, ngram_max=2, max_features=100, sublinear_tf=False)
        for method in ("tfidf-lr", "tfidf-svc"):
            with self.subTest(method=method):
                bundle = self.fit(self.texts, self.labels, method, False, settings)
                vectorizer, head = bundle["vectorizer"], bundle["category_model"]
                self.assertEqual((vectorizer.analyzer, vectorizer.ngram_range, vectorizer.sublinear_tf),
                                 ("char_wb", (1, 2), False))
                self.assertLessEqual(len(vectorizer.vocabulary_), 100)
                self.assertEqual((head.C, head.class_weight), (.25, None))
                self.assertEqual(bundle["information_model"].C, 3.0)
                if method == "tfidf-svc":
                    self.assertEqual(head.max_iter, 800)

    def test_model_roundtrip_code_override_and_integrity_check(self):
        from triage.linear import LinearTextClassifier, save_model
        for method in ("tfidf-lr", "tfidf-svc"):
            bundle = self.fit(self.texts, self.labels, method, True)
            before = LinearTextClassifier(bundle, self.codes)
            path = self.folder / f"{method}.joblib"
            save_model(path, bundle, {"method": method})
            after, _ = LinearTextClassifier.load(path, self.codes, method)
            for text in self.texts:
                self.assertEqual(before.classify(text), after.classify(text))
            self.assertIsNone(after.classify("有狀況").category)
            self.assertEqual(after.classify("e_03").category, "temperature")
            with self.assertRaisesRegex(ValueError, "不一致"):
                LinearTextClassifier.load(path, self.codes, "other-method")
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "雜湊"):
                LinearTextClassifier.load(path, self.codes, method)

    def test_cli_ignores_poisoned_labels_and_uses_frozen_model_only(self):
        from triage.linear import save_model
        path = self.folder / "tfidf-lr.joblib"
        save_model(path, self.fit(self.texts, self.labels, "tfidf-lr", True), {"method": "tfidf-lr"})
        original = {"ticket_id": "1", "created_at": "2026-01-01T00:00:00Z", "reporter": "甲",
                    "raw_text": "空壓機軸承異音很吵"}
        results = []
        for row in (original, {**original, "device_id": "D12", "category": "other", "urgency": "low", "duplicate_of": "future"}):
            source, output = self.folder / "input.jsonl", self.folder / "output.jsonl"
            source.write_text(json.dumps(row), encoding="utf-8")
            process = subprocess.run([sys.executable, str(ROOT / "predict.py"), str(source), str(output),
                                      "--method", "tfidf-lr", "--model-path", str(path)],
                                     cwd=self.folder, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stdout, "")
            results.append(output.read_bytes())
        self.assertEqual(results[0], results[1])
        self.assertIs(json.loads(results[0])["llm_used"], False)

    def test_invalid_training_data_is_rejected(self):
        with self.assertRaises(ValueError):
            self.fit(["abc", "def"], ["noise", "power"], "tfidf-lr", True)
        with self.assertRaises(ValueError):
            self.fit(["abc", "def"], [None, "noise"], "tfidf-lr", True)


if __name__ == "__main__":
    unittest.main()
