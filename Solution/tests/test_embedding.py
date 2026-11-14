"""Batch contracts and optional real offline BGE checks, using synthetic text."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOLUTION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOLUTION))

from triage.classification import BatchClassifications, CategoryMatch
from triage.contracts import Ticket
from triage.devices import DeviceResolver
from triage.embedding import BgeClassifier, BgeEncoder
from triage.io import ROOT, load_json
from triage.pipeline import predict


class BatchPipelineTest(unittest.TestCase):
    def test_batch_classification_preserves_time_order_and_reports_full_service_time(self):
        class Batch:
            mask_device_names = False
            def classify_many(self, texts):
                self.texts = texts
                return BatchClassifications([CategoryMatch("noise", "合成") for _ in texts], [.25] * len(texts))
            def classify(self, text):
                raise AssertionError("不得對已批次分類的文字重新逐筆推論")
        codes = load_json(ROOT / "data/error_codes.json")
        resolver = DeviceResolver(load_json(ROOT / "data/devices.json"), codes, {})
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [Ticket("later", now + timedelta(minutes=1), "甲", "空壓機很吵"),
                Ticket("earlier", now, "甲", "空壓機異音")]
        classifier = Batch()
        results, timings = predict(rows, resolver, classifier)
        self.assertEqual(results[0].duplicate_of, "earlier")
        self.assertEqual(classifier.texts[0], "空壓機異音")
        self.assertTrue(all(t["seconds"] >= .25 and t["batch_classification_seconds"] == .25 for t in timings))
        self.assertEqual(predict([], resolver, classifier), ([], []))

    def test_invalid_encoder_hash_is_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "config.json").write_text("{}")
            (folder / "encoder_manifest.json").write_text(json.dumps({"files_sha256": {"config.json": "invalid"}}))
            with self.assertRaisesRegex(ValueError, "雜湊"):
                BgeEncoder(folder)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "需 numpy")
    def test_batch_head_decisions_and_code_override_share_same_logic(self):
        import numpy as np
        class Encoder:
            def transform(self, texts):
                return np.array([[-1.] if text == "資訊不足" else [1.] for text in texts])
        class Information:
            def predict(self, values):
                return ["unknown" if values[0, 0] < 0 else "known"]
        class Category:
            def predict(self, values):
                return ["noise"]
        classifier = BgeClassifier({"category_model": Category(), "information_model": Information(),
                                    "mask_device_names": False}, {"E-01": {"category": "temperature"}}, Encoder(), batch_size=2)
        texts = ["聲音", "資訊不足", "E-01", "另一聲音"]
        batch = classifier.classify_many(texts)
        self.assertEqual([p.category for p in batch.predictions], ["noise", None, "temperature", "noise"])
        self.assertEqual(batch.predictions, [classifier.classify(text) for text in texts])
        self.assertEqual(batch.batch_seconds[0], batch.batch_seconds[1])
        self.assertEqual(batch.batch_seconds[2], batch.batch_seconds[3])


READY = (importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
         and (SOLUTION / "models/bge-small-zh-v1.5/encoder_manifest.json").is_file())


@unittest.skipUnless(READY, "實際 encoder 測試需本機 BGE 資產與 requirements-embedding.txt")
class RealEncoderTest(unittest.TestCase):
    def test_local_encoder_is_frozen_cpu_normalized_and_model_roundtrips(self):
        import numpy as np
        from training.linear import fit_heads
        from triage.linear import save_model
        directory = SOLUTION / "models/bge-small-zh-v1.5"
        texts = ["聲音很吵", "過熱燙手", "供電跳電", "連線中斷", "把手斷掉", "需要查看"]
        labels = ["noise", "temperature", "power", "software", "other", None]
        with patch("requests.sessions.Session.request", side_effect=AssertionError("離線測試禁止網路")):
            encoder = BgeEncoder(directory)
            vectors = encoder.encode(texts, batch_size=3)
            self.assertEqual(vectors.shape, (6, 512))
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
            self.assertFalse(encoder.model.training)
            self.assertTrue(all(not p.requires_grad and p.device.type == "cpu" for p in encoder.model.parameters()))
            heads = fit_heads(vectors, labels, "bge-lr")
            bundle = {**heads, "mask_device_names": False, "encoder_folder": directory.name, "batch_size": 3}
            before = BgeClassifier(bundle, {}, encoder, batch_size=3)
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                import os
                if os.name == 'nt':
                    import shutil
                    shutil.copytree(directory, folder / directory.name, ignore=shutil.ignore_patterns('.cache'))
                else:
                    (folder / directory.name).symlink_to(directory, target_is_directory=True)
                path = folder / "bge-lr.joblib"
                save_model(path, bundle, {"method": "bge-lr", "encoder": encoder.manifest})
                loaded, _ = BgeClassifier.load(path, {}, "bge-lr")
                self.assertEqual(before.classify_many(texts).predictions, loaded.classify_many(texts).predictions)


if __name__ == "__main__":
    unittest.main()
