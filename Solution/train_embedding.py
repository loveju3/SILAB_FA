#!/usr/bin/env python3
"""C: freeze BGE; compare the same four history-only head/masking recipes as B."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import uuid

import numpy as np

from triage.devices import DeviceResolver
from triage.embedding import BgeClassifier, BgeEncoder
from triage.io import ROOT, atomic_write, load_json
from triage.linear import save_model
from triage.pipeline import predict
from triage.text import classification_text
from training.data import history_split, load_history
from training.evaluation import CATEGORIES, PROVENANCE, category_metrics, macro_f1
from training.linear import fit_heads
from training.settings import LinearSettings

sys.path.insert(0, str(ROOT))
from helper_tool.evaluate import analyze

SOLUTION = Path(__file__).resolve().parent


class CachedEmbeddings:
    """Only for head selection; runtime never carries this text lookup cache."""
    def __init__(self, texts, vectors):
        self.vectors = dict(zip(texts, vectors))

    def transform(self, texts):
        return np.stack([self.vectors[text] for text in texts])


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定 BGE encoder，僅用 history 訓練與比較分類頭")
    parser.add_argument("--model-dir", type=Path, default=SOLUTION / "models")
    parser.add_argument("--report-dir", type=Path, default=SOLUTION / "reports/training-c")
    parser.add_argument("--history", type=Path, default=ROOT / "data/history.jsonl", help="開發用 history.jsonl；不可使用 dev")
    parser.add_argument("--settings-json", default="{}",
                        help="LR／SVC 訓練參數 JSON（見 training/settings.py 的 LinearSettings）；未填使用預設配方")
    args = parser.parse_args(argv)
    try:
        settings = LinearSettings.from_dict(json.loads(args.settings_json), "bge-frozen")
        history = args.history
        tickets, gold = load_history(history)
        codes = load_json(ROOT / "data/error_codes.json")
        resolver = DeviceResolver(load_json(ROOT / "data/devices.json"), codes,
                                  load_json(SOLUTION / "config/device_aliases.json"))
        train, validation, purged = history_split(tickets, resolver)
        labels = [gold[t.ticket_id]["category"] for t in tickets]
        if {labels[i] for i in train} != {*CATEGORIES, None}:
            raise ValueError("訓練分割未涵蓋五類與 null")
        directory = args.model_dir / "bge-small-zh-v1.5"
        if not (directory / "encoder_manifest.json").exists():
            raise ValueError("請先執行 .venv/bin/python Solution/scripts/download_encoder.py")
        began = perf_counter()
        encoder = BgeEncoder(directory)
        load_seconds = perf_counter() - began
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        report_dir = args.report_dir / run_id
        report_dir.mkdir(parents=True)
        files = [Path(__file__), *sorted((SOLUTION / "training").glob("*.py")),
                 *sorted((SOLUTION / "triage").glob("*.py")), SOLUTION / "config/device_aliases.json",
                 ROOT / "data/devices.json", ROOT / "data/error_codes.json",
                 SOLUTION / "requirements-ml.txt", SOLUTION / "requirements-embedding.txt"]
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        manifest = {"run_id": run_id, "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
                    "source_sha256": hashes, "dev_read": False, "provenance": PROVENANCE,
                    "encoder": encoder.manifest, "encoder_load_seconds": load_seconds,
                    "recipe": {"pooling": "CLS", "normalize": "L2", "max_length": 512,
                               "batch_size": 16, "cpu_threads": 2, "retrieval_instruction": False,
                               "frozen_encoder": True, **settings.to_dict("bge-frozen")},
                    "versions": {p: importlib.metadata.version(p) for p in
                                 ("torch", "transformers", "numpy", "scikit-learn", "tokenizers")},
                    "split": {"train_ids": [tickets[i].ticket_id for i in train],
                              "validation_ids": [tickets[i].ticket_id for i in validation],
                              "purged_train_ids": [tickets[i].ticket_id for i in purged]},
                    "selection_metric": "six-class macro-F1; ties prefer masked; no dev selection",
                    "candidates": []}
        cached = {}
        with tempfile.TemporaryDirectory(prefix="silab-c-history-") as temporary:
            folder = Path(temporary)
            gold_path = folder / "history_validation.jsonl"
            atomic_write(gold_path, "".join(json.dumps(gold[tickets[i].ticket_id], ensure_ascii=False) + "\n" for i in validation))
            for masked in (False, True):
                texts = [classification_text(t.raw_text) for t in tickets]
                if masked:
                    texts = [resolver.mask_names(text) for text in texts]
                began = perf_counter()
                vectors = encoder.encode(texts, batch_size=16)
                encoding_seconds = perf_counter() - began
                cached[masked] = vectors
                np.save(report_dir / f"history-{'masked' if masked else 'raw'}-embeddings.npy", vectors)
                for method in ("bge-lr", "bge-svc"):
                    name = f"{method}-{'masked' if masked else 'raw'}"
                    began = perf_counter()
                    heads = fit_heads(vectors[train], [labels[i] for i in train], method, settings)
                    fit_seconds = perf_counter() - began
                    classifier = BgeClassifier({**heads, "mask_device_names": masked}, codes,
                                               CachedEmbeddings(texts, vectors))
                    standalone = classifier.classify_many([texts[i] for i in validation]).predictions
                    raw = heads["category_model"].predict(vectors[validation])
                    _, head_f1 = macro_f1([(labels[i], str(p)) for i, p in zip(validation, raw) if labels[i] is not None], CATEGORIES)
                    predictions, _ = predict(tickets, resolver, classifier)
                    by_id = {p.ticket_id: p for p in predictions}
                    pred_path = folder / f"{name}.jsonl"
                    atomic_write(pred_path, "".join(json.dumps(by_id[tickets[i].ticket_id].to_dict(), ensure_ascii=False) + "\n" for i in validation))
                    summary, errors, _ = analyze(pred_path, gold_path)
                    result = {"name": name, "method": method, "masked": masked,
                              "encoding_seconds": encoding_seconds, "fit_seconds": fit_seconds,
                              "head_only_category_macro_f1": head_f1,
                              **category_metrics([labels[i] for i in validation], [p.category for p in standalone]),
                              "pipeline_metrics": summary["metrics"], "pipeline_total": summary["total"]}
                    manifest["candidates"].append(result)
                    atomic_write(report_dir / f"{name}-errors.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors))
        winners = {method: max((r for r in manifest["candidates"] if r["method"] == method),
                              key=lambda r: (r["six_class_macro_f1"], r["masked"]))["name"]
                   for method in ("bge-lr", "bge-svc")}
        manifest["selected"] = winners
        atomic_write(report_dir / "selection.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifest["artifacts"] = []
        for result in manifest["candidates"]:
            heads = fit_heads(cached[result["masked"]], labels, result["method"], settings)
            bundle = {**heads, "mask_device_names": result["masked"], "encoder_folder": directory.name, "batch_size": 16}
            metadata = {"method": result["method"], "masked": result["masked"], "history_count": len(tickets),
                        "labelled_count": sum(c is not None for c in labels), "null_count": labels.count(None),
                        "encoder": encoder.manifest, "versions": manifest["versions"], "recipe": manifest["recipe"],
                        "history_sha256": manifest["history_sha256"], "source_sha256": hashes,
                        "provenance": PROVENANCE, "history_validation": result, "training_run": run_id}
            saved = save_model(args.model_dir / f"{result['name']}.joblib", bundle, metadata)
            manifest["artifacts"].append({"name": result["name"], "bytes": saved["artifact_bytes"], "sha256": saved["artifact_sha256"]})
            if winners[result["method"]] == result["name"]:
                save_model(args.model_dir / f"{result['method']}.joblib", bundle, metadata)
        atomic_write(report_dir / "training.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"report_dir": str(report_dir), "train": len(train), "validation": len(validation),
                          "purged": len(purged), "selected": winners, "candidates": manifest["candidates"]}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"訓練失敗：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
