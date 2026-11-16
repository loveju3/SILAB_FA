#!/usr/bin/env python3
"""Compare four TF-IDF recipes using history only; refit frozen recipes on history."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import uuid

from triage.devices import DeviceResolver
from triage.io import ROOT, atomic_write, load_json
from triage.linear import LinearTextClassifier, save_model
from triage.pipeline import predict
from triage.text import classification_text
from training.data import history_split, load_history
from training.linear import fit_bundle
from training.evaluation import PROVENANCE, category_metrics
from training.settings import LinearSettings

sys.path.insert(0, str(ROOT))
from baseline.evaluate import CATEGORIES, macro_f1
from helper_tool.evaluate import analyze

SOLUTION = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="僅讀 history；LR/SVC 與設備名稱遮罩四方案比較")
    parser.add_argument("--output-dir", type=Path, default=SOLUTION / "models")
    parser.add_argument("--report-dir", type=Path, default=SOLUTION / "reports/training")
    parser.add_argument("--history", type=Path, default=ROOT / "data/history.jsonl", help="開發用 history.jsonl；不可使用 dev")
    parser.add_argument("--settings-json", default="{}",
                        help="LR／SVC 訓練參數 JSON（見 training/settings.py 的 LinearSettings）；未填使用預設配方")
    args = parser.parse_args(argv)
    try:
        settings = LinearSettings.from_dict(json.loads(args.settings_json), "tfidf")
        history = args.history  # load_history rejects dev names and symlinks.
        tickets, gold = load_history(history)
        codes = load_json(ROOT / "data/error_codes.json")
        resolver = DeviceResolver(load_json(ROOT / "data/devices.json"), codes,
                                  load_json(SOLUTION / "config/device_aliases.json"))
        train, validation, purged = history_split(tickets, resolver)
        train_labels = [gold[tickets[i].ticket_id]["category"] for i in train]
        if set(train_labels) != {*CATEGORIES, None}:
            raise ValueError("分割後訓練集未涵蓋五類與 null，無法進行目前的固定比較")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
        report_dir = args.report_dir / run_id
        report_dir.mkdir(parents=True)
        source_files = [Path(__file__), *sorted((SOLUTION / "training").glob("*.py")),
                        *sorted((SOLUTION / "triage").glob("*.py")),
                        SOLUTION / "config/device_aliases.json", ROOT / "data/devices.json",
                        ROOT / "data/error_codes.json", SOLUTION / "requirements-ml.txt"]
        hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
        manifest = {
            "run_id": run_id, "source": "history.jsonl", "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
            "python": sys.version, "platform": platform.platform(), "source_sha256": hashes,
            "provenance": PROVENANCE, "dev_read": False,
            "split": {"method": "first 70% time boundary; remove earlier groups also present in last 30%; text-only SequenceMatcher >= 0.90, transitive groups",
                      "train_ids": [tickets[i].ticket_id for i in train],
                      "validation_ids": [tickets[i].ticket_id for i in validation],
                      "purged_train_ids": [tickets[i].ticket_id for i in purged],
                      "train_categories": dict(Counter(str(c) for c in train_labels))},
            "settings": settings.to_dict("tfidf"),
            "selection_metric": "standalone six-class macro-F1 including unknown; ties prefer masked; no dev selection",
            "candidates": [],
        }
        validation_labels = [gold[tickets[i].ticket_id]["category"] for i in validation]
        with tempfile.TemporaryDirectory(prefix="silab-history-eval-") as temporary:
            folder = Path(temporary)
            gold_path = folder / "history_validation.jsonl"
            gold_path.write_text("".join(json.dumps(gold[tickets[i].ticket_id], ensure_ascii=False) + "\n"
                                         for i in validation), encoding="utf-8")
            for method in ("tfidf-lr", "tfidf-svc"):
                for masked in (False, True):
                    name = f"{method}-{'masked' if masked else 'raw'}"
                    texts = [classification_text(t.raw_text) for t in tickets]
                    if masked:
                        texts = [resolver.mask_names(text) for text in texts]
                    started = perf_counter()
                    bundle = fit_bundle([texts[i] for i in train], train_labels, method, masked, settings)
                    fit_seconds = perf_counter() - started
                    classifier = LinearTextClassifier(bundle, codes)
                    raw_model = [str(c) for c in bundle["category_model"].predict(
                        bundle["vectorizer"].transform([texts[i] for i in validation]))]
                    standalone = [classifier.classify(texts[i]).category for i in validation]
                    head_pairs = [(g, p) for g, p in zip(validation_labels, raw_model) if g is not None]
                    _, head_f1 = macro_f1(head_pairs, CATEGORIES)
                    predictions, _ = predict(tickets, resolver, classifier)
                    by_id = {p.ticket_id: p for p in predictions}
                    pred_path = folder / f"{name}.jsonl"
                    pred_path.write_text("".join(json.dumps(by_id[tickets[i].ticket_id].to_dict(), ensure_ascii=False) + "\n"
                                                for i in validation), encoding="utf-8")
                    summary, errors, _ = analyze(pred_path, gold_path)
                    result = {"name": name, "method": method, "masked": masked,
                              "fit_seconds": fit_seconds, "vocabulary_size": len(bundle["vectorizer"].vocabulary_),
                              "head_only_category_macro_f1": head_f1,
                              **category_metrics(validation_labels, standalone),
                              "pipeline_metrics": summary["metrics"], "pipeline_total": summary["total"]}
                    manifest["candidates"].append(result)
                    atomic_write(report_dir / f"{name}-errors.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in errors))
        # Freeze choices before final refit; no access to dev in this program.
        winners = {}
        for method in ("tfidf-lr", "tfidf-svc"):
            options = [r for r in manifest["candidates"] if r["method"] == method]
            winners[method] = max(options, key=lambda r: (r["six_class_macro_f1"], r["masked"]))["name"]
        manifest["selected"] = winners
        atomic_write(report_dir / "selection.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        manifest["artifacts"] = []
        for result in manifest["candidates"]:
            texts = [classification_text(t.raw_text) for t in tickets]
            if result["masked"]:
                texts = [resolver.mask_names(text) for text in texts]
            labels = [gold[t.ticket_id]["category"] for t in tickets]
            bundle = fit_bundle(texts, labels, result["method"], result["masked"], settings)
            metadata = {"method": result["method"], "masked": result["masked"], "history_sha256": manifest["history_sha256"],
                        "history_count": len(tickets), "labelled_count": sum(c is not None for c in labels),
                        "null_count": labels.count(None), "source_sha256": hashes, "provenance": PROVENANCE,
                        "recipe": {**settings.to_dict("tfidf"),
                                   "information_gate": "separate balanced LogisticRegression, not category confidence"},
                        "history_validation": result, "training_run": run_id}
            saved = save_model(args.output_dir / f"{result['name']}.joblib", bundle, metadata)
            manifest["artifacts"].append({"name": result["name"], "bytes": saved["artifact_bytes"], "sha256": saved["artifact_sha256"]})
            if winners[result["method"]] == result["name"]:
                save_model(args.output_dir / f"{result['method']}.joblib", bundle, metadata)
        atomic_write(report_dir / "training.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"report_dir": str(report_dir), "train": len(train), "validation": len(validation),
                          "purged": len(purged), "selected": winners, "candidates": manifest["candidates"]}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"訓練失敗：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
