"""History-only encoder/head comparison; validation selects epochs, never fits weights."""
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import save_file

from triage.devices import DeviceResolver
from triage.embedding import BgeClassifier, BgeEncoder
from triage.io import ROOT, atomic_write, file_sha256, load_json
from triage.linear import save_model
from triage.neural import CLASSES, HeadAdapter, NeuralHeads, NeuralBgeClassifier
from triage.pipeline import predict
from triage.text import classification_text
from training.data import history_split, load_history
from training.evaluation import PROVENANCE, category_metrics
from training.linear import fit_heads

SOLUTION = Path(__file__).resolve().parents[1]


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def class_weights(targets, size):
    counts = torch.bincount(targets, minlength=size).float()
    if bool((counts == 0).any()):
        raise ValueError("訓練切分須涵蓋五類與 null；請增加 history 資料")
    return targets.numel() / (size * counts)


def loss_for_batch(category, information, labels, information_labels, category_weights, information_weights, smoothing):
    known = labels >= 0
    # A batch consisting solely of null still trains the information head and encoder.
    category_loss = (torch.nn.functional.cross_entropy(category[known], labels[known],
                     weight=category_weights, label_smoothing=smoothing) if bool(known.any()) else category.sum() * 0)
    information_loss = torch.nn.functional.cross_entropy(information, information_labels,
                                                        weight=information_weights)
    return category_loss + .5 * information_loss


def embeddings_with_grad(encoder, texts, length):
    inputs = encoder.tokenizer(texts, padding=True, truncation=True, max_length=length, return_tensors="pt")
    inputs = {key: value.to(next(encoder.model.parameters()).device) for key, value in inputs.items()}
    vectors = encoder.model(**inputs).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(vectors, p=2, dim=1)


def head_classifier(heads, encoder, codes, batch_size):
    heads.eval()
    return BgeClassifier({"category_model": HeadAdapter(heads.category, CLASSES),
                          "information_model": HeadAdapter(heads.information, ("unknown", "known")),
                          "mask_device_names": False}, codes, encoder, batch_size)


def evaluate_heads(heads, vectors, labels, texts, codes):
    heads.eval()
    with torch.inference_mode():
        cat, info = heads(torch.as_tensor(vectors.copy(), device=next(heads.parameters()).device))
        values = [None if i == 0 else CLASSES[c] for c, i in zip(cat.argmax(-1).tolist(), info.argmax(-1).tolist())]
    # Same public-code override as production; raw head scores are saved separately.
    from triage.linear import LinearTextClassifier
    classifier = LinearTextClassifier({"vectorizer": None, "category_model": None,
                                      "information_model": None, "mask_device_names": False}, codes)
    adjusted = [(match.category if (match := classifier.code_match(t)) else p) for t, p in zip(texts, values)]
    return category_metrics(labels, adjusted), category_metrics(labels, values)


def save_checkpoint(directory, encoder, heads, settings, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    encoder_dir = directory / "encoder"
    encoder.model.save_pretrained(encoder_dir, safe_serialization=True)
    encoder.tokenizer.save_pretrained(encoder_dir)
    files = {p.name: file_sha256(p) for p in encoder_dir.iterdir() if p.is_file() and p.name != "encoder_manifest.json"}
    write_json(encoder_dir / "encoder_manifest.json", {**encoder.manifest, "files_sha256": files,
               "bytes": sum((encoder_dir / name).stat().st_size for name in files),
               "weights_origin": metadata["variant"], "training_run": metadata["run_id"]})
    save_file({k: v.detach().cpu().contiguous() for k, v in heads.state_dict().items()}, str(directory / "heads.safetensors"))
    write_json(directory / "model.json", {**metadata, "format": "bge-neural-v1", "classes": list(CLASSES),
               "settings": asdict(settings), "heads_sha256": file_sha256(directory / "heads.safetensors")})


def train_neural(base, texts, labels, train, validation, cached, settings, codes, output, variant, metadata, notify, device="cpu"):
    seed_all(settings.seed)
    encoder = BgeEncoder(base)
    encoder.model.to(device)
    joint = variant == "finetuned"
    encoder.model.requires_grad_(joint)
    heads = NeuralHeads(encoder.dimension, settings.dropout).to(device)
    targets = torch.tensor([CLASSES.index(c) if c is not None else -1 for c in labels], device=device)
    info_targets = (targets >= 0).long()
    train_targets = targets[train]
    category_weights = class_weights(train_targets[train_targets >= 0], len(CLASSES))
    information_weights = class_weights(info_targets[train], 2)
    parameters = [{"params": list(heads.parameters()), "lr": settings.head_lr}]
    if joint:
        decay, no_decay = [], []
        for name, parameter in encoder.model.named_parameters():
            (no_decay if name.endswith("bias") or "LayerNorm.weight" in name else decay).append(parameter)
        parameters.extend([{"params": decay, "lr": settings.encoder_lr},
                           {"params": no_decay, "lr": settings.encoder_lr, "weight_decay": 0.0}])
    optimizer = torch.optim.AdamW(parameters, weight_decay=settings.weight_decay)
    total_steps = math.ceil(len(train) / settings.batch_size) * settings.epochs
    warmup = max(1, math.ceil(total_steps * .1))
    def scale(step):
        return (step + 1) / warmup if step < warmup else max(0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    before = hashlib.sha256(encoder.model.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()).hexdigest()
    best_score, best_epoch, stale, best_metrics, curve = -1.0, 0, 0, None, []
    best_path = output / variant
    train_labels, val_labels = [labels[i] for i in train], [labels[i] for i in validation]
    for epoch in range(1, settings.epochs + 1):
        began = perf_counter()
        encoder.model.train(joint)
        heads.train()
        order = np.random.default_rng(settings.seed + epoch).permutation(train).tolist()
        total_loss = 0.0
        for start in range(0, len(order), settings.batch_size):
            indices = order[start:start + settings.batch_size]
            optimizer.zero_grad(set_to_none=True)
            vectors = (embeddings_with_grad(encoder, [texts[i] for i in indices], settings.max_length)
                       if joint else torch.as_tensor(cached[indices].copy(), device=device))
            cat, info = heads(vectors)
            loss = loss_for_batch(cat, info, targets[indices], info_targets[indices], category_weights,
                                  information_weights, settings.label_smoothing)
            if not bool(torch.isfinite(loss)):
                raise ValueError("訓練 loss 非有限數，停止並保留失敗紀錄")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(indices)
            notify("batch", variant=variant, epoch=epoch, epochs=settings.epochs,
                   completed=min(start + settings.batch_size, len(order)), total=len(order),
                   message=f"{variant} · epoch {epoch}/{settings.epochs} · batch {start // settings.batch_size + 1}/{math.ceil(len(order) / settings.batch_size)}")
        notify("stage", variant=variant, epoch=epoch,
               message=f"{variant} · epoch {epoch} 訓練完成；計算訓練／驗證指標並保存最佳模型")
        encoder.model.eval()
        train_vectors = encoder.encode([texts[i] for i in train], settings.batch_size) if joint else cached[train]
        val_vectors = encoder.encode([texts[i] for i in validation], settings.batch_size) if joint else cached[validation]
        train_metrics, _ = evaluate_heads(heads, train_vectors, train_labels, [texts[i] for i in train], codes)
        metrics, raw = evaluate_heads(heads, val_vectors, val_labels, [texts[i] for i in validation], codes)
        record = {"variant": variant, "epoch": epoch, "train_loss": total_loss / len(train),
                  "train": train_metrics, "validation": metrics, "raw_head_validation": raw,
                  "seconds": perf_counter() - began, "encoder_lr_next": optimizer.param_groups[-1]["lr"] if joint else 0.0}
        curve.append(record)
        score = metrics["six_class_macro_f1"]
        if score > best_score + 1e-4:
            best_score, best_epoch, best_metrics, stale = score, epoch, metrics, 0
            save_checkpoint(best_path, encoder, heads, settings, {**metadata, "variant": variant,
                            "best_epoch": epoch, "validation": metrics, "train_count": len(train),
                            "refit_full_history": False})
        else:
            stale += 1
        write_json(output / f"{variant}-epochs.json", curve)
        notify("epoch", **record, best_epoch=best_epoch)
        if stale >= settings.patience:
            notify("early_stop", variant=variant, epoch=epoch, best_epoch=best_epoch)
            break
    after = hashlib.sha256(encoder.model.embeddings.word_embeddings.weight.detach().cpu().numpy().tobytes()).hexdigest()
    if joint and before == after:
        raise ValueError("encoder 權重沒有更新，不能宣稱完成聯合微調")
    if not joint and before != after:
        raise ValueError("凍結控制組的 encoder 權重意外改變")
    del encoder, heads, optimizer
    gc.collect()
    return {"variant": variant, "model_path": str(best_path), "best_epoch": best_epoch,
            "validation": best_metrics, "epochs_completed": len(curve), "embedding_weights_changed": before != after,
            "last_epoch_embedding_sha256_before": before, "last_epoch_embedding_sha256_after": after}


def run_comparison(history, output, settings, notify, device="cpu"):
    """Do not refit on validation: publish best train-only checkpoints for review."""
    from helper_tool.evaluate import analyze, write_reports
    began = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    tickets, gold = load_history(history)
    codes = load_json(ROOT / "data/error_codes.json")
    resolver = DeviceResolver(load_json(ROOT / "data/devices.json"), codes,
                              load_json(SOLUTION / "config/device_aliases.json"))
    train, validation, purged = history_split(tickets, resolver)
    labels = [gold[t.ticket_id]["category"] for t in tickets]
    counts = Counter(labels[i] for i in train)
    if set(counts) != {*CLASSES, None} or min(counts.values()) < 2:
        raise ValueError("時間／模板分割後，訓練段每個五類與 null 都至少需要 2 筆")
    if not any(labels[i] is None for i in validation) or not any(labels[i] is not None for i in validation):
        raise ValueError("驗證段須同時有已知類別與 null，才能比較資訊不足判斷")
    source_files = [*sorted((SOLUTION / "training").glob('*.py')), *sorted((SOLUTION / "triage").glob('*.py')),
                    SOLUTION / "train_finetune.py", SOLUTION / "requirements-embedding.txt",
                    ROOT / "data/devices.json", ROOT / "data/error_codes.json", SOLUTION / "config/device_aliases.json"]
    metadata = {"run_id": output.parent.name, "history_sha256": file_sha256(history), "dev_read": False,
                "training_device": device,
                "provenance": PROVENANCE, "settings": asdict(settings), "mask_device_names": False,
                "source_sha256": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_files},
                "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "numpy", "scikit-learn")},
                "selection_metric": "validation six-class macro-F1; min_delta=0.0001; ties retain earlier epoch",
                "validation_scope": "history selection validation, not an independent test; no full-history refit",
                "split": {"train_ids": [tickets[i].ticket_id for i in train],
                          "validation_ids": [tickets[i].ticket_id for i in validation],
                          "purged_train_ids": [tickets[i].ticket_id for i in purged]}}
    write_json(output / "comparison.json", {**metadata, "results": []})
    base = SOLUTION / "models/bge-small-zh-v1.5"
    notify("stage", message="載入固定 BGE 並建立凍結 LR／SVC 對照", train_count=len(train), validation_count=len(validation), purged_count=len(purged))
    encoder = BgeEncoder(base, device=device)
    metadata["base_encoder"] = encoder.manifest
    texts = [classification_text(t.raw_text) for t in tickets]
    cached = encoder.encode(texts, settings.batch_size)
    frozen_dir = output / "frozen"
    frozen_dir.mkdir()
    shutil.copytree(base, frozen_dir / base.name, ignore=shutil.ignore_patterns('.cache'))
    results = []
    gold_path = output / "validation-gold.jsonl"
    atomic_write(gold_path, "".join(json.dumps(gold[tickets[i].ticket_id], ensure_ascii=False) + '\n' for i in validation))
    def evaluate_pipeline(classifier, name):
        predictions, _ = predict(tickets, resolver, classifier)
        by_id = {p.ticket_id: p for p in predictions}
        pred_path = output / f"{name}-validation.jsonl"
        atomic_write(pred_path, "".join(json.dumps(by_id[tickets[i].ticket_id].to_dict(), ensure_ascii=False) + '\n' for i in validation))
        summary, errors, official = analyze(pred_path, gold_path)
        write_reports(output / f"{name}-evaluation", summary, errors, official)
        return {"pipeline_metrics": summary["metrics"], "pipeline_total": summary["total"]}
    for method in ("bge-lr", "bge-svc"):
        heads = fit_heads(cached[train], [labels[i] for i in train], method)
        bundle = {**heads, "mask_device_names": False, "encoder_folder": base.name, "batch_size": settings.batch_size}
        path = frozen_dir / f"{method}.joblib"
        classifier = BgeClassifier(bundle, codes, encoder, settings.batch_size)
        standalone = classifier.classify_many([texts[i] for i in validation]).predictions
        metrics = category_metrics([labels[i] for i in validation], [p.category for p in standalone])
        save_model(path, bundle, {**metadata, "method": method, "encoder": encoder.manifest, "history_validation": metrics,
                                  "refit_full_history": False})
        result = {"variant": method, "model_path": str(path), "validation": metrics, **evaluate_pipeline(classifier, method)}
        results.append(result)
        write_json(output / "comparison.json", {**metadata, "results": results})
        notify("baseline", **result)
    del classifier, encoder
    gc.collect()
    for variant in ("frozen-neural", "finetuned"):
        notify("stage", message="訓練凍結 encoder 的神經頭控制組" if variant == "frozen-neural" else "聯合更新 BGE encoder 與分類頭", variant=variant)
        result = train_neural(base, texts, labels, train, validation, cached, settings, codes, output, variant, metadata, notify, device=device)
        notify("stage", variant=variant, message=f"{variant} · 評估最佳 checkpoint 的完整四欄管線")
        classifier, _ = NeuralBgeClassifier.load(Path(result["model_path"]), codes)
        result.update(evaluate_pipeline(classifier, variant))
        results.append(result)
        del classifier
        gc.collect()
        write_json(output / "comparison.json", {**metadata, "results": results})
    report = {**metadata, "results": results, "total_seconds": perf_counter() - began,
              "note": "The same validation selects epochs; use new held-out data after fixing the recipe. Improvement is not guaranteed."}
    write_json(output / "comparison.json", report)
    return report
