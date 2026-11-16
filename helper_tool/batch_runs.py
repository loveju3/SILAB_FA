"""Run every prediction method on one dataset as a background job.

Each method goes through the same ``run_experiment`` as a single UI run, so
outputs, reports and naming are identical and also appear in 實驗比較. Methods
run one after another (never in parallel) to keep timings comparable and to
respect the server-wide operation lock used by the official evaluator.
"""
from datetime import datetime
import threading
from time import perf_counter
import uuid

from pathlib import Path

from .common import ROOT
from .experiments import METHODS, TAIPEI, run_experiment

ORDER = ("baseline", "solution-a", "tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc", "bge-finetuned")
LABELS = {"baseline": "Baseline · 原始規則", "solution-a": "方案 A · 完整規則",
          "tfidf-lr": "方案 B · TF-IDF＋LR", "tfidf-svc": "方案 B · TF-IDF＋LinearSVC",
          "bge-lr": "方案 C · BGE＋LR", "bge-svc": "方案 C · BGE＋LinearSVC",
          "bge-finetuned": "方案 C · BGE 聯合微調"}
KEEP_JOBS = 3
MODEL_METHODS = ("tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc", "bge-finetuned")
DEFAULT_PATHS = {"tfidf-lr": "Solution/models/tfidf-lr.joblib", "tfidf-svc": "Solution/models/tfidf-svc.joblib",
                 "bge-lr": "Solution/models/bge-lr.joblib", "bge-svc": "Solution/models/bge-svc.joblib",
                 "bge-finetuned": "transfer/finetuned"}
KIND_NAMES = {"tfidf": "TF-IDF 訓練", "bge-frozen": "凍結 BGE 訓練", "bge-finetune": "BGE 聯合微調訓練"}
LINEAR_DEFAULTS = {"c": 1, "class_weight": "balanced", "information_c": 1, "lr_max_iter": 2000, "svc_max_iter": 5000,
                   "analyzer": "char", "ngram_min": 2, "ngram_max": 4, "min_df": 1, "max_features": 0,
                   "sublinear_tf": True}


def _settings_text(job):
    config = job["config"]
    if config["kind"] == "bge-finetune":
        settings = config.get("settings") or {}
        return f"epochs={settings.get('epochs')}、encoder_lr={settings.get('encoder_lr')}"
    linear = config.get("linear_settings")
    if not linear:
        return "預設參數（舊紀錄）"
    changed = [f"{key}={value}" for key, value in linear.items() if LINEAR_DEFAULTS.get(key) != value]
    return "、".join(changed) if changed else "預設參數"


def model_choices(jobs=None) -> dict:
    """Models from succeeded UI training runs, newest first, grouped by prediction method."""
    if jobs is None:
        from . import training_jobs
        jobs = training_jobs.list_jobs()
    choices = {method: [] for method in MODEL_METHODS}
    for job in jobs:
        if job.get("state") != "succeeded" or not job.get("comparison"):
            continue
        stamp = job["job_id"]
        when = f"{stamp[4:6]}/{stamp[6:8]} {stamp[9:11]}:{stamp[11:13]}"
        kind = job["config"]["kind"]
        for result in job["comparison"].get("results", []):
            method = result.get("method") or (result["variant"] if result["variant"] in ("bge-lr", "bge-svc")
                                              else "bge-finetuned")
            path = result.get("model_path")
            if method not in choices or not path or not Path(path).exists():
                continue
            score = result.get("pipeline_total")
            variant = result["variant"]
            selected = " ★ 選型結果" if result.get("selected") else ""
            choices[method].append({
                "path": path, "job_id": stamp, "variant": variant,
                "label": f"{when} · {KIND_NAMES.get(kind, kind)} · {variant}{selected} · {_settings_text(job)}"
                         + (f" · 驗證總分 {score:.4f}" if isinstance(score, (int, float)) else ""),
            })
    for method, items in choices.items():
        # Within one training run, list the automatically selected candidate first.
        items.sort(key=lambda item: (item["job_id"], "★" in item["label"]), reverse=True)
    return {"choices": choices, "defaults": DEFAULT_PATHS}


def resolve_model(method: str, choice, choices: dict) -> tuple[str, str]:
    """Return (model_path, description). Only listed training outputs or the default location are accepted."""
    choice = choice or "latest"
    listed = choices["choices"].get(method, [])
    if choice == "latest":
        if listed:
            return listed[0]["path"], "最新訓練：" + listed[0]["label"]
        choice = "default"
        prefix = "沒有訓練紀錄，改用預設位置："
    else:
        prefix = "預設位置："
    if choice == "default":
        path = ROOT / DEFAULT_PATHS[method]
        # bge-finetuned needs an explicit folder; the others let predict.py find the default file.
        return (str(path) if method == "bge-finetuned" else ""), prefix + DEFAULT_PATHS[method]
    for item in listed:
        if item["path"] == choice:
            return item["path"], item["label"]
    raise ValueError(f"{LABELS.get(method, method)} 選擇的模型不在訓練紀錄中（可能已被刪除），請重新整理後再選")

_guard = threading.Lock()
_jobs: dict[str, dict] = {}
_results: dict[tuple[str, str], dict] = {}
_order: list[str] = []


def _public(job):
    return {key: ([dict(item) for item in value] if key == "items" else value)
            for key, value in job.items() if not key.startswith("_")}


def start_job(payload: dict, reports, operation_lock) -> dict:
    dataset, limit = payload.get("dataset", "history"), payload.get("limit", "native")
    if dataset not in ("dev", "history", "tickets", "custom"):
        raise ValueError("請選擇 dev、history、tickets 或自訂資料集")
    if limit not in ("native", "2gb"):
        raise ValueError("不支援的執行環境")
    methods = payload.get("methods") or list(ORDER)
    if not isinstance(methods, list) or not methods or any(m not in METHODS for m in methods):
        raise ValueError("方法清單不正確")
    base = {key: payload[key] for key in ("dataset", "limit", "output_dir", "dataset_content", "dataset_name")
            if key in payload}
    selections = payload.get("models") or {}
    if not isinstance(selections, dict) or set(selections) - set(MODEL_METHODS):
        raise ValueError("模型選擇格式不正確")
    available = model_choices() if any(m in MODEL_METHODS for m in methods) else {"choices": {}}
    # Resolve every model before starting, so a stale choice fails immediately, not halfway.
    models = {m: resolve_model(m, selections.get(m), available) for m in methods if m in MODEL_METHODS}
    with _guard:
        if any(job["state"] == "running" for job in _jobs.values()):
            raise ValueError("已有「一次跑全部方法」工作執行中，請等待完成")
        job_id = datetime.now(TAIPEI).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        dataset_label = payload.get("dataset_name") if dataset == "custom" else f"{dataset}.jsonl"
        job = {"job_id": job_id, "state": "running", "dataset": str(dataset_label or "dataset.jsonl")[:200],
               "limit": limit, "created_at": datetime.now(TAIPEI).isoformat(), "elapsed_seconds": 0.0,
               "items": [{"method": m, "name": LABELS.get(m, m), "state": "queued",
                          **({"model_path": models[m][0], "model_label": models[m][1]} if m in models else {})}
                         for m in methods],
               "_started": perf_counter()}
        _jobs[job_id] = job
        _order.append(job_id)
        while len(_order) > KEEP_JOBS:
            old = _order.pop(0)
            _jobs.pop(old, None)
            for key in [k for k in _results if k[0] == old]:
                del _results[key]
    threading.Thread(target=_run, args=(job, base, reports, operation_lock), daemon=True).start()
    return get_job(job_id)


def _run(job, base, reports, operation_lock):
    for item in job["items"]:
        method, model_path = item["method"], item.get("model_path", "")
        if method == "bge-finetuned" and not (Path(model_path) / "model.json").is_file():
            item.update(state="skipped", error=f"找不到微調模型 {model_path}（預設模型請先執行 install.py），已略過")
            continue
        item["state"] = "running"
        started = perf_counter()
        try:
            with operation_lock:
                result = run_experiment({**base, "method": method, "model_path": model_path}, reports)
        except Exception as exc:  # one failing method must not stop the others
            item.update(state="failed", error=str(exc)[-600:], wall_seconds=perf_counter() - started)
            continue
        summary = result["summary"]
        item.update(state="succeeded", run_id=result["run_id"], label=result["label"],
                    output_path=result["execution"]["output_path"], count=result["execution"]["count"],
                    total=summary["total"] if summary else None,
                    metrics={k: summary["metrics"][k] for k in ("category", "device", "urgency", "duplicate")}
                    if summary else None,
                    error_count=len(result["errors"]) if summary else None,
                    timing=result["timing"])
        with _guard:
            if job["job_id"] in _jobs:
                _results[(job["job_id"], method)] = result
    job["elapsed_seconds"] = perf_counter() - job["_started"]
    job["state"] = "completed"


def get_job(job_id: str | None = None) -> dict | None:
    with _guard:
        if job_id is None:
            job_id = _order[-1] if _order else None
        job = _jobs.get(job_id) if job_id else None
        if job is None:
            return None
        if job["state"] == "running":
            job["elapsed_seconds"] = perf_counter() - job["_started"]
        return _public(job)


def get_result(job_id: str, method: str) -> dict:
    with _guard:
        result = _results.get((job_id, method))
    if result is None:
        raise ValueError("找不到這個方法的結果（可能尚未完成，或服務已重新啟動）")
    return result
