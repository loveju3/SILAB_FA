"""Execute allowlisted predictors and publish named local evaluation artifacts."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
import uuid

from .common import FIELDS, ROOT, read_records
from .evaluate import analyze, write_reports
from Solution.triage.io import read_tickets, file_sha256

METHODS = {"solution-a": "predict.py", "baseline": "baseline/predict.py",
           "tfidf-lr": "predict.py", "tfidf-svc": "predict.py", "bge-lr": "predict.py", "bge-svc": "predict.py", "bge-finetuned": "predict.py"}
MODEL_METHODS = {"tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc", "bge-finetuned"}
LIMITS = {"native", "2gb"}
TAIPEI = timezone(timedelta(hours=8))


def safe_name(name: str) -> str:
    # Uploaded names are labels, never paths supplied to a shell.
    name = Path(name.replace("\\", "/")).stem
    name = re.sub(r"[^\w\-]+", "-", name, flags=re.UNICODE).strip("-_")
    return name[:60] or "dataset"


def execute(method: str, limit: str, folder: Path, model_path: Path | None = None) -> dict:
    script = METHODS[method]
    options = ["--method", method] if method in MODEL_METHODS else []
    if model_path is not None:
        options.extend(["--model-path", str(model_path)])
    args = [script, str(folder / "input.jsonl"), str(folder / "predictions.jsonl"), *options]
    if method != "baseline":
        args.extend(["--metrics", str(folder / "metrics.json")])
    if limit == "native":
        interpreter = ROOT / ('.venv/Scripts/python.exe' if os.name == 'nt' else '.venv/bin/python')
        python = str(interpreter) if method in MODEL_METHODS and interpreter.is_file() else sys.executable
        command = [python, str(ROOT / args[0]), *args[1:]]
    else:
        if shutil.which("docker") is None:
            raise ValueError("2 GB 模式需要先安裝並啟動 Docker")
        mounts = []
        if model_path is not None:
            options = ["--method", method, "--model-path", f"/models/{model_path.name}"]
            mounts = ["--mount", f"type=bind,src={model_path.parent},dst=/models,readonly"]
        args = [script, "/run/input.jsonl", "/run/predictions.jsonl", *options]
        if method != "baseline":
            args.extend(["--metrics", "/run/metrics.json"])
        container_name = "silab-ui-" + uuid.uuid4().hex[:16]
        # atomic_write creates private files; use the host UID/GID so the UI
        # can read predictions written into Linux/WSL temporary directories.
        user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
        command = ["docker", "run", "--pull=never", "--name", container_name, "--network=none", "--memory=2g",
                   "--memory-swap=2g", "--cpus=2", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "OMP_NUM_THREADS=2", "--env", "OPENBLAS_NUM_THREADS=2",
                   "--env", "MKL_NUM_THREADS=2", "--mount", f"type=bind,src={ROOT},dst=/app,readonly",
                   "--mount", f"type=bind,src={folder},dst=/run", "--workdir", "/app",
                   *user, *mounts, "--entrypoint", "python3", "silab-fa:scheme-c" if method.startswith("bge-") else
                   "silab-fa:scheme-b" if method in MODEL_METHODS else "silab-fa:scheme-a",
                   "Solution/scripts/check_limits.py", "--report", "/run/limits.json", "--run", *args]
    started = perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                env={**os.environ, 'PYTHONUTF8': '1'}, timeout=480)
    except FileNotFoundError as exc:
        raise ValueError("找不到執行工具；2 GB 模式需要先安裝並啟動 Docker") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("預測超過 480 秒，未產生正式輸出") from exc
    finally:
        if limit == "2gb":
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=15)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise ValueError(f"預測失敗（退出碼 {result.returncode}）：{detail}")
    execution = {"method": method, "limit": limit, "wall_seconds": perf_counter() - started}
    if model_path is not None:
        execution["model_path"] = str(model_path)
    if limit == "2gb":
        limits = json.loads((folder / "limits.json").read_text(encoding="utf-8"))
        if (limits.get("memory.max"), limits.get("memory.swap.max"), limits.get("exit_code")) != (
                "2147483648", "0", 0):
            raise ValueError("未驗證 2 GB 與禁止 swap 的限制，不保存為成功結果")
        execution["limits"] = limits
    return execution


def run_experiment(payload: dict, reports: Path) -> dict:
    started = perf_counter()
    dataset = payload.get("dataset", "history")
    method, limit = payload.get("method", "solution-a"), payload.get("limit", "native")
    if dataset not in ("dev", "history", "tickets", "custom"):
        raise ValueError("請選擇 dev、history、tickets 或自訂資料集")
    if not isinstance(method, str) or not isinstance(limit, str) or method not in METHODS or limit not in LIMITS:
        raise ValueError("不支援的方法或限制；目前提供方案 A／B（TF-IDF）／C（BGE）／Baseline，本機／2 GB")
    model_value = payload.get("model_path", "")
    if not isinstance(model_value, str):
        raise ValueError("模型路徑須為字串")
    model_path = None
    if model_value.strip():
        if method not in MODEL_METHODS:
            raise ValueError("此方法不使用模型路徑")
        model_path = Path(model_value.strip()).expanduser()
        model_path = (model_path if model_path.is_absolute() else ROOT / model_path).resolve()
        if not model_path.exists():
            raise ValueError("找不到指定的模型路徑")
    if method == "bge-finetuned" and model_path is None:
        raise ValueError("請選擇訓練紀錄中的模型，或指定神經模型資料夾")
    output_value = payload.get("output_dir", "Solution/predictions")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("請指定預測檔輸出資料夾")
    output_dir = Path(output_value.strip()).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir = output_dir.resolve()
    with tempfile.TemporaryDirectory(prefix="silab-experiment-") as temporary:
        folder = Path(temporary).resolve()
        source = folder / "input.jsonl"
        if dataset == "custom":
            content = payload.get("dataset_content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("請選擇自訂測試資料 JSONL 檔")
            source.write_text(content, encoding="utf-8")
            label = str(payload.get("dataset_name") or "dataset.jsonl")[:200]
        else:
            original = ROOT / "data" / f"{dataset}.jsonl"
            shutil.copyfile(original, source)
            label = original.name
        rows = list(read_records(source))
        if not rows:
            raise ValueError("測試資料不可為空")
        labelled = [all(f in record for f in FIELDS) for _, record in rows]
        has_any_label = any(any(f in record for f in FIELDS) for _, record in rows)
        if has_any_label and not all(labelled):
            raise ValueError("標準答案不完整：每筆都須有四個標籤，或整份資料完全不含標籤")
        # Validate before expensive inference. No output or report is published on failure.
        read_tickets(source)
        output_dir.mkdir(parents=True, exist_ok=True)
        execution = execute(method, limit, folder, model_path) if model_path is not None else execute(method, limit, folder)
        predicted = folder / "predictions.jsonl"
        predictions = {r["ticket_id"]: r for _, r in read_records(predicted)}
        if set(predictions) != {r["ticket_id"] for _, r in rows}:
            raise ValueError("輸出工單 ID 與輸入不一致")
        summary, errors, official = (None, [], "無標準答案，未計分。\n")
        if all(labelled):
            summary, errors, official = analyze(predicted, source)
        score = f"{summary['total']:.6f}" if summary else "未評分"
        timestamp = datetime.now(TAIPEI).strftime("%Y%m%d-%H%M%S-%f")
        filename = f"{timestamp}_{safe_name(label)}_{method}_{limit}_{score}.jsonl"
        output = output_dir / filename
        run_id = datetime.now(TAIPEI).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        run = reports / run_id
        error_map = {e["ticket_id"]: e for e in errors}
        records = []
        for line, row in rows:
            entry = error_map.get(row["ticket_id"], {
                "ticket_id": row["ticket_id"], "source_line": line, "raw_text": row["raw_text"],
                "reporter": row["reporter"], "created_at": row["created_at"],
                "gold": row if summary else None, "prediction": predictions[row["ticket_id"]],
                "wrong_fields": [], "differences": {}, "validation_errors": [],
            })
            entry["source_file"] = label
            records.append(entry)
        if summary:
            summary["prediction_file"], summary["gold_file"] = str(output), label
            write_reports(run, summary, errors, official)
        else:
            run.mkdir(parents=True)
        metadata = {"dataset": label, "output_path": str(output), "filename": filename,
                    "dataset_sha256": file_sha256(source), "created_at": datetime.now(TAIPEI).isoformat(),
                    "scored": summary is not None, "count": len(rows), "timezone": "Asia/Taipei",
                    **execution}
        created = False
        try:
            # Exclusive creation protects previous results, even if a name collides.
            with output.open("xb") as stream:
                created = True
                stream.write(predicted.read_bytes())
            shutil.copyfile(output, run / "predictions.jsonl")
            for name in ("metrics.json", "limits.json"):
                if (folder / name).exists():
                    shutil.copyfile(folder / name, run / name)
            timing = {"total_seconds": perf_counter() - started,
                      "prediction_seconds": execution["wall_seconds"]}
            metadata["timing"] = timing
            (run / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError:
            if created:
                output.unlink(missing_ok=True)
            shutil.rmtree(run)
            raise
    downloads = {"predictions.jsonl": f"/reports/{run_id}/predictions.jsonl"}
    if summary:
        downloads.update({name: f"/reports/{run_id}/{name}" for name in
                          ("summary.json", "errors.jsonl", "errors.csv", "official.txt", "report.md")})
    return {"summary": summary, "errors": errors, "records": records, "label": filename,
            "run_id": run_id, "downloads": downloads, "execution": metadata, "timing": timing}
