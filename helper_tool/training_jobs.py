"""Persistent local training jobs; UI requests never execute model code inline."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from time import perf_counter
import shutil

from .common import ROOT
from Solution.training.settings import FineTuneSettings
from Solution.triage.io import atomic_write, read_tickets

REGISTRY = ROOT / "helper_tool/reports/training"
DEFAULT_OUTPUT = ROOT / "Solution/models/training-runs"
KINDS = {"bge-finetune", "bge-frozen", "tfidf"}
TERMINAL = {"succeeded", "failed", "interrupted"}


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _entry(job_id):
    if not re.fullmatch(r'\d{8}-\d{6}-\d{6}-[a-f0-9]{8}', job_id):
        raise ValueError("訓練工作 ID 不合法")
    registry_file = REGISTRY / f"{job_id}.json"
    return json.loads(registry_file.read_text(encoding="utf-8"))


def get_job(job_id):
    entry = _entry(job_id)
    directory = Path(entry["directory"])
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    # A worker survives normal UI restarts. Detect abrupt termination without
    # treating stale status files as successful training results.
    pid = status.get("pid") or entry.get("pid")
    if status["state"] not in TERMINAL and pid:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            status.update(state="interrupted", message="訓練程序已停止；保留原有紀錄，請建立新工作重試")
            write_json(directory / "status.json", status)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    result = {**entry, **status, "config": config, "events": []}
    events = directory / "events.jsonl"
    if events.exists():
        content = events.read_text(encoding="utf-8")
        lines = content.splitlines()
        # The writer may be appending its last line while this request reads.
        for index, line in enumerate(lines[-80:]):
            try:
                result["events"].append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines[-80:]) - 1 or content.endswith('\n'):
                    raise ValueError("訓練事件紀錄格式損壞，請檢查 events.jsonl")
    comparison = directory / "models/comparison.json"
    if comparison.exists():
        result["comparison"] = json.loads(comparison.read_text(encoding="utf-8"))
    result["elapsed_seconds"] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(config["created_at"])).total_seconds()) if status["state"] not in TERMINAL else status.get("total_seconds")
    return result


def list_jobs():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    return [get_job(p.stem) for p in sorted(REGISTRY.glob('*.json'), reverse=True)[:50]]


def start_job(payload):
    import uuid
    kind = payload.get("kind", "bge-finetune")
    if kind not in KINDS:
        raise ValueError("不支援的訓練方式")
    if payload.get("history_only") is not True:
        raise ValueError("請確認資料為開發用 history，未包含 dev 或其他保留測試集")
    settings = FineTuneSettings.from_dict(payload.get("settings"))
    source = payload.get("source", "local")
    if source not in ("local", "upload"):
        raise ValueError("請選擇本機 history 或上傳 history")
    label = "history.jsonl"
    if source == "upload":
        label = payload.get("history_name")
        if not isinstance(label, str) or not label.lower().endswith('.jsonl') or 'dev' in label.lower():
            raise ValueError("只接受 history JSONL；不可上傳 dev 檔案")
        content = payload.get("history_content")
        if not isinstance(content, str) or not content.strip() or len(content.encode('utf-8')) > 8 * 1024 * 1024:
            raise ValueError("history 檔案須為 1 byte～8 MB")
    else:
        content = (ROOT / "data/history.jsonl").read_text(encoding="utf-8")
    output_value = payload.get("output_dir", str(DEFAULT_OUTPUT))
    if not isinstance(output_value, str) or not output_value.strip():
        raise ValueError("請指定模型保存資料夾")
    output = Path(output_value.strip()).expanduser()
    output = (output if output.is_absolute() else ROOT / output).resolve()
    python = ROOT / ".venv/bin/python"
    if not python.is_file():
        raise ValueError("請先建立專案 .venv 並安裝模型訓練依賴")
    if kind.startswith('bge') and not (ROOT / 'Solution/models/bge-small-zh-v1.5/encoder_manifest.json').is_file():
        raise ValueError("請先下載固定 BGE encoder，訓練不會自行連網下載")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(REGISTRY / '.training.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("已有訓練工作進行中，請等待完成；避免多個訓練互相爭用記憶體") from exc
        output.mkdir(parents=True, exist_ok=True)
        job_id = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f-') + uuid.uuid4().hex[:8]
        directory = output / job_id
        directory.mkdir()
        # Custom output roots must not accidentally add weights or ticket data to Git.
        (directory / '.gitignore').write_text('*\n', encoding='utf-8')
        (directory / 'data').mkdir()
        history = directory / 'data/history.jsonl'
        history.write_text(content, encoding='utf-8')
        try:
            tickets = read_tickets(history)
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
            if not tickets or len(tickets) > 2000:
                raise ValueError("目前訓練支援 1～2,000 筆 history")
            if any('category' not in row or row['category'] not in ('noise','temperature','power','software','other',None) for row in rows):
                raise ValueError("每筆 history 須提供 category：五類之一或 null")
        except (OSError, ValueError, TypeError):
            # Invalid uploads are not training runs, and should leave no model folder.
            shutil.rmtree(directory)
            raise
        config = {"job_id": job_id, "kind": kind, "source": source, "source_name": Path(label.replace('\\','/')).name,
                  "history_only_confirmed": True, "created_at": datetime.now(timezone.utc).isoformat(),
                  "settings": asdict(settings), "count": len(tickets), "output_dir": str(directory),
                  "note": "Uploaded history is user-attested; file names cannot prove data provenance."}
        write_json(directory / 'config.json', config)
        # Snapshot code/config, never the data tree, so uncommitted training runs
        # remain reviewable even after the working copy changes.
        snapshot = directory / 'source'
        for relative in ('Solution/triage', 'Solution/training', 'Solution/config'):
            shutil.copytree(ROOT / relative, snapshot / relative, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative in ('Solution/train.py', 'Solution/train_embedding.py', 'Solution/train_finetune.py',
                         'Solution/requirements-ml.txt', 'Solution/requirements-embedding.txt',
                         'helper_tool/training_jobs.py', 'helper_tool/evaluate.py', 'helper_tool/common.py', 'baseline/evaluate.py',
                         'data/devices.json', 'data/error_codes.json'):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        write_json(directory / 'status.json', {"state": "queued", "message": "等待啟動訓練", "pid": None})
        write_json(REGISTRY / f'{job_id}.json', {"job_id": job_id, "directory": str(directory)})
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        command = [str(python), '-u', str(ROOT / 'Solution/train_finetune.py'), '--run-dir', str(directory), '--lock-fd', str(lock_fd)]
        started = perf_counter()
        try:
            with (directory / 'worker.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(lock_fd,), start_new_session=True)
        except OSError as exc:
            write_json(directory / 'status.json', {"state": "failed", "message": str(exc), "pid": None, "total_seconds": 0})
            raise
        write_json(REGISTRY / f'{job_id}.json', {"job_id": job_id, "directory": str(directory), "pid": process.pid})
        def watch():
            code = process.wait()
            status_path = directory / 'status.json'
            status = json.loads(status_path.read_text(encoding='utf-8'))
            if status['state'] not in TERMINAL:
                write_json(status_path, {**status, "state": "failed", "message": f"訓練程序退出 {code}，詳見 worker.log", "total_seconds": perf_counter() - started})
        threading.Thread(target=watch, daemon=True).start()
    finally:
        # Do not LOCK_UN here: the worker inherits and owns the same file lock.
        os.close(lock_fd)
    return get_job(job_id)
