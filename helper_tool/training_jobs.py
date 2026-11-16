"""Persistent local training jobs; UI requests never execute model code inline."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from time import perf_counter
import shutil
import sys

from .common import ROOT
from .agent_jobs import alive
from Solution.training.settings import FineTuneSettings, LinearSettings, ModalSettings
from Solution.triage.io import atomic_write, read_tickets, load_json

REGISTRY = ROOT / "helper_tool/reports/training"
DEFAULT_OUTPUT = ROOT / "Solution/models/training-runs"
KINDS = {"bge-finetune", "bge-frozen", "tfidf"}
TERMINAL = {"succeeded", "failed", "interrupted"}


BGE_ENCODER_MANIFEST = ROOT / 'Solution/models/bge-small-zh-v1.5/encoder_manifest.json'

def acquire_lock(fd):
    if os.name == 'nt':
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        if os.fstat(fd).st_size == 0:
            os.write(fd, b'0')
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc
    else:
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def write_json(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _entry(job_id):
    if not re.fullmatch(r'\d{8}-\d{6}-\d{6}-[a-f0-9]{8}', job_id):
        raise ValueError("訓練工作 ID 不合法")
    registry_file = REGISTRY / f"{job_id}.json"
    return load_json(registry_file)


def get_job(job_id):
    entry = _entry(job_id)
    directory = Path(entry["directory"])
    if not (directory / 'status.json').is_file():
        raise FileNotFoundError(f"訓練資料夾已不存在（可能已被刪除）：{directory}")
    status = load_json(directory / 'status.json')
    # A worker survives normal UI restarts. Detect abrupt termination without
    # treating stale status files as successful training results.
    pid = status.get("pid") or entry.get("pid")
    if status["state"] not in TERMINAL and pid:
        if not alive(pid):
            status.update(state="interrupted", message="訓練程序已停止；保留原有紀錄，請建立新工作重試")
            # This is a read-time observation only. Persisting it can overwrite
            # a terminal status concurrently being saved by the watcher.
    config = load_json(directory / 'config.json')
    result = {**entry, **status, "config": config, "events": []}
    events = directory / "events.jsonl"
    if events.exists():
        content = events.read_text(encoding="utf-8")
        lines = content.splitlines()
        # The writer may be appending its last line while this request reads.
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
                if record.get('event') != 'batch':
                    result["events"].append(record)
            except json.JSONDecodeError:
                if index != len(lines) - 1 or content.endswith('\n'):
                    raise ValueError("訓練事件紀錄格式損壞，請檢查 events.jsonl")
        result['events'] = result['events'][-80:]
    comparison = directory / "models/comparison.json"
    if comparison.exists():
        result["comparison"] = load_json(comparison)
    result["elapsed_seconds"] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(config["created_at"])).total_seconds()) if status["state"] not in TERMINAL else status.get("total_seconds")
    return result


def list_jobs():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(REGISTRY.glob('*.json'), reverse=True):
        try:
            jobs.append(get_job(path.stem))
        except FileNotFoundError:
            # The user may delete old training folders by hand; their index
            # entries are skipped so the remaining history stays usable.
            continue
        if len(jobs) == 50:
            break
    return jobs


def start_job(payload):
    import uuid
    kind = payload.get("kind", "bge-finetune")
    if kind not in KINDS:
        raise ValueError("不支援的訓練方式")
    backend = payload.get("backend", "cpu")
    if backend not in ("cpu", "gpu", "modal"):
        raise ValueError("請選擇 CPU、GPU 或 Modal")
    if backend != "cpu" and kind != "bge-finetune":
        raise ValueError("GPU／Modal 目前支援 BGE 聯合微調比較；既有 sklearn 訓練請選 CPU")
    if backend == "modal" and payload.get("cloud_upload_confirmed") is not True:
        raise ValueError("請確認允許將本次 history 上傳 Modal 執行訓練（可能產生費用）")
    if payload.get("history_only") is not True:
        raise ValueError("請確認資料為開發用 history，未包含 dev 或其他保留測試集")
    settings = FineTuneSettings.from_dict(payload.get("settings"))
    # LR／SVC parameters only apply to the existing sklearn training kinds.
    linear_settings = (LinearSettings.from_dict(payload.get("linear_settings"), kind).to_dict(kind)
                       if kind != "bge-finetune" else None)
    if kind == "bge-finetune" and payload.get("linear_settings"):
        raise ValueError("聯合微調請使用微調設定，不接受 LR／SVC 設定")
    modal_settings = asdict(ModalSettings.load()) if backend == 'modal' else None
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
    python = (ROOT / '.venv/Scripts/python.exe') if os.name == 'nt' else ROOT / '.venv/bin/python'
    if os.name == 'nt' and not python.is_file():
        runtime = ROOT / 'helper_tool/reports/training-runtime/Scripts/python.exe'
        python = runtime if runtime.is_file() else Path(sys.executable)
    if not python.is_file():
        raise ValueError("請先建立專案 .venv 並安裝模型訓練依賴")
    if kind.startswith('bge') and not BGE_ENCODER_MANIFEST.is_file():
        raise ValueError("請先下載固定 BGE encoder，訓練不會自行連網下載")
    REGISTRY.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(REGISTRY / '.training.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            acquire_lock(lock_fd)
        except BlockingIOError as exc:
            raise ValueError("已有訓練工作進行中，請等待完成；避免多個訓練互相爭用記憶體") from exc
        # POSIX workers inherit flock. Windows serializes startup here, then
        # checks persisted live workers under this same lock on every request.
        if os.name == 'nt' and any(job['state'] not in TERMINAL for job in list_jobs()):
            raise ValueError("已有訓練工作進行中，請等待完成")
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
        config = {"job_id": job_id, "kind": kind, "backend": backend,
                  "python_executable": str(python),
                  "cloud_upload_confirmed": backend == "modal" and payload.get("cloud_upload_confirmed") is True,
                  "source": source, "source_name": Path(label.replace('\\','/')).name,
                  "history_only_confirmed": True, "created_at": datetime.now(timezone.utc).isoformat(),
                  "settings": asdict(settings), "count": len(tickets), "output_dir": str(directory),
                  "note": "Uploaded history is user-attested; file names cannot prove data provenance."}
        if modal_settings is not None:
            config['modal_settings'] = modal_settings
        if linear_settings is not None:
            config['linear_settings'] = linear_settings
        write_json(directory / 'config.json', config)
        # Snapshot code/config, never the data tree, so uncommitted training runs
        # remain reviewable even after the working copy changes.
        snapshot = directory / 'source'
        for relative in ('Solution/triage', 'Solution/training', 'Solution/config'):
            shutil.copytree(ROOT / relative, snapshot / relative, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for relative in ('Solution/train.py', 'Solution/train_embedding.py', 'Solution/train_finetune.py', 'Solution/modal_training.py',
                         'Solution/requirements-ml.txt', 'Solution/requirements-embedding.txt', 'Solution/requirements-modal.txt',
                         'helper_tool/training_jobs.py', 'helper_tool/evaluate.py', 'helper_tool/common.py', 'baseline/evaluate.py',
                         'data/devices.json', 'data/error_codes.json'):
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        write_json(directory / 'status.json', {"state": "queued", "message": "等待啟動訓練", "pid": None})
        write_json(REGISTRY / f'{job_id}.json', {"job_id": job_id, "directory": str(directory)})
        environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "PYTHONUTF8": "1",
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                       "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
        command = [str(python), '-u', str(ROOT / 'Solution/train_finetune.py'), '--run-dir', str(directory),
                   '--lock-fd', str(lock_fd if os.name != 'nt' else -1)]
        started = perf_counter()
        try:
            with (directory / 'worker.log').open('w', encoding='utf-8') as log:
                options = ({'pass_fds': (lock_fd,), 'start_new_session': True} if os.name != 'nt'
                           else {'creationflags': subprocess.CREATE_NO_WINDOW})
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, **options)
        except OSError as exc:
            write_json(directory / 'status.json', {"state": "failed", "message": str(exc), "pid": None, "total_seconds": 0})
            raise
        write_json(REGISTRY / f'{job_id}.json', {"job_id": job_id, "directory": str(directory), "pid": process.pid})
        def watch():
            code = process.wait()
            status_path = directory / 'status.json'
            status = load_json(status_path)
            if status['state'] not in TERMINAL or status['state'] == 'interrupted':
                write_json(status_path, {**status, "state": "failed", "message": f"訓練程序退出 {code}，詳見 worker.log", "total_seconds": perf_counter() - started})
        threading.Thread(target=watch, daemon=True).start()
    finally:
        # Do not LOCK_UN here: the worker inherits and owns the same file lock.
        os.close(lock_fd)
    # Do not read hot worker files in the creation response: a transient read
    # failure must not report a failed POST after a job has actually started.
    return {'job_id': job_id, 'directory': str(directory), 'pid': process.pid,
            'state': 'queued', 'message': '已建立背景訓練工作', 'config': config, 'events': []}
