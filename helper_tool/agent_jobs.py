"""Persisted Agent evaluation workers; HTTP requests only start/poll local jobs."""
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
from time import perf_counter
import uuid

from .common import ROOT
from Solution.triage.io import atomic_write
from agent.process import ChildGuard

REGISTRY = ROOT / "helper_tool/reports/agent"
PATROL_KINDS = {"patrol", "patrol-tickets"}
KINDS = {"history", "synthetic", *PATROL_KINDS}
TERMINAL = {"completed", "failed", "interrupted"}
_lock = threading.Lock()


def save(path, value):
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def directory(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"\d{8}-\d{6}-\d{6}-[a-f0-9]{8}", job_id):
        raise ValueError("Agent 測試工作 ID 不合法")
    return REGISTRY / job_id


def alive(pid):
    if os.name == "nt":
        import ctypes as c
        from ctypes import wintypes as w
        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.GetExitCodeProcess.argtypes = [w.HANDLE, c.POINTER(w.DWORD)]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return c.get_last_error() == 5  # access denied is not proof of exit
        try:
            code = w.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, c.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def registry_lock():
    REGISTRY.mkdir(parents=True, exist_ok=True)
    with _lock, (REGISTRY / ".registry.lock").open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("其他程序正在建立 Agent 測試，請稍後重試") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def get_job(job_id):
    folder = directory(job_id)
    job = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    if job["state"] not in TERMINAL:
        entry = folder / "worker.json"
        pid = job.get("pid")
        if not pid and entry.exists():
            pid = json.loads(entry.read_text(encoding="utf-8"))["pid"]
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(job["created_at"])).total_seconds()
        if (pid and not alive(pid)) or (not pid and age > 15):
            job = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            if job["state"] not in TERMINAL:
                job.update(state="interrupted", message="測試程序已停止，保留已完成結果；可建立新測試")
                save(folder / "status.json", job)
    if job["state"] not in TERMINAL:
        start = job.get("started_at") or job["created_at"]
        job["elapsed_seconds"] = max(0, (datetime.now(timezone.utc) - datetime.fromisoformat(start)).total_seconds())
    else:
        job["elapsed_seconds"] = job.get("timing", {}).get("total_seconds")
    job["report_url"] = f"/api/agent/report?id={job_id}" if job["state"] == "completed" and (folder / "report.json").is_file() else None
    if job["kind"] in PATROL_KINDS and "patrol" not in job and (folder / "report.json").is_file() and job["state"] in TERMINAL:
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        job["patrol"] = {"source_info": report.get("source_info", {"dataset": "synthetic", "count": 12,
                         "file": "歷史合成測試（未保存來源快照）"}),
                         "selected_tickets": report.get("selected_tickets", report.get("output") or [])}
    return job


def list_jobs():
    if not REGISTRY.exists():
        return []
    folders = [p for p in sorted(REGISTRY.iterdir(), reverse=True) if p.is_dir() and (p / "status.json").exists()][:30]
    return [{k: v for k, v in get_job(p.name).items() if k not in {"cases", "patrol"}} for p in folders]


def availability():
    model = ROOT / "agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    server = ROOT / ("agent_assets/runtime/llama-server.exe" if os.name == "nt" else "agent_assets/runtime/llama-server")
    return {"ready": model.is_file() and server.is_file(), "model": str(model), "server": str(server)}


def start_job(payload):
    kind, threads = payload.get("kind", "history"), payload.get("threads", 6)
    if not isinstance(kind, str) or kind not in KINDS or set(payload) - {"kind", "threads"}:
        raise ValueError("請選擇 history、synthetic、patrol 或 patrol-tickets，不接受 dev 或自訂命令")
    if type(threads) is not int or not 1 <= threads <= 12:
        raise ValueError("CPU threads 必須是 1～12 的整數")
    if not availability()["ready"]:
        raise ValueError("缺少本機 Qwen 模型或 CPU runtime。Windows 請先執行 python agent/setup.py；Linux/WSL 需原生 llama-server。")
    if kind == "history" and not (ROOT / "data/history.jsonl").is_file():
        raise ValueError("找不到開發用 history.jsonl")
    if kind == "patrol-tickets" and not (ROOT / "data/tickets.jsonl").is_file():
        raise ValueError("找不到正式巡檢來源 data/tickets.jsonl")
    with registry_lock():
        if any(job["state"] not in TERMINAL for job in list_jobs()):
            raise ValueError("已有 Agent 測試進行中，請等待完成以避免 CPU 互相影響")
        job_id = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d-%H%M%S-%f-") + uuid.uuid4().hex[:8]
        folder = directory(job_id)
        folder.mkdir()
        job = {"job_id": job_id, "kind": kind, "threads": threads, "state": "queued",
               "created_at": datetime.now(timezone.utc).isoformat(), "message": "等待啟動 Agent 測試",
               "total": (1 if kind in PATROL_KINDS else
                         len(json.loads((ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))["cases"])
                         if kind == "synthetic" else 12),
               "completed": 0, "passed": 0, "cases": [], "timing": {}}
        save(folder / "status.json", job)
        command = [sys.executable, "-u", "-m", "helper_tool.agent_jobs", "--worker", job_id]
        try:
            with (folder / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"}, start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            save(folder / "worker.json", {"pid": process.pid})
            threading.Thread(target=process.wait, daemon=True).start()
        except OSError as exc:
            job.update(state="failed", message=str(exc))
            save(folder / "status.json", job)
            raise
    return get_job(job_id)


def worker(job_id):
    started = perf_counter()
    folder = directory(job_id)
    job = json.loads((folder / "status.json").read_text(encoding="utf-8"))
    job.update(state="running", pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(), message="準備資料與載入本機模型")
    save(folder / "status.json", job)
    process, guard, timer = None, None, None
    report_path = folder / "report.json"
    try:
        is_patrol = job["kind"] in PATROL_KINDS
        script = "run_patrol.py" if is_patrol else "run.py"
        command = [sys.executable, "-u", str(ROOT / "evals" / script), "--output", str(report_path), "--threads", str(job["threads"])]
        command.extend(["--server", availability()["server"]])
        if is_patrol:
            command.extend(["--dataset", "tickets" if job["kind"] == "patrol-tickets" else "synthetic"])
        else:
            command.extend(["--suite", job["kind"], "--events"])
        def stop():
            if process is not None and process.poll() is None:
                if os.name == "nt":
                    process.kill()
                    if guard:
                        guard.close()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        with (folder / "runner.stderr.log").open("w", encoding="utf-8") as errors, (folder / "runner.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=errors,
                text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                start_new_session=os.name != "nt", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            guard = ChildGuard(process)
            timer = threading.Timer(130 if is_patrol else 32 * job["total"] + 36, stop)
            timer.daemon = True
            timer.start()
            for line in process.stdout:
                log.write(line)
                log.flush()
                if is_patrol:
                    continue
                event = json.loads(line)
                if event["event"] == "started":
                    job["total"] = event["total"]
                elif event["event"] == "case":
                    job["cases"].append(event["case"])
                    job["completed"] = len(job["cases"])
                    job["passed"] = sum(c["passed"] for c in job["cases"])
                job.update(message=f"已完成 {job['completed']} / {job['total']} 題")
                job["timing"] = {"total_seconds": perf_counter() - started,
                                 "test_seconds": sum(c["elapsed_s"] for c in job["cases"])}
                save(folder / "status.json", job)
            code = process.wait(timeout=5)
        if not report_path.exists() or code not in (0, 1):
            detail = (folder / "runner.stderr.log").read_text(encoding="utf-8")[-1800:]
            raise RuntimeError(f"評測程序退出 {code}，未產生完整報告。{detail}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if is_patrol:
            source = report.get("source_info", {"count": 12, "dataset": "synthetic"})
            job["patrol"] = {"source_info": source, "selected_tickets": report.get("selected_tickets", [])}
            label = "正式工單" if source["dataset"] == "tickets" else "合成工單"
            job["cases"] = [{"id": "patrol-contract", "question": f"從{source['count']}張{label}選10張；檢查格式、原文引用與唯一性（不評排序品質）",
                "expected": "10張不重複工單，引用均來自原文，120秒內完成", "actual": {"answer": report.get("output")},
                "passed": report["contract_passed"], "elapsed_s": report["elapsed_s"], "trace": report.get("trace", ""), "error": report.get("error")}]
        else:
            job["cases"] = report["cases"]
        job.update(state="completed", completed=len(job["cases"]), passed=sum(c["passed"] for c in job["cases"]))
        job["message"] = f"測試完成：{job['passed']} / {job['total']} 通過"
    except Exception as exc:
        job.update(state="failed", message=str(exc))
    finally:
        if timer:
            timer.cancel()
        if process is not None and process.poll() is None:
            stop()
            process.wait(timeout=5)
        if process is not None and process.stdout is not None:
            process.stdout.close()
        if guard:
            guard.close()
        job["timing"] = {"total_seconds": perf_counter() - started,
                         "test_seconds": sum(c["elapsed_s"] for c in job["cases"])}
        save(folder / "status.json", job)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    worker(parser.parse_args().worker)
