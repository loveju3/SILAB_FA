#!/usr/bin/env python3
"""Local, dependency-free UI for scoring and ticket lookup."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import threading
from time import perf_counter
from urllib.parse import parse_qs, quote, urlsplit
import uuid
import webbrowser

sys.dont_write_bytecode = True
try:
    from .common import ROOT, read_records
    from .evaluate import analyze, write_reports
    from .lookup import build_index
except ImportError:
    from common import ROOT, read_records
    from evaluate import analyze, write_reports
    from lookup import build_index
from baseline.predict import main as baseline_predict
# experiments uses package-relative imports, also when this server is a script.
from helper_tool.experiments import run_experiment

WEB = Path(__file__).resolve().parent / "web"
RUNS = Path(__file__).resolve().parent / "reports/ui"
DOWNLOADS = {"summary.json", "errors.jsonl", "errors.csv", "official.txt", "report.md"}
MAX_BODY = 16 * 1024 * 1024


def evaluate_upload(payload):
    started = perf_counter()
    prediction_seconds = None
    gold_name = payload.get("gold", "dev")
    if gold_name not in ("dev", "history", "custom"):
        raise ValueError("請選擇 dev、history 或自訂標準答案")
    with tempfile.TemporaryDirectory(prefix="silab-ui-") as temporary:
        temp = Path(temporary)
        gold = ROOT / "data" / f"{gold_name}.jsonl"
        if gold_name == "custom":
            content = payload.get("gold_content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("請選擇自訂標準答案 JSONL 檔")
            gold = temp / "gold.jsonl"
            gold.write_text(content, encoding="utf-8")
        predictions = temp / "predictions.jsonl"
        if payload.get("baseline") is True:
            for line, record in read_records(gold):
                if not isinstance(record.get("raw_text"), str):
                    raise ValueError(f"標準答案第 {line} 行缺少字串 raw_text，無法執行 Baseline")
            with redirect_stdout(io.StringIO()):
                prediction_started = perf_counter()
                baseline_predict(str(gold), str(predictions))
                prediction_seconds = perf_counter() - prediction_started
            label = f"Baseline · {gold_name}"
        else:
            content = payload.get("prediction_content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("請先選擇預測 JSONL 檔")
            predictions.write_text(content, encoding="utf-8")
            label = str(payload.get("prediction_name") or "predictions.jsonl")[:200]
        summary, errors, official_output = analyze(predictions, gold)
        # Temporary upload paths are not useful provenance after this request.
        summary["prediction_file"] = label
        gold_label = str(payload.get("gold_name") or "自訂標準答案")[:200] if gold_name == "custom" else str(gold)
        summary["gold_file"] = gold_label
        for error in errors:
            error["source_file"] = gold_label
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        write_reports(RUNS / run_id, summary, errors, official_output)
    timing = {"total_seconds": perf_counter() - started, "prediction_seconds": prediction_seconds}
    (RUNS / run_id / "run.json").write_text(
        json.dumps({"label": label, "timing": timing}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"summary": summary, "errors": errors, "label": label, "run_id": run_id,
            "timing": timing,
            "downloads": {name: f"/reports/{run_id}/{name}" for name in sorted(DOWNLOADS)}}


class Handler(BaseHTTPRequestHandler):
    def respond(self, value, status=200):
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def send_bytes(self, content, content_type, status=200, attachment=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        if attachment:
            self.send_header("Content-Disposition", f"attachment; filename=download; filename*=UTF-8''{quote(attachment)}")
        self.end_headers()
        self.wfile.write(content)

    def local_request(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            self.respond({"error": "請從本機網址開啟工具"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            self.respond({"error": "不接受其他網站的請求"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/api/info":
                sources = []
                for name in ("tickets", "dev", "history"):
                    path = ROOT / "data" / f"{name}.jsonl"
                    # Opening the workspace must not inspect the dev holdout.
                    count = None if name == "dev" else (sum(1 for _ in read_records(path)) if path.is_file() else 0)
                    sources.append({"name": name, "count": count, "available": path.is_file()})
                self.respond({"sources": sources, "default_output_dir": str(ROOT / "Solution/predictions"),
                              "devices": json.loads((ROOT / "data/devices.json").read_text(encoding="utf-8"))})
            elif url.path == "/api/lookup":
                ids = parse_qs(url.query).get("id", [""])[0].split()
                if not ids or len(ids) > 30:
                    raise ValueError("請輸入 1～30 個 ticket_id，以空白分隔")
                index = build_index([p for name in ("tickets", "dev", "history")
                                     if (p := ROOT / "data" / f"{name}.jsonl").is_file()])
                self.respond({"results": {tid: index.get(tid, []) for tid in ids}})
            elif url.path.startswith("/reports/"):
                match = re.fullmatch(r"/reports/([0-9]{8}-[0-9]{6}-[a-f0-9]{8})/([a-z.]+)", url.path)
                if not match or match[2] not in DOWNLOADS | {"predictions.jsonl"}:
                    self.respond({"error": "找不到報告"}, 404)
                    return
                name = match[2]
                if name == "predictions.jsonl":
                    name = json.loads((RUNS / match[1] / "run.json").read_text(encoding="utf-8"))["filename"]
                self.send_bytes((RUNS / match[1] / match[2]).read_bytes(), "application/octet-stream", attachment=name)
            else:
                files = {"/": ("index.html", "text/html; charset=utf-8"),
                         "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                         "/style.css": ("style.css", "text/css; charset=utf-8")}
                if url.path not in files:
                    self.respond({"error": "找不到頁面"}, 404)
                    return
                name, content_type = files[url.path]
                self.send_bytes((WEB / name).read_bytes(), content_type)
        except FileNotFoundError:
            self.respond({"error": "找不到資料或報告檔案"}, 404)
        except (OSError, ValueError) as exc:
            self.respond({"error": str(exc)}, 400)

    def do_POST(self):
        if not self.local_request():
            return
        if self.path not in ("/api/evaluate", "/api/run"):
            self.respond({"error": "找不到操作"}, 404)
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("請傳送 JSON")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                raise ValueError("檔案總量需介於 1 byte 與 16 MB 之間")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("請傳送 JSON object")
            self.respond(run_experiment(payload, RUNS) if self.path == "/api/run" else evaluate_upload(payload))
        except (OSError, ValueError, UnicodeError) as exc:
            self.respond({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="啟動 SILAB 開發輔助工具 UI")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    server = None
    # A single request handler preserves the official evaluator's stdout capture.
    for port in range(args.port, min(args.port + 20, 65536)):
        try:
            server = HTTPServer(("127.0.0.1", port), Handler)
            break
        except OSError as exc:
            if exc.errno not in (48, 98):
                raise
    if server is None:
        parser.exit(1, "無法取得本機連接埠，請使用 --port 指定其他埠號。\n")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"SILAB 開發輔助工具：{url}\n保留此視窗即可使用；按 Ctrl+C 結束。", flush=True)
    if not args.no_browser:
        threading.Timer(.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n工具已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
