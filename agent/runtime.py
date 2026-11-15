"""One private CPU llama-server per CLI run; no remote endpoint or auto-download."""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .process import ChildGuard

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = ROOT / "agent_assets/runtime" / ("llama-server.exe" if os.name == "nt" else "llama-server")


class LocalModel:
    def __init__(self, model, server, deadline, threads=6, context=4096):
        self.model, self.server = Path(model), Path(server)
        self.deadline, self.threads = deadline, threads
        self.context = context
        self.process = None
        self.log = None
        self.metrics = []
        self.guard = None
        # Explicitly bypass HTTP_PROXY; only the subprocess loopback address is used.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("超過整體執行時間限制")
        return seconds

    def request(self, route, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.url + route, data=data,
                                         headers={"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=self.remaining()) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            if route == "/health":
                raise
            raise RuntimeError("本機推論錯誤：" + exc.read(1600).decode("utf-8", errors="replace")) from exc
        self.remaining()
        return result

    def __enter__(self):
        if not self.model.is_file() or not self.server.is_file():
            raise FileNotFoundError("缺少本機模型或 llama-server；請先執行 agent/setup.py，推論不會自行下載")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log = tempfile.TemporaryFile(mode="w+b")
        command = [str(self.server.resolve()), "-m", str(self.model.resolve()),
                   "--host", "127.0.0.1", "--port", str(port), "-ngl", "0",
                   "-t", str(self.threads), "-tb", str(self.threads),
                   "-c", str(self.context), "-np", "1", "--no-warmup",
                   "--jinja", "--reasoning", "off", "--reasoning-budget", "0"]
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=self.log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.guard = ChildGuard(self.process)
            while True:
                self.remaining()
                if self.process.poll() is not None:
                    self.log.seek(0)
                    detail = self.log.read().decode("utf-8", errors="replace")[-1800:]
                    raise RuntimeError("llama-server 啟動失敗：" + detail)
                try:
                    self.request("/health")
                    return self
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(min(0.05, self.remaining()))
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def decide(self, messages, schema, max_tokens=180):
        response = self.request("/v1/chat/completions", {
            "model": "local", "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "action", "strict": True, "schema": schema}},
        })
        self.metrics.append({"usage": response.get("usage"), "timings": response.get("timings")})
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("模型輸出遭 token 上限截斷")
        return json.loads(choice["message"]["content"])

    def __exit__(self, *unused):
        if self.process is not None:
            self.metrics.append({"peak_server_rss_bytes": ChildGuard.peak_working_set(self.process),
                                 "peak_cli_rss_bytes": ChildGuard.peak_working_set(),
                                 "rss_note": "各程序峰值，兩者相加是峰值RSS上界；不含OS與其他程序"})
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=2)
        if self.guard is not None:
            self.guard.close()
        if self.log is not None:
            self.log.close()
