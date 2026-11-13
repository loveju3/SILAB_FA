#!/usr/bin/env python3
"""Explicit online setup. Inference itself never downloads weights or binaries."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "agent_assets"
MODEL_REVISION = "dcb19155b962dbb6389f4691a982043a8e651022"
MODEL_NAME = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"
MODEL_SHA256 = "72c5c3cb38fa32d5256e2fe30d03e7a64c6c79e668ad84057e3bd66e250b24fb"
RUNTIME_TAG = "b12366"
RUNTIME_SHA256 = "b6b83e46546c04af52a700ea09638201493448cf52de207b2efefc1cecab625e"
MODEL_URL = ("https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/"
             + MODEL_REVISION + "/" + MODEL_NAME)
RUNTIME_URL = ("https://github.com/ggml-org/llama.cpp/releases/download/"
               + RUNTIME_TAG + "/llama-" + RUNTIME_TAG + "-bin-win-cpu-x64.zip")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, target, expected=None):
    if target.exists() and (expected is None or sha256(target) == expected):
        return
    partial = target.with_suffix(target.suffix + ".part")
    print("Downloading " + target.name, flush=True)
    with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if expected and sha256(partial) != expected:
        raise ValueError("SHA256 mismatch: " + target.name)
    partial.replace(target)


def download_model():
    """Resume a contiguous prefix, fetch six validated byte ranges, then hash."""
    target = ASSETS / MODEL_NAME
    if target.exists() and sha256(target) == MODEL_SHA256:
        return
    partial = target.with_suffix(".gguf.part")
    if target.exists() and not partial.exists():
        target.replace(partial)
    total = 1282439584
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError("本機模型大小超出預期，請檢查 agent_assets 中的檔案")
    step = max(1, (total - offset + 5) // 6)
    ranges = [(begin, min(total - 1, begin + step - 1)) for begin in range(offset, total, step)]
    def fetch(bounds):
        begin, end = bounds
        path = ASSETS / f"model-range-{begin}-{end}.part"
        if path.exists() and path.stat().st_size == end - begin + 1:
            return path
        current = path.stat().st_size if path.exists() else 0
        while current < end - begin + 1:
            first = begin + current
            last = min(end, first + 2 * 1024 * 1024 - 1)
            chunk = path.with_suffix(".chunk")
            headers = path.with_suffix(".headers")
            subprocess.run(["curl.exe", "-sS", "-L", "--fail", "--retry", "3",
                            "--max-time", "90", "--speed-limit", "1024", "--speed-time", "10",
                            "--range", f"{first}-{last}", "-D", str(headers),
                            "-o", str(chunk), MODEL_URL], check=True, timeout=390)
            if f"content-range: bytes {first}-{last}/{total}" not in headers.read_text().lower():
                raise ValueError("伺服器未提供要求的模型分段")
            if chunk.stat().st_size != last - first + 1:
                raise ValueError("模型小分段長度錯誤")
            with path.open("ab") as output, chunk.open("rb") as source:
                shutil.copyfileobj(source, output)
            current += last - first + 1
            chunk.unlink()
            headers.unlink()
        if path.stat().st_size != end - begin + 1:
            raise ValueError("模型分段長度錯誤")
        print(f"Downloaded bytes {begin}-{end}", flush=True)
        return path
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(fetch, ranges))
    with partial.open("ab") as output:
        for path in parts:
            with path.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    if sha256(partial) != MODEL_SHA256:
        raise ValueError("模型 SHA256 不符；不可使用，請重新下載")
    partial.replace(target)
    for path in parts:
        path.unlink()


def main():
    if sys.platform != "win32":
        raise SystemExit("此 setup 下載 Windows CPU runtime；Linux 請自行安裝同版 llama.cpp 並用 --server 指定")
    ASSETS.mkdir(exist_ok=True)
    model = ASSETS / MODEL_NAME
    archive = ASSETS / ("llama-" + RUNTIME_TAG + "-bin-win-cpu-x64.zip")
    download_model()
    download(RUNTIME_URL, archive, RUNTIME_SHA256)
    runtime = ASSETS / "runtime"
    runtime.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            if not (runtime / entry.filename).resolve().is_relative_to(runtime.resolve()):
                raise ValueError("不安全的壓縮檔路徑")
        source.extractall(runtime)
    manifest = {"model_url": MODEL_URL, "model_revision": MODEL_REVISION,
                "model_sha256": sha256(model), "runtime_url": RUNTIME_URL,
                "runtime_tag": RUNTIME_TAG, "runtime_archive_sha256": sha256(archive)}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready: " + str(model))


if __name__ == "__main__":
    main()
