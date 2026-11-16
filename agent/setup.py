#!/usr/bin/env python3
"""Explicit online setup. Inference itself never downloads weights or binaries."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
import zlib

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
MAC_RUNTIME_SHA256 = "6263965248c6c17b942eee1d91d9f810566d2ed1a7ec884130deb2339baee842"
# Official CPU builds for Linux; same tar layout as macOS (llama-<tag>/llama-server + .so files).
LINUX_RUNTIME_SHA256 = {
    "x64": "60c318605a52ad7b15640cae72d6cb6b84b9737d4322be84f4a8e4bde699c1d7",
    "arm64": "5ee16075f1679e04fce6cef6622cbc1eea586add63cff3c512ba3ef0c981eba1",
}


def runtime_asset(system=None, machine=None):
    system = sys.platform if system is None else system
    machine = platform.machine().lower() if machine is None else machine.lower()
    if system == "win32" and machine in {"amd64", "x86_64"}:
        return RUNTIME_URL, RUNTIME_SHA256
    if system == "darwin" and machine in {"arm64", "aarch64"}:
        name = f"llama-{RUNTIME_TAG}-bin-macos-arm64.tar.gz"
        return f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{name}", MAC_RUNTIME_SHA256
    if system.startswith("linux") and machine in {"x86_64", "amd64", "aarch64", "arm64"}:
        arch = "x64" if machine in {"x86_64", "amd64"} else "arm64"
        name = f"llama-{RUNTIME_TAG}-bin-ubuntu-{arch}.tar.gz"
        return (f"https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_TAG}/{name}",
                LINUX_RUNTIME_SHA256[arch])
    raise ValueError("自動安裝支援 Windows x64、Mac Apple Silicon、Linux x64／arm64")


def extract_runtime(archive, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                if not (runtime / entry.filename).resolve().is_relative_to(runtime.resolve()):
                    raise ValueError("不安全的壓縮檔路徑")
            for entry in source.infolist():
                target = runtime / entry.filename
                # Re-running setup must not overwrite matching DLLs while a
                # local inference process is using them on Windows.
                if not entry.is_dir() and target.is_file() and target.stat().st_size == entry.file_size:
                    if zlib.crc32(target.read_bytes()) & 0xffffffff == entry.CRC:
                        continue
                source.extract(entry, runtime)
    else:
        # macOS/Linux release assets contain a versioned top-level folder. Keep all
        # accompanying dylibs and their relative symlinks beside llama-server.
        with tempfile.TemporaryDirectory(dir=runtime.parent) as temporary:
            with tarfile.open(archive, "r:gz") as source:
                source.extractall(temporary, filter="data")
            folder = Path(temporary) / f"llama-{RUNTIME_TAG}"
            if not (folder / "llama-server").is_file():
                raise ValueError("runtime 壓縮檔缺少 llama-server")
            shutil.copytree(folder, runtime, dirs_exist_ok=True, symlinks=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


CERT_HINT = ("Python 找不到 SSL 根憑證。python.org 版 Python 請執行"
             "「/Applications/Python 3.x/Install Certificates.command」後重跑")


def fetch(url, partial):
    """Prefer the system curl: it uses the OS certificate store, while python.org
    builds on macOS ship without CA certificates until Install Certificates runs."""
    curl = shutil.which("curl.exe" if sys.platform == "win32" else "curl")
    if curl:
        subprocess.run([curl, "-fSL", "--retry", "3", "--connect-timeout", "30",
                        "--progress-bar", "-o", str(partial), url], check=True)
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise ValueError(CERT_HINT) from exc
        raise


def download(url, target, expected=None):
    if target.exists() and (expected is None or sha256(target) == expected):
        return
    partial = target.with_suffix(target.suffix + ".part")
    print("Downloading " + target.name, flush=True)
    fetch(url, partial)
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
            subprocess.run(["curl.exe" if sys.platform == "win32" else "curl", "-sS", "-L", "--fail", "--retry", "3",
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
    runtime_url, runtime_sha256 = runtime_asset()
    if sys.platform == "darwin" and sys.version_info < (3, 12):
        raise SystemExit("macOS 安裝請使用 Python 3.12 或 3.13")
    ASSETS.mkdir(exist_ok=True)
    model = ASSETS / MODEL_NAME
    archive = ASSETS / runtime_url.rsplit("/", 1)[1]
    download_model()
    download(runtime_url, archive, runtime_sha256)
    runtime = ASSETS / "runtime"
    extract_runtime(archive, runtime)
    manifest = {"model_url": MODEL_URL, "model_revision": MODEL_REVISION,
                "model_sha256": sha256(model), "runtime_url": runtime_url,
                "runtime_tag": RUNTIME_TAG, "runtime_archive_sha256": sha256(archive)}
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready: " + str(model))


if __name__ == "__main__":
    main()
