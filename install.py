#!/usr/bin/env python3
"""One-command setup: .venv, pinned CPU packages, fine-tuned model and a smoke test.

    python3 install.py                 # Part 1: environment + fine-tuned BGE model
    python3 install.py --with-agent    # also Part 2: Qwen3-1.7B GGUF + llama.cpp (~1.3 GB)
    python3 install.py --check         # verify only; no installs, no network

Works on macOS Apple Silicon, Windows x64 and Linux x64/arm64. This file must stay
runnable by an old system Python (e.g. macOS /usr/bin/python3 3.9): it looks for
Python 3.12/3.13 itself and builds the .venv with that interpreter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "Solution/requirements-embedding.txt"
MODEL = ROOT / "transfer/finetuned"
SUPPORTED = ((3, 12), (3, 13))
TORCH = "torch==2.9.0"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"

MODEL_ZIP = "SILAB_FA_bge_finetuned_20261115-082422.zip"
MODEL_ZIP_SHA256 = "4e721a8722f57720ed87c1cf02274ff1b11e3c43f648044b19843e279d59aa3f"
MODEL_ZIP_URL = "https://github.com/loveju3/SILAB_FA/releases/download/models-20261115/" + MODEL_ZIP
DATA_FILES = ("devices.json", "error_codes.json", "tickets.jsonl", "history.jsonl", "dev.jsonl")
REQUIRED_DATA = ("devices.json", "error_codes.json")


class SetupError(Exception):
    """A failure with a message the reader can act on."""


def say(message=""):
    print(message, flush=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(command, **kwargs):
    env = dict(os.environ, PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    subprocess.run([str(part) for part in command], cwd=str(ROOT), check=True, env=env, **kwargs)


# ---------------------------------------------------------------- platform / Python

def check_platform():
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine != "arm64":
        if not machine.startswith("x86"):
            raise SetupError("無法辨識的 Mac 架構：" + machine)
        raise SetupError("Intel Mac 沒有 torch 2.9 套件，無法執行微調模型。"
                         "請改用 Apple Silicon Mac、Windows x64 或 Linux。")
    if sys.platform == "win32" and machine not in ("amd64", "x86_64"):
        raise SetupError("Windows 只支援 x64（目前是 %s）" % machine)
    if sys.platform.startswith("linux") and machine not in ("x86_64", "amd64", "aarch64", "arm64"):
        raise SetupError("Linux 只支援 x64／arm64（目前是 %s）" % machine)


def python_version(executable):
    try:
        output = subprocess.run([str(executable), "-c",
                                 "import sys, platform; print(sys.version_info[0], sys.version_info[1], platform.machine())"],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    parts = output.stdout.split()
    if output.returncode or len(parts) != 3:
        return None
    return (int(parts[0]), int(parts[1])), parts[2].lower()


def suitable(executable):
    found = python_version(executable)
    if not found or found[0] not in SUPPORTED:
        return False
    # Rosetta / 32-bit interpreters would install the wrong wheels.
    return not (sys.platform == "darwin" and found[1] != "arm64")


def find_python():
    """Return a command (list) for Python 3.12/3.13, preferring the current interpreter."""
    if suitable(sys.executable):
        return [sys.executable]
    candidates = []
    if os.name == "nt":
        candidates += [["py", "-3.13"], ["py", "-3.12"]]
    candidates += [["python3.13"], ["python3.12"]]
    if sys.platform == "darwin":
        candidates += [["/opt/homebrew/bin/python3.13"], ["/opt/homebrew/bin/python3.12"],
                       ["/Library/Frameworks/Python.framework/Versions/3.13/bin/python3"],
                       ["/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"]]
    for command in candidates:
        executable = shutil.which(command[0]) or (command[0] if Path(command[0]).is_file() else None)
        if not executable:
            continue
        if len(command) > 1:  # Windows py launcher: resolve the real interpreter path
            try:
                path = subprocess.run([executable, *command[1:], "-c", "import sys; print(sys.executable)"],
                                      capture_output=True, text=True, timeout=30).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                continue
            executable = path or None
        if executable and suitable(executable):
            return [executable]
    return None


def no_python_message():
    hints = {
        "darwin": "brew install python@3.13，或從 https://www.python.org/downloads/ 安裝 3.13",
        "win32": "winget install Python.Python.3.13，或從 https://www.python.org/downloads/ 安裝（勾選 Add to PATH）",
    }
    hint = hints.get(sys.platform, "sudo apt install python3.12 python3.12-venv，或安裝 uv 後重跑本指令")
    return ("找不到 Python 3.12 或 3.13（目前執行的是 %d.%d）。\n請先安裝：%s"
            % (sys.version_info[0], sys.version_info[1], hint))


# ---------------------------------------------------------------- environment

def create_venv():
    if VENV_PYTHON.is_file() and suitable(VENV_PYTHON):
        say("  沿用既有 .venv")
        return
    if VENV.exists():
        raise SetupError(".venv 已存在但不是 Python 3.12/3.13（或已損毀）。請刪除 .venv 資料夾後重跑")
    python = find_python()
    if python:
        try:
            run(python + ["-m", "venv", VENV])
        except subprocess.CalledProcessError:
            raise SetupError("建立 .venv 失敗。Debian/Ubuntu 請先安裝 python3.12-venv（或 python3.13-venv）")
    elif shutil.which("uv"):
        say("  未找到 Python 3.12/3.13，改由 uv 下載 Python 3.13")
        run(["uv", "venv", "--seed", "--python", "3.13", VENV])
    else:
        raise SetupError(no_python_message())
    if not suitable(VENV_PYTHON):
        raise SetupError(".venv 的 Python 版本不符")


def install_packages():
    say("  首次安裝需下載約 200–500 MB（torch 最大），視網路需數分鐘")
    pip = [VENV_PYTHON, "-m", "pip", "install", "--retries", "2", "--timeout", "60"]
    if sys.platform != "darwin" and platform.machine().lower() in ("x86_64", "amd64"):
        # PyPI's x64 Linux torch bundles CUDA (~GBs); the CPU index keeps it small.
        # macOS and Linux arm64 wheels on PyPI are already CPU builds.
        run(pip + [TORCH, "--index-url", TORCH_CPU_INDEX])
    run(pip + ["-r", REQUIREMENTS])
    run([VENV_PYTHON, "-m", "pip", "check"])


# ---------------------------------------------------------------- model

def validate_model(folder):
    folder = Path(folder)
    info = json.loads((folder / "model.json").read_text(encoding="utf-8"))
    if info.get("format") != "bge-neural-v1" or digest(folder / "heads.safetensors") != info.get("heads_sha256"):
        raise SetupError("微調模型格式或分類頭校驗失敗：" + str(folder))
    encoder = folder / "encoder"
    manifest = json.loads((encoder / "encoder_manifest.json").read_text(encoding="utf-8"))
    files = manifest["files_sha256"]
    required = {"config.json", "model.safetensors", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "vocab.txt"}
    if not required <= set(files):
        raise SetupError("encoder 清單缺少必要檔案")
    for name, expected in files.items():
        if Path(name).name != name or digest(encoder / name) != expected:
            raise SetupError("encoder 校驗失敗：" + name)


def model_ready():
    if not (MODEL / "model.json").is_file():
        return False
    validate_model(MODEL)
    return True


def fetch(url, target):
    """Download with the system curl when present. It uses the OS certificate store;
    python.org Python on macOS has no CA certificates until Install Certificates runs."""
    curl = shutil.which("curl.exe" if os.name == "nt" else "curl")
    if curl:
        subprocess.run([curl, "-fSL", "--retry", "3", "--connect-timeout", "30",
                        "--progress-bar", "-o", str(target), url], check=True)
        return
    try:
        with urllib.request.urlopen(url, timeout=60) as source, Path(target).open("wb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
    except OSError as exc:
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            raise OSError("Python 找不到 SSL 根憑證；python.org 版 Python 請執行"
                          "「/Applications/Python 3.x/Install Certificates.command」")
        raise


def download_model_zip():
    target = ROOT / "transfer" / MODEL_ZIP
    if target.is_file() and digest(target) == MODEL_ZIP_SHA256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    say("  下載 " + MODEL_ZIP_URL)
    try:
        fetch(MODEL_ZIP_URL, partial)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError("模型下載失敗（%s）。可手動下載 %s 放到 repo 根目錄或「下載項目」後重跑，"
                         "或用 --model-zip 指定路徑" % (exc, MODEL_ZIP))
    if digest(partial) != MODEL_ZIP_SHA256:
        partial.unlink()
        raise SetupError("下載的模型 ZIP SHA256 不符，已刪除，請重跑")
    partial.replace(target)
    return target


def find_model_zip(explicit):
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SetupError("找不到 --model-zip 指定的檔案：" + str(path))
        return path
    for path in (ROOT / MODEL_ZIP, ROOT / "transfer" / MODEL_ZIP, Path.home() / "Downloads" / MODEL_ZIP):
        if path.is_file():
            say("  使用本機 ZIP：" + str(path))
            return path
    return download_model_zip()


def extract_model(archive_path):
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=str(MODEL.parent)))
    try:
        with zipfile.ZipFile(str(archive_path)) as archive:
            manifest = json.loads(archive.read("MANIFEST.json"))
            for name, entry in manifest["model_files"].items():
                relative = Path(name)
                if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                    raise SetupError("模型 ZIP 含不合法路徑")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open("finetuned/" + name) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if target.stat().st_size != entry["bytes"] or digest(target) != entry["sha256"]:
                    raise SetupError("模型 ZIP 校驗失敗：" + name)
        validate_model(staging)
        staging.rename(MODEL)
    except (KeyError, zipfile.BadZipFile) as exc:
        raise SetupError("模型 ZIP 格式不符：%s" % exc)
    finally:
        if staging.exists():
            shutil.rmtree(str(staging))


def prepare_model(explicit_zip):
    if model_ready():
        say("  沿用既有 transfer/finetuned（校驗通過）")
        if explicit_zip:
            say("  已有模型，忽略 --model-zip；要換模型請先移開 transfer/finetuned")
        return
    if MODEL.exists():
        raise SetupError("transfer/finetuned 已存在但不完整。請移開或刪除該資料夾後重跑")
    extract_model(find_model_zip(explicit_zip))
    say("  模型已安裝並校驗：transfer/finetuned")


# ---------------------------------------------------------------- self test (runs inside .venv)

def pinned(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r "):
            for item in pinned(Path(path).parent / line[3:].strip()):
                yield item
        else:
            name, version = line.split("==")
            yield name, version


def self_test():
    from importlib import import_module, metadata
    for name, expected in pinned(REQUIREMENTS):
        actual = metadata.version(name)
        if actual.split("+")[0] != expected:
            raise SetupError("%s：預期 %s，實際 %s" % (name, expected, actual))
        import_module({"scikit-learn": "sklearn", "huggingface-hub": "huggingface_hub"}.get(name, name))
    say("  套件版本與載入：OK（Python %s / %s）" % (platform.python_version(), platform.machine()))
    if (MODEL / "model.json").is_file():
        validate_model(MODEL)
        sys.path.insert(0, str(ROOT / "Solution"))
        from triage.neural import NeuralBgeClassifier
        classifier, _ = NeuralBgeClassifier.load(MODEL, {})
        if len(classifier.classify_many(["系統測試：設備發出異常噪音。"]).predictions) != 1:
            raise SetupError("模型推論沒有產生結果")
        say("  微調模型載入與合成文字推論：OK")


# ---------------------------------------------------------------- data / agent

def check_data():
    missing = [name for name in DATA_FILES if not (ROOT / "data" / name).is_file()]
    for name in DATA_FILES:
        say("  %s data/%s" % ("缺少" if name in missing else "OK  ", name))
    return missing


def agent_ready():
    sys.path.insert(0, str(ROOT))
    try:
        from agent.setup import MODEL_NAME
        from agent.runtime import DEFAULT_SERVER
    except Exception:  # noqa: BLE001 - agent package needs Python 3.12+
        return False
    return (ROOT / "agent_assets" / MODEL_NAME).is_file() and Path(DEFAULT_SERVER).is_file()


def check_agent_runtime():
    server = ROOT / "agent_assets/runtime" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    result = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=30)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
        hint = ("；Linux 官方 llama.cpp 需要 glibc 2.38 以上（Ubuntu 24.04+）"
                if "GLIBC" in detail[0] else "")
        raise SetupError("llama-server 無法啟動：%s%s" % (detail[0], hint))
    say("  llama-server 可執行：OK")


# ---------------------------------------------------------------- main

def main(argv=None):
    parser = argparse.ArgumentParser(description="建立 .venv、安裝套件與微調模型，並做推論測試")
    parser.add_argument("--with-agent", action="store_true", help="另外下載第二部分 Agent 的 Qwen 模型與 llama.cpp（約 1.3 GB）")
    parser.add_argument("--model-zip", help="模型 ZIP 路徑；預設依序找 repo 根目錄、下載項目，最後從 GitHub Release 下載")
    parser.add_argument("--check", action="store_true", help="只檢查既有安裝，不安裝也不連網")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    stage = "檢查系統"
    try:
        if args.self_test:
            self_test()
            return 0
        say("== SILAB FA 環境安裝（%s / %s）==" % (platform.system(), platform.machine()))
        check_platform()
        steps = ["建立 .venv（Python 3.12/3.13）", "安裝固定版本 CPU 套件", "準備微調模型 transfer/finetuned",
                 "驗證套件與模型推論", "檢查 data/ 資料檔"]
        if args.with_agent:
            steps.append("下載 Agent 模型與 llama.cpp")
        for index, stage in enumerate(steps, 1):
            say("[%d/%d] %s" % (index, len(steps), stage))
            if index == 1:
                if args.check:
                    if not VENV_PYTHON.is_file():
                        raise SetupError("尚未建立 .venv，請執行 python3 install.py")
                else:
                    create_venv()
            elif index == 2 and not args.check:
                install_packages()
            elif index == 3:
                if args.check:
                    if not model_ready():
                        raise SetupError("尚未安裝 transfer/finetuned，請執行 python3 install.py")
                else:
                    prepare_model(args.model_zip)
            elif index == 4:
                run([VENV_PYTHON, Path(__file__).resolve(), "--self-test"])
            elif index == 5:
                missing = check_data()
                absent = [name for name in REQUIRED_DATA if name in missing]
                if absent:
                    raise SetupError("請把題目附的 %s 放進 data/ 後重跑" % "、".join(absent))
            elif index == 6:
                run([VENV_PYTHON, ROOT / "agent/setup.py"])
                check_agent_runtime()
    except SetupError as exc:
        say("\n安裝結果：FAIL（階段：%s）\n原因：%s" % (stage, exc))
        return 1
    except subprocess.CalledProcessError as exc:
        say("\n安裝結果：FAIL（階段：%s）\n原因：指令失敗（exit %s），請看上方訊息；修正後重跑會沿用已完成的部分"
            % (stage, exc.returncode))
        return 1
    except (OSError, ValueError, KeyError) as exc:
        say("\n安裝結果：FAIL（階段：%s）\n原因：%s" % (stage, exc))
        return 1
    except KeyboardInterrupt:
        say("\n安裝結果：中斷（階段：%s）" % stage)
        return 130
    say("\n安裝結果：PASS")
    if missing:
        say("提醒：data/ 尚缺 %s，相關步驟會略過" % "、".join(missing))
    say("\n接下來（不需要啟用 .venv，腳本會自動使用它）：")
    say("  python3 run_demo.py                       # 一鍵產出預測、dev 評分與示範")
    say("  python3 predict.py data/tickets.jsonl tickets_predictions.jsonl")
    if args.with_agent or agent_ready():
        say('  python3 ask.py "B1 動力室有哪些設備？" --trace')
        say("  python3 patrol.py")
    else:
        say("  第二部分 Agent 需要再執行：python3 install.py --with-agent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
