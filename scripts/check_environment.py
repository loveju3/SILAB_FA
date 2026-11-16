"""Check local prerequisites without reading any ticket data.

`ready` covers what the default two-argument predict.py needs (packages, public
tables, fine-tuned model). Optional extras (other B/C models, Agent runtime,
ticket files, Git) are listed under `optional_missing` and do not fail the check.
"""
import hashlib
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime import DEFAULT_SERVER
from agent.setup import MODEL_NAME, MODEL_SHA256


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    problems, optional, versions = [], [], {}
    if sys.version_info[:2] not in {(3, 12), (3, 13)}:
        problems.append('Use Python 3.12 or 3.13 for the pinned test environment')
    if sys.platform == 'win32' and not sys.flags.utf8_mode:
        problems.append('Windows: set $env:PYTHONUTF8 = "1" before invoking Python')
    if sys.platform == 'darwin' and platform.machine().lower() not in {'arm64', 'aarch64'}:
        problems.append('macOS requires native Apple Silicon (arm64) Python')
    expected_versions = {'torch': '2.9.0', 'transformers': '4.57.1', 'sklearn': '1.7.2',
                         'safetensors': '0.6.2', 'numpy': '2.3.4', 'scipy': '1.16.3'}
    for name, expected in expected_versions.items():
        try:
            module = importlib.import_module(name)
            versions[name] = module.__version__
            if versions[name].split('+')[0] != expected:
                problems.append(f'{name}: expected {expected}, got {versions[name]}')
        except (ImportError, OSError) as exc:
            problems.append(f'{name}: {exc}')
    for name in ('data/devices.json', 'data/error_codes.json'):
        if not (ROOT / name).is_file():
            problems.append('Missing: ' + name)
    for name in ('data/tickets.jsonl', 'data/history.jsonl', 'data/dev.jsonl'):
        if not (ROOT / name).is_file():
            optional.append('Missing: ' + name)
    for method in ('tfidf-lr', 'tfidf-svc', 'bge-lr', 'bge-svc'):
        folder = ROOT/'Solution/models'
        if not (folder/(method+'.joblib')).is_file():
            optional.append(f'Optional model not installed: Solution/models/{method}.joblib')
            continue
        try:
            metadata = json.loads((folder/(method+'.json')).read_text(encoding='utf-8'))
            if digest(folder/(method+'.joblib')) != metadata['artifact_sha256']:
                raise ValueError('classifier hash mismatch')
        except (OSError, ValueError, KeyError) as exc:
            problems.append(method + ': ' + str(exc))
    for folder in (ROOT/'Solution/models/bge-small-zh-v1.5', ROOT/'transfer/finetuned/encoder'):
        if 'Solution' in folder.parts and not folder.exists():
            optional.append('Optional encoder not installed: Solution/models/bge-small-zh-v1.5')
            continue
        try:
            manifest = json.loads((folder/'encoder_manifest.json').read_text(encoding='utf-8'))
            for name, expected in manifest['files_sha256'].items():
                if Path(name).name != name or digest(folder/name) != expected:
                    raise ValueError('encoder hash mismatch: ' + name)
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f'{folder.relative_to(ROOT)}: {exc}')
    try:
        folder = ROOT/'transfer/finetuned'
        metadata = json.loads((folder/'model.json').read_text(encoding='utf-8'))
        if metadata['format'] != 'bge-neural-v1' or digest(folder/'heads.safetensors') != metadata['heads_sha256']:
            raise ValueError('fine-tuned model format or head hash mismatch')
    except (OSError, ValueError, KeyError) as exc:
        problems.append('transfer/finetuned: ' + str(exc))
    if not (ROOT/'agent_assets'/MODEL_NAME).is_file():
        optional.append('Agent not installed: run python3 install.py --with-agent')
    else:
        try:
            if digest(ROOT/'agent_assets'/MODEL_NAME) != MODEL_SHA256:
                raise ValueError('Qwen model hash mismatch')
            result = subprocess.run([str(DEFAULT_SERVER), '--version'], capture_output=True, timeout=15)
            if result.returncode:
                raise ValueError(f'CPU runtime --version exit {result.returncode}')
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            problems.append('Agent: ' + str(exc))
    try:
        subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        optional.append('Git checkout required by Evals reports: ' + str(exc))
    print(json.dumps({'ready': not problems, 'python': sys.version, 'platform': platform.platform(),
                      'packages': versions, 'problems': problems, 'optional_missing': optional,
                      'scope': 'Prerequisites and asset hashes only; no dev reads, inference accuracy or performance verdict'},
                     ensure_ascii=False, indent=2))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
