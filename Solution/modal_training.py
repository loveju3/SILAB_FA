"""Optional Modal GPU runner. Upload only the selected, explicitly authorized history.

The image includes a code snapshot, public tables and the pinned base encoder;
never mount the project's data/reports directories or credential files.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import sys


def remote_python_version():
    """Serialized functions require the same Python minor version as the client."""
    version = f'{sys.version_info.major}.{sys.version_info.minor}'
    if version not in ('3.12', '3.13'):
        raise ValueError(f'目前固定的訓練依賴支援 Python 3.12／3.13；此訓練環境為 {version}，請改用相容的 Python')
    return version


def remote_train(history_content, settings_dict, run_id):
    """Stream real training events, then a chunked checkpoint archive."""
    import queue
    import threading
    import sys
    sys.path[:0] = ['/workspace/Solution', '/workspace']
    from training.finetuning import run_comparison
    from training.settings import FineTuneSettings

    with tempfile.TemporaryDirectory(prefix='silab-') as temporary:
        directory = Path(temporary) / run_id
        directory.mkdir()
        history = directory / 'history.jsonl'
        history.write_text(history_content, encoding='utf-8')
        messages = queue.Queue()

        def train():
            try:
                run_comparison(history, directory / 'models', FineTuneSettings.from_dict(settings_dict),
                               lambda event, **values: messages.put(('event', event, values)), device='cuda')
            except Exception as exc:
                messages.put(('error', f'{type(exc).__name__}: {exc}'))
            finally:
                messages.put(('done',))

        thread = threading.Thread(target=train, daemon=True)
        thread.start()
        while True:
            message = messages.get()
            if message[0] == 'done':
                break
            if message[0] == 'error':
                raise RuntimeError(message[1])
            yield message
        thread.join()
        yield ('event', 'stage', {'message': 'Modal 訓練完成；封裝與下載模型、評估紀錄'})
        archive = directory / 'models.zip'
        # Modal-mounted files can have Unix-epoch mtimes. copytree preserves
        # those dates, but ZIP can only represent 1980 onward. Clamp timestamps
        # without changing the checkpoint bytes or their integrity hashes.
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED, strict_timestamps=False) as bundle:
            for path in sorted((directory / 'models').rglob('*')):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(directory)))
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
                yield ('artifact', chunk)
        yield ('complete', digest.hexdigest())


def restore_archive(archive, directory):
    """Extract only regular model files within this run; reject malformed bundles."""
    directory = Path(directory).resolve()
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        if sum(info.file_size for info in bundle.infolist()) > 2 * 1024**3:
            raise ValueError('Modal 模型封裝超過 2 GiB')
        for info in bundle.infolist():
            name = info.filename
            target = (directory / name).resolve()
            if (name in seen or '\\' in name or not name.startswith('models/')
                    or not target.is_relative_to(directory / 'models')
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Modal 模型封裝包含不合法路徑')
            seen.add(name)
        bundle.extractall(directory)
    path = directory / 'models/comparison.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if len(report.get('results', [])) != 4:
        raise ValueError('Modal 比較結果不完整')
    for result in report['results']:
        remote = result['model_path'].replace('\\', '/')
        relative = remote.split('/models/', 1)[-1]
        local = (directory / 'models' / relative).resolve()
        if not local.is_relative_to(directory / 'models') or not local.exists():
            raise ValueError('Modal 回傳模型路徑無效')
        result['model_path'] = str(local)
    report['training_backend'] = 'modal'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run_modal(directory, settings, notify, modal_settings=None):
    from Solution.training.settings import ModalSettings
    from Solution.triage.io import atomic_write
    resources = ModalSettings.load() if modal_settings is None else ModalSettings.from_dict(modal_settings)
    python_version = remote_python_version()
    try:
        import modal
    except ImportError as exc:
        raise ValueError(f'訓練 Python 缺少 Modal 或其依賴（{exc}）。目前環境：{sys.executable}；'
                         '請以此 Python 執行 -m pip install -r Solution/requirements-modal.txt。'
                         '套件載入成功後，再用 -m modal setup 登入；此錯誤發生在雲端連線之前') from exc
    directory = Path(directory)
    atomic_write(directory / 'modal-runtime.json', json.dumps(
        {'resources': asdict(resources), 'modal_sdk': modal.__version__, 'python_version': python_version,
         'client_python_version': sys.version, 'python_executable': sys.executable},
        ensure_ascii=False, indent=2) + '\n')
    root = Path(__file__).resolve().parents[1]
    snapshot = directory / 'source'
    dependencies = []
    for name in ('requirements-ml.txt', 'requirements-embedding.txt'):
        dependencies.extend(line.strip() for line in (snapshot / 'Solution' / name).read_text().splitlines()
                            if line.strip() and not line.startswith(('#', '-r')))
    image = (modal.Image.debian_slim(python_version=python_version).pip_install(*dependencies)
             .env({'PYTHONPATH': '/workspace:/workspace/Solution', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false'})
             .add_local_dir(snapshot, '/workspace', ignore=['**/__pycache__/**', '**/*.pyc'])
             .add_local_dir(root / 'Solution/models/bge-small-zh-v1.5',
                            '/workspace/Solution/models/bge-small-zh-v1.5', ignore=['.cache/**']))
    app = modal.App(resources.app_name)
    train = app.function(image=image, gpu=resources.gpu, cpu=resources.cpu, memory=resources.memory_mb,
                         timeout=resources.timeout_seconds, serialized=True)(remote_train)
    notify('stage', message=f'連線 Modal · {resources.gpu}；準備 GPU 環境並傳送本次授權的 history（首次建置較久）')
    digest = hashlib.sha256()
    complete = None
    archive = directory / 'modal-models.zip.part'
    with app.run(), archive.open('wb') as stream:
        for message in train.remote_gen((directory / 'data/history.jsonl').read_text(encoding='utf-8'),
                                        asdict(settings), directory.name):
            if message[0] == 'event':
                notify(message[1], **message[2])
            elif message[0] == 'artifact':
                digest.update(message[1])
                stream.write(message[1])
            elif message[0] == 'complete':
                complete = message[1]
    if complete is None or digest.hexdigest() != complete:
        raise ValueError('Modal 模型下載不完整；保留本機紀錄，請重新建立工作')
    restore_archive(archive, directory)
    archive.unlink()
    notify('stage', message='Modal 模型與評估報告已下載至本機')
