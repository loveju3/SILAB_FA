"""Read saved experiment aggregates only; never open predictions or error rows."""
import math
from pathlib import Path
import re

from Solution.triage.io import load_json


def number(value, *, score=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        return None
    return value if not score or value <= 1 else None


def list_experiments(root: Path):
    experiments, skipped = [], 0
    for folder in sorted(root.glob('*'), reverse=True):
        if not re.fullmatch(r'\d{8}-\d{6}-[a-f0-9]{8}', folder.name) or folder.is_symlink() or not folder.is_dir():
            continue
        metadata_path, summary_path = folder / 'run.json', folder / 'summary.json'
        if not metadata_path.is_file() or metadata_path.is_symlink() or summary_path.is_symlink():
            continue
        try:
            metadata = load_json(metadata_path)
            summary = load_json(summary_path) if summary_path.is_file() else {}
            if not isinstance(metadata, dict) or not isinstance(summary, dict):
                raise ValueError('報告須為 JSON object')
            metrics = summary.get('metrics') or {}
            timing = metadata.get('timing') or {}
            if not isinstance(metrics, dict) or not isinstance(timing, dict):
                raise ValueError('指標或耗時格式不正確')
            digest = metadata.get('dataset_sha256')
            digest = digest if isinstance(digest, str) and re.fullmatch(r'[a-f0-9]{64}', digest) else None
            dataset = metadata.get('dataset') or summary.get('gold_file') or '未記錄'
            count = metadata.get('count', summary.get('gold_count'))
            experiments.append({
                'run_id': folder.name, 'label': metadata.get('filename') or metadata.get('label') or folder.name,
                'created_at': metadata.get('created_at'),
                'dataset': Path(str(dataset).replace('\\', '/')).name,
                'dataset_sha256': digest,
                'count': count if type(count) is int and count >= 0 else None,
                'method': metadata.get('method') or '未記錄',
                'limit': metadata.get('limit') or '未記錄',
                'model_path': metadata.get('model_path'),
                'total': number(summary.get('total'), score=True),
                'metrics': {key: number(metrics.get(key), score=True) for key in ('category', 'device', 'urgency', 'duplicate')},
                'timing': {key: number(timing.get(key)) for key in ('total_seconds', 'prediction_seconds')},
            })
        except (OSError, ValueError, TypeError):
            # A partial/corrupt report must not hide other completed experiments.
            skipped += 1
    return {'experiments': experiments, 'skipped': skipped}
