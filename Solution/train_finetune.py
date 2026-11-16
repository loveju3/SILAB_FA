#!/usr/bin/env python3
"""Offline worker for persistent UI training jobs; no dev input option."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from time import perf_counter
import traceback

SOLUTION = Path(__file__).resolve().parent
ROOT = SOLUTION.parent
sys.path.insert(0, str(ROOT))
from Solution.triage.io import atomic_write
from Solution.training.settings import FineTuneSettings, LinearSettings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args()
    # The inherited descriptor holds the cross-server training lock until exit.
    if args.lock_fd >= 0:
        os.fstat(args.lock_fd)
    directory = args.run_dir.resolve()
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    began = perf_counter()
    progress = {"percent": 0, "label": "準備訓練", "determinate": False}
    def status(state, message):
        atomic_write(directory / 'status.json', json.dumps({"state": state, "message": message, "pid": os.getpid(),
                     "total_seconds": perf_counter() - began, "progress": progress}, ensure_ascii=False) + '\n')
    def notify(event, **values):
        if event == 'batch':
            offset = 25 if values['variant'] == 'frozen-neural' else 55
            percent = offset + 30 * ((values['epoch'] - 1 + values['completed'] / values['total']) / values['epochs'])
            progress.update(percent=max(progress['percent'], percent), determinate=True)
        elif event == 'baseline':
            progress.update(percent=15 if values['variant'] == 'bge-lr' else 25, determinate=True)
        elif event == 'completed':
            progress.update(percent=100, determinate=True)
        progress['label'] = values.get('message') or f"{values.get('variant', '')} {event} {values.get('epoch', '')}".strip()
        record = {"event": event, "at": datetime.now(timezone.utc).isoformat(), **values}
        with (directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            stream.flush()
        status('running', values.get('message') or f"{values.get('variant', '')} {event} {values.get('epoch', '')}".strip())
    try:
        status('running', '檢查 history 與訓練設定')
        settings = FineTuneSettings.from_dict(config['settings'])
        history = directory / 'data/history.jsonl'
        if config['kind'] == 'bge-finetune':
            backend = config.get('backend', 'cpu')
            if backend == 'modal':
                if config.get('cloud_upload_confirmed') is not True:
                    raise ValueError('未授權本次 history 雲端上傳')
                from modal_training import run_modal
                run_modal(directory, settings, notify, modal_settings=config.get('modal_settings'))
            else:
                import torch
                if backend == 'gpu' and not torch.cuda.is_available():
                    raise ValueError('CUDA GPU 不可用；請安裝 CUDA 版 PyTorch 與驅動，或改選 CPU')
                from training.finetuning import run_comparison
                run_comparison(history, directory / 'models', settings, notify, device='cuda' if backend == 'gpu' else 'cpu')
        else:
            script, option = ('train.py', '--output-dir') if config['kind'] == 'tfidf' else ('train_embedding.py', '--model-dir')
            if config['kind'] == 'bge-frozen':
                shutil.copytree(SOLUTION / 'models/bge-small-zh-v1.5', directory / 'models/bge-small-zh-v1.5',
                                ignore=shutil.ignore_patterns('.cache'))
            linear = LinearSettings.from_dict(config.get('linear_settings'), config['kind'])
            command = [sys.executable, str(SOLUTION / script), '--history', str(history), option, str(directory / 'models'),
                       '--report-dir', str(directory / 'reports'),
                       '--settings-json', json.dumps(linear.to_dict(config['kind']))]
            notify('stage', message='執行既有四候選比較，完成後保存到本次時間戳目錄', command=command)
            subprocess.run(command, check=True, **({'pass_fds': (args.lock_fd,)} if args.lock_fd >= 0 else {}))
            reports = list((directory / 'reports').glob('*/training.json'))
            if len(reports) != 1:
                raise ValueError('訓練報告缺失或不唯一，不能標記為完成')
            manifest = json.loads(reports[0].read_text(encoding='utf-8'))
            comparison = {"refit_full_history": True, "split": manifest['split'],
                          "linear_settings": linear.to_dict(config['kind']),
                          "provenance": manifest.get('provenance', ''), "results": [
                {"variant": r['name'], "method": r['method'],
                 "model_path": str(directory / 'models' / (r['name'] + '.joblib')),
                 "validation": {key: r[key] for key in ('category_macro_f1', 'six_class_macro_f1', 'unknown_recall')},
                 "pipeline_total": r['pipeline_total'], "pipeline_metrics": r['pipeline_metrics'],
                 "selected": r['name'] in manifest['selected'].values()}
                for r in manifest['candidates']]}
            atomic_write(directory / 'models/comparison.json', json.dumps(comparison, ensure_ascii=False, indent=2) + '\n')
        notify('completed', message='訓練完成，模型與紀錄已保存')
        status('succeeded', '完成；模型位於本次資料夾的 models/，未覆寫既有預設模型')
        return 0
    except Exception as exc:
        traceback.print_exc()
        status('failed', f'{type(exc).__name__}: {exc}；詳見 worker.log')
        return 2
    finally:
        if args.lock_fd >= 0:
            os.close(args.lock_fd)


if __name__ == '__main__':
    sys.exit(main())
