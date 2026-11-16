#!/usr/bin/env python3
"""Offline A/B/C CLI: python3 Solution/predict.py INPUT.jsonl OUTPUT.jsonl."""
from time import perf_counter

PROCESS_START = perf_counter()

import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys

from triage.factory import METHODS, build_classifier
from triage.devices import DeviceResolver
from triage.io import ROOT, atomic_write, file_sha256, load_json, read_tickets
from triage.metrics import latency_summary, peak_rss_bytes
from triage.pipeline import predict

SOLUTION = Path(__file__).resolve().parent
FINETUNED = ROOT / "transfer/finetuned"
MEMORY_LIMITS = {"2g": 2 * 1024 ** 3}


def choose_method(args):
    """Default to the fine-tuned model when installed; 2 GB mode uses scheme A."""
    if args.mem_limit:
        if args.method not in (None, "rules"):
            raise ValueError("--mem-limit 2g 固定使用方案 A（rules），不可與其他 --method 併用")
        apply_memory_limit(MEMORY_LIMITS[args.mem_limit])
        return "rules", None
    if args.method:
        if args.method == "bge-finetuned" and args.model_path is None:
            return args.method, FINETUNED
        return args.method, args.model_path
    if (FINETUNED / "model.json").is_file():
        return "bge-finetuned", FINETUNED
    print("警告：找不到 transfer/finetuned 微調模型（請先執行 python3 install.py），改用方案 A 規則版。",
          file=sys.stderr)
    return "rules", None


def apply_memory_limit(limit):
    """Best-effort process cap on Linux; budget.md documents the Docker 2 GiB verification."""
    if not sys.platform.startswith("linux"):
        return
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    if soft == resource.RLIM_INFINITY or soft > limit:
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="離線分流：規則／TF-IDF／BGE 線性分類；不讀取輸入標籤")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metrics", type=Path, help="額外寫入量測與逐筆耗時 JSON")
    parser.add_argument("--method", choices=METHODS,
                        help="預設：已安裝 transfer/finetuned 時用 bge-finetuned，否則 rules")
    parser.add_argument("--model-path", type=Path,
                        help="模型路徑；bge-finetuned 預設 transfer/finetuned，其他預設 Solution/models/<method>.joblib")
    parser.add_argument("--mem-limit", choices=sorted(MEMORY_LIMITS),
                        help="2 GB 演練模式：固定使用方案 A（見 budget.md）")
    args = parser.parse_args(argv)
    try:
        args.method, args.model_path = choose_method(args)
        if args.input.resolve() == args.output.resolve():
            raise ValueError("輸出不可覆蓋輸入")
        if args.metrics and args.metrics.resolve() in {args.input.resolve(), args.output.resolve()}:
            raise ValueError("量測檔不可覆蓋輸入或預測")
        paths = [ROOT / "data/devices.json", ROOT / "data/error_codes.json",
                 SOLUTION / "config/device_aliases.json", SOLUTION / "config/category_rules.json"]
        devices, codes, aliases, rules = [load_json(p) for p in paths]
        classifier, model_info, model_paths = build_classifier(args.method, codes, rules, args.model_path)
        if args.output.resolve() in {p.resolve() for p in model_paths} or (
                args.metrics and args.metrics.resolve() in {p.resolve() for p in model_paths}):
            raise ValueError("輸出不可覆蓋模型或模型資訊")
        tickets = read_tickets(args.input)
        loaded = perf_counter()
        predictions, timings = predict(tickets, DeviceResolver(devices, codes, aliases),
                                       classifier)
        inferred = perf_counter()
        atomic_write(args.output, "".join(json.dumps(p.to_dict(), ensure_ascii=False) + "\n"
                                          for p in predictions))
        written = perf_counter()
        if args.metrics:
            latency = latency_summary(timings)
            if getattr(classifier, "timing_description", None):
                latency["method"] = classifier.timing_description
            artifacts = [*paths, Path(__file__), *sorted((SOLUTION / "triage").glob("*.py")), *model_paths]
            report = {
                **model_info, "python": sys.version, "platform": platform.platform(),
                "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "artifacts_sha256": {str(p.resolve()): file_sha256(p) for p in artifacts},
                "load_seconds": loaded - PROCESS_START,
                "pipeline_seconds": inferred - loaded,
                "serialization_write_seconds": written - inferred,
                "elapsed_to_output_seconds": written - PROCESS_START,
                "timing_scope": "starts at first Python statement; external runner measures full process startup and metrics write",
                "peak_rss_bytes": peak_rss_bytes(), "latency": latency, "tickets": timings,
            }
            atomic_write(args.metrics, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
