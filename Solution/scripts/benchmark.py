#!/usr/bin/env python3
"""Run independent processes; save artifacts and external wall-clock timings."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/tickets.jsonl")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--method", choices=("rules", "tfidf-lr", "tfidf-svc", "bge-lr", "bge-svc", "bge-finetuned"), default="rules")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Solution/experiments/runs")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必須 >= 1")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = args.output_dir / run_id
    directory.mkdir(parents=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True)
    result = {"run_id": run_id, "git_head": git.stdout.strip(), "git_status": dirty.stdout,
              "platform": platform.platform(), "python": sys.version,
              "method": "fresh Python process each run; no explicit warm-up; OS file cache not cleared",
              "runs": []}
    for i in range(args.runs):
        output, metrics = directory / f"predictions-{i}.jsonl", directory / f"metrics-{i}.json"
        command = [sys.executable, str(ROOT / "Solution/predict.py"), str(args.input.resolve()),
                   str(output), "--metrics", str(metrics), "--method", args.method]
        start = perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, timeout=480)
        elapsed = perf_counter() - start
        result["runs"].append({"command": command, "external_wall_seconds": elapsed,
                               "exit_code": process.returncode, "stderr": process.stderr})
        (directory / "experiment.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if process.returncode:
            raise RuntimeError(process.stderr)
    print(directory)


if __name__ == "__main__":
    main()
