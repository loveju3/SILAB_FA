#!/usr/bin/env python3
"""Host-side Docker verification; mounts only this project and its results."""
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter
import uuid

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="silab-fa:scheme-a")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--probe", action="store_true", help="先執行獨立超限測試")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs 必須 >= 1")
    directory = ROOT / "Solution/experiments/runs" / ("budget-" + uuid.uuid4().hex[:12])
    directory.mkdir(parents=True)
    records = []

    def run(label, script_args):
        name = "silab-fa-" + uuid.uuid4().hex[:16]
        command = ["docker", "run", "--name", name, "--memory=2g", "--memory-swap=2g",
                   "--cpus=2", "--network=none", "--env", "PYTHONDONTWRITEBYTECODE=1",
                   "--env", "OMP_NUM_THREADS=2", "--env", "OPENBLAS_NUM_THREADS=2",
                   "--env", "MKL_NUM_THREADS=2", "--mount", f"type=bind,src={ROOT},dst=/app,readonly",
                   "--mount", f"type=bind,src={directory},dst=/results", "--workdir", "/app",
                   "--entrypoint", "python3", args.image, "Solution/scripts/check_limits.py", *script_args]
        started = perf_counter()
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=600)
            # Select fields explicitly: never export image environment variables.
            template = '{"state":{{json .State}},"memory":{{.HostConfig.Memory}},"memory_swap":{{.HostConfig.MemorySwap}},"nano_cpus":{{.HostConfig.NanoCpus}},"network":{{json .HostConfig.NetworkMode}},"image_id":{{json .Image}}}'
            inspected = subprocess.run(["docker", "inspect", "--format", template, name],
                                       capture_output=True, text=True, check=True)
            record = {"label": label, "command": command, "exit_code": process.returncode,
                      "docker_wall_seconds": perf_counter() - started, "stdout": process.stdout,
                      "stderr": process.stderr, "container": json.loads(inspected.stdout)}
            records.append(record)
            (directory / "docker_runs.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
            return record
        finally:
            subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True)

    if args.probe:
        record = run("probe", ["--probe"])
        if record["exit_code"] != 137 or not record["container"]["state"]["OOMKilled"]:
            raise RuntimeError("未取得預期的 OOM 證據；查看 docker_runs.json")
    for dataset, count in (("dev", 1), ("tickets", args.runs)):
        for i in range(count):
            label = f"{dataset}-{i}"
            record = run(label, ["--report", f"/results/{label}-limits.json", "--run",
                                 "Solution/predict.py", f"data/{dataset}.jsonl",
                                 f"/results/{label}.jsonl", "--metrics", f"/results/{label}-metrics.json"])
            if record["exit_code"]:
                raise RuntimeError(f"{label} 執行失敗；查看 docker_runs.json")
    print(directory)


if __name__ == "__main__":
    main()
