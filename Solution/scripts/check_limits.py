#!/usr/bin/env python3
"""Run inside a cgroup v2 container; report or deliberately exceed its limit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true", help="配置並觸及超限記憶體；預期由 OOM 終止")
    parser.add_argument("--report", type=Path, help="保存限制、峰值與子程序退出碼")
    parser.add_argument("--run", nargs=argparse.REMAINDER, help="在已驗證限制內執行 Python 腳本")
    args = parser.parse_args()
    if args.probe and args.run:
        parser.error("--probe 與 --run 不可一起使用")
    root = Path("/sys/fs/cgroup")
    files = ("memory.max", "memory.swap.max", "memory.current", "memory.peak", "memory.events", "cpu.max")
    state = {name: (root / name).read_text().strip() if (root / name).exists() else None for name in files}
    print(json.dumps(state, indent=2), flush=True)
    if state["memory.max"] != "2147483648" or state["memory.swap.max"] != "0":
        print("未確認 2 GiB、無 swap 的 cgroup v2 限制", file=sys.stderr)
        return 2
    if args.probe:
        blocks = []
        # bytearray touches pages, unlike an untouched virtual-memory reservation.
        for _ in range(40):
            blocks.append(bytearray(64 * 1024 * 1024))
        print("錯誤：配置超限後未被終止", file=sys.stderr)
        return 3
    if args.run:
        start = perf_counter()
        result = subprocess.run([sys.executable, *args.run], check=False)
        state["process_wall_seconds"] = perf_counter() - start
        state["exit_code"] = result.returncode
        state["memory.peak"] = (root / "memory.peak").read_text().strip()
        state["memory.events_after"] = (root / "memory.events").read_text().strip()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        return result.returncode if result.returncode >= 0 else 128 - result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
