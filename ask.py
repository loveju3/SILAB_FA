#!/usr/bin/env python3
"""Cold-process offline Agent entry point. stdout is JSON; diagnostics go to stderr."""
import argparse
import json
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from agent.core import Agent
from agent.runtime import LocalModel, ROOT, DEFAULT_SERVER
from agent.tools import Store


def arguments():
    parser = argparse.ArgumentParser(description="本機 CPU 智慧問答 Agent")
    parser.add_argument("question")
    parser.add_argument("--model", type=Path, default=ROOT / "agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf")
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--tickets", type=Path, default=ROOT / "data/tickets.jsonl",
                        help="正式預設 tickets；開發可明確指定 history 或合成資料，拒絕 dev")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=29.0)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--trace", action="store_true")
    return parser


def main(argv=None):
    args = arguments().parse_args(argv)
    if not args.question.strip():
        raise ValueError("問題不可空白")
    if args.threads < 1 or not 0 < args.timeout <= 29 or not 2 <= args.max_steps <= 8:
        raise ValueError("threads >= 1；timeout 為 (0,29]；max-steps 為 2..8")
    deadline = STARTED + args.timeout
    store = Store(args.tickets, args.data_dir)
    with LocalModel(args.model, args.server, deadline, args.threads) as model:
        result = Agent(store, model, deadline, args.trace, args.max_steps).ask(args.question)
    if args.trace:
        print(json.dumps({"elapsed_s": time.monotonic() - STARTED,
                          "model_calls": model.metrics}, ensure_ascii=False), file=sys.stderr)
    if time.monotonic() >= deadline:
        raise TimeoutError("含關閉模型的總時間超出限制")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"錯誤：{exc}", file=sys.stderr)
        sys.exit(2)
