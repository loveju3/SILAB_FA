#!/usr/bin/env python3
"""One command to reproduce the deliverables after `python3 install.py`.

    python3 run_demo.py            # Part 1 (dev score, tickets, 2 GB) + Part 2 demo if installed
    python3 run_demo.py --tests    # also run the development unit tests
    python3 run_demo.py --evals    # also run the Agent evals (slow: one cold start per question)

Outputs: tickets_predictions.jsonl, tickets_predictions_2gb.jsonl (repo root) and
reports/demo/ (dev predictions, metrics). dev.jsonl is only scored with the official
baseline/evaluate.py; this script never prints individual dev rows.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import _venv

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "reports/demo"
QUESTION = "B1 動力室有哪些設備？"


def run(title, command, results):
    shown = ["python" if c == sys.executable else str(c).replace(str(ROOT) + os.sep, "") for c in command]
    print("\n" + "=" * 70 + "\n" + title + "\n$ " + " ".join(shown), flush=True)
    started = time.perf_counter()
    env = dict(os.environ, PYTHONUTF8="1")
    code = subprocess.run([str(c) for c in command], cwd=str(ROOT), env=env).returncode
    seconds = time.perf_counter() - started
    results.append((title, "PASS" if code == 0 else "FAIL (exit %d)" % code, seconds))
    return code == 0


def skip(title, reason, results):
    print("\n" + "=" * 70 + "\n" + title + "\n略過：" + reason, flush=True)
    results.append((title, "略過：" + reason, 0.0))


def agent_installed():
    from agent.runtime import DEFAULT_SERVER
    from agent.setup import MODEL_NAME
    return (ROOT / "agent_assets" / MODEL_NAME).is_file() and Path(DEFAULT_SERVER).is_file()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests", action="store_true", help="另外跑開發單元測試")
    parser.add_argument("--evals", action="store_true", help="另外跑 Agent 評測（synthetic + history）")
    parser.add_argument("--skip-agent", action="store_true", help="不跑第二部分示範")
    args = parser.parse_args(argv)
    python, results = sys.executable, []
    OUT.mkdir(parents=True, exist_ok=True)
    predict, evaluate = ROOT / "predict.py", ROOT / "baseline/evaluate.py"
    has = {name: (DATA / name).is_file() for name in ("dev.jsonl", "tickets.jsonl", "history.jsonl")}

    # Part 1 -----------------------------------------------------------------
    if has["dev.jsonl"]:
        for label, extra in (("預設（微調 BGE）", []), ("2 GB 模式（--mem-limit 2g）", ["--mem-limit", "2g"])):
            name = "dev_predictions%s.jsonl" % ("_2gb" if extra else "")
            if run("第一部分 dev 預測：" + label, [python, predict, DATA / "dev.jsonl", OUT / name, *extra], results):
                run("第一部分 dev 評分：" + label, [python, evaluate, OUT / name, DATA / "dev.jsonl"], results)
    else:
        skip("第一部分 dev 評分", "data/dev.jsonl 不存在", results)
    if has["tickets.jsonl"]:
        run("第一部分 tickets 預測（兩參數預設）→ tickets_predictions.jsonl",
            [python, predict, DATA / "tickets.jsonl", ROOT / "tickets_predictions.jsonl",
             "--metrics", OUT / "tickets_metrics.json"], results)
        run("第一部分 tickets 預測（--mem-limit 2g）→ tickets_predictions_2gb.jsonl",
            [python, predict, DATA / "tickets.jsonl", ROOT / "tickets_predictions_2gb.jsonl",
             "--mem-limit", "2g", "--metrics", OUT / "tickets_2gb_metrics.json"], results)
    else:
        skip("第一部分 tickets 預測", "data/tickets.jsonl 不存在", results)

    # Part 2 -----------------------------------------------------------------
    if args.skip_agent:
        skip("第二部分 Agent", "--skip-agent", results)
    elif not agent_installed():
        skip("第二部分 Agent", "尚未安裝，請先執行 python3 install.py --with-agent", results)
    elif not has["tickets.jsonl"]:
        skip("第二部分 Agent", "data/tickets.jsonl 不存在", results)
    else:
        run("第二部分 ask.py（trace 在 stderr，答案 JSON 在 stdout）",
            [python, ROOT / "ask.py", QUESTION, "--trace"], results)
        run("第二部分 patrol.py", [python, ROOT / "patrol.py"], results)
        if args.evals:
            run("第二部分 Evals：synthetic", [python, ROOT / "evals/run.py", "--suite", "synthetic"], results)
            if has["history.jsonl"]:
                run("第二部分 Evals：history", [python, ROOT / "evals/run.py", "--suite", "history"], results)

    # Tests ------------------------------------------------------------------
    if args.tests:
        run("開發測試：Solution + helper_tool", [python, ROOT / "Solution/scripts/test_development.py"], results)
        run("開發測試：agent", [python, "-m", "unittest", "discover", "-s", "agent/tests"], results)
        run("開發測試：evals", [python, "-m", "unittest", "discover", "-s", "evals", "-p", "test_*.py"], results)
        run("開發測試：install.py", [python, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"], results)

    print("\n" + "=" * 70 + "\n總結", flush=True)
    for title, status, seconds in results:
        if status.startswith("略過"):
            print("  略過                     %s（%s）" % (title, status[3:]))
        else:
            print("  %-14s %7.1fs  %s" % (status, seconds, title))
    print("\n輸出：tickets_predictions.jsonl、tickets_predictions_2gb.jsonl、reports/demo/", flush=True)
    return 1 if any(status.startswith("FAIL") for _, status, _ in results) else 0


if __name__ == "__main__":
    _venv.ensure()
    sys.exit(main())
