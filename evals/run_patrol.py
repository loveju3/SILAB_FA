#!/usr/bin/env python3
"""Real-model patrol contract smoke test, not a risk-ranking accuracy score."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evals/results/patrol_smoke.json")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--server", type=Path)
    args = parser.parse_args()
    fixture = json.loads((ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))
    rows = fixture["tickets"] + [{**row, "ticket_id": f"SYN{index + 9:03}"}
                                  for index, row in enumerate(fixture["tickets"][:4])]
    index = {row["ticket_id"]: row for row in rows}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "source": "12 hand-authored synthetic tickets; contract smoke only, no ranking ground truth",
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    report["config"] = {"threads": args.threads, "timeout_s": 118, "context": 8192}
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT / "patrol.py", *sorted((ROOT / "agent").glob("*.py"))]}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="silab-patrol-") as temp:
            path = Path(temp) / "synthetic.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            command = [sys.executable, str(ROOT / "patrol.py"), "--tickets", str(path), "--trace",
                       "--threads", str(args.threads)]
            if args.server:
                command.extend(["--server", str(args.server)])
            result = subprocess.run(command,
                                    capture_output=True, text=True, encoding="utf-8", timeout=122,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        report.update(exit_code=result.returncode, trace=result.stderr)
        output = json.loads(result.stdout) if result.returncode == 0 else None
        report["output"] = output
        valid = isinstance(output, list) and len(output) == 10
        if valid:
            valid = len({item["ticket_id"] for item in output}) == 10
            for item in output:
                quote = re.search("「(.*?)」", item["reason"])
                valid = valid and item["ticket_id"] in index and item["priority"] in {"high", "medium", "low"}
                valid = valid and bool(quote) and quote.group(1) in index[item["ticket_id"]]["raw_text"]
        report["contract_passed"] = bool(valid)
    except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
        report.update(contract_passed=False, error=str(exc))
    report["elapsed_s"] = round(time.monotonic() - started, 3)
    report["contract_passed"] = report["contract_passed"] and report["elapsed_s"] <= 120
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"patrol contract: {'PASS' if report['contract_passed'] else 'FAIL'} {report['elapsed_s']:.3f}s")
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
