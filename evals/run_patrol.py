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
sys.path.insert(0, str(ROOT))
import _venv  # noqa: E402

if __name__ == "__main__":
    _venv.ensure()


def load_source(dataset):
    """Return raw four-field records and their original positions, never labels."""
    fields = ("ticket_id", "created_at", "reporter", "raw_text")
    if dataset == "tickets":
        data = (ROOT / "data/tickets.jsonl").read_bytes()
        entries = [(line_no, json.loads(line)) for line_no, line in
                   enumerate(data.decode("utf-8").splitlines(), 1) if line.strip()]
        name = "data/tickets.jsonl"
    elif dataset == "synthetic":
        fixture = json.loads((ROOT / "evals/fixtures.json").read_text(encoding="utf-8"))
        start = len(fixture["tickets"]) + 1  # padding IDs must not collide with fixture IDs
        values = fixture["tickets"] + [{**row, "ticket_id": f"SYN{index + start:03}"}
                                      for index, row in enumerate(fixture["tickets"][:4])]
        data = json.dumps(values, ensure_ascii=False).encode("utf-8")
        entries = list(enumerate(values, 1))
        name = "合成巡檢輸入快照"
    else:
        raise ValueError("巡檢來源只接受 tickets 或 synthetic")
    rows, positions = [], {}
    for line_no, value in entries:
        row = {key: value[key] for key in fields}
        if any(not isinstance(v, str) for v in row.values()) or not row["ticket_id"] or row["ticket_id"] in positions:
            raise ValueError("工單四欄型別錯誤或編號空白／重複")
        rows.append(row)
        positions[row["ticket_id"]] = {"record_index": len(rows), "source_line": line_no, "record": row}
    return rows, positions, {"dataset": dataset, "file": name, "count": len(rows),
                             "sha256": hashlib.sha256(data).hexdigest()}


def selected_records(output, positions):
    if not isinstance(output, list):
        return []
    return [{"rank": rank, "ticket_id": item.get("ticket_id"), "priority": item.get("priority"),
             "reason": item.get("reason"), **positions.get(item.get("ticket_id"), {})}
            for rank, item in enumerate(output, 1) if isinstance(item, dict)]


def contract_valid(output, index):
    if not isinstance(output, list) or len(output) != 10:
        return False
    seen = set()
    for item in output:
        if not isinstance(item, dict):
            return False
        key, reason = item.get("ticket_id"), item.get("reason")
        if not isinstance(key, str) or key not in index or key in seen or not isinstance(reason, str):
            return False
        if item.get("priority") not in ("high", "medium", "low"):
            return False
        quote = re.search("「(.*?)」", reason, re.DOTALL)
        if not quote or not quote.group(1) or quote.group(1) not in index[key]["raw_text"]:
            return False
        seen.add(key)
    return True


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "evals/results/patrol_smoke.json")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--dataset", choices=["synthetic", "tickets"], default="synthetic")
    args = parser.parse_args()
    rows, positions, source_info = load_source(args.dataset)
    index = {row["ticket_id"]: row for row in rows}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "source": ("Raw tickets snapshot; contract checks only, ranking requires human review" if args.dataset == "tickets"
                         else "12 hand-authored synthetic tickets; contract smoke only, no ranking ground truth"),
              "source_info": source_info,
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    report["config"] = {"threads": args.threads, "timeout_s": 118, "context": 8192,
                        "candidate_policy": "critical-device-lexical-v3-safety-first", "candidate_limit": 30,
                        "prefilter": "full-text-safety-term-count-desc-then-created-at-desc",
                        "selection": "llm-top10"}
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT / "patrol.py", *sorted((ROOT / "agent").glob("*.py"))]}
    report["source_sha256"]["data/devices.json"] = hashlib.sha256((ROOT / "data/devices.json").read_bytes()).hexdigest()
    try:
        with tempfile.TemporaryDirectory(prefix="silab-patrol-") as temp:
            path = Path(temp) / "synthetic.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            report["source_info"]["inference_snapshot_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            command = [sys.executable, str(ROOT / "patrol.py"), "--tickets", str(path), "--trace",
                       "--threads", str(args.threads)]
            if args.server:
                command.extend(["--server", str(args.server)])
            result = subprocess.run(command,
                                    capture_output=True, text=True, encoding="utf-8", timeout=122,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        report.update(exit_code=result.returncode, trace=result.stderr)
        for line in result.stderr.splitlines():
            if line.startswith('{'):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if "eligible_count" in event:
                    report["candidate_summary"] = {key: event[key] for key in
                        ("eligible_count", "candidate_count", "candidate_ids")}
        output = json.loads(result.stdout) if result.returncode == 0 else None
        report["output"] = output
        report["selected_tickets"] = selected_records(output, positions)
        report["contract_passed"] = contract_valid(output, index)
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
