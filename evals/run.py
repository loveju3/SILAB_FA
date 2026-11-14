#!/usr/bin/env python3
"""Cold-process Evals: synthetic records only; saves local, versioned reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="只跑指定 case ID；不指定時跑全部12題")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--suite", choices=["synthetic", "history"], default="synthetic")
    parser.add_argument("--events", action="store_true", help="以 JSONL 輸出進度供本機測試 UI 使用")
    parser.add_argument("--server", type=Path, help="指定本機 CPU llama-server 路徑")
    args = parser.parse_args()
    if args.events and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    fixture_path = ROOT / "evals/fixtures.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if args.suite == "history":
        from history_suite import build_history_suite
        fixture = build_history_suite(ROOT)
        fixture_path = ROOT / "data/history.jsonl"
    cases = [c for c in fixture["cases"] if not args.case or c["id"] == args.case]
    if not cases:
        parser.error("未知 case")
    model = ROOT / "agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = [ROOT / "ask.py", *sorted((ROOT / "agent").glob("*.py")),
                    *sorted((ROOT / "evals").glob("*.py"))]
    manifest_path = ROOT / "agent_assets/manifest.json"
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "git_revision": version,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "fixture_sha256": digest(fixture_path), "suite": args.suite, "source": fixture["provenance"],
              "config": {"threads": args.threads, "timeout_s": 29, "context": 4096,
                         "max_steps": 4, "thinking": False, "server": str(args.server) if args.server else "default"},
              "model_sha256": digest(model) if model.exists() else None, "cases": []}
    report["assets"] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    if args.events:
        print(json.dumps({"event": "started", "total": len(cases)}, ensure_ascii=False), flush=True)
    with tempfile.TemporaryDirectory(prefix="silab-agent-evals-") as temp:
        tickets = Path(temp) / "synthetic.jsonl"
        tickets.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fixture["tickets"]), encoding="utf-8")
        ids = {r["ticket_id"] for r in fixture["tickets"]}
        for case in cases:
            started = time.monotonic()
            command = [sys.executable, str(ROOT / "ask.py"), case["question"], "--tickets", str(tickets),
                       "--threads", str(args.threads), "--trace"]
            if args.server:
                command.extend(["--server", str(args.server)])
            row = {"id": case["id"], "question": case["question"], "expected": case["answer"]}
            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                        timeout=32, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                row.update(exit_code=result.returncode, trace=result.stderr)
                answer = json.loads(result.stdout) if result.returncode == 0 else None
                row["actual"] = answer
                evidence = answer.get("evidence", []) if answer else []
                needs_evidence = not case["id"].startswith("public-") and case["answer"] not in (0, [])
                valid = (isinstance(evidence, list) and len(evidence) <= 20 and
                         all(isinstance(key, str) and key in ids for key in evidence) and
                         len(evidence) == len(set(evidence)) and (bool(evidence) or not needs_evidence))
                row["passed"] = bool(answer and type(answer["answer"]) is type(case["answer"]) and
                                     answer["answer"] == case["answer"] and valid)
            except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
                row.update(passed=False, error=str(exc))
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            row["passed"] = row["passed"] and row["elapsed_s"] <= 30
            report["cases"].append(row)
            if args.events:
                print(json.dumps({"event": "case", "case": row}, ensure_ascii=False), flush=True)
            else:
                print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'} {row['elapsed_s']:.3f}s", flush=True)
    report["passed"] = sum(c["passed"] for c in report["cases"])
    report["total"] = len(cases)
    output = args.output or ROOT / "evals/results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.events:
        print(json.dumps({"event": "finished", "passed": report["passed"], "total": report["total"]}), flush=True)
    else:
        print(f"{report['passed']}/{report['total']} — {output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
