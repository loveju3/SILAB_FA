#!/usr/bin/env python3
"""Cold-process development Evals; saves local, versioned reports."""
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
sys.path.insert(0, str(ROOT))
import _venv  # noqa: E402

if __name__ == "__main__":
    _venv.ensure()


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grade_answer(case, answer, ids):
    """Score outputs, never treat a process error or null output as a refusal."""
    if not isinstance(answer, dict) or "answer" not in answer:
        return False
    evidence = answer.get("evidence")
    if (not isinstance(evidence, list) or len(evidence) > 20 or
            not all(isinstance(key, str) and key in ids for key in evidence) or
            len(evidence) != len(set(evidence))):
        return False
    needs_evidence = case.get("evidence_required",
        not case["id"].startswith("public-") and case["answer"] not in (0, []))
    if needs_evidence and not evidence:
        return False
    if "expected_evidence" in case and set(evidence) != set(case["expected_evidence"]):
        return False
    if "allowed_tools" in case:
        used = answer.get("tools_used")
        if not isinstance(used, list) or any(tool not in case["allowed_tools"] for tool in used):
            return False
    return type(answer["answer"]) is type(case["answer"]) and answer["answer"] == case["answer"]


def select_cases(cases, case_id=None, split="dev"):
    """--case picks one case explicitly; otherwise filter by split (unmarked cases are dev)."""
    if case_id:
        return [c for c in cases if c["id"] == case_id]
    return [c for c in cases if split == "all" or c.get("split", "dev") == split]


def summarize(rows):
    """Pass counts per category, for the report's per-question-type table."""
    summary = {}
    for row in rows:
        entry = summary.setdefault(row.get("category", "uncategorized"), {"passed": 0, "total": 0})
        entry["total"] += 1
        entry["passed"] += bool(row["passed"])
    return summary


def tools_matched(case, answer):
    """Diagnostic only (not graded): did the Agent call every expected tool?"""
    if "expected_tools" not in case:
        return None
    used = answer.get("tools_used") if isinstance(answer, dict) else None
    return isinstance(used, list) and all(tool in used for tool in case["expected_tools"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="只跑指定 case ID（不受 --split 限制）；不指定時依 --split 篩選")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev",
                        help="dev（預設）供開發；holdout 只在 ask.py 版本凍結後跑一次；all 全部")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--suite", choices=["synthetic", "history"], default="synthetic")
    parser.add_argument("--events", action="store_true", help="以 JSONL 輸出進度供本機測試 UI 使用")
    parser.add_argument("--server", type=Path, help="指定本機 CPU llama-server 路徑")
    parser.add_argument("--model", type=Path, default=ROOT / "agent_assets/Qwen_Qwen3-1.7B-Q4_K_M.gguf",
                        help="指定本機 GGUF 模型（預設 Qwen3-1.7B Q4_K_M）")
    args = parser.parse_args()
    if args.events and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    fixture_path = ROOT / "evals/fixtures.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if args.suite == "history":
        from history_suite import build_history_suite
        fixture = build_history_suite(ROOT)
        fixture_path = ROOT / "data/history.jsonl"
    cases = select_cases(fixture["cases"], args.case, args.split)
    if not cases:
        parser.error("未知 case" if args.case else "此 split 沒有題目")
    model = args.model
    version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source_paths = [ROOT / "ask.py", *sorted((ROOT / "agent").glob("*.py")),
                    *sorted((ROOT / "evals").glob("*.py"))]
    manifest_path = ROOT / "agent_assets/manifest.json"
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "git_revision": version,
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "fixture_sha256": digest(fixture_path), "suite": args.suite, "source": fixture["provenance"],
              "split": args.split if not args.case else f"case:{args.case}",
              "evaluation_data_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                  [ROOT / "evals/fixtures.json", ROOT / "data/devices.json", ROOT / "data/error_codes.json"]},
              "config": {"threads": args.threads, "timeout_s": 29, "context": 4096,
                         "max_steps": 4, "thinking": False, "server": str(args.server) if args.server else "default",
                         "model": model.name},
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
                       "--threads", str(args.threads), "--model", str(model), "--trace"]
            if args.server:
                command.extend(["--server", str(args.server)])
            row = {"id": case["id"], "question": case["question"], "expected": case["answer"],
                   **{key: case[key] for key in ("rationale", "category", "evidence_required",
                       "expected_evidence", "allowed_tools", "split", "paired_with",
                       "expected_tools") if key in case}}
            try:
                result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                        timeout=32, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                row.update(exit_code=result.returncode, trace=result.stderr)
                answer = json.loads(result.stdout) if result.returncode == 0 else None
                row["actual"] = answer
                row["passed"] = grade_answer(case, answer, ids)
                row["tools_matched"] = tools_matched(case, answer)
            except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
                row.update(passed=False, error=str(exc))
            row["elapsed_s"] = round(time.monotonic() - started, 3)
            row["passed"] = row["passed"] and row["elapsed_s"] <= 30
            report["cases"].append(row)
            if args.events:
                print(json.dumps({"event": "case", "case": row}, ensure_ascii=False), flush=True)
            else:
                print(f"{case['id']}: {'PASS' if row['passed'] else 'FAIL'} {row['elapsed_s']:.3f}s", flush=True)
                print(f"  預期答案：{json.dumps(case['answer'], ensure_ascii=False)}", flush=True)
                if row.get("exit_code") not in (None, 0) or row.get("error"):
                    reason = row.get("error") or (row.get("trace") or "").strip().splitlines()[-1:]
                    print(f"  執行失敗（不是答錯）：exit={row.get('exit_code')} "
                          f"{reason if isinstance(reason, str) else ''.join(reason)}", flush=True)
                if isinstance(row.get("actual"), dict):
                    print(f"  實際答案：{json.dumps(row['actual'].get('answer'), ensure_ascii=False)}", flush=True)
                if row.get("tools_matched") is not None:
                    print(f"  預期工具：{'有' if row['tools_matched'] else '缺'}呼叫 {case['expected_tools']}（診斷用，不計分）", flush=True)
                print(f"  設計理由：{case.get('rationale', '舊題目未記錄設計理由')}", flush=True)
    report["passed"] = sum(c["passed"] for c in report["cases"])
    report["total"] = len(cases)
    report["by_category"] = summarize(report["cases"])
    report["grading_note"] = ("本評分要求 evidence 完全吻合，比隱藏題只檢查 evidence 格式更嚴；"
                              "expected_tools 只做診斷，不影響通過與否。")
    output = args.output or ROOT / "evals/results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.events:
        print(json.dumps({"event": "finished", "passed": report["passed"], "total": report["total"]}), flush=True)
    else:
        for category, entry in sorted(report["by_category"].items()):
            print(f"  {category}: {entry['passed']}/{entry['total']}")
        print(f"{report['passed']}/{report['total']} — {output}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
