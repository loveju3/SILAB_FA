"""Build a local-only development suite from history, never dev.

Expected answers are calculated independently of Agent tools. Only public tables
and the four inference fields are used; the source's answer labels are discarded.
"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


def build_history_suite(root):
    path = Path(root) / "data/history.jsonl"
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                record = json.loads(line)
                rows.append({key: record[key] for key in ("ticket_id", "created_at", "reporter", "raw_text")})
    if not rows:
        raise ValueError("history 不可為空")
    counts = Counter(row["reporter"] for row in rows)
    leaders = sorted(name for name, count in counts.items() if count == max(counts.values()))
    reporter = sorted(counts)[0]
    dates = [datetime.fromisoformat(r["created_at"].replace("Z", "+00:00")).astimezone(timezone.utc) for r in rows]
    boundary = min(dates).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=3)
    public = json.loads((Path(root) / "evals/fixtures.json").read_text(encoding="utf-8"))["cases"][:6]
    cases = public + [
        {"id": "history-count", "question": "目前輸入資料共有幾張工單？", "answer": len(rows)},
        {"id": "history-date", "question": f"{boundary:%Y-%m-%d} 之後有幾張工單？包含當日零點。",
         "answer": sum(dt >= boundary for dt in dates)},
        {"id": "history-top", "question": "哪位回報人的報修工單最多？同分時回所有人名。",
         "answer": leaders[0] if len(leaders) == 1 else leaders},
        {"id": "history-reporter", "question": f"回報人「{reporter}」總共回報了幾張工單？", "answer": counts[reporter]},
        {"id": "history-literal", "question": "原文字面包含「空壓機」的工單共有幾張？只匹配這三個字。",
         "answer": sum("空壓機" in row["raw_text"] for row in rows)},
        {"id": "history-empty", "question": "原文字面包含「SYNTHETIC_NO_MATCH_7F2A」的工單共有幾張？",
         "answer": sum("SYNTHETIC_NO_MATCH_7F2A" in row["raw_text"] for row in rows)},
    ]
    return {"provenance": "history four-field projection and public tables; no dev access or answer labels",
            "tickets": rows, "cases": cases}
