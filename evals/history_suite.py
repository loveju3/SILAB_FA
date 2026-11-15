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
    public_ids = {"public-location", "public-name", "public-model", "public-code", "public-code-variant", "public-empty"}
    public = [case for case in json.loads((Path(root) / "evals/fixtures.json").read_text(encoding="utf-8"))["cases"]
              if case["id"] in public_ids]
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
    reasons = [
        "驗證 history 完整四欄投影的工單總數，不使用答案標籤。",
        "驗證 history 日期篩選包含 UTC 當日零點。",
        "驗證 history 回報人排名，同分時回傳全部人名。",
        "驗證 history 指定完整回報人姓名的計數。",
        "驗證 history 僅匹配指定原文字詞，不使用語意分類標籤。",
        "驗證 history 對指定罕見字串的字面計數，不預先假定一定為零。",
    ]
    for case, reason in zip(cases[len(public):], reasons):
        case.update(rationale=reason, category="history_query")
    return {"provenance": "history four-field projection and public tables; no dev access or answer labels",
            "tickets": rows, "cases": cases}
