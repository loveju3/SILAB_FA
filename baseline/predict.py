#!/usr/bin/env python3
"""目前線上的做法：關鍵詞比對。寫在 2024 年，之後沒人動過。

    python3 baseline/predict.py data/dev.jsonl out.jsonl
    python3 baseline/evaluate.py out.jsonl data/dev.jsonl

它會壞在兩件事上：錯字打不到關鍵詞、指涉模糊時猜不出是哪台。
"""
import json
import pathlib
import sys

KEYWORDS = [
    ("noise", ["吵", "聲音", "異音", "震", "響", "叫", "摩擦", "敲", "嗡嗡"]),
    ("temperature", ["熱", "燙", "溫度", "不冷", "散熱", "悶", "高溫", "降溫"]),
    ("power", ["跳電", "斷電", "沒電", "電壓", "保險絲", "停電", "斷路", "電源", "跳脫"]),
    ("software", ["當機", "連線", "畫面", "程式", "更新", "讀值", "離線", "逾時", "重開"]),
]
SAFETY = ["冒煙", "焦味", "漏電", "火花", "起火", "燒", "觸電", "爆"]
HEDGE = ["好像", "不確定", "應該", "可能", "大概", "好像", "似乎"]


def guess_device(text, devices):
    for d in devices:
        if d["name"] in text or d["location"] in text:
            return d["device_id"]
    return None


def guess_category(text):
    for cat, words in KEYWORDS:
        if any(w in text for w in words):
            return cat
    return "other"


def guess_urgency(text):
    if any(w in text for w in SAFETY):
        return "high"
    if any(w in text for w in HEDGE):
        return "low"
    return "normal"


SUGGESTION = {
    "power": "請電工檢查迴路與斷路器",
    "temperature": "請檢查散熱與冷卻水",
    "noise": "請安排檢修異音來源",
    "software": "請資訊人員檢查系統",
    "other": "請現場確認狀況",
}


def main(inp, out):
    here = pathlib.Path(__file__).resolve().parent
    for cand in (here / "data" / "devices.json",          # 被複製到 repo 根目錄時
                 here.parent / "data" / "devices.json",   # 留在 baseline/ 時
                 pathlib.Path("data/devices.json")):      # 從 repo 根目錄執行時
        if cand.exists():
            devices = json.loads(cand.read_text(encoding="utf-8"))
            break
    else:
        raise SystemExit("找不到 data/devices.json，請在 repo 根目錄執行")
    with open(inp, encoding="utf-8") as f, open(out, "w", encoding="utf-8") as o:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec["raw_text"]
            cat = guess_category(text)
            o.write(json.dumps({
                "ticket_id": rec["ticket_id"],
                "device_id": guess_device(text, devices),
                "category": cat,
                "urgency": guess_urgency(text),
                "duplicate_of": None,
                "llm_used": False,
                # 只依類別套一句通用建議：沒讀文字，所以不可能具體
                "suggestion": SUGGESTION.get(cat, "請現場確認"),
            }, ensure_ascii=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
