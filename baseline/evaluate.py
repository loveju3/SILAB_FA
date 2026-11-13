#!/usr/bin/env python3
"""評分。

    python3 baseline/evaluate.py <你的預測檔> <有答案的原始檔>

預測檔每行要有 ticket_id 和五個欄位：device_id、category、urgency、duplicate_of、llm_used
（llm_used 是給我們看你有沒有用 LLM、用了多少，boolean）
"""
import collections
import json
from collections import Counter
import sys

CATEGORIES = ["noise", "temperature", "power", "software", "other"]
URGENCIES = ["low", "normal", "high"]


def load(path):
    out = collections.OrderedDict()
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[r["ticket_id"]] = r
    return out


def macro_f1(pairs, classes):
    res = {}
    for c in classes:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        res[c] = (prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0, tp + fn)
    return res, sum(v[2] for v in res.values()) / len(classes)


def main(pred_path, gold_path):
    pred = load(pred_path)
    gold = load(gold_path)

    need = {"ticket_id", "device_id", "category", "urgency", "duplicate_of", "llm_used"}
    missing_fields = [t for t, p in pred.items() if not need <= set(p)]
    bad_shape = list(missing_fields)

    # 型別錯誤也算無效預測（寫成字串的 true/false 是最常見的手滑）
    def _bad_type(p):
        return not isinstance(p.get("llm_used"), bool)
    bad_type = [t for t, p in pred.items() if t not in bad_shape and _bad_type(p)]
    if bad_type:
        print(f"警告：有 {len(bad_type)} 筆 llm_used 不是 JSON 布林（視為無效預測）："
              f"{bad_type[:3]}")
        bad_shape += bad_type

    # 重複的 ticket_id 會被後一筆覆蓋，先講清楚
    dup_ids = [t for t, n in Counter(json.loads(l)["ticket_id"]
                                     for l in open(pred_path, encoding="utf-8")
                                     if l.strip()).items() if n > 1]
    if dup_ids:
        print(f"警告：有 {len(dup_ids)} 個 ticket_id 重複出現（只算最後一筆）：{dup_ids[:3]}")

    # suggestion 不計分（由人讀），缺了只警告，不影響任何分數
    sug = [p.get("suggestion") for t, p in pred.items() if t not in bad_shape]
    filled = [s for s in sug if isinstance(s, str) and s.strip()]
    print(f"suggestion 已填 {len(filled)}/{len(sug)} 筆（此欄不計分，由人閱讀）")
    if len(filled) < len(sug):
        print(f"  ⚠ 有 {len(sug)-len(filled)} 筆沒填 suggestion——不扣分，但面試時會被問到")
    for s in filled[:3]:
        print(f"  例：{s}")
    if missing_fields:
        print(f"警告：有 {len(missing_fields)} 筆缺少必要欄位（視為無效預測）："
              f"{missing_fields[:3]}")
    missing = [t for t in gold if t not in pred]
    extra = [t for t in pred if t not in gold]
    if missing:
        print(f"警告：有 {len(missing)} 筆沒預測到（一律不給分，正確答案是 null 的也不算你對）")
    if extra:
        print(f"警告：有 {len(extra)} 筆不在原始檔裡")

    dev_ok = dev_n = abst_ok = abst_n = 0
    cat_pairs, urg_pairs = [], []
    dupe_tp = dupe_fp = dupe_fn = 0
    llm_used = 0

    for tid, g in gold.items():
        p = pred.get(tid, {}) if tid not in bad_shape else {}
        if p.get("llm_used") is True:
            llm_used += 1
        if g["device_id"] is None and g["category"] is None:
            abst_n += 1
            # 要真的輸出這一筆、而且 device 和 category 都說不知道，才算棄答正確
            abst_ok += (tid in pred and tid not in bad_shape
                        and p.get("device_id") is None and p.get("category") is None)
        else:
            dev_n += 1
            # 缺行或格式錯的不能白拿分（標準答案 device 為 null 時 None == None 會誤判為對）
            dev_ok += (tid in pred and tid not in bad_shape
                       and p.get("device_id") == g["device_id"])
        if g["category"] is not None:
            cat_pairs.append((g["category"], p.get("category")))
        if g["urgency"] is not None:
            urg_pairs.append((g["urgency"], p.get("urgency")))
        gd, pd = g["duplicate_of"], p.get("duplicate_of")
        if gd and pd == gd:
            dupe_tp += 1
        elif gd and pd:
            dupe_fp += 1                     # 指錯對象，算錯
            dupe_fn += 1
        elif gd:
            dupe_fn += 1
        elif pd:
            dupe_fp += 1

    cats, cat_f1 = macro_f1(cat_pairs, CATEGORIES)
    urgs, urg_f1 = macro_f1(urg_pairs, URGENCIES)
    dupe_f1 = 2 * dupe_tp / (2 * dupe_tp + dupe_fp + dupe_fn) if dupe_tp + dupe_fp + dupe_fn else 0.0
    # 說不知道要算進 device 分：躲掉的比猜錯好，但不能靠躲拿分
    dev_acc = (dev_ok + abst_ok) / (dev_n + abst_n) if dev_n + abst_n else 0.0

    print(f"\n{'欄位':22s}{'分數':>8s}   說明")
    print(f"{'device_id':22s}{dev_acc:8.3f}   {dev_ok}/{dev_n}，另 {abst_ok}/{abst_n} 筆正確說不知道")
    print(f"{'category (macro-F1)':22s}{cat_f1:8.3f}")
    for c in CATEGORIES:
        prec, rec, f1, n = cats[c]
        print(f"    {c:18s}{f1:8.3f}   prec={prec:.2f} rec={rec:.2f} n={n}")
    print(f"{'urgency (macro-F1)':22s}{urg_f1:8.3f}")
    for c in URGENCIES:
        prec, rec, f1, n = urgs[c]
        print(f"    {c:18s}{f1:8.3f}   prec={prec:.2f} rec={rec:.2f} n={n}")
    print(f"{'duplicate_of (F1)':22s}{dupe_f1:8.3f}")
    print(f"{'該說不知道時':22s}{abst_ok / abst_n if abst_n else 0:8.3f}   {abst_ok}/{abst_n}")

    total = 0.4 * cat_f1 + 0.2 * dev_acc + 0.2 * urg_f1 + 0.2 * dupe_f1
    print(f"\n總分 = 0.4×category + 0.2×device + 0.2×urgency + 0.2×duplicate = {total:.3f}")
    print(f"LLM 使用率: {llm_used}/{len(gold)} = {llm_used / len(gold):.1%}")
    print("\n提醒：分數只是其中一項。全部丟 LLM 也跑得出高分，但撐不住預算的分數"
          "在 README 裡要自己講清楚。")
    return total


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
