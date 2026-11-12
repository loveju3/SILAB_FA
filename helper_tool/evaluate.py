#!/usr/bin/env python3
"""Score predictions with the official evaluator and export error analysis."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys

try:
    from .common import FIELDS, ROOT, dump_json, read_records
except ImportError:
    from common import FIELDS, ROOT, dump_json, read_records

# The original evaluator remains the authoritative source of the total score.
sys.path.insert(0, str(ROOT))
from baseline.evaluate import main as official_evaluate, macro_f1, CATEGORIES, URGENCIES


def analyze(pred_path, gold_path):
    gold_rows = list(read_records(gold_path))
    pred_rows = list(read_records(pred_path))
    if not gold_rows:
        raise ValueError("標準答案檔不能是空檔")
    for line, gold in gold_rows:
        missing = set(FIELDS) - gold.keys()
        if missing:
            raise ValueError(f"{gold_path}:{line}: 缺少標籤 {sorted(missing)}；請使用 dev 或 history")
        if gold["category"] not in [None, *CATEGORIES] or gold["urgency"] not in URGENCIES:
            raise ValueError(f"{gold_path}:{line}: category 或 urgency 標籤不合法")
        for field in ("device_id", "duplicate_of"):
            if gold[field] is not None and not isinstance(gold[field], str):
                raise ValueError(f"{gold_path}:{line}: {field} 必須是字串或 null")
    gold_ids = {row["ticket_id"] for _, row in gold_rows}
    pred = {row["ticket_id"]: row for _, row in pred_rows}
    pred_lines = {row["ticket_id"]: line for line, row in pred_rows}
    required = {"ticket_id", *FIELDS, "llm_used"}
    invalid = {}
    for line, row in pred_rows:
        # Reject non-scalar field values before the official evaluator uses them.
        for field in FIELDS:
            if field in row and row[field] is not None and not isinstance(row[field], str):
                raise ValueError(f"{pred_path}:{line}: {field} 必須是字串或 null")
        reasons = [f"缺少必要欄位：{field}" for field in sorted(required - row.keys())]
        if "llm_used" in row and not isinstance(row["llm_used"], bool):
            reasons.append("llm_used 不是 JSON boolean")
        if reasons:
            invalid[row["ticket_id"]] = reasons

    # Capture its original output for reproducibility, without changing baseline/.
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        total = official_evaluate(str(pred_path), str(gold_path))

    errors, wrong_counts, confusions = [], Counter(), {field: Counter() for field in FIELDS}
    category_pairs, urgency_pairs = [], []
    device_ok = abstention_ok = abstention_n = tp = fp = fn = 0
    for line, gold in gold_rows:
        tid = gold["ticket_id"]
        present = tid in pred
        valid = present and tid not in invalid
        raw_prediction = pred.get(tid)
        prediction = raw_prediction if valid else {}
        abstention = gold["device_id"] is None and gold["category"] is None
        device_correct = valid and prediction.get("device_id") == gold["device_id"]
        if abstention:
            abstention_n += 1
            device_correct = device_correct and prediction.get("category") is None
            abstention_ok += bool(device_correct)
        device_ok += bool(device_correct)
        if gold["category"] is not None:
            category_pairs.append((gold["category"], prediction.get("category")))
        if gold["urgency"] is not None:
            urgency_pairs.append((gold["urgency"], prediction.get("urgency")))
        gd, pd = gold["duplicate_of"], prediction.get("duplicate_of")
        if gd and pd == gd:
            tp += 1
        elif gd and pd:
            fp += 1
            fn += 1
        elif gd:
            fn += 1
        elif pd:
            fp += 1

        details = {}
        for field in FIELDS:
            if not valid:
                reason = "missing_prediction" if not present else "invalid_prediction"
            elif field == "device_id" and not device_correct:
                reason = "abstention_requires_both_null" if abstention else "wrong_value"
            elif field == "category" and gold[field] is None:
                continue  # Official category F1 excludes gold-null rows.
            elif prediction.get(field) == gold[field]:
                continue
            elif field == "duplicate_of":
                reason = "false_positive" if not gd else "false_negative" if not pd else "wrong_target"
            else:
                reason = "wrong_value"
            wrong_counts[field] += 1
            actual = prediction.get(field) if valid else "<missing>" if not present else "<invalid>"
            confusions[field][(json.dumps(gold[field], ensure_ascii=False),
                               json.dumps(actual, ensure_ascii=False), reason)] += 1
            details[field] = {"expected": gold[field], "predicted": actual, "reason": reason}
        if details:
            errors.append({
                "ticket_id": tid, "source_file": str(gold_path), "source_line": line,
                "prediction_line": pred_lines.get(tid),
                "created_at": gold.get("created_at"), "reporter": gold.get("reporter"),
                "raw_text": gold.get("raw_text"), "wrong_fields": list(details),
                "differences": details, "gold": gold, "prediction": raw_prediction,
                "validation_errors": invalid.get(tid, []),
            })

    cats, cat_f1 = macro_f1(category_pairs, CATEGORIES)
    urgs, urg_f1 = macro_f1(urgency_pairs, URGENCIES)
    dupe_f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    metrics = {"category": cat_f1, "device": device_ok / len(gold_rows),
               "urgency": urg_f1, "duplicate": dupe_f1}
    summary = {
        "prediction_file": str(pred_path), "gold_file": str(gold_path),
        "total": total, "metrics": metrics,
        "weights": {"category": .4, "device": .2, "urgency": .2, "duplicate": .2},
        "gold_count": len(gold_rows), "prediction_count": len(pred_rows),
        "error_ticket_count": len(errors),
        "wrong_field_counts": {field: wrong_counts[field] for field in FIELDS},
        "missing_ids": [g["ticket_id"] for _, g in gold_rows if g["ticket_id"] not in pred],
        "extra_ids": [tid for tid in pred if tid not in gold_ids],
        "invalid_predictions": invalid,
        "suggestion_warning_ids": [tid for tid, p in pred.items()
                                   if not isinstance(p.get("suggestion"), str) or not p["suggestion"].strip()],
        "abstention": {"correct": abstention_ok, "count": abstention_n},
        "duplicate_counts": {"tp": tp, "fp": fp, "fn": fn},
        "per_class": {
            "category": {c: dict(zip(("precision", "recall", "f1", "support"), v)) for c, v in cats.items()},
            "urgency": {c: dict(zip(("precision", "recall", "f1", "support"), v)) for c, v in urgs.items()},
        },
        "confusions": {field: [{"expected": json.loads(g), "predicted": json.loads(p), "reason": reason, "count": n}
                               for (g, p, reason), n in counter.most_common()]
                       for field, counter in confusions.items()},
    }
    return summary, errors, buffer.getvalue()


def write_reports(output_dir, summary, errors, official_output):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(dump_json(summary) + "\n", encoding="utf-8")
    (output_dir / "errors.jsonl").write_text(
        "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors), encoding="utf-8")
    (output_dir / "official.txt").write_text(official_output, encoding="utf-8")
    with (output_dir / "errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        columns = ["ticket_id", "source_line", "raw_text", "wrong_fields", "validation_errors"]
        columns += [f"{field}_{side}" for field in FIELDS for side in ("gold", "predicted", "reason")]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for error in errors:
            row = {key: error[key] for key in columns[:3]}
            row["wrong_fields"] = ", ".join(error["wrong_fields"])
            row["validation_errors"] = "; ".join(error["validation_errors"])
            for field in FIELDS:
                difference = error["differences"].get(field, {})
                row[f"{field}_gold"] = json.dumps(error["gold"][field], ensure_ascii=False)
                row[f"{field}_predicted"] = json.dumps(
                    error["prediction"].get(field) if error["prediction"] else None, ensure_ascii=False)
                row[f"{field}_reason"] = difference.get("reason", "")
            writer.writerow(row)
    lines = ["# 評分與錯誤分析", "", f"總分：**{summary['total']:.6f}**（{summary['total'] * 100:.2f}/100）", "",
             "| 指標 | 分數 | 權重 |", "|---|---:|---:|"]
    for field, value in summary["metrics"].items():
        lines.append(f"| {field} | {value:.6f} | {summary['weights'][field]} |")
    lines += ["", f"有錯工單：{len(errors)}/{summary['gold_count']}。欄位錯誤數不能相加當作工單數。",
              "", "category 標準答案為 null 時不列入分類錯誤；雙 null 棄答錯誤列入 device。",
              "缺失或無效預測在四欄均標示問題；duplicate 的 null/null 不算 TP。",
              "多餘 ID 不計分，請另看 summary.json；錯誤統計是診斷筆數，不是 F1 的扣分量。", "",
              f"缺失 ID：{len(summary['missing_ids'])}；多餘 ID：{len(summary['extra_ids'])}；"
              f"無效預測：{len(summary['invalid_predictions'])}；"
              f"建議缺漏：{len(summary['suggestion_warning_ids'])}。", "",
              "## 各欄錯誤與常見混淆", ""]
    for field in FIELDS:
        lines += [f"### {field}：{summary['wrong_field_counts'][field]} 筆", ""]
        for item in summary["confusions"][field][:10]:
            lines.append(f"- 正解 `{json.dumps(item['expected'], ensure_ascii=False)}` → "
                         f"預測 `{json.dumps(item['predicted'], ensure_ascii=False)}`：{item['count']} 筆"
                         f"（{item['reason']}）")
        lines.append("")
    lines += ["全部錯題與原文見 errors.csv / errors.jsonl；完整類別 precision、recall、F1 見 summary.json。", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="官方評分＋逐筆錯誤分析（不需要第三方套件）")
    parser.add_argument("predictions", type=Path, help="預測 JSONL")
    parser.add_argument("gold", type=Path, nargs="?", default=ROOT / "data/dev.jsonl", help="預設 data/dev.jsonl")
    parser.add_argument("--output-dir", type=Path, help="報告目錄；同目錄重跑會覆寫報告")
    args = parser.parse_args(argv)
    try:
        summary, errors, official_output = analyze(args.predictions, args.gold)
        output_dir = args.output_dir or ROOT / "helper_tool/reports" / args.predictions.stem
        # Avoid overwriting either input through a report filename collision.
        targets = { (output_dir / name).resolve() for name in
                   ("summary.json", "errors.jsonl", "errors.csv", "official.txt", "report.md") }
        if args.predictions.resolve() in targets or args.gold.resolve() in targets:
            raise ValueError("報告檔名與輸入檔衝突，請指定其他 --output-dir")
        write_reports(output_dir, summary, errors, official_output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"錯誤：{exc}\n")
    print(official_output, end="")
    print(f"\n精確總分：{summary['total']:.6f}（{summary['total'] * 100:.2f}/100）")
    print(f"有錯工單：{len(errors)}/{summary['gold_count']}；各欄問題：{summary['wrong_field_counts']}")
    print(f"報告目錄：{output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
