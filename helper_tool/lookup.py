#!/usr/bin/env python3
"""Find tickets by exact ticket_id, with file and line provenance."""
import argparse
from pathlib import Path
import sys

try:
    from .common import ROOT, dump_json, read_records
except ImportError:
    from common import ROOT, dump_json, read_records


def build_index(paths):
    index = {}
    for path in paths:
        for line, record in read_records(path):
            index.setdefault(record["ticket_id"], []).append({
                "source_file": str(Path(path).resolve()), "source_line": line, "record": record,
            })
    return index


def display(ticket_ids, index, as_json=False):
    results = {ticket_id: index.get(ticket_id, []) for ticket_id in ticket_ids}
    if as_json:
        print(dump_json(results))
    else:
        for ticket_id, matches in results.items():
            if not matches:
                print(f"{ticket_id}：查無資料")
            for match in matches:
                print(f"\n{ticket_id} — {match['source_file']}:{match['source_line']}")
                print(dump_json(match["record"]))
    return all(results.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description="依 ticket_id 查詢工單；未提供 ID 時進入互動模式")
    parser.add_argument("ticket_ids", nargs="*", help="一個或多個工單編號，精確比對")
    parser.add_argument("--files", nargs="+", type=Path, help="指定 JSONL；預設查 tickets、dev、history")
    parser.add_argument("--json", action="store_true", help="輸出 JSON（須提供工單編號）")
    args = parser.parse_args(argv)
    if args.json and not args.ticket_ids:
        parser.error("--json 須同時提供 ticket_id")
    paths = args.files or [ROOT / "data" / f"{name}.jsonl" for name in ("tickets", "dev", "history")]
    try:
        index = build_index(paths)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"錯誤：{exc}\n")
    if args.ticket_ids:
        return 0 if display(args.ticket_ids, index, args.json) else 1
    print("輸入 ticket_id（可用空白分隔多筆），輸入 q 離開。")
    while True:
        try:
            text = input("ticket_id> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("q", "quit", "exit"):
            break
        if text:
            display(text.split(), index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
