"""Shared UTF-8 JSONL reader; paths supplied by users are relative to cwd."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("device_id", "category", "urgency", "duplicate_of")


def read_records(path):
    """Yield (line number, record), rejecting ambiguous IDs and malformed JSON."""
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 格式錯誤：{exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: 每筆必須是 JSON object")
            ticket_id = record.get("ticket_id")
            if not isinstance(ticket_id, str) or not ticket_id.strip():
                raise ValueError(f"{path}:{line_number}: ticket_id 必須是非空字串")
            if ticket_id in seen:
                raise ValueError(f"{path}:{line_number}: ticket_id 重複：{ticket_id}")
            seen.add(ticket_id)
            yield line_number, record


def dump_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)
