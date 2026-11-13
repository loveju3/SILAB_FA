"""Deterministic queries over public tables and the four raw ticket fields.

No predictions, trained classifiers, scheme A aliases, or answer labels are used.
Ticket text searches are lexical matches, not ground-truth semantic labels.
"""
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = "2026-10-14T15:10:00Z"


def normalize(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value).casefold())


def contains_device(text, terms):
    for term in terms:
        if term.isascii():
            if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", text):
                return True
        elif term in text:
            return True
    return False


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("時間必須含時區；日期邊界請轉成 UTC 00:00:00Z")
    return dt.astimezone(timezone.utc)


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


STRING = {"type": "string"}
SCHEMAS = {
    "get_device": object_schema({
        "identifier": {"type": "string", "description": "設備ID、名稱或型號；不可填位置。不篩設備時省略。"},
        "field": {"type": "string", "enum": ["device_id", "name", "location", "model", "critical"]},
    }, ["identifier", "field"]),
    "devices_at_location": object_schema({
        "location": {"type": "string", "description": "完整設備位置，例如頂樓、B1 動力室；回設備ID清單。"},
    }, ["location"]),
    "list_devices": object_schema({
        "selection": {"type": "string", "enum": ["all", "critical", "noncritical"]},
    }, ["selection"]),
    "get_error_code": object_schema({
        "code": STRING,
        "field": {"type": "string", "enum": ["category", "meaning", "model"]},
    }, ["code"]),
    "query_tickets": object_schema({
        "device_id": STRING,
        "reporter": {"type": "string", "description": "逐字照抄完整回報人名稱，不可省略前綴或姓氏。"},
        "since": STRING, "until": STRING,
        "contains_any": {"type": "array", "items": STRING, "maxItems": 12,
                         "description": "問題限定原文包含A或B時，必須填[A,B]；省略代表沒有文字條件。"},
        "operation": {"type": "string", "enum": ["count", "top_reporter", "reporters", "ticket_ids", "read"]},
    }, ["operation"]),
    "get_ticket_details": object_schema({
        "ticket_ids": {"type": "array", "items": STRING, "minItems": 1, "maxItems": 20},
    }, ["ticket_ids"]),
}


def validate(value, schema):
    """Validate the deliberately small schema subset used by our tools."""
    kind = schema["type"]
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if type(value) is not types[kind]:
        raise ValueError(f"預期 {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("未知欄位值")
    if kind == "object":
        if set(value) - set(schema["properties"]):
            raise ValueError("不允許的參數")
        if set(schema["required"]) - set(value):
            raise ValueError("缺少必要參數")
        for key, child in value.items():
            validate(child, schema["properties"][key])
    if kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10000):
            raise ValueError("陣列長度不合法")
        for child in value:
            validate(child, schema["items"])


class Store:
    def __init__(self, tickets_path=None, data_dir=ROOT / "data"):
        self.devices = json.loads((Path(data_dir) / "devices.json").read_text(encoding="utf-8"))
        self.codes = json.loads((Path(data_dir) / "error_codes.json").read_text(encoding="utf-8"))
        self.tickets_path = Path(tickets_path or Path(data_dir) / "tickets.jsonl").resolve()
        # Extra guard against accidental evaluation-data access, including symlinks.
        if "dev" in self.tickets_path.stem.casefold():
            raise ValueError("Agent 不接受 dev 資料；開發請使用 history 或合成資料")
        self._tickets = None

    @property
    def tickets(self):
        if self._tickets is None:
            rows, seen = [], set()
            with self.tickets_path.open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    row = {k: source[k] for k in ("ticket_id", "created_at", "reporter", "raw_text")}
                    if any(not isinstance(v, str) for v in row.values()):
                        raise ValueError("工單四個欄位必須為字串")
                    if not row["ticket_id"] or row["ticket_id"] in seen:
                        raise ValueError("工單編號空白或重複")
                    timestamp(row["created_at"])
                    seen.add(row["ticket_id"])
                    rows.append(row)
            self._tickets = sorted(rows, key=lambda r: timestamp(r["created_at"]))
        return self._tickets

    def call(self, name, args):
        if name not in SCHEMAS:
            raise ValueError("未知工具")
        validate(args, SCHEMAS[name])
        return getattr(self, name)(**args)

    def get_devices(self, identifier="", location="", critical=None, field="device_id", single=False):
        rows = [d for d in self.devices if
                (not identifier or normalize(identifier) in {
                    normalize(d[k]) for k in ("device_id", "name", "model")}) and
                (not location or normalize(location) in normalize(d["location"])) and
                (critical is None or d["critical"] == critical)]
        values = list(dict.fromkeys(d[field] for d in rows))
        if single and len(values) > 1:
            raise ValueError("符合多個值；請縮小範圍或使用清單")
        return {"answer": (values[0] if values else None) if single else values,
                "evidence": [], "records": rows, "source": "devices.json"}

    def get_device(self, identifier, field):
        return self.get_devices(identifier=identifier, field=field, single=True)

    def devices_at_location(self, location):
        return self.get_devices(location=location)

    def list_devices(self, selection):
        critical = {"all": None, "critical": True, "noncritical": False}[selection]
        return self.get_devices(critical=critical)

    def get_error_code(self, code, field="category"):
        code = re.sub(r"[-_\s]", "", unicodedata.normalize("NFKC", code)).upper()
        key = next((k for k in self.codes if k.replace("-", "") == code), None)
        row = self.codes.get(key)
        return {"answer": row[field] if row else None, "evidence": [],
                "record": row, "source": "error_codes.json"}

    def query_tickets(self, operation, device_id="", reporter="", since="", until="", contains_any=None):
        start, end = timestamp(since) if since else None, timestamp(until) if until else None
        if start and end and start >= end:
            raise ValueError("since 必須早於 until（since 含邊界，until 不含）")
        if contains_any and any(not term.strip() for term in contains_any):
            raise ValueError("文字條件不可為空白")
        terms = []
        if device_id:
            device = next((d for d in self.devices if d["device_id"] == device_id), None)
            if device is None:
                raise ValueError("未知設備編號")
            terms = [normalize(device[k]) for k in ("device_id", "name", "model")]
        matches = []
        for row in self.tickets:
            text, dt = normalize(row["raw_text"]), timestamp(row["created_at"])
            if (start and dt < start) or (end and dt >= end):
                continue
            if reporter and row["reporter"] != reporter:
                continue
            if terms and not contains_device(text, terms):
                continue
            if contains_any and not any(normalize(term) in text for term in contains_any if term.strip()):
                continue
            matches.append(row)
        counts = Counter(r["reporter"] for r in matches)
        top = sorted(k for k, v in counts.items() if v == max(counts.values(), default=0))
        answer = {"count": len(matches), "top_reporter": top[0] if len(top) == 1 else top,
                  "reporters": sorted(counts), "ticket_ids": [r["ticket_id"] for r in matches],
                  "read": [r["ticket_id"] for r in matches[-20:]]}[operation]
        cited = matches[-20:] if operation == "read" else [
            r for r in matches if operation != "top_reporter" or r["reporter"] in top]
        return {"answer": answer, "evidence": [r["ticket_id"] for r in cited[:20]],
                "total": len(matches), "records": matches[-20:] if operation == "read" else [],
                "applied_filters": {"device_id": device_id, "reporter": reporter,
                                    "since": since, "until": until, "contains_any": contains_any or []},
                "truncated": operation == "read" and len(matches) > 20,
                "note": "依原始工單完整集合計算；文字／設備條件僅為字面匹配，不代表語意分類或指涉已確認。"}

    def get_ticket_details(self, ticket_ids):
        index = {r["ticket_id"]: r for r in self.tickets}
        if any(key not in index for key in ticket_ids):
            raise ValueError("引用不存在的工單")
        return {"answer": ticket_ids, "evidence": ticket_ids,
                "records": [index[key] for key in ticket_ids]}
