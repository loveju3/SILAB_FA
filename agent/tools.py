"""Deterministic queries over public tables and the four raw ticket fields.

No answer labels or precomputed predictions are read. Ticket text searches are lexical
matches, not ground-truth semantic labels. The one exception is ticket_status: it runs
the part-1 classifier (predict.py) on the matched raw texts and reports that the
category is a model judgement, not a label.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
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
    "ticket_status": object_schema({
        "device": {"type": "string", "description": "設備編號、名稱或型號；回該設備最近工單的故障類別。"},
        "since": {"type": "string", "description": "問題明確給期間才填；省略代表快照前 7 天。"},
    }, ["device"]),
}
STATUS_DAYS = 7
STATUS_LIMIT = 50  # bound classifier input; the newest tickets in the window are kept


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
        self.deadline = None  # time.monotonic() bound for the classifier subprocess

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

    def classify(self, rows):
        """Category per ticket from the part-1 CLI (scheme A rules), raw four fields only."""
        timeout = 60.0 if self.deadline is None else self.deadline - time.monotonic() - 1.5
        if timeout <= 1:
            raise TimeoutError("剩餘時間不足以執行分類器")
        with tempfile.TemporaryDirectory(prefix="silab-status-") as temp:
            source, output, metrics = (Path(temp) / name for name in ("in.jsonl", "out.jsonl", "m.json"))
            source.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            try:
                # Scheme A rules (~0.4 s): C-FT is more accurate but its 5-12 s load put ask.py's
                # 30 s limit at risk on a busy CPU (two eval timeouts), and a timeout is a wrong answer.
                result = subprocess.run([sys.executable, str(ROOT / "predict.py"), str(source), str(output),
                                         "--method", "rules", "--metrics", str(metrics)],
                                        capture_output=True, text=True,
                                        encoding="utf-8", timeout=timeout,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("分類器超時") from exc
            if result.returncode != 0:
                raise RuntimeError("分類器執行失敗：" + (result.stderr.strip().splitlines() or ["?"])[-1])
            categories = {json.loads(line)["ticket_id"]: json.loads(line)["category"]
                          for line in output.read_text(encoding="utf-8").splitlines() if line.strip()}
            scheme = json.loads(metrics.read_text(encoding="utf-8")).get("scheme", "unknown")
        return categories, scheme

    def ticket_status(self, device, since=""):
        """Most common category of a device's recent tickets (lexical device match).

        Window: since (default snapshot - 7 days) up to the snapshot. Ties go to the newest
        tied ticket's category; an empty window falls back to the device's newest ticket.
        """
        values = self.get_devices(identifier=device, field="device_id", single=True)["answer"]
        if values is None:
            raise ValueError("未知設備；device 只接受設備編號、名稱或型號")
        record = next(d for d in self.devices if d["device_id"] == values)
        snapshot = timestamp(SNAPSHOT)
        start = timestamp(since) if since else snapshot - timedelta(days=STATUS_DAYS)
        if start >= snapshot:
            raise ValueError("since 必須早於資料快照時間")
        terms = [normalize(record[k]) for k in ("device_id", "name", "model")]
        matched = [r for r in self.tickets if timestamp(r["created_at"]) <= snapshot and
                   contains_device(normalize(r["raw_text"]), terms)]
        window = [r for r in matched if timestamp(r["created_at"]) >= start][-STATUS_LIMIT:]
        basis = "window"
        if not window:
            window, basis = matched[-1:], "latest_outside_window"
        if not window:
            return {"answer": None, "evidence": [], "category_counts": {}, "total": 0,
                    "note": f"{record['device_id']}（{record['name']}）在快照前沒有字面匹配的工單。"}
        categories, scheme = self.classify(window)
        newest_first = window[::-1]
        counts = Counter(c for c in (categories[r["ticket_id"]] for r in window) if c is not None)
        answer = None
        if counts:
            best = max(counts.values())
            # Newest ticket among the tied categories decides the tie.
            answer = next(categories[r["ticket_id"]] for r in newest_first
                          if counts.get(categories[r["ticket_id"]]) == best)
        latest = newest_first[0]
        excerpt = latest["raw_text"][:40] + ("…" if len(latest["raw_text"]) > 40 else "")
        scope = (f"{start.strftime('%Y-%m-%dT%H:%M:%SZ')} 起至快照 {len(window)} 張" if basis == "window"
                 else "期間內沒有工單，改看最新一張")
        summary = "、".join(f"{k} {v}" for k, v in counts.most_common()) or "無法判斷類別"
        return {"answer": answer, "evidence": [r["ticket_id"] for r in newest_first[:20]],
                # No raw records: the model only needs to finish, and ~860 extra prompt tokens
                # cost ~6 s of the 30 s budget. The note carries the newest excerpt for people.
                "category_counts": dict(counts), "basis": basis, "total": len(window),
                "note":(f"{record['device_id']}（{record['name']}）{scope}：{summary}。"
                         f"最新 {latest['ticket_id']}（{latest['created_at']}）：「{excerpt}」。"
                         f"類別由第一部分分類器（{scheme}）依原文判斷，不是原文標籤；"
                         "設備為字面匹配，錯字、簡體或代稱的工單可能漏掉。")}
