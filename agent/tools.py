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


DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")
TIME_WORDS = re.compile(r"\d{4}-\d{1,2}|\d{1,2}月|\d{1,2}[日號]|今天|昨天|前天|本週|本周|這週|最近|上週|today", re.I)


def boundary(value, name):
    """since/until: a full UTC timestamp, or a bare YYYY-MM-DD meaning that UTC day's 00:00."""
    try:
        if DATE_ONLY.fullmatch(value):
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return timestamp(value)
    except ValueError as exc:
        raise ValueError(f"{name} 格式錯誤；請填 YYYY-MM-DD 或含時區的時間") from exc


# Layer 1: deterministic spelling variants. Simplified forms of every character used in the
# public device names and locations (devices.json), so 「B1 动力室」 equals 「B1 動力室」.
S2T = str.maketrans("动楼机厅仓库后门货码头检员产线区卫顶压盘电湿环发输带马达监饮冻柜",
                    "動樓機廳倉庫後門貨碼頭檢員產線區衛頂壓盤電濕環發輸帶馬達監飲凍櫃")


def lookup_key(value):
    return re.sub(r"[-_\s]", "", normalize(value).translate(S2T))


# Identifiers stay free text: a grammar enum of public values made the 1.7B model force
# unknown targets (D13, 二樓機房) onto a similar real one. Tools match by lookup_key and an
# unmatched value deterministically yields null / an empty list.


def placeholder_check(value):
    """Example values in the prompt are <...> placeholders; copying one is never a real argument."""
    if isinstance(value, dict):
        for child in value.values():
            placeholder_check(child)
    elif isinstance(value, list):
        for child in value:
            placeholder_check(child)
    elif isinstance(value, str) and "<" in value and ">" in value:
        raise ValueError("參數含範例佔位符；請填題目中的實際值，題目沒提到的條件就省略")


def object_schema(properties, required=()):
    return {"type": "object", "properties": properties,
            "required": list(required), "additionalProperties": False}


STRING = {"type": "string"}
SCHEMAS = {
    "get_device": object_schema({
        "identifier": {"type": "string", "description": "照抄題目中的設備編號、名稱或型號。"},
        "field": {"type": "string", "enum": ["device_id", "name", "location", "model", "critical"]},
    }, ["identifier", "field"]),
    "devices_at_location": object_schema({
        "location": {"type": "string", "description": "照抄題目中的位置。"},
    }, ["location"]),
    "list_devices": object_schema({
        "selection": {"type": "string", "enum": ["all", "critical", "noncritical"]},
    }, ["selection"]),
    "get_error_code": object_schema({
        "code": {"type": "string", "description": "照抄題目中的錯誤碼。"},
        "field": {"type": "string", "enum": ["category", "meaning", "model"]},
    }, ["code"]),
    "query_tickets": object_schema({
        "device_id": {"type": "string", "description": "題目中的設備編號、名稱或型號；題目沒提設備就省略。"},
        "reporter": {"type": "string", "description": "逐字照抄題目中的完整回報人名稱。"},
        "day": {"type": "string", "description": "只問某一天當天：YYYY-MM-DD，或 today（快照日）。"},
        "last_days": {"type": "integer", "description": "最近N天，從快照回推。"},
        "since": {"type": "string", "description": "某日之後（含當日），YYYY-MM-DD。"},
        "until": {"type": "string", "description": "到此時間前（不含），YYYY-MM-DD。"},
        "contains_any": {"type": "array", "items": STRING, "maxItems": 12,
                         "description": "問題限定原文包含A或B時，必須填[A,B]；省略代表沒有文字條件。"},
        "operation": {"type": "string", "enum": ["count", "top_reporter", "reporters", "ticket_ids", "read"]},
    }, ["operation"]),
    "get_ticket_details": object_schema({
        "ticket_ids": {"type": "array", "items": STRING, "minItems": 1, "maxItems": 20},
        "field": {"type": "string", "enum": ["raw_text", "reporter", "created_at"],
                  "description": "問單一工單的某欄位時填。"},
    }, ["ticket_ids"]),
    "ticket_status": object_schema({
        "device": {"type": "string", "description": "題目中的設備編號、名稱或型號；回該設備最近工單的故障類別。"},
        "since": {"type": "string", "description": "問題明確給期間才填，YYYY-MM-DD；省略代表快照前 7 天。"},
    }, ["device"]),
}
STATUS_DAYS = 7
STATUS_LIMIT = 50  # bound classifier input; the newest tickets in the window are kept


def validate(value, schema):
    """Validate the deliberately small schema subset used by our tools."""
    kind = schema["type"]
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int,
             "null": type(None)}
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
        placeholder_check(args)
        return getattr(self, name)(**args)

    def resolve_device(self, value):
        """Device ID for an ID/name/model (spelling variants allowed), else None."""
        key = lookup_key(value)
        return next((d["device_id"] for d in self.devices if key in {
            lookup_key(d[k]) for k in ("device_id", "name", "model")}), None)

    def misplaced(self, value):
        """Explain a value that clearly belongs to another parameter or tool, else None."""
        if self.resolve_device(value):
            return f"「{value}」是設備，請改用 device_id"
        if any(r["ticket_id"] == value for r in self.tickets):
            return f"「{value}」是工單編號，請改用 get_ticket_details"
        if TIME_WORDS.search(value):
            return f"「{value}」是時間，請改用 day／last_days／since／until"
        return None

    def get_devices(self, identifier="", location="", critical=None, field="device_id", single=False):
        rows = [d for d in self.devices if
                (not identifier or lookup_key(identifier) in {
                    lookup_key(d[k]) for k in ("device_id", "name", "model")}) and
                (not location or lookup_key(location) in lookup_key(d["location"])) and
                (critical is None or d["critical"] == critical)]
        values = list(dict.fromkeys(d[field] for d in rows))
        if single and len(values) > 1:
            raise ValueError("符合多個值；請縮小範圍或使用清單")
        return {"answer": (values[0] if values else None) if single else values,
                "evidence": [], "records": rows, "source": "devices.json"}

    def get_device(self, identifier, field):
        return self.get_devices(identifier=identifier, field=field, single=True)

    def devices_at_location(self, location):
        result = self.get_devices(location=location)
        if not result["answer"] and self.resolve_device(location):
            raise ValueError(f"「{location}」是設備不是位置；問設備在哪裡請用 get_device field=location")
        return result

    def list_devices(self, selection):
        critical = {"all": None, "critical": True, "noncritical": False}[selection]
        return self.get_devices(critical=critical)

    def get_error_code(self, code, field="category"):
        code = re.sub(r"[-_\s]", "", unicodedata.normalize("NFKC", code)).upper()
        key = next((k for k in self.codes if k.replace("-", "") == code), None)
        row = self.codes.get(key)
        return {"answer": row[field] if row else None, "evidence": [],
                "record": row, "source": "error_codes.json"}

    def date_range(self, day="", last_days=None, since="", until=""):
        """Resolve date arguments against the data snapshot (never the computer clock)."""
        snapshot = timestamp(SNAPSHOT)
        # Overlapping date arguments: the most specific wins (day > last_days > since). An error
        # here made the model resend the same mixed arguments until it ran out of steps.
        if day:
            last_days = None
        elif last_days is not None:
            since = ""
        if day:
            if day.casefold() == "today":
                day = SNAPSHOT[:10]
            if not DATE_ONLY.fullmatch(day):
                raise ValueError("day 格式為 YYYY-MM-DD 或 today")
            since, until = day, (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d")
        start = end = None
        if last_days is not None:
            if not 1 <= last_days <= 3650:
                raise ValueError("last_days 必須是 1 以上的天數")
            start = snapshot - timedelta(days=last_days)
        if since:
            start = boundary(since, "since")
        if until:
            end = boundary(until, "until")
        if start and start >= snapshot:
            raise ValueError(f"開始時間不早於資料快照 {SNAPSHOT}，不會有工單；「今天」是快照日 "
                             f"{SNAPSHOT[:10]}，不是電腦日期，請改正日期條件")
        if start and end and start >= end:
            raise ValueError("since 必須早於 until（since 含邊界，until 不含）")
        fmt = lambda value: value.strftime("%Y-%m-%dT%H:%M:%SZ") if value else ""
        return start, end, fmt(start), fmt(end)

    def query_tickets(self, operation, device_id="", reporter="", day="", last_days=None,
                      since="", until="", contains_any=None):
        start, end, since, until = self.date_range(day, last_days, since, until)
        if contains_any and any(not term.strip() for term in contains_any):
            raise ValueError("文字條件不可為空白")
        terms = []
        resolved = self.resolve_device(device_id) if device_id else None
        unknown_device = bool(device_id) and resolved is None
        if resolved:
            device_id = resolved
            device = next(d for d in self.devices if d["device_id"] == device_id)
            terms = [normalize(device[k]) for k in ("device_id", "name", "model")]
        if reporter and reporter not in {r["reporter"] for r in self.tickets}:
            reason = self.misplaced(reporter)
            if reason:
                raise ValueError(reason)
            key = normalize(reporter)
            similar = sorted({r["reporter"] for r in self.tickets if key in normalize(r["reporter"])})[:5]
            if similar:
                raise ValueError(f"沒有回報人「{reporter}」；請逐字使用完整名稱，可能是：{'、'.join(similar)}")
        matches = []
        for row in [] if unknown_device else self.tickets:
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
        result = {"answer": answer, "evidence": [r["ticket_id"] for r in cited[:20]],
                  "total": len(matches), "records": matches[-20:] if operation == "read" else [],
                  "applied_filters": {"device_id": device_id, "reporter": reporter,
                                      "since": since, "until": until, "contains_any": contains_any or []},
                  "truncated": operation == "read" and len(matches) > 20,
                  "note": "依原始工單完整集合計算；文字／設備條件僅為字面匹配，不代表語意分類或指涉已確認。"}
        if unknown_device:
            result["note"] = "設備不在公開設備表中，沒有可對應的工單。"
        elif not matches:
            result["hint"] = "結果為空；確認每個條件都出自題目，題目沒提到的條件要省略"
        return result

    def get_ticket_details(self, ticket_ids, field=None):
        index = {r["ticket_id"]: r for r in self.tickets}
        missing = [key for key in ticket_ids if key not in index]
        if field and len(ticket_ids) != 1:
            raise ValueError("field 只能搭配單一工單")
        if missing and len(missing) < len(ticket_ids):
            raise ValueError("部分工單不存在：" + "、".join(missing))
        if missing:
            return {"answer": None, "evidence": [], "records": [],
                    "note": "查無工單：" + "、".join(missing)}
        records = [index[key] for key in ticket_ids]
        return {"answer": records[0][field] if field else ticket_ids, "evidence": ticket_ids,
                "records": records}

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
            return {"answer": None, "evidence": [], "category_counts": {}, "total": 0,
                    "note": "設備不在公開設備表中。"}
        record = next(d for d in self.devices if d["device_id"] == values)
        snapshot = timestamp(SNAPSHOT)
        start = boundary(since, "since") if since else snapshot - timedelta(days=STATUS_DAYS)
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
