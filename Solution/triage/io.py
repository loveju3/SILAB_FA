"""Filesystem boundary: project only the four allowed input fields."""
import json
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from time import sleep

from .contracts import Ticket

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tickets(path: Path) -> list[Ticket]:
    tickets, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("每行必須是 JSON object")
                values = {k: row[k] for k in ("ticket_id", "created_at", "reporter", "raw_text")}
                if any(not isinstance(v, str) for v in values.values()):
                    raise ValueError("四個輸入欄位必須為字串")
                if not values["ticket_id"].strip() or not values["reporter"].strip():
                    raise ValueError("ticket_id/reporter 不可空白")
                timestamp = datetime.fromisoformat(values.pop("created_at").replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("created_at 必須包含時區")
                ticket = Ticket(created_at=timestamp.astimezone(timezone.utc), **values)
                if ticket.ticket_id in seen:
                    raise ValueError(f"ticket_id 重複：{ticket.ticket_id}")
                seen.add(ticket.ticket_id)
                tickets.append(ticket)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return tickets


def load_json(path: Path):
    for attempt in range(8):
        try:
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)
        except PermissionError:
            if os.name != 'nt' or attempt == 7:
                raise
            sleep(.01 * 2**attempt)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A failed run must not leave a partial file that looks like valid predictions.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
        for attempt in range(8):
            try:
                os.replace(temp_path, path)
                break
            except PermissionError:
                # Windows readers and antivirus scans may briefly deny replace.
                # Keep the previous complete JSON visible while retrying.
                if os.name != 'nt' or attempt == 7:
                    raise
                sleep(.01 * 2**attempt)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
