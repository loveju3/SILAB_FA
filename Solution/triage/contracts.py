"""Input and output contracts shared by the CLI and prediction pipeline."""
from dataclasses import dataclass, asdict
from datetime import datetime

CATEGORIES = {"noise", "temperature", "power", "software", "other"}
URGENCIES = {"low", "normal", "high"}


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    created_at: datetime
    reporter: str
    raw_text: str


@dataclass(frozen=True)
class Prediction:
    ticket_id: str
    device_id: str | None
    category: str | None
    urgency: str
    duplicate_of: str | None
    llm_used: bool
    suggestion: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_predictions(tickets: list[Ticket], predictions: list[Prediction],
                         device_ids: set[str]) -> None:
    if [t.ticket_id for t in tickets] != [p.ticket_id for p in predictions]:
        raise ValueError("輸出必須依輸入順序，每個 ticket_id 恰好一筆")
    order = {t.ticket_id: i for i, t in enumerate(sorted(tickets, key=lambda t: t.created_at))}
    by_id = {p.ticket_id: p for p in predictions}
    reporters = {t.ticket_id: t.reporter for t in tickets}
    for p in predictions:
        if p.device_id is not None and p.device_id not in device_ids:
            raise ValueError(f"{p.ticket_id}: 未知設備")
        if p.category is not None and p.category not in CATEGORIES:
            raise ValueError(f"{p.ticket_id}: 未知類別")
        if p.urgency not in URGENCIES or type(p.llm_used) is not bool:
            raise ValueError(f"{p.ticket_id}: urgency/llm_used 不合法")
        if not isinstance(p.suggestion, str) or not 0 < len(p.suggestion) <= 40:
            raise ValueError(f"{p.ticket_id}: suggestion 須為 1–40 字")
        if p.duplicate_of is not None:
            if p.device_id is None or p.category is None:
                raise ValueError(f"{p.ticket_id}: 未知設備/類別不能建立 duplicate")
            if p.duplicate_of not in order or order[p.duplicate_of] >= order[p.ticket_id]:
                raise ValueError(f"{p.ticket_id}: duplicate 必須指向前單")
            prior = by_id[p.duplicate_of]
            if (prior.device_id, prior.category, reporters[prior.ticket_id]) != (
                    p.device_id, p.category, reporters[p.ticket_id]):
                raise ValueError(f"{p.ticket_id}: duplicate 必須同人、同設備、同類別")
