"""Ordered urgency, reporter context and suggestions, without file access."""
from dataclasses import dataclass

from .classification import CategoryMatch
from .contracts import Ticket
from .devices import DeviceMatch
from .text import matches

HIGH_1 = "整條產線停線 冒煙 焦味 快燒起來 整區停電 有人受困 已經停機 漏水停不下來 火災警報 火花 燒掉 全部作業都停了 溫度還在往上 不能等 線上的東西都卡住了".split()
HIGH_2 = "漏電 積水 漏油 起火 短路 觸電 傾倒 爆裂 有毒 高溫 異味 發熱".split()
CRITICAL = "跳脫 斷電 停機 停線 關機 不亮 沒電 燒掉".split()
REPEAT = "第三次 到底 已經報修 老問題 還沒好 又來了".split()
HEDGE = "好像 好像是 不確定 不太確定 應該是 應該吧 可能是 大概 似乎 也許 不曉得 聽同事說的 看起來像 我猜 別人跟我說".split()
REFERENCE = "那台 那邊 同一台 同一個問題 同個問題 老問題 又壞 又來了 上次那件 之前修過".split()
SAME_PROBLEM = "同一個問題 同個問題 老問題 上次那件".split()


def urgency(text: str, critical: bool) -> str:
    for words in (HIGH_1, HIGH_2, CRITICAL if critical else [], REPEAT):
        if any(matches(w, text, missing=True) for w in words):
            return "high"
    if any(matches(w, text, missing=True) for w in HEDGE):
        return "low"
    return "normal"


@dataclass(frozen=True)
class Context:
    device_id: str | None
    category: str | None
    source_ticket_id: str | None


def resolve_context(ticket: Ticket, text: str, device: DeviceMatch,
                    category: CategoryMatch, previous: Context | None) -> tuple[Context, CategoryMatch]:
    selected = next(iter(device.candidates)) if len(device.candidates) == 1 else None
    source = ticket.ticket_id if selected else None
    refers = any(matches(w, text, missing=True) for w in REFERENCE)
    if selected is None and previous and previous.device_id and refers:
        if not device.candidates or previous.device_id in device.candidates:
            selected, source = previous.device_id, previous.source_ticket_id
    # Class carry-over needs explicit same-problem evidence, never just "again".
    if (category.category is None and selected and previous and selected == previous.device_id
            and any(matches(w, text) for w in SAME_PROBLEM)):
        category = CategoryMatch(previous.category, "前單同問題")
    return Context(selected, category.category, source), category


def suggestion(ticket: Ticket, category: CategoryMatch, level: str) -> str:
    # Quote current raw text, not a normalized/invented diagnosis.
    excerpt = " ".join(ticket.raw_text.split())[:16] or "未提供現象"
    actions = {
        "noise": "檢查聲音與振動來源",
        "temperature": "檢查散熱與冷卻狀態",
        "power": "請電工檢查供電與保護裝置",
        "software": "核對畫面、通訊及系統紀錄",
        "other": "檢查所述零件或現場異常",
        None: "補充設備、位置及故障現象",
    }
    action = "先確認安全並通知維修" if level == "high" else actions[category.category]
    return f"針對「{excerpt}」，{action}"[:40]
