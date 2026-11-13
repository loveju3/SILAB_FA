"""Time-ordered history split with near-template groups kept apart."""
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from triage.contracts import CATEGORIES
from triage.io import read_tickets
from triage.text import classification_text


def load_history(path: Path):
    if path.name != "history.jsonl" or path.is_symlink():
        raise ValueError("訓練只接受 history.jsonl，不接受 dev 或其他資料集")
    tickets = read_tickets(path)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    for row in rows:
        if "category" not in row or row["category"] not in CATEGORIES | {None}:
            raise ValueError("history category 必須是五類之一或 null")
    labels = {row["ticket_id"]: row for row in rows}
    ordered = sorted(tickets, key=lambda t: t.created_at)
    return ordered, labels


def history_split(tickets, resolver, fraction=.7, similarity=.9):
    if not 0 < fraction < 1 or not 0 < similarity <= 1 or len(tickets) < 2:
        raise ValueError("分割參數或 history 筆數不合法")
    cutoff = max(1, min(len(tickets) - 1, int(len(tickets) * fraction)))
    boundary = tickets[cutoff].created_at
    train_candidates = [i for i, t in enumerate(tickets) if t.created_at < boundary]
    validation = [i for i, t in enumerate(tickets) if t.created_at >= boundary]
    # Group using text only, never category/device gold labels. Mask equipment so
    # a shared symptom template does not leak merely through a different name.
    signatures = [re.sub(r"[\W_]+", "", resolver.mask_names(classification_text(t.raw_text)))
                  for t in tickets]
    parents = list(range(len(tickets)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, a in enumerate(signatures):
        for j in range(i):
            b = signatures[j]
            if (not a and not b) or (a and b and SequenceMatcher(None, a, b, autojunk=False).ratio() >= similarity):
                parents[find(i)] = find(j)
    validation_groups = {find(i) for i in validation}
    train = [i for i in train_candidates if find(i) not in validation_groups]
    purged = [i for i in train_candidates if find(i) in validation_groups]
    if not train or not validation:
        raise ValueError("時間／模板分割後沒有可用訓練或驗證資料")
    return train, validation, purged
