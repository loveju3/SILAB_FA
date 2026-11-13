"""Shared history-only selection metrics for TF-IDF and frozen embeddings."""
import sys

from triage.io import ROOT

sys.path.insert(0, str(ROOT))
from baseline.evaluate import CATEGORIES, macro_f1

PROVENANCE = "訓練與選型僅使用 history；沿用 A 的正規化、別名、遮罩及其他規則曾受 dev 觀察影響，非全流程獨立未觀察 dev 的結果。"


def category_metrics(gold, predictions):
    pairs = [(g, p) for g, p in zip(gold, predictions) if g is not None]
    _, five = macro_f1(pairs, CATEGORIES)
    six_pairs = [(g or "unknown", p or "unknown") for g, p in zip(gold, predictions)]
    classes, six = macro_f1(six_pairs, [*CATEGORIES, "unknown"])
    return {"category_macro_f1": five, "six_class_macro_f1": six,
            "unknown_precision": classes["unknown"][0], "unknown_recall": classes["unknown"][1],
            "unknown_support": classes["unknown"][3]}
