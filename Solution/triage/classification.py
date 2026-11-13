"""Rule classifier: positive symptom evidence distinguishes other from unknown."""
from dataclasses import dataclass
from typing import Protocol

from .text import error_codes, matches


@dataclass(frozen=True)
class CategoryMatch:
    category: str | None
    evidence: str | None


class CategoryClassifier(Protocol):
    mask_device_names: bool

    def classify(self, text: str) -> CategoryMatch: ...


class RuleClassifier:
    mask_device_names = True

    def __init__(self, rules: list[dict], codes: dict):
        self.rules = rules
        self.codes = codes

    def classify(self, text: str) -> CategoryMatch:
        known = [self.codes[c]["category"] for c in error_codes(text) if c in self.codes]
        if len(set(known)) == 1:
            return CategoryMatch(known[0], next(c for c in error_codes(text) if c in self.codes))
        hits = [(rule["weight"], len(term), -index, rule["category"], term)
                for index, rule in enumerate(self.rules) for term in rule["terms"]
                if matches(term, text, missing=True)]
        if not hits:
            return CategoryMatch(None, None)
        _, _, _, category, evidence = max(hits)
        return CategoryMatch(category, evidence)
