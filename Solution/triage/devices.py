"""Evidence-based device candidates, separate from reporter context."""
from dataclasses import dataclass
import re

from .text import error_codes, matches, normalize


@dataclass(frozen=True)
class DeviceMatch:
    candidates: frozenset[str]
    evidence: tuple[str, ...]


class DeviceResolver:
    def __init__(self, devices: list[dict], codes: dict, aliases: dict):
        self.devices = {d["device_id"]: d for d in devices}
        self.codes = codes
        self.aliases = aliases

    def candidates(self, text: str) -> DeviceMatch:
        text = normalize(text)
        explicit, locations, code_ids, evidence = set(), set(), set(), []
        for device_id, device in self.devices.items():
            names = [device["name"], device["model"], *self.aliases.get(device_id, [])]
            id_hit = re.search(rf"(?<![a-z0-9]){device_id.lower()}(?![a-z0-9])", text)
            if id_hit or any(matches(n, text, missing=True) for n in names):
                explicit.add(device_id)
                evidence.append(f"direct:{device_id}")
            if matches(device["location"], text, missing=True):
                locations.add(device_id)
        for code in error_codes(text):
            if code in self.codes:
                code_ids.update(d["device_id"] for d in self.devices.values()
                                if d["model"] == self.codes[code]["model"])
                evidence.append(f"code:{code}")
        # Explicit names outrank incidental/contradictory codes. Multiple named
        # devices remain ambiguous unless location or code singles out one.
        candidates = explicit or code_ids or locations
        for hint in (locations, code_ids):
            intersection = candidates & hint
            if len(candidates) > 1 and intersection:
                candidates = intersection
        if locations:
            evidence.append("location:" + ",".join(sorted(locations)))
        return DeviceMatch(frozenset(candidates), tuple(evidence))

    def mask_names(self, text: str) -> str:
        terms = [normalize(term) for d in self.devices.values()
                 for term in (d["name"], d["location"], d["model"],
                              *self.aliases.get(d["device_id"], []))]
        for term in sorted(set(terms), key=len, reverse=True):
            text = text.replace(term, " ")
        return text
