"""Measurements with explicit scope and platform-dependent RSS units."""
import math
import platform
import resource


def latency_summary(timings: list[dict]) -> dict:
    values = sorted(row["seconds"] for row in timings)
    def percentile(fraction):
        return values[max(0, math.ceil(len(values) * fraction) - 1)] if values else None
    return {"count": len(values), "p50_seconds": percentile(.5),
            "p95_seconds": percentile(.95), "max_seconds": values[-1] if values else None,
            "method": "nearest-rank; per ticket includes normalization, rules, context and suggestion; excludes load, sort, validation and serialization"}


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if platform.system() == "Darwin" else value * 1024)
