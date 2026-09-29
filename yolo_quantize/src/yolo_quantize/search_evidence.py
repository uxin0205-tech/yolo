"""Read both metric families from the hash-pinned raw validator artifact."""

import hashlib
import json
import math
from pathlib import Path

from .metric_gate import FULL35_MAP50_95_KEYS, FULL35_MAP50_KEYS


def read_dual_search_metrics(record):
    path = Path(record["metrics_report"])
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["metrics_report_sha256"]:
        raise ValueError("raw metrics report SHA-256 drifted")
    metrics = json.loads(raw)["metrics"]
    keys = (*FULL35_MAP50_KEYS, *FULL35_MAP50_95_KEYS)
    if not set(keys).issubset(metrics):
        raise ValueError("raw report is missing required dual metrics")
    result = {k: float(metrics[k]) for k in keys}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in result.values()):
        raise ValueError("raw metrics must be finite in [0,1]")
    return result
