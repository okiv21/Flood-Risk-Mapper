"""Contract for one LGA forecast and independently observed onset windows."""

from datetime import datetime, timedelta, timezone
from math import isfinite

RAINFALL = [
    "past_1d_mm",
    "past_3d_mm",
    "past_7d_mm",
    "past_14d_mm",
    "next_24h_mm",
    "next_48h_mm",
    "next_72h_mm",
    "max_hourly_72h_mm",
]
FEATURES = RAINFALL + ["susceptibility", "built_up_fraction", "month"]


def instant(text):
    value = datetime.fromisoformat(text)
    if value.utcoffset() is None:
        raise ValueError("A timezone offset is required")
    return value.astimezone(timezone.utc)


def features(row):
    issue = instant(row["issue_utc"])
    start = instant(row["window_start_utc"])
    if (
        not issue <= start < issue + timedelta(hours=1)
        or start.minute
        or start.second
        or start.microsecond
    ):
        raise ValueError("Forecast must start at the next complete UTC hour")
    if instant(row["window_end_utc"]) != start + timedelta(hours=72):
        raise ValueError("Require exactly 72 hours")
    for k in [
        "forecast_available_utc",
        "antecedent_available_utc",
        "static_available_utc",
    ]:
        if instant(row[k]) > issue:
            raise ValueError(f"Input arrived after issue time: {k}")
    if (
        not row["lga_id"]
        or not row["forecast_source"]
        or not row["antecedent_source"]
        or not row["susceptibility_version"]
    ):
        raise ValueError("Missing input provenance")
    values = {k: float(row[k]) for k in FEATURES if k != "month"}
    if any(not isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Missing or invalid feature")
    if values["susceptibility"] > 1 or values["built_up_fraction"] > 1:
        raise ValueError("Expected fractions")
    for sequence in [RAINFALL[:4], RAINFALL[4:7]]:
        if any(values[a] > values[b] for a, b in zip(sequence, sequence[1:])):
            raise ValueError("Inconsistent cumulative rain")
    if values["max_hourly_72h_mm"] > values["next_72h_mm"]:
        raise ValueError("Hourly rain exceeds window total")
    return {**values, "month": start.month}


def outcome(row):
    """Unknown windows remain missing. Start is inclusive, end exclusive."""
    start, end = instant(row["window_start_utc"]), instant(row["window_end_utc"])
    if not row.get("evidence_id") or not row.get("reviewer"):
        return None
    kind = row.get("evidence_kind")
    if kind not in ("onset_interval", "continuous_event_free"):
        return None
    lo, hi = instant(row["evidence_start_utc"]), instant(row["evidence_end_utc"])
    if hi < lo:
        raise ValueError("Evidence interval reversed")
    if kind == "onset_interval" and start <= lo <= hi < end:
        return 1
    if kind == "continuous_event_free" and lo <= start and hi >= end:
        return 0
    return None
