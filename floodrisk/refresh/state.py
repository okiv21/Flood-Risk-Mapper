"""Publish immutable snapshots and evaluate their freshness at read time."""

from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import os
import tempfile

MAX_AGE_HOURS = {"forecast": 9, "recent_rain": 36, "satellite": 36, "sources": 24 * 40}


def utc(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError("Timezone required")
    return dt.astimezone(timezone.utc)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def publish(root, kind, snapshot, now=None):
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    now = now or datetime.now(timezone.utc)
    summary = json.loads((snapshot / "summary.json").read_text())
    timestamp = utc(summary["issue_utc"])
    if not timedelta(0) <= now - timestamp <= timedelta(hours=MAX_AGE_HOURS[kind]):
        raise ValueError("Snapshot too old or dated in the future")
    if summary.get("window_end_utc") and utc(summary["window_end_utc"]) <= now:
        raise ValueError("Cannot publish an expired forecast")
    manifest = {
        str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(snapshot.rglob("*"))
        if p.is_file() and p.name != "batch.lock"
    }
    entry = {
        "kind": kind,
        "snapshot": str(snapshot.relative_to(root)),
        "issue_utc": summary["issue_utc"],
        "files": manifest,
        "window_end_utc": summary.get("window_end_utc"),
    }
    latest = root / kind / "latest.json"
    if latest.exists() and utc(json.loads(latest.read_text())["issue_utc"]) > timestamp:
        raise ValueError("Cannot replace a newer snapshot")
    atomic_json(latest, entry)
    return entry


def freshness(root, kind, now=None):
    now = now or datetime.now(timezone.utc)
    path = Path(root) / kind / "latest.json"
    if not path.exists():
        return {"kind": kind, "status": "missing"}
    entry = json.loads(path.read_text())
    age = (now - utc(entry["issue_utc"])).total_seconds() / 3600
    expired = entry.get("window_end_utc") and utc(entry["window_end_utc"]) <= now
    status = "fresh" if 0 <= age <= MAX_AGE_HOURS[kind] and not expired else "stale"
    return {
        "kind": kind,
        "status": status,
        "age_hours": age,
        "issue_utc": entry["issue_utc"],
        "window_end_utc": entry.get("window_end_utc"),
    }
