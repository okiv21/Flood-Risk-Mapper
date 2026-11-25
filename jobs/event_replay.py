"""Retrieve original model runs preceding two independently reported Nigerian floods."""

import csv
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path("reports/event_replay")
CASES = {
    "Mokwa": "2025-05-28",
    "Maiduguri": "2024-09-09",
}
URL = "https://single-runs-api.open-meteo.com/v1/forecast"


def summarize(data, run):
    hours = data["hourly"]["time"]
    values = data["hourly"]["precipitation"]
    # Accumulations end at each timestamp; exclude the initialization hour.
    expected = [
        (run + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(73)
    ]
    if (
        hours[:73] != expected
        or data["utc_offset_seconds"] != 0
        or data["hourly_units"]["precipitation"] != "mm"
        or len(values) != len(hours)
    ):
        raise ValueError("Unexpected forecast coverage or units")
    rain = values[1:73]
    if any(v is None or not math.isfinite(v) or v < 0 for v in rain):
        raise ValueError("Incomplete or invalid forecast precipitation")
    return {f"forecast_{h}h_mm": round(sum(rain[:h]), 2) for h in (24, 48, 72)}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with Path("reports/hotspots_oct22/lgas.csv").open() as f:
        sites = {row["name"]: row for row in csv.DictReader(f) if row["name"] in CASES}
    session = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
            )
        ),
    )
    rows = []
    for site, date in CASES.items():
        for days in (3, 2, 1):
            run = datetime.fromisoformat(date).replace(tzinfo=timezone.utc) - timedelta(
                days=days
            )
            params = dict(
                latitude=sites[site]["latitude"],
                longitude=sites[site]["longitude"],
                models="ecmwf_ifs",
                hourly="precipitation",
                forecast_hours=73,
                run=run.strftime("%Y-%m-%dT%H:%M"),
            )
            raw = ROOT / f"{site.lower()}_{run.date()}.json"
            if not raw.exists():
                response = session.get(URL, params=params, timeout=(12, 30))
                response.raise_for_status()
                summarize(response.json(), run)
                raw.write_bytes(response.content)
            data = json.loads(raw.read_bytes())
            rows.append(
                dict(
                    site=site,
                    lga_id=sites[site]["lga_id"],
                    model_init_utc=run.isoformat(),
                    query=requests.Request("GET", URL, params=params).prepare().url,
                    raw_file=raw.name,
                    sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
                    **summarize(data, run),
                )
            )
    with (ROOT / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
