"""Archive a single national batch of as-retrieved Open-Meteo rainfall features."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import fcntl
import time

import requests
from floodrisk.features.rainfall import (
    forecast_features,
    antecedent_features,
    nonnegative,
)


def fetch(registry, output):
    with registry.open() as f:
        locations = list(csv.DictReader(f))
    if len(locations) != 774 or len({r["lga_id"] for r in locations}) != 774:
        raise ValueError("Incomplete LGA registry")
    if output.exists():
        raise ValueError("Snapshot already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    endpoint = "https://api.open-meteo.com/v1/forecast"
    stage = output.with_name(output.name + ".partial")
    stage.mkdir(exist_ok=True)
    with (stage / "batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        for i in range(0, len(locations), 10):
            batch = locations[i : i + 10]
            params = {
                axis: ",".join(format(float(row[axis]), ".5f") for row in batch)
                for axis in ["latitude", "longitude"]
            }
            params.update(
                hourly="precipitation", past_days=14, forecast_days=5, timezone="UTC"
            )
            saved = stage / f"batch_{i:03d}.json"
            cached = json.loads(saved.read_text()) if saved.exists() else None
            now = datetime.now(timezone.utc)
            if (
                cached
                and cached["request"] == params
                and now - datetime.fromisoformat(cached["retrieved_utc"])
                < timedelta(hours=3)
            ):
                items = cached["items"]
                received = datetime.fromisoformat(cached["retrieved_utc"])
            else:
                for attempt in range(4):
                    try:
                        response = requests.get(
                            endpoint, params=params, timeout=(15, 60)
                        )
                        response.raise_for_status()
                        break
                    except requests.RequestException as exc:
                        code = (
                            exc.response.status_code
                            if exc.response is not None
                            else None
                        )
                        if attempt == 3 or (
                            code is not None and code != 429 and code < 500
                        ):
                            raise
                        time.sleep(60 if code == 429 else 2**attempt)
                received = datetime.now(timezone.utc)
                items = response.json()
                if len(items) != len(batch):
                    raise ValueError("Location count mismatch")
                temp = saved.with_suffix(".tmp")
                temp.write_text(
                    json.dumps(
                        dict(
                            request=params,
                            retrieved_utc=received.isoformat(),
                            items=items,
                        )
                    )
                    + "\n"
                )
                temp.replace(saved)
            print(f"Archived {i + len(batch)}/{len(locations)} locations", flush=True)
            for j, (location, item) in enumerate(zip(batch, items)):
                if item.get("location_id", 0) != j:
                    raise ValueError("Location order mismatch")
                results.append((location, item, received.isoformat()))
        issue = datetime.now(timezone.utc)
        if any(
            issue - datetime.fromisoformat(received) > timedelta(hours=3)
            for _, _, received in results
        ):
            raise ValueError(
                "Batch contains rainfall older than three hours; retry to refresh cached inputs"
            )
        if len(results) != len(locations) or len(
            {r[0]["lga_id"] for r in results}
        ) != len(locations):
            raise ValueError("Incomplete or duplicated rainfall locations")
        rows = []
        for location, item, received in results:
            forecast = forecast_features(item, issue)
            hourly = item["hourly"]
            amounts = dict(zip(hourly["time"], hourly["precipitation"]))
            # Rain at each timestamp represents the preceding hour.
            midnight = issue.replace(hour=0, minute=0, second=0, microsecond=0)
            daily = []
            for n in range(14, 0, -1):
                day = midnight - timedelta(days=n)
                mm = sum(
                    nonnegative(
                        amounts[(day + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M")]
                    )
                    for h in range(1, 25)
                )
                daily.append(dict(date=day.date().isoformat(), mm=mm))
            rows.append(
                dict(
                    lga_id=location["lga_id"],
                    issue_utc=issue.isoformat(),
                    window_start_utc=forecast["window_start_utc"],
                    window_end_utc=forecast["window_end_utc"],
                    forecast_available_utc=received,
                    antecedent_available_utc=received,
                    forecast_source=endpoint,
                    antecedent_source="Open-Meteo past-hours model estimates",
                    **antecedent_features(daily, midnight.date().isoformat()),
                    **forecast["features"],
                )
            )
        with (stage / "rainfall.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(
            lgas=len(rows),
            issue_utc=issue.isoformat(),
            window_start_utc=forecast["window_start_utc"],
            window_end_utc=forecast["window_end_utc"],
            batches=(len(rows) + 9) // 10,
            support="One interior point per LGA; model grid point rainfall. Past 14 days are as-retrieved model estimates.",
            source=endpoint,
            forecast_run_id=None,
            unit="mm",
            product="rainfall features only",
        )
        (stage / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        stage.rename(output)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["registry", "output"]:
        p.add_argument("--" + name, required=True, type=Path)
    a = p.parse_args()
    print(fetch(a.registry, a.output))
