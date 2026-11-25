"""Evaluate the preregistered point-rainfall and calendar rules."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from jobs.satellite_replay import fetch, HOURS, OUT


def score(project):
    with (OUT / "onitsha_south.csv").open() as f:
        cases = list(csv.DictReader(f))
    with Path("reports/hotspots_oct22/lgas.csv").open() as f:
        point = next(r for r in csv.DictReader(f) if r["lga_id"] == cases[0]["lga_id"])
    training = [
        r for r in cases if int(r["year"]) <= 2021 and r["proxy"] != "unresolved"
    ]
    predictions = []
    for year in range(2018, 2026):
        targets = []
        for r in cases:
            if int(r["year"]) != year:
                continue
            for lead in (24, 48, 72):
                t = datetime.fromisoformat(r["observed_utc"]) - timedelta(hours=lead)
                run = t.replace(hour=t.hour // 6 * 6, minute=0, second=0, microsecond=0)
                targets.append(
                    dict(
                        key=f"{r['radar_id']}|{lead}",
                        run_ms=int(run.timestamp() * 1000),
                        lat=float(point["latitude"]),
                        lon=float(point["longitude"]),
                        lead=lead,
                        row=r,
                    )
                )
        if not targets:
            continue
        path = OUT / f"onitsha_gfs_{year}.json"
        blocks = (
            json.loads(path.read_text()) if path.exists() else fetch(project, targets)
        )
        for target in targets:
            b = sorted(
                (b for b in blocks if b["key"] == target["key"]),
                key=lambda b: b["hour"],
            )
            if [v["hour"] for v in b] != HOURS or any(
                v["run_ms"] != target["run_ms"]
                or v["valid_ms"] != target["run_ms"] + 3600000 * v["hour"]
                or not math.isfinite(v["mm"])
                or v["mm"] < 0
                for v in b
            ):
                raise ValueError(f"Incomplete archive: {target['key']}")
            lead = target["lead"]
            total = sum(v["mm"] for v in b if lead - 24 < v["hour"] <= lead)
            r = target["row"]
            refs = [x["proxy"] for x in training if x["month"] == r["month"]]
            calendar = (
                "unavailable"
                if not refs
                else "above"
                if refs.count("above") > len(refs) / 2
                else "below_or_equal"
            )
            predictions.append(
                dict(
                    year=year,
                    period="train"
                    if year < 2022
                    else "validation"
                    if year == 2022
                    else "test",
                    month=r["month"],
                    lead_h=lead,
                    radar_id=r["radar_id"],
                    target=r["proxy"],
                    model_init_ms=target["run_ms"],
                    rain_24h_mm=total,
                    rainfall_rule="above" if total >= 25 else "below_or_equal",
                    calendar_rule=calendar,
                )
            )
        if not path.exists():
            path.write_text(json.dumps(blocks, indent=2) + "\n")
    with (OUT / "onitsha_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, predictions[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(predictions)
    summary = []
    for period in ["validation", "test"]:
        for lead in (24, 48, 72):
            for rule in ["rainfall_rule", "calendar_rule"]:
                counts = Counter()
                for r in predictions:
                    if r["period"] != period or r["lead_h"] != lead:
                        continue
                    actual, pred = r["target"], r[rule]
                    if actual == "unresolved" or pred == "unavailable":
                        counts["excluded"] += 1
                    else:
                        counts[
                            {
                                (True, True): "tp",
                                (True, False): "fn",
                                (False, True): "fp",
                                (False, False): "tn",
                            }[(actual == "above", pred == "above")]
                        ] += 1
                summary.append(
                    dict(
                        period=period,
                        lead_h=lead,
                        rule=rule,
                        **{k: counts[k] for k in ["tp", "fn", "fp", "tn", "excluded"]},
                    )
                )
    with (OUT / "onitsha_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, summary[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    score(p.parse_args().project)
