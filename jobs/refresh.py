"""Run one scheduled acquisition and publish its validated snapshot."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from floodrisk.refresh.state import atomic_json, freshness, publish


def recent_rain(root, output, now):
    status = freshness(root, "forecast", now)
    if status["status"] != "fresh":
        raise ValueError(
            "A fresh complete forecast archive is required for recent rainfall"
        )
    pointer = json.loads((root / "forecast/latest.json").read_text())
    source = root / pointer["snapshot"]
    import csv
    import hashlib

    path = source / "rainfall.csv"
    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != pointer["files"]["rainfall.csv"]
    ):
        raise ValueError("Archived rainfall changed")
    with path.open() as f:
        rows = list(csv.DictReader(f))
    fields = [
        "lga_id",
        "issue_utc",
        "antecedent_available_utc",
        "antecedent_source",
        "past_1d_mm",
        "past_3d_mm",
        "past_7d_mm",
        "past_14d_mm",
        "max_daily_14d_mm",
    ]
    output.mkdir(parents=True)
    with (output / "recent_rain.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output / "summary.json",
        {
            "issue_utc": pointer["issue_utc"],
            "derived_at_utc": now.isoformat(),
            "lgas": len(rows),
            "parent_snapshot": pointer["snapshot"],
            "source": "Open-Meteo past-hours model estimates",
            "observation_type": "model estimate",
            "support": "One interior grid point per LGA; 14 complete preceding UTC days",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind", choices=["forecast", "recent_rain", "satellite", "sources", "status"]
    )
    parser.add_argument("--root", type=Path, default=Path("data/refresh"))
    parser.add_argument(
        "--registry", type=Path, default=Path("reports/hotspots_oct22/lgas.csv")
    )
    parser.add_argument("--project", default="manifest-ivy-453612-b3")
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.kind == "status":
        report = [
            freshness(args.root, kind, now)
            for kind in ["forecast", "recent_rain", "satellite", "sources"]
        ]
        print(json.dumps(report, indent=2))
        atomic_json(
            args.root / "status.json",
            {"checked_utc": now.isoformat(), "products": report},
        )
        return
    output = args.root / args.kind / now.strftime("%Y%m%dT%H%M%S%fZ")
    try:
        if args.kind == "forecast":
            from floodrisk.acquire.lga_rainfall import fetch

            fetch(args.registry, output)
        elif args.kind == "recent_rain":
            recent_rain(args.root, output, now)
        else:
            from floodrisk.refresh.catalog import collect

            collect(args.kind, args.project, args.root, output, now)
        entry = publish(args.root, args.kind, output)
        atomic_json(
            args.root / args.kind / "attempt.json",
            {
                "attempt_utc": now.isoformat(),
                "status": "success",
                "snapshot": entry["snapshot"],
            },
        )
        print(json.dumps(entry, indent=2))
    except Exception as error:
        atomic_json(
            args.root / args.kind / "attempt.json",
            {
                "attempt_utc": now.isoformat(),
                "status": "failed",
                "error_type": type(error).__name__,
            },
        )
        raise


if __name__ == "__main__":
    main()
