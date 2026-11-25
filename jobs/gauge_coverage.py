"""Audit published Nigerian daily streamflow observations for forecast evaluation."""

import argparse
import csv
from datetime import date, timedelta
from hashlib import sha256
from math import isfinite
from pathlib import Path

FILES = {"Hadejia": 11450, "Jamaare": 11449, "Kaduna": 11446, "Oroo": 11447}


def inspect(path):
    lines = path.read_text().splitlines()
    missing = float(lines[1].split()[1])
    observed = set()
    seen = set()
    for line in lines[5:]:
        y, m, d, h, minute, q = line.split()
        day = date(int(y), int(m), int(d))
        if day in seen or int(h) or int(minute):
            raise ValueError(f"Duplicate or unexpected observation time: {day}")
        seen.add(day)
        v = float(q)
        if v == missing:
            continue
        if not isfinite(v) or v < 0:
            raise ValueError(f"Invalid discharge: {day}")
        observed.add(day)
    return dict(
        basin=path.stem,
        grdc_id_in_header=lines[0].split()[0],
        first_observation=min(observed),
        last_observation=max(observed),
        observed_days=len(observed),
        missing_rows=len(seen) - len(observed),
        complete_three_day_blocks=sum(
            all(d + timedelta(days=i) in observed for i in [1, 2]) for d in observed
        ),
        observed_days_from_2023=sum(d.year >= 2023 for d in observed),
        sha256=sha256(path.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/flood_reference/bonn"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    for basin, ident in FILES.items():
        path = args.root / (basin + ".day")
        url = f"https://bonndata.uni-bonn.de/api/access/datafile/{ident}"
        if not path.exists():
            import requests

            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        rows.append(dict(**inspect(path), source=url))
    output = Path("reports/gauge_review/coverage.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
