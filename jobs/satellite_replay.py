"""Replay original GFS precipitation at the eight October 2022 radar windows."""

import argparse
import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import ee
import numpy as np
import rasterio

from floodrisk.deploy.tiled import file_hash

OUT = Path("reports/satellite_replay")
HOURS = list(range(6, 73, 6))


def fetch(project, targets):
    ee.Initialize(project=project)
    sets = []
    for row in targets:
        location = ee.Geometry.Point([row["lon"], row["lat"]])
        run_ms = row["run_ms"]
        images = (
            ee.ImageCollection("NOAA/GFS0P25")
            .filter(ee.Filter.eq("creation_time", run_ms))
            .filter(ee.Filter.inList("forecast_hours", HOURS))
            .select("total_precipitation_surface")
        )

        def sample(obj):
            image = ee.Image(obj)
            rain = image.reduceRegion(
                ee.Reducer.first(), location, crs=image.projection(), maxPixels=4
            )
            return ee.Feature(
                None,
                dict(
                    key=row["key"],
                    source_id=image.id(),
                    run_ms=image.get("creation_time"),
                    valid_ms=image.get("forecast_time"),
                    hour=image.get("forecast_hours"),
                    mm=rain.get("total_precipitation_surface"),
                ),
            )

        sets.append(ee.FeatureCollection(images.toList(12).map(sample)))
    response = ee.FeatureCollection(sets).flatten().getInfo()
    return [f["properties"] for f in response["features"]]


def main(project):
    source = Path("reports/hotspots_oct22")
    changes = json.loads((source / "paired_change.json").read_text())
    observations = {
        o["site"]: o for o in json.loads((source / "observations.json").read_text())
    }
    targets = []
    for change in changes:
        site = change["site"]
        obs = observations[site]
        settings = obs["settings"]
        mask = Path("data/scenes/hotspots_oct22") / site / "change_context.tif"
        if file_hash(mask) != change["change_sha256"]:
            raise ValueError("Changed radar grid")
        with rasterio.open(mask) as ds:
            cells = ds.read(1)
            area = (
                abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d)
                / 1e6
            )
            known = np.count_nonzero(cells == 2) * area
            uncertain = np.count_nonzero((cells == 3) | (cells == 255)) * area
        # If unknown pixels could flip the answer, keep it unresolved.
        label = (
            "above"
            if known > 0.5
            else ("below_or_equal" if known + uncertain <= 0.5 else "unresolved")
        )
        observed = datetime.fromisoformat(obs["observed_utc"])
        for lead in (24, 48, 72):
            limit = observed - timedelta(hours=lead)
            run = limit.replace(
                hour=limit.hour // 6 * 6, minute=0, second=0, microsecond=0
            )
            targets.append(
                dict(
                    key=f"{site}|{lead}",
                    site=site,
                    lon=settings["center_lon"],
                    lat=settings["center_lat"],
                    observed_utc=observed.isoformat(),
                    nominal_lead_h=lead,
                    init_lead_h=round((observed - run).total_seconds() / 3600, 3),
                    run_ms=int(run.timestamp() * 1000),
                    known_change_km2=round(known, 4),
                    uncertain_km2=round(uncertain, 4),
                    footprint_target=label,
                    radar_sha256=change["change_sha256"],
                )
            )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "gfs_blocks.json"
    blocks = json.loads(path.read_text()) if path.exists() else fetch(project, targets)
    for row in targets:
        group = sorted(
            (b for b in blocks if b["key"] == row["key"]), key=lambda b: b["hour"]
        )
        if [b["hour"] for b in group] != HOURS or any(
            b["run_ms"] != row["run_ms"]
            or b["valid_ms"] != b["run_ms"] + 3600000 * b["hour"]
            or not math.isfinite(b["mm"])
            or b["mm"] < 0
            for b in group
        ):
            raise ValueError(f"Incomplete forecast: {row['key']}")
        h = row["nominal_lead_h"]
        row["rain_until_last_six_hour_boundary_mm"] = round(
            sum(b["mm"] for b in group if b["hour"] <= h), 3
        )
    if len(blocks) != len(targets) * 12:
        raise ValueError("Unexpected forecast block count")
    path.write_text(json.dumps(blocks, indent=2) + "\n")
    with (OUT / "cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, targets[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    main(parser.parse_args().project)
