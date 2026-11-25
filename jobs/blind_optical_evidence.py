"""Catalogue optical evidence near the acquisition of each sealed Nigerian radar chip."""

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import ee

ROOT = Path("reports/reserved_nigeria")
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"


def main(project):
    ee.Initialize(project=project)
    chips = json.loads((ROOT / "manifest.json").read_text())["chips"]
    sets = []
    for chip in chips:
        roi = ee.Geometry.Rectangle(chip["bounds"], proj=chip["crs"], geodesic=False)
        stamp = re.search(r"_(\d{8}T\d{6})_", chip["source"])[1]
        radar = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        instant = ee.Date(int(radar.timestamp() * 1000))
        scenes = (
            ee.ImageCollection(COLLECTION)
            .filterBounds(roi)
            .filterDate(instant.advance(-3, "day"), instant.advance(3, "day"))
            .filter(ee.Filter.contains(leftField=".geo", rightValue=roi, maxError=10))
        )

        def summarize(obj):
            im = ee.Image(obj)
            scl = im.select("SCL")
            # SCL vegetation, bare ground and water. Cloud/shadow and missing are excluded.
            clear = (
                scl.gte(4)
                .And(scl.lte(6))
                .unmask(0, sameFootprint=False)
                .rename("clear")
            )
            frac = clear.reduceRegion(
                ee.Reducer.mean(), roi, crs=scl.projection(), scale=20, maxPixels=250000
            )
            return ee.Feature(
                None,
                {
                    "chip_id": chip["chip_id"],
                    "source_index": im.get("system:index"),
                    "optical_ms": im.get("system:time_start"),
                    "clear_fraction": frac.get("clear"),
                },
            )

        sets.append(ee.FeatureCollection(scenes.toList(100).map(summarize)))
    response = ee.FeatureCollection(sets).flatten().getInfo()
    candidates = [r["properties"] for r in response["features"]]
    rows = []
    for chip in chips:
        stamp = re.search(r"_(\d{8}T\d{6})_", chip["source"])[1]
        radar = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        options = [r for r in candidates if r["chip_id"] == chip["chip_id"]]
        options.sort(
            key=lambda r: (
                -r["clear_fraction"],
                abs(r["optical_ms"] / 1000 - radar.timestamp()),
                r["source_index"],
            )
        )
        for i, r in enumerate(options):
            opt = datetime.fromtimestamp(r["optical_ms"] / 1000, timezone.utc)
            rows.append(
                dict(
                    chip_id=chip["chip_id"],
                    radar_utc=radar.isoformat(),
                    rank=i + 1,
                    asset=COLLECTION + "/" + r["source_index"],
                    optical_utc=opt.isoformat(),
                    offset_hours=round((opt - radar).total_seconds() / 3600, 3),
                    clear_fraction=r["clear_fraction"],
                )
            )
        if not options:
            rows.append(
                dict(
                    chip_id=chip["chip_id"],
                    radar_utc=radar.isoformat(),
                    rank="",
                    asset="",
                    optical_utc="",
                    offset_hours="",
                    clear_fraction="",
                )
            )
    with (ROOT / "optical_candidates.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    main(p.parse_args().project)
