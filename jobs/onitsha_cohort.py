"""Measure a whole-LGA satellite proxy with a fixed seasonal sampling rule."""

import argparse
import calendar
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import ee
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform, transform_geom
from shapely.geometry import shape, box

from jobs.radar_maps import build
from floodrisk.acquire.earthengine import grid_settings
from floodrisk.acquire.water_reference import compare
from floodrisk.deploy.tiled import file_hash
from floodrisk.features.lga_observations import BOUNDARY_SHA

OUT = Path("reports/satellite_replay")
CACHE = Path("data/predictions/onitsha_south_cohort")
LGA = "59680162B59497895637604"
CRS = "EPSG:32632"


def main(project):
    src = Path("data/boundaries/nga_adm2_verified.geojson")
    if file_hash(src) != BOUNDARY_SHA:
        raise ValueError("Boundary hash mismatch")
    polygon = next(
        f["geometry"]
        for f in json.loads(src.read_text())["features"]
        if f["properties"]["shapeID"] == LGA
    )
    geom = transform_geom("EPSG:4326", CRS, polygon)
    bounds = shape(geom).bounds
    lon, lat = transform(
        CRS, "EPSG:4326", [(bounds[0] + bounds[2]) / 2], [(bounds[1] + bounds[3]) / 2]
    )
    settings = dict(
        center_lon=lon[0],
        center_lat=lat[0],
        crs=CRS,
        scale_m=10,
        width=512,
        height=512,
        start="2018-01-01",
        end="2026-01-01",
    )
    _, grid_bounds = grid_settings(settings)
    if not box(*grid_bounds).covers(shape(geom)):
        raise ValueError("Incomplete LGA")
    ee.Initialize(project=project)
    roi = ee.Geometry.Rectangle(grid_bounds, proj=CRS, geodesic=False)
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterDate(settings["start"], settings["end"])
        .filter(ee.Filter.contains(leftField=".geo", rightValue=roi, maxError=10))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", 30))
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .filter(ee.Filter.eq("platform_number", "A"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    )
    entries = (
        collection.reduceColumns(
            ee.Reducer.toList(2), ["system:time_start", "system:index"]
        )
        .get("list")
        .getInfo()
    )
    rows = []
    for year in range(2018, 2026):
        masks = {}
        ids = {}
        for month in (2, 7, 10):
            anchor = datetime(year, month, 15, tzinfo=timezone.utc)
            possible = [
                (abs(t / 1000 - anchor.timestamp()), i, t)
                for t, i in entries
                if datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m")
                == anchor.strftime("%Y-%m")
            ]
            if not possible:
                continue
            _, idx, t = min(possible)
            folder = CACHE / idx
            if not (folder / "map.json").exists():
                build(
                    {
                        "site": "Onitsha South",
                        "system:index": idx,
                        "system:time_start": t,
                    },
                    settings,
                    folder,
                    project,
                )
            meta = json.loads((folder / "map.json").read_text())
            image = folder / "prediction/water_mask.tif"
            if (
                meta["source"]["system:index"] != idx
                or file_hash(image) != meta["files"]["prediction/water_mask.tif"]
            ):
                raise ValueError("Invalid cached mask")
            ref = CACHE / "reference"
            if not ref.exists():
                compare(project, folder / "input/acquisition.json", image, ref)
            with (
                rasterio.open(image) as src,
                rasterio.open(ref / "historical_occurrence.tif") as hist,
            ):
                if (src.transform, src.shape, src.crs) != (
                    hist.transform,
                    hist.shape,
                    hist.crs,
                ):
                    raise ValueError("Grid mismatch")
                inside = geometry_mask([geom], src.shape, src.transform, invert=True)
                masks[month] = src.read(1)
                occ = hist.read(1)
            ids[month] = (idx, meta["observed_utc"])
            if month == 2 or 2 not in masks:
                continue
            a, b = masks[2], masks[month]
            gain = inside & (a == 0) & (b == 1)
            known = int((gain & (occ < 90)).sum())
            uncertain = int(
                ((gain & (occ == 255)) | (inside & ((a == 255) | (b == 255)))).sum()
            )
            rows.append(
                dict(
                    year=year,
                    month=month,
                    lga_id=LGA,
                    radar_id=idx,
                    dry_reference=ids[2][0],
                    observed_utc=ids[month][1],
                    lga_pixels=int(inside.sum()),
                    known_gained_pixels=known,
                    unknown_pixels=uncertain,
                    proxy="above"
                    if known > 5000
                    else "below_or_equal"
                    if known + uncertain <= 5000
                    else "unresolved",
                )
            )
            print(year, month, rows[-1]["proxy"], flush=True)
    if not rows:
        raise ValueError("No paired scenes")
    with (OUT / "onitsha_south.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    main(p.parse_args().project)
