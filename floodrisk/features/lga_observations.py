"""Catalogue LGAs and count observed water classes within the sampled hotspots."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import box, shape, mapping
from shapely.strtree import STRtree

BOUNDARY_SHA = "bef7f2cfa45e012f4772eaa61c7b99e5188aeba4d5c6badae7e9f9aae8c02fcd"


def table(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build(boundaries, observations, root, output, states=None):
    blob = boundaries.read_bytes()
    if hashlib.sha256(blob).hexdigest() != BOUNDARY_SHA:
        raise ValueError("Expected pinned 2022 GRID3 boundaries")
    features = json.loads(blob)["features"]
    areas = [shape(f["geometry"]) for f in features]
    index = STRtree(areas)
    if states is not None:
        raw_states = states.read_bytes()
        if (
            hashlib.sha256(raw_states).hexdigest()
            != "64fa218ac3d453cc1e66412ff461c5dfa1a4a1ade0da93b239a9891b587d28f9"
        ):
            raise ValueError("Expected pinned 2022 GRID3 state boundaries")
        state_features = json.loads(raw_states)["features"]
        state_index = STRtree([shape(f["geometry"]) for f in state_features])
    records = []
    for f, area in zip(features, areas):
        point = area.representative_point()
        parent = {}
        if states is not None:
            matches = state_index.query(point, predicate="within")
            if len(matches) != 1:
                raise ValueError(
                    f"Ambiguous parent state: {f['properties']['shapeName']}"
                )
            props = state_features[int(matches[0])]["properties"]
            parent = dict(state=props["shapeName"], state_id=props["shapeID"])
        records.append(
            dict(
                lga_id=f["properties"]["shapeID"],
                name=f["properties"]["shapeName"],
                **parent,
                longitude=point.x,
                latitude=point.y,
            )
        )
    if len(set(r["lga_id"] for r in records)) != 774:
        raise ValueError("Unexpected LGA count")
    output.mkdir(parents=True, exist_ok=True)
    table(output / "lgas.csv", records, list(records[0]))
    rows = []
    for obs in observations:
        path = root / obs["site"] / "reference" / "water_context.tif"
        if hashlib.sha256(path.read_bytes()).hexdigest() != obs["context_sha256"]:
            raise ValueError("Changed context raster")
        with rasterio.open(path) as src:
            footprint = shape(
                transform_geom(src.crs, "EPSG:4326", mapping(box(*src.bounds)))
            )
            candidates = sorted(index.query(footprint, predicate="intersects").tolist())
            polygons = [
                (transform_geom("EPSG:4326", src.crs, mapping(areas[i])), i + 1)
                for i in candidates
            ]
            zones = (
                rasterize(
                    polygons,
                    out_shape=src.shape,
                    transform=src.transform,
                    dtype="uint16",
                )
                if polygons
                else np.zeros(src.shape)
            )
            pixels = src.read(1)
            for zone in np.unique(zones):
                cells = pixels[zones == zone]
                row = dict(
                    site=obs["site"],
                    observed_utc=obs["observed_utc"],
                    lga_id=records[int(zone) - 1]["lga_id"] if zone else "",
                    pixel_area_m2=100,
                )
                row.update(
                    {
                        f"class_{k}_pixels": int((cells == k).sum())
                        for k in [0, 1, 2, 3, 255]
                    }
                )
                rows.append(row)
    table(output / "observed_pixels_by_lga.csv", rows, list(rows[0]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["boundaries", "observations", "root", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument(
        "--states", type=Path, help="Pinned ADM1 polygons for state attribution"
    )
    a = p.parse_args()
    build(
        a.boundaries, json.loads(a.observations.read_text()), a.root, a.output, a.states
    )
