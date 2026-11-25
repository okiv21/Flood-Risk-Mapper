"""Identify LGAs where observed pixels alone exceed the radar proxy threshold."""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping, shape
from shapely.strtree import STRtree

from floodrisk.deploy.tiled import file_hash
from floodrisk.features.lga_observations import BOUNDARY_SHA

OUT = Path("reports/satellite_replay")


def main():
    source = Path("data/boundaries/nga_adm2_verified.geojson")
    if file_hash(source) != BOUNDARY_SHA:
        raise ValueError("Boundary version mismatch")
    polygons = json.loads(source.read_text())["features"]
    geoms = [shape(p["geometry"]) for p in polygons]
    tree = STRtree(geoms)
    rows = []
    for obs in json.loads(
        Path("reports/hotspots_oct22/paired_change.json").read_text()
    ):
        path = Path("data/scenes/hotspots_oct22") / obs["site"] / "change_context.tif"
        if file_hash(path) != obs["change_sha256"]:
            raise ValueError("Change raster mismatch")
        with rasterio.open(path) as src:
            pixels = src.read(1)
            footprint = shape(
                transform_geom(src.crs, "EPSG:4326", mapping(box(*src.bounds)))
            )
            indices = sorted(tree.query(footprint, predicate="intersects"))
            zones = rasterize(
                [
                    (
                        transform_geom("EPSG:4326", src.crs, mapping(geoms[i])),
                        int(i) + 1,
                    )
                    for i in indices
                ],
                out_shape=src.shape,
                transform=src.transform,
                dtype="uint16",
            )
            area = abs(
                src.transform.a * src.transform.e - src.transform.b * src.transform.d
            )
            for z in np.unique(zones):
                props = polygons[int(z) - 1]["properties"] if z else {}
                local = pixels[zones == z]
                count = int(np.count_nonzero(local == 2))
                rows.append(
                    dict(
                        site=obs["site"],
                        observed_utc=obs["observed_utc"],
                        lga_id=props.get("shapeID", ""),
                        lga_name=props.get("shapeName", ""),
                        grid_pixels=len(local),
                        known_gained_pixels=count,
                        known_gained_km2=round(count * area / 1e6, 4),
                        unknown_pixels=int(np.isin(local, [3, 255]).sum()),
                        lga_target=(
                            "unassigned"
                            if not z
                            else "above"
                            if count * area > 500000
                            else "unresolved_partial_coverage"
                        ),
                    )
                )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "lga_bounds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
