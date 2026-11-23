"""Measure overlap of mapped surface features with the eight water observations."""

import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from floodrisk.acquire.surface import acquire
from floodrisk.deploy.tiled import file_hash


def run(project, root, observations, output):
    results = []
    for obs in observations:
        folder = root / obs["site"]
        context = folder / "reference/water_context.tif"
        if file_hash(context) != obs["context_sha256"]:
            raise ValueError("Context hash mismatch")
        surface = folder / "surface"
        if not surface.exists():
            acquire(project, folder / "scene/acquisition.json", surface)
        metadata = json.loads((surface / "surface.json").read_text())
        if file_hash(surface / "surface_features.tif") != metadata["sha256"]:
            raise ValueError("Cached surface raster changed")
        with (
            rasterio.open(context) as ref,
            rasterio.open(surface / "surface_features.tif") as src,
        ):
            if (
                src.crs != ref.crs
                or src.shape != ref.shape
                or src.transform != ref.transform
            ):
                raise ValueError("Grids differ")
            classes = ref.read(1)
            land, occupancy = src.read()
            categories = {}
            for k in [0, 1, 2, 3, 255]:
                mask = classes == k
                categories[str(k)] = dict(
                    grid_pixels=int(mask.sum()),
                    built_up_pixels=int(((land == 50) & mask).sum()),
                    occupied_pixels=int(((occupancy == 1) & mask).sum()),
                    missing_cover_pixels=int(((land == -9999) & mask).sum()),
                    missing_occupancy_pixels=int(((occupancy == -9999) & mask).sum()),
                )
        results.append(
            dict(
                site=obs["site"],
                observed_utc=obs["observed_utc"],
                water_context=categories,
                surface_sha256=file_hash(surface / "surface_features.tif"),
                context_sha256=obs["context_sha256"],
                mapped_centroids_in_scene=sum(
                    json.loads(
                        (surface / "building_centroid_counts.json").read_text()
                    ).values()
                ),
                source_timing="WorldCover 2021 and Open Buildings v3 (2023); historical building inventory at the 2022 radar date is unverified",
            )
        )
        print(obs["site"], flush=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    for k in ["root", "observations", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.project, a.root, json.loads(a.observations.read_text()), a.output)
