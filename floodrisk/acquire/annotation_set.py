"""Freeze a blind Nigerian pixel-annotation packet, independent of the official splits."""

import argparse
import csv
import json
from pathlib import Path

import rasterio
from rasterio.windows import Window
from rasterio.warp import transform_bounds
from floodrisk.acquire.earthengine import download, grid_settings
from floodrisk.datasets.sen1floods11 import read_split
from floodrisk.deploy.tiled import file_hash

# Four disjoint 10.24 km scenes in each landscape. Select without model output.
REGIONS = {
    "lagos_lagoon": [(3.31, 6.60), (3.63, 6.54), (3.80, 6.58), (3.97, 6.62)],
    "niger_benue": [(6.69, 7.98), (6.91, 7.82), (6.75, 7.60), (6.70, 7.41)],
    "bayelsa": [(6.09, 4.95), (6.39, 5.09), (6.20, 4.76), (6.40, 4.77)],
}
MODEL_SHA = "73fb31bbf88120a8f1e1cee20fd7c8b7f78add716a1954afaaa213f1e5bcdb08"
REVIEW_FIELDS = [
    "chip_id",
    "mask_file",
    "mask_sha256",
    "reviewer",
    "evidence_uri",
    "evidence_observed_date",
    "review_date",
]


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def prepare(project, root, cache, report):
    excluded = []
    for name in sorted(
        {
            n
            for split in ["train", "valid", "test", "bolivia"]
            for n, _ in read_split(root, split)
            if n.startswith("Nigeria_")
        }
    ):
        with rasterio.open(root / "S1Hand" / name) as src:
            excluded.append(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    for scene in Path("data/scenes").rglob("acquisition.json"):
        if cache.resolve() in scene.resolve().parents:
            continue
        acquisition = json.loads(scene.read_text())
        settings = acquisition.get("settings")
        if settings:
            _, bounds = grid_settings(settings)
            excluded.append(transform_bounds(settings["crs"], "EPSG:4326", *bounds))
    report.mkdir(parents=True, exist_ok=True)
    rows = []
    for region, sites in REGIONS.items():
        for i, (lon, lat) in enumerate(sites, 1):
            group = f"{region}_{i}"
            folder = cache / group
            settings = dict(
                center_lon=lon,
                center_lat=lat,
                width=1024,
                height=1024,
                scale_m=10,
                crs=f"EPSG:{32601 + int((lon + 180) // 6)}",
                start="2024-08-01",
                end="2024-11-01",
                selection_target=f"2024-{8 + (i - 1) // 2:02d}-{8 if i % 2 else 23:02d}",
            )
            _, bounds = grid_settings(settings)
            geographic = transform_bounds(settings["crs"], "EPSG:4326", *bounds)
            if any(intersects(geographic, b) for b in excluded):
                raise ValueError(
                    f"Reserved footprint overlaps existing source: {group}"
                )
            excluded.append(geographic)
            source = folder / "source"
            a = (
                json.loads((source / "acquisition.json").read_text())
                if source.exists()
                else download(project, source, settings)
            )
            raster = source / "sentinel1_vv_vh_db.tif"
            if settings != a["settings"] or file_hash(raster) != a["sha256"]:
                raise ValueError("Reserved source changed")
            with rasterio.open(raster) as src:
                for y in [0, 512]:
                    for x in [0, 512]:
                        chip = f"{group}_{y // 512}_{x // 512}"
                        window = Window(x, y, 512, 512)
                        dest = folder / (chip + ".tif")
                        if not dest.exists():
                            profile = {
                                **src.profile,
                                "height": 512,
                                "width": 512,
                                "transform": src.window_transform(window),
                            }
                            temp = dest.with_suffix(".tmp")
                            with rasterio.open(temp, "w", **profile) as dst:
                                dst.write(src.read(window=window))
                                for n, desc in enumerate(src.descriptions, 1):
                                    dst.set_band_description(n, desc)
                            temp.replace(dest)
                        rows.append(
                            dict(
                                chip_id=chip,
                                region=region,
                                group=group,
                                image=str(dest.relative_to(cache)),
                                sha256=file_hash(dest),
                                source=a["source"]["system:index"],
                                parent_sha256=a["sha256"],
                                crs=src.crs.to_string(),
                                bounds=list(src.window_bounds(window)),
                            )
                        )
            print(group, flush=True)
    manifest = dict(
        target="observed water on the radar acquisition date",
        model_sha256=MODEL_SHA,
        threshold=0.55,
        selection="12 disjoint scene footprints; 4 adjacent chips per scene; purposive locations; no model-based selection",
        classes={"0": "dry", "1": "water", "255": "uncertain or missing"},
        chips=rows,
    )
    path = report / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Previously frozen packet differs")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    queue = report / "review.csv"
    if not queue.exists():
        with queue.open("x", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({"chip_id": r["chip_id"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    for key in ["root", "cache", "report"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    prepare(a.project, a.root, a.cache, a.report)
