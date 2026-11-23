"""Build one dated 5.12 km radar observation at each project hotspot."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from floodrisk.acquire.earthengine import download
from floodrisk.acquire.water_reference import compare
from floodrisk.deploy.tiled import file_hash, predict_scene

SITES = {
    "Lagos": (3.45, 6.55),
    "Lokoja": (6.74, 7.79),
    "Port Harcourt": (7.04, 4.81),
    "Yenagoa": (6.27, 4.93),
    "Anambra (Onitsha)": (6.78, 6.14),
    "Kano": (8.52, 12.0),
    "Maiduguri": (13.15, 11.85),
    "Abuja": (7.49, 9.06),
}


def read(path):
    return json.loads(path.read_text())


def run(project, root, model, config):
    results = []
    for site, (lon, lat) in SITES.items():
        folder = root / site
        scene = folder / "scene"
        raster = scene / "sentinel1_vv_vh_db.tif"
        settings = dict(
            center_lon=lon,
            center_lat=lat,
            crs=f"EPSG:{32601 + int((lon + 180) // 6)}",
            width=512,
            height=512,
            scale_m=10,
            start="2022-10-01",
            end="2022-11-01",
            selection_target="2022-10-15",
        )
        acquisition = (
            read(scene / "acquisition.json")
            if scene.exists()
            else download(project, scene, settings)
        )
        if (
            settings != acquisition["settings"]
            or file_hash(raster) != acquisition["sha256"]
        ):
            raise ValueError(f"Cached scene differs at {site}")
        pred = folder / "prediction"
        prediction = (
            read(pred / "prediction.json")
            if pred.exists()
            else predict_scene(raster, model, config, pred)
        )
        if (
            prediction["input_sha256"] != acquisition["sha256"]
            or prediction["model_sha256"] != file_hash(model)
            or prediction["config_sha256"] != file_hash(config)
        ):
            raise ValueError(f"Cached prediction differs at {site}")
        reference = folder / "reference"
        mask = pred / "water_mask.tif"
        context = (
            read(reference / "reference.json")
            if reference.exists()
            else compare(project, scene / "acquisition.json", mask, reference)
        )
        if context["mask_sha256"] != file_hash(mask):
            raise ValueError(f"Cached water reference differs at {site}")
        results.append(
            dict(
                site=site,
                settings=settings,
                acquisition=acquisition["source"],
                observed_utc=datetime.fromtimestamp(
                    acquisition["source"]["system:time_start"] / 1000, timezone.utc
                ).isoformat(),
                input_sha256=acquisition["sha256"],
                mask_sha256=context["mask_sha256"],
                context_sha256=file_hash(reference / "water_context.tif"),
                model_sha256=prediction["model_sha256"],
                threshold=prediction["threshold"],
                counts=context["counts"],
            )
        )
        print(site, results[-1]["observed_utc"], flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["project", "root", "model", "config", "report"]:
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    results = run(args.project, Path(args.root), Path(args.model), Path(args.config))
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
