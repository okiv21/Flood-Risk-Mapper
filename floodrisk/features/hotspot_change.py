"""Compare October water against earlier, same-orbit engineering references."""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from floodrisk.acquire.earthengine import download
from floodrisk.deploy.tiled import file_hash, predict_scene


def run(project, root, observations, model, config, output):
    rows = []
    for obs in observations:
        folder = root / obs["site"]
        reference = folder / "earlier_scene"
        acquisition = obs["acquisition"]
        settings = {
            **obs["settings"],
            "start": "2022-01-01",
            "end": "2022-03-01",
            "selection_target": "2022-02-15",
            "relative_orbit": acquisition["relativeOrbitNumber_start"],
            "orbit_pass": acquisition["orbitProperties_pass"],
        }
        a = (
            json.loads((reference / "acquisition.json").read_text())
            if reference.exists()
            else download(project, reference, settings)
        )
        image = reference / "sentinel1_vv_vh_db.tif"
        if a["settings"] != settings or file_hash(image) != a["sha256"]:
            raise ValueError("Cached earlier scene changed")
        path = folder / "earlier_prediction"
        p = (
            json.loads((path / "prediction.json").read_text())
            if path.exists()
            else predict_scene(image, model, config, path)
        )
        if (
            p["model_sha256"] != obs["model_sha256"]
            or p["input_sha256"] != a["sha256"]
            or p["config_sha256"] != file_hash(config)
        ):
            raise ValueError("Predictions use incompatible artifacts")
        context = folder / "reference/water_context.tif"
        if file_hash(context) != obs["context_sha256"]:
            raise ValueError("October context changed")
        with (
            rasterio.open(context) as src,
            rasterio.open(path / "water_mask.tif") as prior,
        ):
            if (
                src.crs != prior.crs
                or src.shape != prior.shape
                or src.transform != prior.transform
            ):
                raise ValueError("Grid mismatch")
            later, early = src.read(1), prior.read(1)
            valid = (later != 255) & (early != 255)
            changes = np.where(valid, np.where(early == 0, later, 0), 255).astype(
                "uint8"
            )
            target = folder / "change_context.tif"
            temp = target.with_suffix(".tmp")
            with rasterio.open(temp, "w", **src.profile) as dst:
                dst.write(changes, 1)
                dst.set_band_description(
                    1,
                    "0 no dry-to-water change; 1/2/3 change in high/lower/unknown historical occurrence; 255 missing",
                )
            temp.replace(target)
            rows.append(
                dict(
                    site=obs["site"],
                    observed_utc=obs["observed_utc"],
                    change_sha256=file_hash(target),
                    october_source=acquisition["system:index"],
                    earlier_source=a["source"]["system:index"],
                    earlier_input_sha256=a["sha256"],
                    earlier_mask_sha256=file_hash(path / "water_mask.tif"),
                    model_sha256=p["model_sha256"],
                    paired_valid_pixels=int(valid.sum()),
                    both_water=int(((early == 1) & (later > 0) & valid).sum()),
                    earlier_water_only=int(((early == 1) & (later == 0) & valid).sum()),
                    october_water_only_by_historical_class={
                        str(k): int(((early == 0) & (later == k)).sum())
                        for k in [1, 2, 3]
                    },
                )
            )
        print(obs["site"], flush=True)
    output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    for key in ["root", "observations", "model", "config", "output"]:
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    run(
        a.project,
        a.root,
        json.loads(a.observations.read_text()),
        a.model,
        a.config,
        a.output,
    )
