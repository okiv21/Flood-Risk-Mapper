"""Rebuild the matched-orbit radar comparison surrounding the Maiduguri flood."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from jobs.radar_maps import build

SCENES = [
    "S1A_IW_GRDH_1SDV_20240905T045539_20240905T045604_055525_06C676_AAD8",
    "S1A_IW_GRDH_1SDV_20240917T045540_20240917T045605_055700_06CD64_64BE",
]


def main(project):
    settings = next(
        row["settings"]
        for row in json.loads(
            Path("reports/hotspots_oct22/observations.json").read_text()
        )
        if row["site"] == "Maiduguri"
    )
    root = Path("data/predictions/maiduguri_2024")
    report = Path("reports/event_replay")
    masks, records = [], []
    for index in SCENES:
        time = datetime.strptime(index.split("_")[4], "%Y%m%dT%H%M%S").replace(
            tzinfo=timezone.utc
        )
        scene = {
            "site": "Maiduguri",
            "system:index": index,
            "system:time_start": int(time.timestamp() * 1000),
        }
        folder = root / index
        if not (folder / "map.json").exists():
            build(scene, settings, folder, project)
        acquisition = json.loads((folder / "input/acquisition.json").read_text())
        record = json.loads((folder / "map.json").read_text())
        if record["source"] != scene or acquisition["source"]["system:index"] != index:
            raise ValueError("Unexpected radar source")
        records.append(dict(acquisition=acquisition["source"], prediction=record))
        with rasterio.open(folder / "prediction/water_mask.tif") as src:
            grid = (src.crs, src.transform, src.shape)
            if masks and grid != first_grid:
                raise ValueError("Grid mismatch")
            first_grid = grid
            masks.append(src.read(1))
    orbit = lambda r: (
        r["acquisition"]["relativeOrbitNumber_start"],
        r["acquisition"]["orbitProperties_pass"],
    )
    if orbit(records[0]) != orbit(records[1]):
        raise ValueError("Orbit mismatch")
    a, b = masks
    valid = (a != 255) & (b != 255)
    change = np.where(valid, a * 2 + b, 4)
    colors = ["#eeeae1", "#0077ba", "#c78d53", "#203b5b", "#888888"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained")
    for ax, m, title in zip(
        axes,
        [a, b, change],
        ["5 September · before", "17 September · after", "Change between scenes"],
    ):
        palette = colors if ax == axes[2] else [colors[0], colors[3], colors[4]]
        image = m if ax == axes[2] else np.where(m == 255, 2, m)
        ax.imshow(
            image,
            cmap=ListedColormap(palette),
            vmin=0,
            vmax=len(palette) - 1,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.axis("off")
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(color=c, label=l)
            for c, l in zip(
                colors,
                ["Dry both", "Water gained", "Water lost", "Water both", "No data"],
            )
        ],
        loc="outside lower center",
        ncol=5,
    )
    fig.suptitle("Predicted surface water · Maiduguri study window · September 2024")
    fig.savefig(report / "maiduguri_change.png", dpi=160)
    plt.close(fig)
    (report / "maiduguri_change.json").write_text(
        json.dumps(
            dict(
                inputs=records,
                valid_pixels=int(valid.sum()),
                gained_pixels=int((change == 1).sum()),
                lost_pixels=int((change == 2).sum()),
                both_water=int((change == 3).sum()),
                pixel_area_m2=abs(first_grid[1].a * first_grid[1].e),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    main(parser.parse_args().project)
