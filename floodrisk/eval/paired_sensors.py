"""Compare optical and radar models on common valid pixels from held-out scenes."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
from floodrisk.datasets.optical import OpticalChips
from floodrisk.datasets.sen1floods11 import Sen1Floods11
from floodrisk.deploy.tiled import file_hash
from floodrisk.eval.metrics import PooledIoU
from floodrisk.models.unet import build_unet


def load(path, channels):
    model = build_unet(pretrained=False, in_channels=channels)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def compare(root, cache, radar, optical, output):
    torch.set_num_threads(4)
    models = [load(radar, 2), load(optical, 4)]
    rows = []
    results = {}
    for split in ["valid", "test", "bolivia"]:
        sar, ms = Sen1Floods11(root, split), OpticalChips(root, split, cache)
        pools = {
            k: PooledIoU() for k in ["radar_common", "optical_common", "optical_native"]
        }
        count = 0
        for i, (name, _) in enumerate(sar.items):
            x_sar, _ = sar[i]
            x_optical, optical_labels = ms[i]
            y = optical_labels.numpy()
            with rasterio.open(root / "S1Hand" / name) as src:
                pixels = src.read(masked=True)
                usable = np.isfinite(pixels.data).all(axis=0) & ~np.ma.getmaskarray(
                    pixels
                ).any(axis=0)
            common = np.where(usable, y, -1)
            with torch.inference_mode():
                predictions = [
                    model(x[None])[0, 0].numpy() > 0
                    for model, x in zip(models, [x_sar, x_optical])
                ]
            pools["optical_native"].update(predictions[1], y)
            n = int((common != -1).sum())
            count += n
            row = dict(split=split, chip=name, common_labelled_pixels=n)
            for key, pred in zip(["radar_common", "optical_common"], predictions):
                pools[key].update(pred, common)
                metric = PooledIoU()
                metric.update(pred, common)
                row[key + "_intersection"] = metric.water_inter
                row[key + "_union"] = metric.water_union
            rows.append(row)
        results[split] = dict(
            chips=len(sar),
            common_labelled_pixels=count,
            **{k: v.compute() for k, v in pools.items()},
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_chips.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(
        threshold=0.5,
        radar_sha256=file_hash(radar),
        optical_sha256=file_hash(optical),
        metrics=results,
        protocol="CPU FP32; pooled IoU; same official chips; common masks require valid labels, valid optical and finite unmasked radar; clouds retained",
    )
    (output / "paired_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["root", "cache", "radar", "optical", "output"]:
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(compare(a.root, a.cache, a.radar, a.optical, a.output), indent=2))
