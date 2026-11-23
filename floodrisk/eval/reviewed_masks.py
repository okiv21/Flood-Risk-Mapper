"""Evaluate the sealed radar model only after the complete blind mask review."""

import argparse
import csv
from datetime import date, datetime
import json
from pathlib import Path
import re

import numpy as np
import rasterio
from floodrisk.deploy.tiled import file_hash


def read_review(path, manifest):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    indexed = {r["chip_id"]: r for r in rows}
    if len(rows) != len(indexed) or set(indexed) != {
        r["chip_id"] for r in manifest["chips"]
    }:
        raise ValueError("Review rows must match the frozen chip IDs exactly")
    for source in manifest["chips"]:
        r = indexed[source["chip_id"]]
        if not all(
            str(r.get(k) or "").strip()
            for k in [
                "mask_file",
                "mask_sha256",
                "reviewer",
                "evidence_uri",
                "evidence_observed_date",
                "review_date",
            ]
        ):
            raise ValueError(f"Incomplete independent review for {source['chip_id']}")
        observed = date.fromisoformat(r["evidence_observed_date"])
        scene = datetime.strptime(
            re.search(r"_(\d{8})T", source["source"])[1], "%Y%m%d"
        ).date()
        if date.fromisoformat(r["review_date"]) < max(observed, scene):
            raise ValueError("Review precedes its observation")
    return indexed


def run(manifest_path, review_path, cache, mask_root, model, output):
    manifest = json.loads(manifest_path.read_text())
    review = read_review(review_path, manifest)
    if file_hash(model) != manifest["model_sha256"]:
        raise ValueError("Model differs from pre-review checkpoint")
    import onnxruntime as ort
    from floodrisk.datasets.sen1floods11 import preprocess_s1

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    confusion = {}
    evaluated = []
    for record in manifest["chips"]:
        r = review[record["chip_id"]]
        image = cache / record["image"]
        mask = (mask_root / r["mask_file"]).resolve()
        if not mask.is_relative_to(mask_root.resolve()):
            raise ValueError("Mask outside annotation root")
        if file_hash(image) != record["sha256"] or file_hash(mask) != r["mask_sha256"]:
            raise ValueError("Raster checksum mismatch")
        with rasterio.open(image) as sar, rasterio.open(mask) as truth:
            if (
                truth.count != 1
                or truth.dtypes[0] != "uint8"
                or truth.nodata != 255
                or sar.crs != truth.crs
                or sar.transform != truth.transform
                or sar.shape != truth.shape
            ):
                raise ValueError(
                    "Expected aligned single-band uint8 labels with 255 nodata"
                )
            labels = truth.read(1)
            pixels = sar.read(masked=True)
            if not np.isin(labels, [0, 1, 255]).all():
                raise ValueError("Unexpected mask class")
            usable = (
                (labels != 255)
                & ~np.ma.getmaskarray(pixels).any(axis=0)
                & np.isfinite(pixels.data).all(axis=0)
            )
            if not usable.any():
                raise ValueError("Mask contains no evaluable pixels")
            x = preprocess_s1(pixels.filled(np.nan))[None]
            logits = session.run(None, {"sar": x})[0][0, 0]
            if not np.isfinite(logits).all():
                raise ValueError("Nonfinite model output")
            water = (1 / (1 + np.exp(-np.clip(logits, -80, 80))))[usable] > manifest[
                "threshold"
            ]
            y = labels[usable] == 1
            counts = np.array(
                [
                    np.sum(water & y),
                    np.sum(water & ~y),
                    np.sum(~water & y),
                    np.sum(~water & ~y),
                ]
            )
            group = record["source"]
            confusion[group] = (
                confusion.get(group, np.zeros(4, dtype=np.int64)) + counts
            )
            evaluated.append(
                dict(
                    chip_id=r["chip_id"],
                    mask_sha256=r["mask_sha256"],
                    source=group,
                    counts=counts.tolist(),
                )
            )
    groups = np.stack(list(confusion.values()))
    tp, fp, fn, tn = groups.sum(axis=0)
    if tp + fp + fn == 0:
        raise ValueError("Water IoU undefined: empty pooled water union")
    water_iou = tp / (tp + fp + fn)
    draws = np.random.default_rng(42).integers(
        0, len(groups), size=(10000, len(groups))
    )
    totals = groups[draws].sum(axis=1)
    union = totals[:, :3].sum(axis=1)
    bootstrap = totals[union > 0, 0] / union[union > 0]
    report = dict(
        water_iou=water_iou,
        dry_iou=float(tn / (tn + fp + fn)) if tn + fp + fn else None,
        water_iou_scene_bootstrap_95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        defined_bootstrap_draws=len(bootstrap),
        groups=len(groups),
        chips=evaluated,
        review_sha256=file_hash(review_path),
        model_sha256=manifest["model_sha256"],
        threshold=manifest["threshold"],
        target=manifest["target"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["manifest", "reviews", "cache", "mask-root", "model", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    run(a.manifest, a.reviews, a.cache, a.mask_root, a.model, a.output)
