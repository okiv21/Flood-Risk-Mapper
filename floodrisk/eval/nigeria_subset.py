"""Evaluate the official Nigerian test chips using the frozen FP32 configuration."""

import argparse
from pathlib import Path
import csv, json
import numpy as np
import onnxruntime as ort
import rasterio
from floodrisk.datasets.sen1floods11 import Sen1Floods11
from floodrisk.eval.metrics import PooledIoU
from floodrisk.deploy.tiled import file_hash


def evaluate(root, model, out):
    out.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    ds = Sen1Floods11(root, "test")
    rows = []
    totals = {k: PooledIoU() for k in ["unet_fp32_055", "otsu"]}
    for i, (s1, label_name) in enumerate(ds.items):
        if not s1.startswith("Nigeria_"):
            continue
        x, y = ds[i]
        label = y.numpy()
        logits = session.run(None, {"sar": x.numpy()[None]})[0][0, 0]
        pred = 1 / (1 + np.exp(-np.clip(logits, -80, 80))) > 0.55
        with rasterio.open(
            root
            / "S1OtsuLabelHand"
            / label_name.replace("LabelHand", "S1OtsuLabelHand")
        ) as src:
            otsu = src.read(1) == 1
        for method, p in [("unet_fp32_055", pred), ("otsu", otsu)]:
            valid = label != -1
            t = label == 1
            counts = dict(
                tp=int((p & t & valid).sum()),
                fp=int((p & ~t & valid).sum()),
                fn=int((~p & t & valid).sum()),
                tn=int((~p & ~t & valid).sum()),
            )
            totals[method].update(p, label)
            rows.append(dict(chip=s1, method=method, **counts))
    with (out / "per_chip.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    results = {}
    for method, metric in totals.items():
        a = np.array(
            [
                [r[k] for k in ["tp", "fp", "fn", "tn"]]
                for r in rows
                if r["method"] == method
            ]
        )
        rng = np.random.default_rng(42)
        draws = rng.integers(0, len(a), size=(10000, len(a)))
        s = a[draws].sum(axis=1)
        samples = s[:, 0] / np.maximum(s[:, 0] + s[:, 1] + s[:, 2], 1)
        results[method] = dict(
            **metric.compute(),
            chip_bootstrap_95_interval=np.percentile(samples, [2.5, 97.5]).tolist(),
        )
    report = dict(
        chips=len(rows) // 2,
        scope="Official Sen1Floods11 Nigeria test-country subset; Nigeria also appears in train/validation; not a new geographic holdout",
        model_sha256=file_hash(model),
        threshold=0.55,
        results=results,
        bootstrap="10000 paired chip resamples; seed42; only4chips, event independence not established; interval is exploratory",
        limitation="Does not replace a larger independent Nigerian test set; no flood forecast or urban pluvial accuracy established",
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = list(results)
    values = [results[k]["iou_water"] for k in methods]
    low = [
        values[i] - results[k]["chip_bootstrap_95_interval"][0]
        for i, k in enumerate(methods)
    ]
    high = [
        results[k]["chip_bootstrap_95_interval"][1] - values[i]
        for i, k in enumerate(methods)
    ]
    bars = ax.bar(
        ["U-Net FP32, t=0.55", "Provided Otsu"],
        values,
        color=["#1976d2", "#ef8a17"],
        yerr=[low, high],
        capsize=6,
    )
    ax.bar_label(bars, fmt="%.4f")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Pooled water IoU")
    ax.set_title(
        "Nigeria official-test subset | only 4 chips\nExploratory chip-bootstrap intervals; not a new independent test set"
    )
    fig.tight_layout()
    fig.savefig(out / "nigeria_subset_iou.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.root, args.model, args.output)
