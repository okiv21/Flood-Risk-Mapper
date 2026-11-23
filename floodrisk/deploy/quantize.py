"""Compare dynamic and static INT8 ONNX against FP32 without calibration leakage."""

import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
from floodrisk.deploy.tiled import file_hash
from floodrisk.eval.metrics import PooledIoU


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["root", "model", "config", "output"]:
        p.add_argument("--" + key, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        p.error("Use a new output directory")
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_dynamic,
        quantize_static,
        QuantType,
        QuantFormat,
        CalibrationDataReader,
        CalibrationMethod,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
    from floodrisk.datasets.sen1floods11 import Sen1Floods11

    config = json.loads(a.config.read_text())
    if file_hash(a.model) != config["onnx_sha256"]:
        raise ValueError("Model/config hash mismatch")
    a.output.mkdir(parents=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1

    def session(path):
        return ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )

    train = Sen1Floods11(a.root, "train")
    valid = Sen1Floods11(a.root, "valid")
    indices = (
        np.random.default_rng(42)
        .choice(len(train), min(32, len(train)), replace=False)
        .tolist()
    )
    calibration = [train.items[i][0] for i in indices]
    dynamic = a.output / "m1_s1_dynamic_int8.onnx"
    static = a.output / "m1_s1_static_int8.onnx"
    sample = valid[0][0].numpy()[None]
    dynamic_status = {}
    try:
        quantize_dynamic(
            str(a.model), str(dynamic), weight_type=QuantType.QInt8, per_channel=True
        )
        session(dynamic).run(None, {"sar": sample})
        dynamic_status = {"runnable": True, "bytes": dynamic.stat().st_size}
    except Exception as exc:
        dynamic_status = {
            "runnable": False,
            "error": str(exc),
            "exception_type": type(exc).__name__,
        }
    preprocessed = a.output / "m1_s1_preprocessed.onnx"
    quant_pre_process(str(a.model), str(preprocessed), skip_symbolic_shape=True)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(indices)

        def get_next(self):
            i = next(self.items, None)
            return None if i is None else {"sar": train[i][0].numpy()[None]}

    quantize_static(
        str(preprocessed),
        str(static),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    onnx.checker.check_model(str(static))
    sessions = {"fp32": session(a.model), "static_int8": session(static)}

    def evaluate_split(split, store_rows=False):
        ds = Sen1Floods11(a.root, split)
        metrics = {name: PooledIoU() for name in sessions}
        rows = []
        for i in range(len(ds)):
            x, y = ds[i]
            x = x.numpy()[None]
            label = y.numpy()
            for name, s in sessions.items():
                logits = s.run(None, {"sar": x})[0][0, 0]
                if not np.isfinite(logits).all():
                    raise ValueError("Non-finite logits")
                pred = 1 / (1 + np.exp(-np.clip(logits, -80, 80))) > config["threshold"]
                metrics[name].update(pred, label)
                if store_rows:
                    valid_mask = label != -1
                    truth = label == 1
                    rows.append(
                        dict(
                            split=split,
                            method=name,
                            chip=ds.items[i][0],
                            tp=int((pred & truth & valid_mask).sum()),
                            fp=int((pred & ~truth & valid_mask).sum()),
                            fn=int((~pred & truth & valid_mask).sum()),
                            tn=int((~pred & ~truth & valid_mask).sum()),
                        )
                    )
            if (i + 1) % 20 == 0:
                print(split, i + 1, "/", len(ds), flush=True)
        return {k: v.compute() for k, v in metrics.items()}, rows

    validation, _ = evaluate_split("valid")
    timings = {k: [] for k in sessions}
    for s in sessions.values():
        for _ in range(3):
            s.run(None, {"sar": sample})
    for i in range(20):
        names = list(sessions) if i % 2 == 0 else list(reversed(sessions))
        for name in names:
            start = time.perf_counter()
            sessions[name].run(None, {"sar": sample})
            timings[name].append(1000 * (time.perf_counter() - start))
    latency = {
        k: dict(median_ms=float(np.median(v)), p95_ms=float(np.percentile(v, 95)))
        for k, v in timings.items()
    }
    # Fixed acceptance rule: <=1 IoU percentage-point loss, faster median, >=30% smaller.
    drop = validation["fp32"]["iou_water"] - validation["static_int8"]["iou_water"]
    accepted = (
        drop <= 0.01
        and latency["static_int8"]["median_ms"] < latency["fp32"]["median_ms"]
        and static.stat().st_size <= 0.7 * a.model.stat().st_size
    )
    report = dict(
        platform=platform.platform(),
        threshold=config["threshold"],
        fp32_sha256=file_hash(a.model),
        static_int8_sha256=file_hash(static),
        calibration_split="train",
        calibration_seed=42,
        calibration_chips=calibration,
        static_method="QDQ S8S8 per-channel MinMax; 32 training chips; graph preprocessing before quantisation",
        dynamic=dynamic_status,
        validation=validation,
        validation_water_iou_loss=drop,
        acceptance_rule="Validation water IoU loss <=0.01, faster median CPU latency, size <=70% FP32",
        accepted_on_validation=bool(accepted),
        latency=latency,
        bytes={"fp32": a.model.stat().st_size, "static_int8": static.stat().st_size},
        timing_scope="4 CPU threads, batch 1, same first validation chip, 3 warmups, 20 alternating forward passes; no IO or preprocessing",
        primary_artifact="Existing FP32 remains the deployment default; quantised artifact requires explicit opt-in",
    )
    # Once acceptance is frozen, record benchmark held-out comparisons, without retuning.
    rows = []
    report["held_out"] = {}
    if accepted:
        for split in ["test", "bolivia"]:
            metrics, records = evaluate_split(split, True)
            report["held_out"][split] = metrics
            rows.extend(records)
    if rows:
        with (a.output / "held_out_per_chip.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (a.output / "quantization.json").write_text(json.dumps(report, indent=2) + "\n")
    metadata = dict(
        config,
        onnx_sha256=file_hash(static),
        artifact=static.name,
        dtype="int8_qdq",
        accepted_on_validation=bool(accepted),
        calibration_report="quantization.json",
    )
    (a.output / "inference_config.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    with (a.output / "latency_samples.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["repeat", "fp32_ms", "static_int8_ms"])
        writer.writerows(
            (i + 1, timings["fp32"][i], timings["static_int8"][i]) for i in range(20)
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in [(axes[0], "iou_water", "Validation water IoU")]:
        bars = ax.bar(
            ["FP32", "INT8"],
            [validation[k][key] for k in ["fp32", "static_int8"]],
            color=["#1976d2", "#ef8a17"],
        )
        ax.bar_label(bars, fmt="%.4f")
        ax.set_ylim(0, 1)
        ax.set_title(title)
    for ax, values, title, fmt in [
        (
            axes[1],
            [latency[k]["median_ms"] for k in ["fp32", "static_int8"]],
            "Median CPU forward pass (ms)",
            "%.1f",
        ),
        (
            axes[2],
            [a.model.stat().st_size / 1e6, static.stat().st_size / 1e6],
            "Artifact size (MB)",
            "%.1f",
        ),
    ]:
        bars = ax.bar(["FP32", "INT8"], values, color=["#1976d2", "#ef8a17"])
        ax.bar_label(bars, fmt=fmt)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.2)
    fig.suptitle(
        "INT8 experiment | calibration: train only | acceptance chosen on validation"
    )
    fig.tight_layout()
    fig.savefig(a.output / "quantization_tradeoff.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
