# M1-S1 training record

Source: user-downloaded `history.csv` and `best.pt`, associated with Kaggle script version `365653491`. The CSV contains 60 consecutive epochs. Best validation water IoU is **0.6895**, with **0.8227 mIoU**, at epoch **49**. Final epoch validation water IoU is **0.6744** and mIoU **0.8146**.

Logged epoch durations total **624.7 seconds**; this excludes setup and is not an end-to-end runtime benchmark. Metrics in the CSV are rounded to four decimal places.

See [history.csv](history.csv) for every epoch and [summary.json](summary.json) for the artifact hash and machine-readable summary. The checkpoint is retained locally at repository-root `best.pt` and excluded from Git. It loaded successfully with strict state-dictionary matching into the current U-Net architecture and was evaluated on CPU.

The current source defaults specify U-Net/ResNet-34, two SAR channels, masked BCE + Dice, AdamW, and 60 epochs. Exact training commit and environment were not supplied, so those defaults are not independently verified run metadata. Checkpoint selection in the current code uses validation water IoU; the supplied file is expected to be the best checkpoint, pending verification.

## Held-out evaluation

Evaluated locally on 2026-11-16 using dataset version 1, CPU float32 inference, probability threshold > 0.5, and pooled IoU excluding label -1.

| Split | Chips | Water IoU | mIoU |
|---|---:|---:|---:|
| Test | 90 | 0.697312 | 0.823634 |
| Bolivia | 15 | 0.690794 | 0.808678 |

[Evaluation JSON](evaluation.json) records unrounded metrics, package versions, and checkpoint/split hashes. Compared with the rounded Otsu results, test water IoU improves by approximately 15.1 percentage points; Bolivia water IoU is approximately 0.6 points lower. This is mixed evidence of geographic generalisation, not a uniform improvement.

The checkpoint matches the architecture, but its exact training revision and correspondence to epoch 49 cannot be independently proved from a state dictionary alone.


## Inference and intended use

The operational artifact for historical observations is the FP32 ONNX U-Net with two input channels in VV/VH order. Apply the training-time dB clipping and normalization. The graph emits logits. Apply sigmoid outside the graph and classify scores strictly greater than 0.55 as water. The georeferenced inference path preserves missing-data masks. Radar shadow, built surfaces, permanent water and infrequent revisit times complicate flood interpretation. Use the model for dated research observations. [Configuration and artifact hashes](onnx_fp32/inference_config.json)

The Nigerian subset contains four official test chips and has wide chip-bootstrap intervals. A separate Nigerian human-annotated test set remains outstanding. The static INT8 version is smaller but slower on the measured Intel CPU, so FP32 remains the default. [Nigerian subset](nigeria_subset/summary.json) · [Quantization comparison](int8_experiment/quantization.json)
