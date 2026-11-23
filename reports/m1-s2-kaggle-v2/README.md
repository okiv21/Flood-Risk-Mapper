# Sentinel-2 optical comparison

Kaggle ran the locally maintained notebook for 60 epochs on a CUDA worker. The U-Net uses a ResNet-34 encoder and four Sentinel-2 reflectance bands in red, green, blue and near-infrared order. Reflectance is divided by 10,000 and clipped to [0, 1]. The loss ignores missing optical inputs and unknown labels. Nonzero cloud-obscured observations remain in the evaluation. [Local training notebook](../../notebooks/16_optical/16-optical-comparison.ipynb) · [Private Kaggle run, version 2](https://www.kaggle.com/code/okimosadi/flood-mapper-optical-comparison/versions/2)

| Official split | Chips | Water IoU | Mean IoU |
|---|---:|---:|---:|
| Validation | 89 | 0.8071 | 0.8906 |
| Test | 90 | 0.8171 | 0.8946 |
| Bolivia | 15 | 0.8033 | 0.8822 |

These figures use the Kaggle checkpoint, threshold 0.50 and each chip’s optical valid-pixel mask. Epochs 44 and 60 both round to validation water IoU 0.8071 in the CSV. The training code selects a checkpoint using the unrounded value. [Epoch history](history_s2.csv) · [Kaggle evaluation](evaluation_s2.json)

The downloaded weights are 97,931,411 bytes with SHA-256 `337a08f8eb9dea2351142dbe9759783551085f0ea06e998eb21c3ae9f03053b1`. Download `best_s2.pt` from the private run to reproduce inference. The optical model estimates water at acquisition time. It cannot supply imagery through cloud cover or estimate flood onset.

## Reproduce the paired CPU evaluation

Use the versioned radar and optical checkpoints and the same official split files. The local optical cache uses the original Sen1Floods11 public bucket and is independent of the training job's temporary files.

```bash
python -c 'import sys; from pathlib import Path; from floodrisk.datasets.optical import prepare; prepare(Path(sys.argv[1]), Path(sys.argv[2]), splits=("valid", "test", "bolivia"))' "$SEN1FLOODS11_ROOT" data/scenes/optical_test
python -m floodrisk.eval.paired_sensors --root "$SEN1FLOODS11_ROOT" --cache data/scenes/optical_test --radar best.pt --optical reports/m1-s2-kaggle-v2/best_s2.pt --output reports/m1-s2-kaggle-v2
```


On the common valid pixels, optical/radar water IoU was **0.8172/0.6923** on test and **0.8067/0.7091** on Bolivia. There were 20,440,638 common test pixels and 2,819,585 common Bolivia pixels. The model comparison uses CPU FP32. Small differences from the Kaggle mixed-precision evaluation are recorded separately. [Counts by chip](paired_chips.csv) · [Comparison notebook](../../notebooks/18_sensor_comparison/18-sensor-comparison.ipynb)

The FP32 ONNX export passed probability and mask checks on all 89 validation chips. Its input is four normalized optical bands. The radar GeoTIFF entry point expects VV/VH. Generate a fresh export with `python -m floodrisk.deploy.export --root "$SEN1FLOODS11_ROOT" --optical-cache data/scenes/optical_test --checkpoint reports/m1-s2-kaggle-v2/best_s2.pt --config reports/m1-s2-kaggle-v2/optical_config.json --output data/scenes/optical_export`. [Benchmark](onnx_fp32/benchmark.json) · [Input contract](onnx_fp32/inference_config.json)
