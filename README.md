# Flood Risk Mapper

Flood risk mapping for Nigeria from Sentinel-1 radar imagery.

## Data: Sen1Floods11 (hand-labelled)

- Train chips: 252
- NaN pixels: 2.01%
- No-data labels: 13.26% (ignored in loss and metrics)
- Water share (of valid pixels): 9.51%

Normalisation (computed on the training split):

| Band | Clip range (dB) | Mean | Std |
|---|---|---|---|
| VV | (-30, 5) | -10.394 | 4.020 |
| VH | (-42, -5) | -17.232 | 4.691 |

## Results (M1 water segmentation)

IoU is masked (no-data excluded) and pooled over the whole split.

| Method | Test water IoU | Test mIoU | Bolivia water IoU | Bolivia mIoU |
|---|---|---|---|---|
| Otsu threshold (S1 VH) | 0.546 | 0.731 | 0.697 | 0.814 |
| U-Net (ours) | TBD | TBD | TBD | TBD |
