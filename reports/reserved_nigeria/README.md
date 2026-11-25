# Blind Nigerian water-mask review packet

The prepared packet contains 48 new 512 × 512 Sentinel-1 chips. Twelve 10.24 km parent windows cover Lagos lagoon edges, the Niger–Benue area and Bayelsa. Four adjacent chips come from each parent. Two parents use the same satellite acquisition, yielding 11 source acquisitions for grouping. All 48 image hashes and shapes were checked. The review form currently contains zero completed masks. The windows and 2024 target dates were chosen before prediction. The preparation script rejects overlap with official Nigerian benchmark chips, the project's earlier radar footprints and the other reserved windows. Purposive selection in three landscapes limits geographic generalization.

The radar model and 0.55 threshold are frozen in the manifest. The packet contains inputs only. No human mask, score or flood-onset label is implied by inclusion. Satellite images, including any supporting optical observations, can contain ambiguous water boundaries.

## Independent review

1. Use the `image` paths in the manifest under the local packet directory. Interpret surface water on the acquisition date without viewing the model predictions. Use an independently identified reference appropriate to the place and time. Record its observation date and a traceable URI. Mark ambiguous or obscured pixels unknown.
2. Save one aligned single-band uint8 GeoTIFF per chip, using 0 for dry, 1 for water and 255 for missing or uncertain pixels. Preserve the chip's dimensions, affine transform and CRS. Set nodata to 255.
3. Fill in a copy of the review CSV. Supply the mask path relative to the annotation directory, SHA-256, reviewer, independent evidence URI, evidence date and review date. Retain the source material for audit. A model-generated mask cannot serve as its own evaluation target.
4. After all chips have been reviewed, run the evaluator. It checks the hashes, target codes and grid alignment. It aggregates intersections and unions over valid pixels and bootstraps whole source acquisitions so adjacent chips are resampled together.

```bash
export SSL_CERT_FILE="$(python -m certifi)"
packet=data/scenes/reserved_nigeria
report=reports/reserved_nigeria
python -m floodrisk.acquire.annotation_set --project "$EE_PROJECT" --root "$SEN1FLOODS11_ROOT" --cache "$packet" --report "$report"
python -m floodrisk.eval.reviewed_masks --manifest "$report/manifest.json" --reviews "$report/review.csv" --cache "$packet" --mask-root "$ANNOTATION_DIR" --model reports/m1-s1-kaggle-365653491/onnx_fp32/m1_s1_fp32.onnx --output "$report/evaluation.json"
```

[Sealed chip IDs and provenance](manifest.json) · [Review form](review.csv)

## Optical evidence available to reviewers

The [candidate table](optical_candidates.csv) lists full-footprint Sentinel-2 surface-reflectance observations acquired within 72 hours of each sealed radar chip. It contains the Earth Engine asset ID, acquisition time, offset from radar time, and share of the chip assigned vegetation, bare ground or water by the 20 m scene classifier. Candidates are ranked within each chip by that share, then by proximity in time. The classifier supplies a visibility screen; a person still has to inspect whether water boundaries are interpretable.

In the 76 chip–image matches, all 48 chips had at least one candidate, but 30 had zero qualifying pixels even in their best candidate. Only two best candidates exceeded 50%, and none reached 80%. An image shortly before or after a radar pass can also show a different water boundary. The review CSV remains blank until independent annotation. Rebuild the catalogue with `python -m jobs.blind_optical_evidence --project YOUR_EE_PROJECT`.

[Download the sealed review packet](https://github.com/okiv21/Flood-Risk-Mapper/releases/tag/blind-nigeria-review-v1). Extract the ZIP to use its 48 GeoTIFFs, chip manifest, blank review form and dated optical candidate list. The extracted directory is the evaluator's `--cache`. Keep completed annotations in a separate directory and supply a copy of the completed review form to `--reviews`. The packet ZIP SHA-256 is `05365bbd7e7186cbccb0156f7e661188704375f4167e67cdad081b741b403c73`.
