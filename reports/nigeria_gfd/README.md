# Nigerian terrain benchmark

**Question.** Within the area observed during a documented flood event, can terrain distinguish locations where a satellite mapped water? The response is the Global Flood Database event-maximum water class, sampled at 250 m after excluding permanent water and poor observations. The intended application is retrospective research. The experiment does not estimate the annual probability of flooding or the chance of a flood in the next 72 hours.

**Sources and sampling.** Select records whose primary country is Nigeria and intersect each footprint with the LSIB Nigeria boundary. Require three or more clear views, a stored clear fraction of at least 0.5, and four valid terrain inputs. The sampler checks the clear fraction against the measured 0–1 range. Elevation and slope come from SRTM. MERIT provides adjusted elevation and upstream area. The adjusted elevation band represents absolute height. Terrain and the permanent-water mask are retrospective products. The original flood mapping procedure also uses terrain filtering, which can inflate apparent terrain discrimination. [Global Flood Database and licence](https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1) · [SRTM](https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003) · [MERIT Hydro](https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1)

**Weighting.** Each of 11 events contributed 500 mapped-water and 1,500 other eligible sample points. Weight each class by its eligible population divided by its number of samples, then scale each event's total weight to one. This avoids treating the deliberately oversampled 25% water fraction as prevalence. Event periods sometimes overlap. The nominal 50 km spatial blocks use UTM zone 32 across Nigeria and do not impose a buffer at block boundaries.

**Evaluation.** Seven events through 2009 trained a fixed LightGBM classifier. Five disjoint block folds within these years test geographic transfer. Two events in 2012 and one in 2015 fit an isotonic score adjustment. The fitted classifier and adjustment were then applied to the 2018 event and an external 2022 case. The event-based split permits the same area to appear in different years. Spatial and temporal evaluations answer different questions.

| Evaluation population | Eligible points | ROC-AUC | Average precision | Brier loss |
|---|---:|---:|---:|---:|
| Training-era spatial folds, unadjusted | 14,000 | 0.9195 | 0.1749 | 0.0102 |
| 2018 event, adjusted | 2,000 | 0.9715 | 0.6201 | 0.0072 |
| Lokoja 2022, prior dry/seasonal class required | 227 | 0.3272 | 0.6413 | 0.6785 |
| Lokoja 2022, unknown prior class included | 3,228 | 0.9394 | 0.6397 | 0.1193 |

**Interpretation.** ROC-AUC measures how often a mapped-water point ranks above a comparison point. Average precision depends on how common mapped water is. For a constant score, average precision equals the water fraction, which was 1.28% (weighted) in 2018, 74.0% in the restricted Lokoja population and 13.2% in the expanded population. Brier loss averages squared probability error and is better when smaller. These scores describe agreement with remote sensing products. The 2022 comparison cannot establish accuracy against field observations.

**Why two Lokoja results?** UNOSAT published a five-day VIIRS maximum water extent. We used the analysis footprint, cloud obstruction and maximum extent layers from a third-party GIS mirror, with a 0.0005-degree geometry simplification tolerance. The first calculation required a known prior dry or seasonal JRC class. It excluded much of the landscape. After inspecting its coverage, we repeated the check including unknown prior class and retained both results. Of 3,228 eligible points in the wider check, 3,001 had unknown prior class. Both calculations exclude documented prior permanent water. The model weights and calibration stayed fixed. The wider comparison was exploratory. The sharp change with geographic coverage precludes using these scores to issue LGA warnings. [UNOSAT product](https://unosat.org/products/3418) · [Geometry mirror](https://services3.arcgis.com/iuNbZYJOrAYBrPyC/arcgis/rest/services/FL20221019NGA_SHP/FeatureServer) · [Annual water classes](https://developers.google.com/earth-engine/datasets/catalog/JRC_GSW1_4_YearlyHistory)

**Artifacts.** The text booster accepts four features in the evaluation manifest's order. Apply the saved isotonic knots after prediction to reproduce adjusted scores. The evaluation manifest records the model hash, event partitions, fold membership and baseline comparisons. Raw event samples and downloaded geometries remain local. [Model](terrain_model.txt) · [Metrics and calibration](evaluation.json) · [Source inventory](source_inventory.json) · [Restricted 2022 check](lokoja_2022.json) · [Expanded 2022 check](lokoja_2022_expanded.json)

## Reproduce

Install the acquisition and pre-app requirements in the project environment. The Intel macOS LightGBM wheel requires OpenMP. On the tested installation the scikit-learn wheel supplies a compatible library, which can be located for the current shell with the first command below.

```bash
export DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/sklearn/.dylibs"
export SSL_CERT_FILE="$(python -m certifi)"
python -m floodrisk.acquire.flood_reference --project "$EE_PROJECT" --output data/flood_reference/nigeria_gfd_fraction
python -m floodrisk.train_susceptibility --samples data/flood_reference/nigeria_gfd_fraction --output reports/nigeria_gfd
python -m floodrisk.eval.lokoja_reference --project "$EE_PROJECT" --model-dir reports/nigeria_gfd --cache data/flood_reference/unosat_lokoja --output reports/nigeria_gfd/lokoja_2022.json
python -m floodrisk.eval.lokoja_reference --project "$EE_PROJECT" --model-dir reports/nigeria_gfd --cache data/flood_reference/unosat_lokoja --output reports/nigeria_gfd/lokoja_2022_expanded.json --include-unknown
```
