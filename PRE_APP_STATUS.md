# Pre-application readiness

| Deliverable | Evidence available | Remaining dependency |
|---|---|---|
| Historical radar map for a first interface | Eight matched-orbit February/October pairs, dated 10 m hotspot windows, context categories, validated local PMTiles, 774 LGA lookup entries and partial-footprint counts | Expansion to full cities, multiple seasons and national coverage remains a separate batch campaign |
| Radar model | Kaggle-trained U-Net, official test and Bolivia evaluations, four Nigerian official test chips, working FP32 CPU runtime and measured INT8 alternative | The independently annotated 40–60-chip Nigerian test set needs new reviewed masks. The Lokoja queue has zero accepted labels |
| Optical comparison | Local source notebook, four-band preprocessing, real optical-chip/label read check, official split loader | Private Kaggle GPU execution and its resulting checkpoint and metrics |
| Terrain susceptibility | 11 historical events, design-weighted classifier and baselines, five spatial folds, frozen 2018 evaluation, external 2022 Lokoja comparison | National 100 m full-feature inference, full-season external validation, building exposure and area-weighted LGA scores remain open |
| Rain and 72-hour event risk | A complete 774-LGA national rainfall snapshot, as-of window checks, temporal training and local batch scoring entry points | Reviewed onset and continuously observed event-free windows, contemporaneous forecast and static inputs for all periods, and a validated model. No real probability, lead time or false-alarm rate has been established |
| Delivery | Local files for a historical observation interface | UI, public hosting and notifications belong to the application phase |

## First interface boundary

The October 2022 tile archives and their index can supply a historical observation viewer. Use the acquisition date and water-context legend supplied with each archive. The wider warning product requires separately collected outcome evidence. The spec places the first map interface before national multi-season inference and the operational daily risk model.

## Forecast data contract

One record represents one LGA and one issue time. Supply timezone-aware issue, window start and exclusive end, availability times for the forecast, antecedent and static inputs, source identifiers, a versioned susceptibility estimate on a 0–1 scale, built-up fraction, preceding 1/3/7/14-day rain, next 24/48/72-hour forecast rain and maximum hourly rain. Keep the underlying forecast response as retrieved. For training, record the reviewer, evidence ID, observation interval and evidence kind. A reviewed onset interval has to fit inside the horizon. An event-free record must cover the entire horizon. An absent report has an unknown outcome.

The research training command is `python -m floodrisk.train_event --source windows.csv --output model_dir`. It reserves 2022 for isotonic calibration and 2023 onward for testing, drops windows crossing split boundaries and rejects evidence reused across partitions. The scoring entry point is `python -m jobs.daily_score --inputs issue.csv --registry reports/hotspots_oct22/lgas.csv --model-dir model_dir --output scores.csv`. A synthetic integration check exercised 774 rows and discarded its temporary outputs. Real training and daily risk scores are pending. A forecast archive assembled from the initial hours of successive runs cannot reproduce the full 72-hour forecast available on an earlier date. [Open-Meteo historical archive documentation](https://open-meteo.com/en/docs/historical-forecast-api)
