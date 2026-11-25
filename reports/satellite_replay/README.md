# Satellite-target replay

The spec proposes a 0.5 km² threshold for newly mapped water. We use it as written. A positive label means the frozen radar model classified more than 5,000 10 m pixels as changing from dry in the dry-season image to water in the target image, with historical water occurrence below 90%. A negative label requires that even counting all uncertain pixels could not exceed 5,000. Where missing data could change the answer, the label is unresolved. This measures a satellite proxy. A real flood can start and recede between passes.

## October 2022

[Eight footprint observations](cases.csv) and [their LGA intersections](lga_bounds.csv) use the original project grids, pinned boundaries and same-orbit reference masks. The 5.12 km squares contain parts of 22 LGAs. An LGA can cross the threshold within a partial view, but a small change in that view cannot establish a negative for the whole LGA. Areas outside the polygons remain unassigned. The area rule uses pixel centres and nominal projected area.

We retrieved [the archived GFS precipitation blocks](gfs_blocks.json) at each window centre from the original model runs initialized at least 24, 48 and 72 hours before acquisition. Select the latest six-hour initialization at or before each cut-off. Sum disjoint six-hour accumulations up to the final six-hour boundary before the pass. The remaining fraction of a cycle is excluded. Initialization and operational delivery times differ. The weather model has approximately 28 km grid spacing, so it cannot resolve rainfall within a 5 km window. [Earth Engine GFS documentation](https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25)

## Whole-LGA pilot protocol

Onitsha South was selected after inspecting October 2022. Its entire boundary fits within one model input. This choice favors a site with known surface-water change. Fix the extraction to 2018–2025, Sentinel-1A ascending relative orbit 30, choosing the eligible full-footprint acquisition nearest the 15th of February, July and October in each calendar month. Use each year's February baseline. Record missing pairs and unresolved labels. Use the fixed 1984–2021 JRC occurrence product only in defining the retrospective target. It is never a forecast input.

For a simple rainfall benchmark, flag a sampled date when the last four six-hour GFS blocks ending before the radar pass sum to at least 25 mm. The 25 mm cut-off is an illustrative fixed rule with no established local alert meaning. Use the LGA registry point. Compare this rule with a calendar-month majority baseline estimated on resolved 2018–2021 samples, resolving ties as below threshold. Report the 2022 calibration-period cases separately; neither baseline is tuned there. Evaluate 2023–2025 by lead and report raw counts with denominators. With only two target dates per year and one selected LGA, these counts cannot establish operational warning performance.

## Held-out results

The [hosted benchmark](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45661708757) completed all 24 planned acquisitions, 16 February-to-wet-season comparisons and 48 forecast replays. The [selection ledger](onitsha_selection.json) records timestamps, source scenes, hashes and candidate counts. Each of the six held-out dates was resolved even after allowing for unknown pixels. July was below the threshold in each held-out year; October was above in each year.

At each nominal lead (24, 48 and 72 hours), each rule yielded the same counts: **0 hits, 3 misses, 0 false positives, 3 true negatives**. The rainfall rule never reached 25 mm in the selected 24-hour forecast interval. The calendar rule also never flagged: the four training Octobers split evenly and the registered tie rule selected below threshold. These are six unique radar observations, reused at three forecast leads. The 18 predictions do not represent 18 independent outcomes. [Per-date targets](onitsha_south.csv) · [Forecasts and decisions](onitsha_predictions.csv) · [Counts](onitsha_metrics.csv)

The comparison is against a change in modeled surface water within one selected LGA. It provides no measured onset time, and no rule triggered on a held-out date, so there is no successful warning lead time to report. The site was selected after looking at 2022, limiting the interpretation of calibration-period performance. Any threshold chosen after seeing the held-out years requires fresh test data.
