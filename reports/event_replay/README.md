# Rainfall forecast replay for two Nigerian floods

These two cases answer a narrow question: how much rainfall did a weather model predict at the project's LGA point before a documented flood? Each row uses a complete ECMWF forecast initialized before the event. The model initialization time is available in the archive. Its historical delivery time to an alerting system is unknown.

| Location | Event evidence and date precision | Independent satellite observation |
|---|---|---|
| Mokwa, Niger | [NEMA](https://nema.gov.ng/flash-flood-in-mokwa-niger-state-nemas-early-warning-response-efforts-and-the-role-of-state-local-governments-in-flood-prevention/) reports a flash flood following intense rain on 28 May 2025. Use 28 May as the calendar anchor. | [UNOSAT](https://unosat.org/products/4137) mapped water in Mokwa Town using optical imagery from 31 May. This observation postdates the event. |
| Maiduguri, Borno | UNICEF describes effects on [9 September](https://www.unicef.org/nigeria/media/14381/file/Nigeria%20SitAn%202024%20Report_pdf.pdf) and extensive inundation on [10 September 2024](https://www.unicef.org/documents/nigeria-situation-report-maiduguri-flood-response-10-sep-23-sep-2024). Use the earlier date as the anchor. A breach of the Alau Dam contributed to this event. | [UNOSAT](https://unosat.org/static/unosat_filesystem/3967/UNOSAT_Preliminary_Assessment_Report_FL20240902NGA_Maiduguri_13Sep2024.pdf) identified increased water along the Ngadda River on 11 September at 10:01 UTC. |

The three forecasts initialized at 00 UTC on the three dates preceding each anchor predict the following totals for their own successive 72-hour windows:

| Location | Three calendar days before | Two days before | One day before |
|---|---:|---:|---:|
| Mokwa | 0.5 mm | 6.3 mm | 5.6 mm |
| Maiduguri | 8.0 mm | 3.0 mm | 11.1 mm |

The underlying locations are one interior registry point per LGA. Conditions in the town, upstream basin and rest of the LGA can differ. The three forecasts have different end times. The earliest windows can close before the reported flood. These numbers alone establish neither warning lead time nor model accuracy. Both cases were selected because flooding was reported. Evaluating missed events and false alarms requires a prespecified decision rule, comparable event and continuously monitored non-event periods, and a held-out evaluation population.

Run `python -m jobs.event_replay` to regenerate [the results](runs.csv). Each row links to its [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api) request and identifies the local response by SHA-256. Precipitation in millimetres is accumulated from hour-ending steps 1 through 24, 48 or 72. Missing, negative and non-finite values are rejected.

A separate radar replay uses the project's existing Maiduguri window, a fixed U-Net and the same descending orbit on 5 and 17 September. All 262,144 pixels have valid predictions on both dates. The independent optical observations on 11 and 12 September fall between these passes. Comparing the 17 September mask directly with those earlier outlines would mix model error with actual changes in water extent. The observed change applies only to the study window. Rebuild with `python -m jobs.maiduguri_replay --project YOUR_EE_PROJECT`. [Copernicus activation EMSR753](https://mapping.emergency.copernicus.eu/activations/EMSR753/)

## Independent outcome records needed for the warning evaluation

The [blank collection sheet](outcome_evidence_template.csv) is for independently documented observations. A reviewer should record the boundary identifier from the [774-LGA registry](../hotspots_oct22/lgas.csv), observation method, geographic coverage, any monitoring gaps, evidence ID, source URI and UTC times with offsets. Retain the supporting record. Use `onset_interval` only when the source bounds the onset; a date on which flooding was first mapped can be later. Use `continuous_event_free` only for periods with evidence covering the intended area and entire forecast window. A source that covers one neighborhood or gauge supports only that narrower inference. Silence in a disaster bulletin leaves the outcome unknown.

The sheet records evidence separately from forecasts. Join reviewed records to issue-time features only after checking location and time. Keep evidence IDs together across splits. The specified experiment needs independently established positive and negative windows for training (2016–2021), calibration (2022), and final testing (2023 onward). A single reviewed case can start the register; no case by itself establishes national warning skill.
