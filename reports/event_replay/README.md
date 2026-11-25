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
