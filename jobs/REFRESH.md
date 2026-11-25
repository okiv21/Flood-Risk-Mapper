# Scheduled data refresh

GitHub Actions runs the refresh jobs while the laptop is off. All times below are UTC. Add one hour for Lagos.

| Product | Schedule | Maximum age for a fresh status |
|---|---|---|
| Forecast rainfall, all 774 LGA points | 00:17, 06:17, 12:17, 18:17 daily | 9 hours, and before forecast expiry |
| Recent rainfall estimates | 01:47 daily | 36 hours |
| Sentinel-1 inventory for the eight project windows | 02:47 daily | 36 hours for the inventory check |
| Terrain, water, land-cover and building release metadata | 03:47 on the first of each month | 40 days |

The recent-rainfall job derives preceding 1/3/7/14-day estimates from the latest complete forecast archive. Its timestamp stays tied to the parent collection. These estimates come from weather models. They are distinct from finalized CHIRPS measurements.

Each successful job publishes a prerelease containing an immutable ZIP archive with source responses, derived tables, timestamps and file hashes. A later run restores the latest successful archive and verifies its hashes. Failed acquisition leaves the prior archive intact. GitHub diagnostic artifacts expire after 30 days, while release assets remain available until explicitly deleted. These small research archives will need storage monitoring as they accumulate.

The Data freshness workflow independently checks all four products every two hours and fails if any is missing or overdue.

Run `python -m jobs.refresh status` after restoring archives to calculate their age at the current time. Snapshot timestamps are preserved across failures. A missing or stale product must be displayed as such by its eventual consumer. Workflow schedules can be delayed, and GitHub disables scheduled jobs in inactive public repositories after 60 days. The Actions status and calculated age must be monitored. [GitHub schedule behavior](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)

## Credentials

Rainfall and release-catalog checks use public sources. Satellite queries use Google Workload Identity Federation. Configure repository variables `EE_WORKLOAD_IDENTITY_PROVIDER` and `EE_SERVICE_ACCOUNT` with the Google provider resource and service-account email. The provider must restrict the GitHub repository identity to this repository. The service account needs Earth Engine access in `manifest-ivy-453612-b3`. The configured identity is `flood-refresh@manifest-ivy-453612-b3.iam.gserviceaccount.com`, with Earth Engine Viewer and Service Usage Consumer access. The provider restricts the repository ID, owner ID, main branch and refresh workflow path. This setup uses short-lived credentials. The Cloud Resource Manager, IAM and STS APIs were enabled in the existing project during setup.

## Manual execution

Use Actions → Refresh data → Run workflow to choose a product. Local commands are `python -m jobs.refresh forecast`, `python -m jobs.refresh recent_rain`, `python -m jobs.refresh satellite`, and `python -m jobs.refresh sources`. Local satellite execution can reuse the authenticated Earth Engine session. Archived data live under ignored `data/refresh/`.

Satellite inventories cover suitable full-footprint VV/VH images available in the past 30 days. Acquisition time is preserved separately from check time. A fresh inventory can contain older imagery. After a successful inventory, the water-maps job runs the frozen radar model for each unpublished site and scene in the rolling window. Each map release includes acquisition time, radar input, water scores, mask and provenance. Static-source changes are recorded for review before adoption. Retraining requires reviewed evidence and validation.

## Verification on 24 November 2026

Manually dispatched hosted runs passed for [forecast rainfall](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45358711485), [recent rainfall](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45391929073), [source releases](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45392173975), and [satellite inventory](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45392862995). The [independent freshness check](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45393023122) also passed. These verify hosted execution; they do not yet demonstrate a history of scheduled cron executions.

A [second satellite run](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45393053543) restored the previous archive and found 43 eligible scenes across eight windows, with zero new scenes. Its release archive was downloaded again and its file hashes verified. The local suite passed all 29 tests, including stale/expired snapshot handling and protection against replacing newer data with older data.

On 25 November, the release archives confirmed two forecast updates triggered by the schedule. The [first overnight health check](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45485196801) flagged a 12.1-hour-old forecast against the nine-hour limit. After the [next automatic forecast publication](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45500996617), the [following health check](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45550732674) found all four products fresh. A [second scheduled forecast run](https://github.com/okiv21/Flood-Risk-Mapper/actions/runs/45577515514) also published successfully. This verifies automatic forecast publication and stale-data detection. At the time of this review, hosted verification for daily radar, daily recent rainfall and monthly sources was still based on manual dispatch.
