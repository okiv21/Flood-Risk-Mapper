# First real Lagos radar scene

Status: **authentication, project access and live acquisition verified** with `manifest-ivy-453612-b3`. The first Lagos-area scene (2022-08-12) has been downloaded and processed. See the [real-scene inspection](../reports/lagos_2022/README.md).

## One-time access setup

1. Open [Google's Earth Engine registration page](https://console.cloud.google.com/earth-engine) with your Google account.
2. Create or select a Cloud project. Use a descriptive project name such as `Nigeria Flood Risk Mapper`; note the actual **project ID**, which may differ from the name.
3. Complete Earth Engine registration and its use-category questionnaire accurately. Use noncommercial registration if your actual use qualifies. Ensure the Earth Engine API is enabled. See [Google's access guide](https://developers.google.com/earth-engine/guides/access).
4. Authenticate in your local terminal, with the project environment active:

```bash
cd "/Users/mac/Documents/Ml PROJECTS/Flood-Risk-Mapper"
source .venv/bin/activate
python -m pip install -r requirements/acquisition.txt
python -m floodrisk.acquire.authenticate --project manifest-ivy-453612-b3
```

Complete Google's browser login flow locally. The authentication helper uses the installed CA bundle with TLS verification enabled and requests Earth Engine/Cloud access only. Do not put credentials in the repository or chat. The account and project must have the permissions described in the [authentication guide](https://developers.google.com/earth-engine/guides/auth).

## Acquisition plan

The initial region is a 10.24×10.24 km grid around longitude **3.45**, latitude **6.55**, in the Lagos area. It is an engineering study area, not an LGA boundary. The output grid is **1024×1024 at 10m**, UTM zone 31N (EPSG:32631), with origin snapped to a 10m grid.

Search interval: 2022-08-01 through 2022-11-01 (exclusive end). Select the earliest full-footprint, IW, 10m, dual VV/VH acquisition among the first 50 chronological candidates. This is a historical acquisition rule, not identification of a confirmed flood event. The exact date, orbit, platform and scene index will be recorded after a successful query.

[Earth Engine's Sentinel-1 GRD collection](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) supplies calibrated, terrain-corrected sigma-zero backscatter in dB. VV is exported first, VH second. Do not apply another log conversion to this collection. Matching units does not establish identical processing or geographic conditions to Sen1Floods11.

The direct download is limited to 8MiB of two-band float32 pixel data (below the documented [32MB request limit](https://developers.google.com/earth-engine/apidocs/ee-image-getdownloadurl)). This is a small-area workflow, not national acquisition infrastructure.

## Commands

Inspect the plan without credentials or a remote request:

```bash
python -m floodrisk.acquire.earthengine --project pending --plan
```

After authentication, substitute your actual project ID:

```bash
python -m floodrisk.acquire.earthengine \
  --project YOUR_PROJECT_ID \
  --output data/scenes/lagos_2022
```

The downloader checks the output CRS, pixel grid, band count and at least 99% usable coverage, then saves `sentinel1_vv_vh_db.tif` and `acquisition.json`. Missing pixels use -9999 no-data. Scene identifiers, source properties, grid settings and the file hash are recorded; ephemeral download URLs are not saved. Existing output directories are preserved.

The output can then be processed with `floodrisk.deploy.tiled` at halo 0 and 64 for a real-scene comparison. Such a comparison can show context sensitivity and map alignment; it cannot establish segmentation accuracy without independent human labels. Permanent-water removal remains necessary for interpreting water as flood extent.

## Checks completed

Offline tests verify the 10m projected grid, its 10.24km extent and centre, download-size limit and date ordering. Eight project tests pass. The live query, download, raster validation, and provenance writing also completed successfully. The client mutates its download-options dictionary, so a copy is passed to preserve JSON-serializable grid metadata.
