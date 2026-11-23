# October 2022 historical map package

Each site covers a 512 × 512 pixel grid at 10 m spacing. The closest eligible full-footprint Sentinel-1 acquisition to 15 October was selected within October, without inspecting flood labels. Dates differ by site. Each PMTiles v3 file contains georeferenced PNG tiles at zoom levels 8–14. Nearest-neighbour sampling preserves water categories. Higher zoom displays the same underlying observations. The archives passed a read-back and PNG integrity check for all 178 tiles.

| Pixel class | Meaning |
|---|---|
| 0 | Radar model predicts dry |
| 1 | Predicted water at a location with at least 90% historical water occurrence |
| 2 | Predicted water where reported historical occurrence is below 90% |
| 3 | Predicted water where historical occurrence is unavailable |
| 255 | Missing radar observation; transparent on the map |

EC JRC/Google supplies historical occurrence for 1984–2021. The source is 30 m and is aligned to the 10 m radar grid. Water can be seasonal, tidal or permanent. A single pass does not establish a flood onset or building damage.

The LGA crosswalk uses the 2022 GRID3 Nigeria boundaries distributed by geoBoundaries under CC BY 4.0. The registry stores 774 unique shape IDs with their names, parent state or Federal Capital Territory, and one interior point each. Each point matched exactly one of 37 GRID3 ADM1 polygons. Five pairs of LGAs share a name, including Surulere in Lagos and Surulere in Oyo. A point may sit away from the most populated part of an LGA. The pixel table assigns by pixel centre and keeps points outside the polygons separate. Multiplying counts by 100 gives nominal square metres on the projected grid. Only the parts of 22 LGAs within the observation windows contribute counts. [LGA boundary metadata](../../data/cases/nga_adm2_source.json) · [State boundary metadata](../../data/cases/nga_adm1_source.json)

[Tile index, spatial bounds and hashes](tiles/index.json) · [Dated scene manifests](observations.json) · [LGA registry](lgas.csv) · [Pixel counts by LGA](observed_pixels_by_lga.csv)

## Reproduce

From the repository root, with the project virtual environment active and an authenticated Earth Engine project:

```bash
export SSL_CERT_FILE="$(python -m certifi)"
run=reports/m1-s1-kaggle-365653491/onnx_fp32
root=data/scenes/hotspots_oct22
out=reports/hotspots_oct22
python -m floodrisk.acquire.hotspots --project "$EE_PROJECT" --root "$root" --model "$run/m1_s1_fp32.onnx" --config "$run/inference_config.json" --report "$out/observations.json"
python -m floodrisk.deploy.hotspot_tiles --root "$root" --observations "$out/observations.json" --output "$out/tiles"
mkdir -p data/boundaries
curl -fL 'https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/gbOpen/NGA/ADM2/geoBoundaries-NGA-ADM2.geojson' -o data/boundaries/nga_adm2_verified.geojson
curl -fL 'https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/9469f09/releaseData/gbOpen/NGA/ADM1/geoBoundaries-NGA-ADM1.geojson' -o data/boundaries/nga_adm1_verified.geojson
python -m floodrisk.features.lga_observations --states data/boundaries/nga_adm1_verified.geojson --boundaries data/boundaries/nga_adm2_verified.geojson --observations "$out/observations.json" --root "$root" --output "$out"
python -m floodrisk.features.hotspot_surface --project "$EE_PROJECT" --root "$root" --observations "$out/observations.json" --output "$out/surface_overlap.json"
```


## Earlier-scene comparison

Each site has an earlier Sentinel-1 reference selected near 15 February 2022, using the October pass direction and relative orbit. In the paired areas, 40,570 pixels changed from predicted dry to predicted water where historical water occurrence was below 90%. Another 52,824 changed where historical occurrence was unavailable. A change between two acquisition dates can reflect seasonal water, a genuine flood, or model error. The pair alone cannot identify an onset date. [Paired observations and hashes](paired_change.json)

Reproduce with `python -m floodrisk.features.hotspot_change --project "$EE_PROJECT" --root "$root" --observations "$out/observations.json" --model "$run/m1_s1_fp32.onnx" --config "$run/inference_config.json" --output "$out/paired_change.json"` from the shell setup above.


The companion change archives encode 0 for no predicted dry-to-water change, 1–3 for changes grouped by historical occurrence, and 255 for missing data. Each archive carries its earlier scene ID and October acquisition date. Build them with `python -m floodrisk.deploy.hotspot_tiles --root "$root" --observations "$out/paired_change.json" --output "$out/change_tiles" --changes`. [Change archive index](change_tiles/index.json)

## Mapped buildings

Across the eight windows, 248,062 Open Buildings detections with confidence of at least 0.75 had centroids inside the study areas. The surface overlay counts 10 m grid cells where predicted October water overlaps the rasterized footprints. For example, 63 occupied pixels overlapped water in the lower historical-occurrence category. Multiple pixels can belong to a single building. WorldCover dates to 2021 and the Open Buildings release to 2023. Individual damage and the October 2022 building inventory are unknown. [Overlap counts for every site and water class](surface_overlap.json)
