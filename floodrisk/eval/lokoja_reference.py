"""Compare the frozen terrain model to an external October 2022 water map near Lokoja."""

import argparse
import hashlib
import json
from pathlib import Path
import ee
import numpy as np
import pandas as pd
import requests
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union
from lightgbm import Booster
from floodrisk.train_susceptibility import FEATURES, metrics

SERVICE = "https://services3.arcgis.com/iuNbZYJOrAYBrPyC/arcgis/rest/services/FL20221019NGA_SHP/FeatureServer"
BOUNDS = [6.45, 7.50, 7.04, 8.08]


def footprint(layer, cache):
    params = dict(
        f="geojson",
        where="1=1",
        geometry=",".join(map(str, BOUNDS)),
        geometryType="esriGeometryEnvelope",
        inSR=4326,
        outSR=4326,
        spatialRel="esriSpatialRelIntersects",
        outFields="*",
        returnGeometry="true",
        maxAllowableOffset=0.0005,
    )
    path = cache / f"layer_{layer}.geojson"
    if not path.exists():
        response = requests.get(
            f"{SERVICE}/{layer}/query", params=params, timeout=(30, 240)
        )
        response.raise_for_status()
        doc = response.json()
        if doc.get("type") != "FeatureCollection" or doc.get("exceededTransferLimit"):
            raise ValueError("Incomplete reference geometry")
        path.write_bytes(response.content)
    doc = json.loads(path.read_bytes())
    geometry = unary_union(
        [
            shapely.make_valid(shape(f["geometry"])).intersection(box(*BOUNDS))
            for f in doc["features"]
        ]
    )
    return geometry, hashlib.sha256(path.read_bytes()).hexdigest()


def run(project, model_dir, cache, output, include_unknown=False):
    cache.mkdir(parents=True, exist_ok=True)
    polygons = {}
    hashes = {}
    for layer in [2, 3, 4]:
        polygons[layer], hashes[layer] = footprint(layer, cache)
    source = cache / ("terrain_with_unknown.csv" if include_unknown else "terrain.csv")
    if not source.exists():
        ee.Initialize(project=project)
        srtm = ee.Image("USGS/SRTMGL1_003")
        merit = ee.Image("MERIT/Hydro/v1_0_1")
        # 2021 annual classes: 0 unknown, 1 dry, 2 seasonal, 3 permanent.
        water = (
            ee.ImageCollection("JRC/GSW1_4/YearlyHistory")
            .filterDate("2021-01-01", "2022-01-01")
            .first()
            .select("waterClass")
        )
        stack = (
            ee.Image.cat(
                [
                    srtm.select("elevation"),
                    ee.Terrain.slope(srtm),
                    merit.select("hnd"),
                    merit.select("upa").add(1).log(),
                ]
            )
            .rename(FEATURES)
            .toFloat()
        )
        if include_unknown:
            annual = water.unmask(0).rename("prior_water_class")
            stack = stack.addBands(annual).updateMask(annual.neq(3))
        else:
            stack = stack.updateMask(water.eq(1).Or(water.eq(2)))
        pts = stack.sample(
            region=ee.Geometry.Rectangle(BOUNDS, geodesic=False),
            scale=250,
            projection="EPSG:32632",
            numPixels=3500,
            seed=42,
            geometries=True,
            tileScale=4,
        ).getInfo()
        rows = []
        for f in pts["features"]:
            lon, lat = f["geometry"]["coordinates"]
            rows.append({**f["properties"], "longitude": lon, "latitude": lat})
        pd.DataFrame(rows).to_csv(source, index=False)
    frame = pd.read_csv(source)
    points = shapely.points(frame.longitude, frame.latitude)
    eligible = shapely.covers(polygons[2], points) & ~shapely.covers(
        polygons[3], points
    )
    data = frame[eligible].copy()
    data["target"] = shapely.covers(polygons[4], points[eligible]).astype(int)
    data["sample_weight"] = 1
    manifest = json.loads((model_dir / "evaluation.json").read_text())
    model = model_dir / "terrain_model.txt"
    if hashlib.sha256(model.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Changed terrain model")
    raw = Booster(model_file=str(model)).predict(data[FEATURES])
    scores = np.interp(raw, manifest["isotonic"]["x"], manifest["isotonic"]["y"])
    report = dict(
        source_product="https://unosat.org/products/3418",
        geometry_mirror=SERVICE,
        layer_sha256=hashes,
        model_sha256=manifest["model_sha256"],
        terrain_sample_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        window="2022-10-13/2022-10-17",
        bounds_lonlat=BOUNDS,
        samples=len(frame),
        eligible=len(data),
        mapped_water=int(data.target.sum()),
        prior_water_class_counts=data.prior_water_class.value_counts().to_dict()
        if include_unknown
        else None,
        criteria=(
            "2021 JRC yearly class dry, seasonal or unknown"
            if include_unknown
            else "2021 JRC yearly class dry or seasonal"
        )
        + "; valid terrain; inside analysis extent; outside cloud obstruction",
        sampling="Seed 42 random sample of 250 m UTM zone 32 grid; equal weights; vector generalisation tolerance 0.0005 degree",
        raw=metrics(data, raw),
        calibrated=metrics(data, scores),
        constant_train_rate=metrics(
            data,
            np.full(
                len(data),
                manifest["partitions"]["train"]["weighted_reference_fraction"],
            ),
        ),
        scope="Single river confluence and event; agreement with published VIIRS mapped extent. Timing within the event, dry streets and field accuracy remain unknown.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--include-unknown", action="store_true")
    for k in ["model-dir", "cache", "output"]:
        p.add_argument("--" + k, required=True, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            run(a.project, a.model_dir, a.cache, a.output, a.include_unknown), indent=2
        )
    )
