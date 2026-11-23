"""Sample independent, coarse published event-water references for an exploratory M2 benchmark."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from floodrisk.deploy.tiled import file_hash


def acquire(project, output):
    import ee
    from rasterio.warp import transform

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ee.Initialize(project=project)
    ee.data.setDeadline(180000)
    country_source = "USDOS/LSIB_SIMPLE/2017"
    country = (
        ee.FeatureCollection(country_source)
        .filter(ee.Filter.eq("country_na", "Nigeria"))
        .geometry()
    )
    collection = (
        ee.ImageCollection("GLOBAL_FLOOD_DB/MODIS_EVENTS/V1")
        .filter(ee.Filter.eq("dfo_country", "Nigeria"))
        .sort("system:time_start")
    )
    objects = collection.toList(100)
    n = collection.size().getInfo()
    srtm = ee.Image("USGS/SRTMGL1_003")
    hydro = ee.Image("MERIT/Hydro/v1_0_1")
    features = ee.Image.cat(
        [
            srtm.select("elevation").rename("elevation_m"),
            ee.Terrain.slope(srtm).rename("slope_degrees"),
            hydro.select("hnd").rename("adjusted_elevation_m"),
            hydro.select("upa").add(1).log().rename("log1p_upstream_area_km2"),
        ]
    ).toFloat()
    settings = dict(
        source="GLOBAL_FLOOD_DB/MODIS_EVENTS/V1",
        country="Nigeria",
        country_boundary=country_source,
        quality="stored clear_perc fraction >=0.5 and clear_views >=3; jrc_perm_water ==0; all feature bands valid",
        sample_scale_m=250,
        projection="EPSG:32632",
        requested_per_class={"0": 1500, "1": 500},
        seed=42,
        target="Published event maximum water, outside GFD permanent-water mask; coarse satellite reference, not human truth",
        features=[
            "elevation_m",
            "slope_degrees",
            "adjusted_elevation_m",
            "log1p_upstream_area_km2",
        ],
        weighting="Class population / actual class sample count / total valid population; equal event-mixture weighting",
        limitations="Selected flood events only; no 72h onset labels, no verified event-free dates; not operational risk calibration",
        temporal="Retrospective static feature products and GFD permanent-water mask; historical issue-time availability not established",
    )
    config = output / "settings.json"
    if config.exists() and json.loads(config.read_text()) != settings:
        raise ValueError("Existing reference configuration differs")
    config.write_text(json.dumps(settings, indent=2) + "\n")
    all_rows = []
    events = []
    for i in range(n):
        image = ee.Image(objects.get(i))
        meta = image.toDictionary(
            ["id", "system:time_start", "system:time_end", "dfo_main_cause"]
        ).getInfo()
        event_id = str(meta["id"])
        csv_path = output / f"event_{event_id}.csv"
        meta_path = output / f"event_{event_id}.json"
        if csv_path.exists() and meta_path.exists():
            with csv_path.open() as f:
                rows = list(csv.DictReader(f))
            record = json.loads(meta_path.read_text())
            if (
                file_hash(csv_path) != record.get("sha256")
                or len(rows) != record["samples"]
            ):
                raise ValueError(f"Incomplete or changed samples for event {event_id}")
            all_rows.extend(rows)
            events.append(record)
            print("Reused event", event_id, len(rows), flush=True)
            continue
        region = country.intersection(image.geometry(), 1000)
        quality_range = (
            image.select("clear_perc")
            .reduceRegion(
                ee.Reducer.minMax(),
                region,
                scale=1000,
                crs="EPSG:32632",
                maxPixels=10000000,
                tileScale=8,
            )
            .getInfo()
        )
        if (
            quality_range.get("clear_perc_min", 0) < 0
            or quality_range.get("clear_perc_max", 0) > 1.00001
        ):
            raise ValueError(
                "Unexpected clear_perc units; expected measured source fraction [0,1]"
            )
        quality = (
            image.select("clear_perc")
            .gte(0.5)
            .And(image.select("clear_views").gte(3))
            .And(image.select("jrc_perm_water").eq(0))
        )
        label = image.select("flooded").rename("target").toInt()
        stack = (
            features.addBands(label)
            .updateMask(quality)
            .updateMask(features.mask().reduce(ee.Reducer.min()))
        )
        region = country.intersection(image.geometry(), 1000)
        population = (
            stack.select("target")
            .reduceRegion(
                ee.Reducer.frequencyHistogram(),
                region,
                scale=250,
                crs="EPSG:32632",
                maxPixels=100000000,
                tileScale=8,
            )
            .getInfo()
            .get("target", {})
        )
        population = {str(int(float(k))): float(v) for k, v in population.items()}
        if not population or population.get("1", 0) == 0:
            record = dict(
                metadata=meta,
                population=population,
                status="no eligible positive reference pixels; excluded",
            )
            meta_path.write_text(json.dumps(record, indent=2) + "\n")
            events.append(record)
            print("No eligible positive pixels", event_id, flush=True)
            continue
        samples = stack.stratifiedSample(
            numPoints=0,
            classBand="target",
            classValues=[0, 1],
            classPoints=[1500, 500],
            region=region,
            scale=250,
            projection="EPSG:32632",
            seed=42,
            dropNulls=True,
            geometries=True,
            tileScale=8,
        ).getInfo()
        rows = []
        for feature in samples["features"]:
            row = dict(feature["properties"])
            lon, lat = feature["geometry"]["coordinates"]
            row.update(longitude=lon, latitude=lat)
            row.update(
                event_id=event_id,
                event_start_utc=datetime.fromtimestamp(
                    meta["system:time_start"] / 1000, timezone.utc
                ).isoformat(),
                event_end_utc=datetime.fromtimestamp(
                    meta["system:time_end"] / 1000, timezone.utc
                ).isoformat(),
            )
            x, y = transform("EPSG:4326", "EPSG:32632", [lon], [lat])
            row["block_50km"] = f"{int(x[0] // 50000)}_{int(y[0] // 50000)}"
            row["target"] = int(row["target"])
            rows.append(row)
        observed = {
            str(label): sum(r["target"] == label for r in rows) for label in [0, 1]
        }
        total = sum(population.values())
        for row in rows:
            key = str(row["target"])
            row["sample_weight"] = population[key] / observed[key] / total
        with csv_path.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        record = dict(
            metadata=meta,
            observed_clear_fraction_range=quality_range,
            population=population,
            sample_counts=observed,
            samples=len(rows),
            sha256=file_hash(csv_path),
        )
        meta_path.write_text(json.dumps(record, indent=2) + "\n")
        events.append(record)
        all_rows.extend(rows)
        print("Acquired event", event_id, observed, flush=True)
    if all_rows:
        with (output / "samples.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            writer.writeheader()
            writer.writerows(all_rows)
    report = dict(
        settings=settings,
        event_inventory=events,
        events_with_samples=len({str(r["event_id"]) for r in all_rows}),
        samples=len(all_rows),
        class_counts={
            str(k): sum(int(r["target"]) == k for r in all_rows) for k in [0, 1]
        },
    )
    (output / "inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    report = acquire(a.project, a.output)
    print(report["samples"], "samples")


if __name__ == "__main__":
    main()
