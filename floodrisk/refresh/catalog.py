"""Daily Sentinel-1 inventory and monthly static-source release checks."""

from datetime import timedelta
import json
import os
from pathlib import Path
import requests
from floodrisk.refresh.state import atomic_json


def collect(kind, project, root, output, now):
    output.mkdir(parents=True)
    if kind == "sources":
        # The authoritative catalog tree exposes new collection/version identifiers.
        url = "https://api.github.com/repos/google/earthengine-catalog/git/trees/main"
        response = requests.get(url, params={"recursive": "1"}, timeout=90)
        response.raise_for_status()
        tree = response.json()
        if tree.get("truncated") or "tree" not in tree:
            raise ValueError("Incomplete Earth Engine catalog tree")
        prefixes = [
            "COPERNICUS_DEM",
            "MERIT_Hydro",
            "ESA_WorldCover",
            "GOOGLE_Research_open-buildings",
            "JRC_GSW",
        ]
        entries = [
            item
            for item in tree["tree"]
            if item["path"].endswith((".jsonnet", ".libsonnet"))
            and any(p in item["path"] for p in prefixes)
        ]
        if not entries:
            raise ValueError("No expected source catalog entries found")
        previous = root / "sources/latest.json"
        prior = {}
        if previous.exists():
            pointer = json.loads(previous.read_text())
            prior = {
                x["path"]: x["sha"]
                for x in json.loads(
                    (root / pointer["snapshot"] / "catalog.json").read_text()
                )["entries"]
            }
        atomic_json(
            output / "changes.json",
            {
                "new_or_changed": [
                    x["path"] for x in entries if prior.get(x["path"]) != x["sha"]
                ],
                "removed": sorted(set(prior) - {x["path"] for x in entries}),
                "initial_inventory": not bool(prior),
            },
        )
        atomic_json(
            output / "catalog.json",
            {"source": url, "revision": tree["sha"], "entries": entries},
        )
        atomic_json(
            output / "summary.json",
            {
                "issue_utc": now.isoformat(),
                "entries": len(entries),
                "policy": "Detect release identifiers and metadata changes. Adoption and model retraining require validation.",
            },
        )
        return
    import ee

    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        import google.auth

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/earthengine"]
        )
        ee.Initialize(credentials=credentials, project=project)
    else:
        ee.Initialize(project=project)
    ee.data.setDeadline(120000)
    observations = json.loads(
        Path("reports/hotspots_oct22/observations.json").read_text()
    )
    from floodrisk.acquire.earthengine import grid_settings

    items = []
    start = (now - timedelta(days=30)).isoformat()
    for obs in observations:
        settings = obs["settings"]
        _, bounds = grid_settings(settings)
        region = ee.Geometry.Rectangle(bounds, proj=settings["crs"], geodesic=False)
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, now.isoformat())
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        size = collection.size().getInfo()
        if size > 200:
            raise ValueError("Unexpectedly large scene inventory")
        images = collection.sort("system:time_start").toList(200)
        for i in range(size):
            image = ee.Image(images.get(i))
            if image.geometry().contains(region, maxError=10).getInfo():
                item = image.toDictionary(
                    [
                        "system:index",
                        "system:time_start",
                        "relativeOrbitNumber_start",
                        "orbitProperties_pass",
                    ]
                ).getInfo()
                items.append({"site": obs["site"], **item})
    atomic_json(output / "scenes.json", items)
    atomic_json(
        output / "summary.json",
        {
            "issue_utc": now.isoformat(),
            "window_start_utc": start,
            "sites": len(observations),
            "eligible_scenes": len(items),
            "scope": "Full-footprint dual-polarization scenes in the last 30 days; inventory for incremental processing",
        },
    )
