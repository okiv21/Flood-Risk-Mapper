"""Verify that the published archives, LGA counts and experiment partitions agree."""

import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "reports"


class PublishedArtifactsTest(unittest.TestCase):
    def test_tiles_and_area_counts_match_each_observation(self):
        folder = ROOT / "hotspots_oct22"
        sites = {
            r["site"]: r for r in json.loads((folder / "observations.json").read_text())
        }
        tiles = json.loads((folder / "tiles/index.json").read_text())
        with (folder / "observed_pixels_by_lga.csv").open() as f:
            counts = list(csv.DictReader(f))
        self.assertEqual(len(sites), 8)
        self.assertEqual({r["site"] for r in tiles}, set(sites))
        for row in tiles:
            blob = (folder / "tiles" / row["file"]).read_bytes()
            self.assertEqual(blob[:8], b"PMTiles\x03")
            self.assertEqual(hashlib.sha256(blob).hexdigest(), row["sha256"])
            for key, total in sites[row["site"]]["counts"].items():
                self.assertEqual(
                    sum(
                        int(c[f"class_{key}_pixels"])
                        for c in counts
                        if c["site"] == row["site"]
                    ),
                    total,
                )

    def test_spatial_folds_and_temporal_events_are_disjoint(self):
        r = json.loads((ROOT / "nigeria_gfd/evaluation.json").read_text())
        partitions = r["partitions"]
        events = [eid for p in partitions.values() for eid in p["events"]]
        self.assertEqual(len(events), len(set(events)))
        blocks = [
            key for fold in r["spatial_validation"]["folds"] for key in fold["blocks"]
        ]
        self.assertEqual(len(blocks), len(set(blocks)))
        self.assertEqual(
            sum(f["rows"] for f in r["spatial_validation"]["folds"]),
            partitions["train"]["rows"],
        )
        for f in ["lokoja_2022.json", "lokoja_2022_expanded.json"]:
            self.assertEqual(
                json.loads((ROOT / "nigeria_gfd" / f).read_text())["model_sha256"],
                r["model_sha256"],
            )

    def test_archived_rainfall_covers_the_registry(self):
        with (ROOT / "hotspots_oct22/lgas.csv").open() as f:
            ids = {r["lga_id"] for r in csv.DictReader(f)}
        with (ROOT / "rainfall/national_20261123/rainfall.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(ids), 774)
        self.assertEqual(len(rows), len(ids))
        self.assertEqual({r["lga_id"] for r in rows}, ids)
        self.assertEqual(len({r["issue_utc"] for r in rows}), 1)
