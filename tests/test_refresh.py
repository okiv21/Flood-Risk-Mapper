import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from floodrisk.refresh.state import atomic_json, freshness, publish


class RefreshStateTests(unittest.TestCase):
    def test_failed_or_old_snapshot_cannot_replace_latest(self):
        now = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "forecast/first"
            atomic_json(
                first / "summary.json",
                {
                    "issue_utc": now.isoformat(),
                    "window_end_utc": (now + timedelta(hours=72)).isoformat(),
                },
            )
            entry = publish(root, "forecast", first, now)
            old = root / "forecast/old"
            atomic_json(
                old / "summary.json",
                {"issue_utc": (now - timedelta(hours=1)).isoformat()},
            )
            with self.assertRaises(ValueError):
                publish(root, "forecast", old, now)
            self.assertEqual(
                json.loads((root / "forecast/latest.json").read_text()), entry
            )
            self.assertEqual(freshness(root, "forecast", now)["status"], "fresh")
            self.assertEqual(
                freshness(root, "forecast", now + timedelta(hours=10))["status"],
                "stale",
            )

    def test_expired_window_and_future_timestamp_are_stale(self):
        now = datetime(2026, 11, 24, 12, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(freshness(root, "forecast", now)["status"], "missing")
            atomic_json(
                root / "forecast/latest.json",
                {"issue_utc": now.isoformat(), "window_end_utc": now.isoformat()},
            )
            self.assertEqual(freshness(root, "forecast", now)["status"], "stale")
            atomic_json(
                root / "forecast/latest.json",
                {"issue_utc": (now + timedelta(hours=1)).isoformat()},
            )
            self.assertEqual(freshness(root, "forecast", now)["status"], "stale")
