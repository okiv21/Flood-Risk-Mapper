import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jobs import radar_maps


class RadarRefreshTests(unittest.TestCase):
    def test_failed_scene_retries_without_republishing_other_sites(self):
        scenes = [
            {"site": name, "system:index": "same", "system:time_start": 1000}
            for name in ["Lagos", "Abuja"]
        ]
        completed = set()
        attempts = []

        def build(scene, settings, folder, project):
            attempts.append(scene["site"])
            if attempts == ["Lagos"]:
                raise ValueError("Temporary acquisition failure")
            folder.mkdir()
            (folder / "map.json").write_text("{}")

        def release(command, check):
            self.assertTrue(Path(command[4]).is_file())
            completed.add(command[3])

        with tempfile.TemporaryDirectory() as tmp:
            inventory = Path(tmp) / "scenes.json"
            inventory.write_text(json.dumps(scenes))
            with (
                patch("sys.argv", ["radar_maps", str(inventory), "--project", "test"]),
                patch.object(radar_maps, "build", side_effect=build),
                patch.object(radar_maps.subprocess, "run", side_effect=release),
                patch.object(
                    radar_maps.subprocess,
                    "check_output",
                    side_effect=lambda *a, **kw: "\n".join(completed),
                ),
            ):
                with self.assertLogs(level="ERROR"), self.assertRaises(SystemExit):
                    radar_maps.main()
                radar_maps.main()
                radar_maps.main()
        self.assertEqual(attempts, ["Lagos", "Abuja", "Lagos"])
