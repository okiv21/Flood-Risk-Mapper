import unittest
from floodrisk.scoring.windows import features, outcome, RAINFALL


class EventWindowsTest(unittest.TestCase):
    def setUp(self):
        self.row = dict(
            issue_utc="2022-01-01T05:23:00Z",
            window_start_utc="2022-01-01T06:00:00Z",
            window_end_utc="2022-01-04T06:00:00Z",
            lga_id="example",
            susceptibility=0.1,
            built_up_fraction=0.25,
            forecast_source="fixture",
            antecedent_source="fixture",
            susceptibility_version="fixture",
        )
        self.row.update(
            {
                k: self.row["issue_utc"]
                for k in [
                    "antecedent_available_utc",
                    "forecast_available_utc",
                    "static_available_utc",
                ]
            }
        )
        self.row.update(dict.fromkeys(RAINFALL, 1))

    def test_future_input_is_rejected(self):
        self.assertEqual(features(self.row)["month"], 1)
        with self.assertRaises(ValueError):
            features({**self.row, "forecast_available_utc": "2022-01-01T05:24:00Z"})

    def test_detection_window_cannot_establish_onset(self):
        obs = {
            **self.row,
            "reviewer": "reviewer",
            "evidence_id": "fixture",
            "evidence_kind": "onset_interval",
            "evidence_start_utc": "2021-12-31T00:00:00Z",
            "evidence_end_utc": "2022-01-02T00:00:00Z",
        }
        self.assertIsNone(outcome(obs))
        self.assertEqual(
            outcome({**obs, "evidence_start_utc": obs["window_start_utc"]}), 1
        )
        self.assertIsNone(
            outcome(
                {
                    **obs,
                    "evidence_start_utc": obs["window_start_utc"],
                    "evidence_end_utc": obs["window_end_utc"],
                }
            )
        )

    def test_negative_needs_continuous_window_coverage(self):
        obs = {
            **self.row,
            "reviewer": "reviewer",
            "evidence_id": "fixture",
            "evidence_kind": "continuous_event_free",
            "evidence_start_utc": self.row["window_start_utc"],
            "evidence_end_utc": self.row["window_end_utc"],
        }
        self.assertEqual(outcome(obs), 0)
        self.assertIsNone(
            outcome({**obs, "evidence_start_utc": "2022-01-01T07:00:00Z"})
        )
        self.assertIsNone(outcome({**obs, "reviewer": ""}))
