import tempfile
from pathlib import Path
import unittest
from floodrisk.eval.reviewed_masks import read_review


class BlindReviewTest(unittest.TestCase):
    def test_unreviewed_or_wrong_id_cannot_enter_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "review.csv"
            manifest = {
                "chips": [
                    {
                        "chip_id": "reserved",
                        "source": "S1A_IW_GRDH_1SDV_20240811T000000",
                    }
                ]
            }
            path.write_text("chip_id,reviewer\nreserved,\n")
            with self.assertRaisesRegex(ValueError, "Incomplete independent review"):
                read_review(path, manifest)
            path.write_text("chip_id,reviewer\nsubstitute,A\n")
            with self.assertRaisesRegex(ValueError, "frozen chip IDs"):
                read_review(path, manifest)
