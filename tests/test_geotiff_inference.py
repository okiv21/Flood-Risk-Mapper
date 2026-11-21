"""Geospatial integration checks against the actual local ONNX export."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine

from floodrisk.deploy.predict import predict

PROJECT = Path(__file__).resolve().parents[1]
EXPORT = PROJECT / 'reports/m1-s1-kaggle-365653491/onnx_fp32'


@unittest.skipUnless((EXPORT / 'm1_s1_fp32.onnx').is_file(), 'Local ONNX export is required')
class GeoTIFFInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.transform = Affine(10, 2, 300000, 1, -10, 700000)
        self.input = self.folder / 'radar.tif'
        self.raw = np.full((3, 512, 512), -15, dtype='float32')
        self.raw[1, 0, 0] = np.nan
        self.raw[2, 0, 1] = -9999
        self.write(self.raw)

    def write(self, raw, crs='EPSG:32631'):
        with rasterio.open(self.input, 'w', driver='GTiff', width=512, height=512,
                           count=3, dtype='float32', crs=crs,
                           transform=self.transform, nodata=-9999) as dst:
            dst.write(raw)

    def run_prediction(self, output='prediction'):
        return predict(self.input, EXPORT / 'm1_s1_fp32.onnx', EXPORT / 'inference_config.json',
                       self.folder / output, vv_band=2, vh_band=3)

    def test_georeferencing_nodata_and_score_mask_agreement(self):
        report = self.run_prediction()
        self.assertEqual(report['nodata_pixels'], 2)
        self.assertEqual(report['vv_band'], 2)
        with rasterio.open(self.input) as src, \
                rasterio.open(self.folder / 'prediction/water_mask.tif') as mask, \
                rasterio.open(self.folder / 'prediction/water_score.tif') as score:
            self.assertEqual(mask.crs, src.crs)
            self.assertEqual(mask.transform, src.transform)
            self.assertEqual(mask.bounds, src.bounds)
            self.assertEqual(score.transform, src.transform)
            self.assertEqual(score.crs, src.crs)
            self.assertEqual(mask.dtypes, ('uint8',))
            self.assertEqual(score.dtypes, ('float32',))
            self.assertEqual(mask.nodata, 255)
            self.assertEqual(score.nodata, -9999)
            m, s = mask.read(1), score.read(1)
            valid = m != 255
            self.assertTrue(np.array_equal(valid, score.read_masks(1) > 0))
            self.assertTrue(np.array_equal(m[valid], (s[valid] > report['threshold']).astype('uint8')))
            self.assertTrue(((s[valid] >= 0) & (s[valid] <= 1)).all())
            self.assertEqual(m[0, 0], 255)
            self.assertEqual(m[0, 1], 255)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.run_prediction()

    def test_reject_empty_input_and_missing_crs(self):
        self.write(np.full_like(self.raw, -9999))
        with self.assertRaisesRegex(ValueError, 'no valid radar'):
            self.run_prediction()
        self.assertFalse((self.folder / 'prediction').exists())
        self.write(self.raw, crs=None)
        with self.assertRaisesRegex(ValueError, 'coordinate reference'):
            self.run_prediction()

    def test_reject_mismatched_model_manifest(self):
        config = json.loads((EXPORT / 'inference_config.json').read_text())
        config['onnx_sha256'] = 'wrong'
        path = self.folder / 'bad_config.json'
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'hash'):
            predict(self.input, EXPORT / 'm1_s1_fp32.onnx', path, self.folder / 'prediction')
        self.assertFalse((self.folder / 'prediction').exists())


if __name__ == '__main__':
    unittest.main()
