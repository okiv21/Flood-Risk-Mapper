"""Offline checks for the real-scene download request."""
import unittest
from floodrisk.acquire.earthengine import DEFAULTS, grid_settings


class AcquisitionGridTests(unittest.TestCase):
    def test_projected_grid_is_ten_meter_and_lagos_centered(self):
        from rasterio.warp import transform
        grid,bounds=grid_settings(DEFAULTS)
        self.assertEqual(grid['dimensions'],[1024,1024])
        self.assertEqual(grid['crs'],'EPSG:32631')
        self.assertEqual(grid['crs_transform'][0],10)
        self.assertEqual(grid['crs_transform'][4],-10)
        self.assertEqual(bounds[2]-bounds[0],10240)
        self.assertEqual(bounds[3]-bounds[1],10240)
        x,y=transform('EPSG:4326',grid['crs'],[3.45],[6.55])
        self.assertLessEqual(abs((bounds[0]+bounds[2])/2-x[0]),5)
        self.assertLessEqual(abs((bounds[1]+bounds[3])/2-y[0]),5)

    def test_download_limit_and_date_order(self):
        with self.assertRaises(ValueError):grid_settings(dict(DEFAULTS,width=10000))
        with self.assertRaises(ValueError):grid_settings(dict(DEFAULTS,end='2022-01-01'))
