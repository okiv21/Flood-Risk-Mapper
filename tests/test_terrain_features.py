"""Protect terrain units, valid zeros, negative elevations and no-data handling."""
import unittest
import numpy as np
from floodrisk.acquire.terrain import summarize


class TerrainTests(unittest.TestCase):
    def test_missing_does_not_become_zero_and_negative_elevation_is_valid(self):
        stack=np.array([[[-2,0],[-9999,4]],[[0,1],[2,3]],
                        [[-1,0],[1,2]],[[0,10],[100,1000]]],dtype='float32')
        stats=summarize(stack)
        self.assertEqual(stats['surface_elevation_m']['valid_pixels'],3)
        self.assertEqual(stats['surface_elevation_m']['median'],0)
        self.assertEqual(stats['surface_elevation_m']['min'],-2)
        self.assertEqual(stats['upstream_area_km2']['min'],0)

    def test_invalid_physical_ranges_and_empty_band_are_rejected(self):
        for band,value in [(1,-1),(1,91),(3,-1)]:
            stack=np.ones((4,2,2),dtype='float32');stack[band,0,0]=value
            with self.assertRaises(ValueError):summarize(stack)
        stack=np.ones((4,2,2),dtype='float32');stack[0]=np.nan
        with self.assertRaises(ValueError):summarize(stack)


class TerrainCellTests(unittest.TestCase):
    def test_cell_centres_partial_edges_and_missing_values(self):
        import csv,tempfile
        from pathlib import Path
        import rasterio
        from rasterio.transform import from_origin
        from floodrisk.features.terrain import build_cells
        from floodrisk.acquire.terrain import BANDS
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);raster=folder/'features.tif';output=folder/'cells.csv'
            raw=np.ones((4,12,13),dtype='float32');raw[0,:10,:10]=-9999
            with rasterio.open(raster,'w',driver='GTiff',width=13,height=12,count=4,
                dtype='float32',crs='EPSG:32631',transform=from_origin(500000,700000,10,10),nodata=-9999) as dst:
                dst.write(raw)
                for i,name in enumerate(BANDS,1):dst.set_band_description(i,name)
            schema=build_cells(raster,output)
            with output.open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(schema['cells'],4)
            self.assertEqual(float(rows[0]['center_x']),500050)
            self.assertEqual(float(rows[0]['center_y']),699950)
            self.assertEqual(rows[0]['surface_elevation_m_mean'],'')
            self.assertEqual(float(rows[0]['surface_elevation_m_valid_fraction']),0)
            self.assertEqual(float(rows[-1]['width_m']),30)
            self.assertEqual(float(rows[-1]['height_m']),20)
            self.assertEqual(sum(int(r['pixels']) for r in rows),156)
