import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
from floodrisk.features.surface import surface_shares,augment


class SurfaceTests(unittest.TestCase):
    def test_class_shares_and_missing_values(self):
        cover=np.ma.array([50,80,10,255],mask=[0,0,0,1])
        occupancy=np.ma.array([0,1,1,255],mask=[0,0,0,1])
        features=surface_shares(cover,occupancy)
        self.assertAlmostEqual(features['cover_built_up_fraction'],1/3)
        self.assertAlmostEqual(sum(v for k,v in features.items() if k.startswith('cover_')),1)
        self.assertEqual(features['worldcover_valid_fraction'],0.75)
        self.assertAlmostEqual(features['building_occupancy_fraction'],2/3)
        features=surface_shares(np.ma.masked_all(4),np.ma.masked_all(4))
        self.assertIsNone(features['cover_built_up_fraction'])
        self.assertIsNone(features['building_occupancy_fraction'])
        with self.assertRaises(ValueError):surface_shares(np.array([123]),np.array([0]))

    def test_count_join_partial_cell_and_grid_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);raster=folder/'surface.tif';table=folder/'terrain.csv';counts=folder/'counts.json'
            raw=np.zeros((2,2,13),dtype='float32');raw[0]=50;raw[1,0,0]=1
            with rasterio.open(raster,'w',driver='GTiff',width=13,height=2,count=2,dtype='float32',
                crs='EPSG:32631',transform=from_origin(500000,700000,10,10),nodata=-9999) as dst:
                dst.write(raw);dst.set_band_description(1,'worldcover_code');dst.set_band_description(2,'building_occupancy_10m')
            def write_rows(second_x):
                with table.open('w') as f:
                    writer=csv.writer(f);writer.writerow(['cell_row','cell_col','center_x','center_y','pixels'])
                    writer.writerows([[0,0,500050,699990,20],[0,1,second_x,699990,6]])
            write_rows(500115);counts.write_text(json.dumps({'0_1':3}))
            report=augment(table,raster,counts,folder/'out.csv')
            self.assertEqual(report['mapped_building_centroid_count'],3)
            with (folder/'out.csv').open() as f:rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]['mapped_building_centroid_count'],'0')
            self.assertEqual(rows[1]['mapped_building_centroid_count'],'3')
            write_rows(500116)
            with self.assertRaisesRegex(ValueError,'grids differ'):augment(table,raster,counts,folder/'bad.csv')
            self.assertFalse((folder/'bad.csv').exists())
