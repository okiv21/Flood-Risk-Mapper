"""Check tile coverage and placement independently of model quality."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from affine import Affine

from floodrisk.deploy.predict import predict, preprocess
from floodrisk.deploy.tiled import predict_scene, tile_windows
from test_geotiff_inference import EXPORT


class WindowTests(unittest.TestCase):
    def test_coverage_partial_edges_and_small_scenes(self):
        for width,height in [(17,23),(512,512),(800,777),(1024,1024)]:
            for halo in [0,64]:
                coverage=np.zeros((height,width),dtype='uint8')
                for core,context in tile_windows(width,height,halo):
                    row,col=int(core.row_off),int(core.col_off)
                    coverage[row:row+int(core.height),col:col+int(core.width)] += 1
                    self.assertEqual(context.col_off+halo,core.col_off)
                    self.assertEqual(context.row_off+halo,core.row_off)
                self.assertTrue((coverage==1).all())
        with self.assertRaises(ValueError):list(tile_windows(10,10,256))


@unittest.skipUnless((EXPORT/'m1_s1_fp32.onnx').is_file(),'Local ONNX export required')
class TiledIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.transform=Affine(10,2,300000,1,-10,700000)
        self.config=EXPORT/'inference_config.json'
        self.model=EXPORT/'m1_s1_fp32.onnx'

    def write(self,raw):
        p=self.folder/'input.tif'
        with rasterio.open(p,'w',driver='GTiff',width=raw.shape[2],height=raw.shape[1],
                           count=2,dtype='float32',crs='EPSG:32631',transform=self.transform,nodata=-9999) as dst:
            dst.write(raw)
        return p

    def test_context_cropping_and_empty_core_with_coordinate_echo(self):
        # A pointwise echo makes any shift, lost pixel, or padding leak detectable.
        raw=np.zeros((2,777,800),dtype='float32')
        raw[0]=-20+np.arange(800)[None,:]*0.01
        raw[1]=-17
        raw[:,:384,:384]=-9999
        raw[:,500,500]=np.nan
        source=self.write(raw)
        class Echo:
            def __init__(self,*a,**kw):pass
            def run(self,names,feeds):return [feeds['sar'][:,:1]]
        with patch('onnxruntime.InferenceSession',Echo):
            report=predict_scene(source,self.model,self.config,self.folder/'tiled',halo=64)
        valid=np.isfinite(raw).all(axis=0)&(raw!=-9999).all(axis=0)
        import json
        settings=json.loads(self.config.read_text())['preprocessing']
        expected=1/(1+np.exp(-preprocess(raw,valid,settings)[0,0]))
        with rasterio.open(self.folder/'tiled/water_score.tif') as dst:
            score=dst.read(1)
            self.assertEqual(dst.transform,self.transform)
            self.assertEqual(dst.crs.to_epsg(),32631)
            self.assertEqual(dst.shape,(777,800))
            np.testing.assert_allclose(score[valid],expected[valid],atol=1e-7)
            self.assertTrue((score[~valid]==-9999).all())
        self.assertEqual(report['skipped_empty_tiles'],1)
        self.assertEqual(report['valid_pixels'],int(valid.sum()))
        with self.assertRaises(ValueError):predict_scene(source,self.model,self.config,self.folder/'tiled')

    def test_zero_halo_matches_real_single_chip_path(self):
        raw=np.full((2,512,512),-15,dtype='float32');raw[:,0,0]=np.nan
        source=self.write(raw)
        predict(source,self.model,self.config,self.folder/'single')
        predict_scene(source,self.model,self.config,self.folder/'tiled',halo=0)
        for name in ['water_mask.tif','water_score.tif']:
            with rasterio.open(self.folder/'single'/name) as a,rasterio.open(self.folder/'tiled'/name) as b:
                np.testing.assert_array_equal(a.read(),b.read())
                self.assertEqual(a.bounds,b.bounds)
