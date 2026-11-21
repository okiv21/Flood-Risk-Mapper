"""Align historical JRC water occurrence to an acquired scene and compare a water mask."""
import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio

from floodrisk.deploy.tiled import file_hash


def compare(project, acquisition_path, mask_path, output):
    import ee
    import requests

    acquisition_path,mask_path,output=map(Path,[acquisition_path,mask_path,output])
    if output.exists():raise ValueError('Output directory already exists')
    acquisition=json.loads(acquisition_path.read_text())
    grid=dict(acquisition['download_grid'])
    ee.Initialize(project=project)
    # Masked occurrence includes locations with no detected water history.
    # Preserve that distinction rather than silently treating it as known dry land.
    image=ee.Image('JRC/GSW1_4/GlobalSurfaceWater').select('occurrence').unmask(255,sameFootprint=False).toByte()
    url=image.getDownloadURL(dict(grid))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.water-reference-',dir=output.parent) as temp:
        stage=Path(temp)/'complete';stage.mkdir()
        downloaded=stage/'download.tif'
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with downloaded.open('wb') as f:
                for block in response.iter_content(1024*1024):f.write(block)
        with rasterio.open(downloaded) as ref,rasterio.open(mask_path) as mask:
            if ref.crs!=mask.crs or ref.shape!=mask.shape or not ref.transform.almost_equals(mask.transform):
                raise ValueError('Reference and prediction grids differ')
            occurrence=ref.read(1)
            predicted=mask.read(1)
            if not np.isin(occurrence,list(range(101))+[255]).all():raise ValueError('Invalid occurrence values')
            if not np.isin(predicted,[0,1,255]).all():raise ValueError('Invalid water mask')
            profile=mask.profile.copy();profile.update(dtype='uint8',nodata=255,compress='deflate')
        classes=np.full(predicted.shape,255,dtype='uint8')
        classes[predicted==0]=0
        classes[(predicted==1)&(occurrence>=90)&(occurrence<=100)]=1
        classes[(predicted==1)&(occurrence<90)]=2
        classes[(predicted==1)&(occurrence==255)]=3
        for name,array,description in [
            ('historical_occurrence.tif',occurrence,'JRC 1984-2021 water occurrence percent; 255 unavailable/masked'),
            ('water_context.tif',classes,'0 predicted dry; 1 water within >=90% reference; 2 outside reference; 3 reference unavailable; 255 radar no-data')]:
            with rasterio.open(stage/name,'w',**profile) as dst:
                dst.write(array,1);dst.set_band_description(1,description)
                dst.update_tags(attribution='Source: EC JRC/Google',reference_native_resolution_m='30',
                                rule='historical occurrence >=90 percent; exploratory high-frequency-water rule')
        downloaded.unlink()
        report=dict(source='JRC/GSW1_4/GlobalSurfaceWater',band='occurrence',period='1984-2021',
                    attribution='Source: EC JRC/Google',native_resolution_m=30,aligned_grid=grid,
                    resampling='Earth Engine default nearest neighbour; 10m alignment does not add source detail',
                    reference_threshold_percent=90,mask_sha256=file_hash(mask_path),
                    reference_sha256=file_hash(stage/'historical_occurrence.tif'),
                    counts={str(i):int((classes==i).sum()) for i in [0,1,2,3,255]},
                    meaning='Exploratory high-frequency-water exclusion; outside-reference pixels are not confirmed flooding',
                    unavailable='255 includes masked locations where JRC never detected water; absence is not proof of adequate observations')
        (stage/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
        if output.exists():raise ValueError('Output directory appeared during download')
        stage.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','acquisition','mask','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args()
    print(json.dumps(compare(a.project,a.acquisition,a.mask,a.output),indent=2))


if __name__=='__main__':main()
