"""Download terrain/drainage features aligned to the small acquired radar scene."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from floodrisk.deploy.tiled import file_hash


BANDS=['surface_elevation_m','slope_degrees','hydro_adjusted_elevation_m','upstream_area_km2']


def summarize(stack):
    if stack.ndim!=3 or stack.shape[0]!=4:raise ValueError('Expected four terrain bands')
    result={}
    for name,array in zip(BANDS,stack):
        valid=np.isfinite(array)&(array!=-9999)
        if name in ['slope_degrees','upstream_area_km2'] and (array[valid]<0).any():
            raise ValueError('Negative slope or drainage area')
        if name=='slope_degrees' and (array[valid]>90).any():raise ValueError('Slope exceeds 90 degrees')
        if not valid.any():raise ValueError(f'No valid pixels for {name}')
        result[name]=dict(valid_pixels=int(valid.sum()),missing_pixels=int((~valid).sum()),
            min=float(array[valid].min()),median=float(np.median(array[valid])),max=float(array[valid].max()))
    return result


def acquire(project,acquisition_path,output):
    import ee
    import requests
    a=json.loads(Path(acquisition_path).read_text());grid=dict(a['download_grid'])
    output=Path(output)
    if output.exists():raise ValueError('Output directory already exists')
    if grid['dimensions'][0]*grid['dimensions'][1]*4*4>24*1024*1024:
        raise ValueError('Small-area direct terrain download must be <=24MiB')
    ee.Initialize(project=project)
    t=grid['crs_transform'];width,height=grid['dimensions']
    roi=ee.Geometry.Rectangle([t[2],t[5]+t[4]*height,t[2]+t[0]*width,t[5]],proj=grid['crs'],geodesic=False)
    dem_source='COPERNICUS/DEM/GLO30_2024_1'
    tiles=ee.ImageCollection(dem_source).filterBounds(roi.buffer(1000)).sort('system:index')
    # Calculate slope at 30m in a metre-based projection, not on replicated 10m pixels.
    dem=tiles.select('DEM').mosaic().setDefaultProjection(crs=grid['crs'],scale=30).reproject(crs=grid['crs'],scale=30)
    hydro=ee.Image('MERIT/Hydro/v1_0_1')
    image=ee.Image.cat([dem.rename(BANDS[0]),ee.Terrain.slope(dem).rename(BANDS[1]),
        hydro.select('hnd').rename(BANDS[2]),hydro.select('upa').rename(BANDS[3])]).toFloat()
    url=image.unmask(-9999,sameFootprint=False).getDownloadURL(dict(grid))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.terrain-',dir=output.parent) as temp:
        stage=Path(temp)/'complete';stage.mkdir()
        raw_path=stage/'download.tif'
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with raw_path.open('wb') as f:
                for chunk in response.iter_content(1024*1024):f.write(chunk)
        with rasterio.open(raw_path) as src:
            if src.count!=4 or src.shape!=(height,width) or src.crs!=rasterio.crs.CRS.from_string(grid['crs']) or not src.transform.almost_equals(rasterio.Affine(*t)):
                raise ValueError('Downloaded terrain grid differs from requested grid')
            stack=src.read().astype('float32');profile=src.profile.copy()
        stats=summarize(stack)
        stack[~np.isfinite(stack)]=-9999
        profile.update(driver='GTiff',dtype='float32',nodata=-9999,compress='deflate')
        destination=stage/'terrain_features.tif'
        with rasterio.open(destination,'w',**profile) as dst:
            dst.write(stack)
            for i,name in enumerate(BANDS,1):dst.set_band_description(i,name)
            dst.update_tags(dem_source=dem_source,hydro_source='MERIT/Hydro/v1_0_1',
                            warning='hnd is adjusted elevation; not assumed to be relative height above drainage')
        raw_path.unlink()
        report=dict(bands=BANDS,statistics=stats,grid=grid,sha256=file_hash(destination),
            sources=dict(dem=dem_source,dem_tiles=tiles.aggregate_array('system:index').getInfo(),hydro='MERIT/Hydro/v1_0_1'),
            resolutions=dict(dem_nominal_m=30,slope_computation_m=30,hydro_arcseconds=3,aligned_grid_m=10),
            method='30m projected DEM/slope; default nearest-neighbour alignment to 10m; no new source detail',
            semantics='DSM includes buildings/vegetation; hnd retained as hydrologically adjusted elevation, not relative HAND; upa is upstream area, not river discharge',
            temporal='2024_1 processed DEM edition used for feature engineering; as-of availability for 2022 not established',
            licenses=dict(merit='CC-BY-NC 4.0 selected for noncommercial research; source redistribution restrictions apply',
                copernicus='Copernicus DEM licence and attribution required'),
            interpretation='Feature acquisition and inspection only; no susceptibility model or risk score trained')
        (stage/'terrain.json').write_text(json.dumps(report,indent=2)+'\n')
        if output.exists():raise ValueError('Output directory appeared during acquisition')
        stage.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','acquisition','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(acquire(a.project,a.acquisition,a.output),indent=2))


if __name__=='__main__':main()
