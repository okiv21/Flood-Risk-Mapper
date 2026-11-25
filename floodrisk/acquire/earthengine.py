"""Download a small Lagos Sentinel-1 VV/VH dB scene from Earth Engine."""
import argparse
from datetime import date
import json
import tempfile
from pathlib import Path

from floodrisk.deploy.tiled import file_hash


DEFAULTS = dict(center_lon=3.45, center_lat=6.55, width=1024, height=1024,
                scale_m=10, crs='EPSG:32631', start='2022-08-01', end='2022-11-01')


def grid_settings(settings):
    """Construct a snapped 10m grid; restrict direct downloads to 8MiB of data."""
    from rasterio.warp import transform
    w,h=settings['width'],settings['height']
    if w < 1 or h < 1 or w*h*2*4 > 8*1024*1024:
        raise ValueError('Direct download grid must fit within 8 MiB for two float32 bands')
    if date.fromisoformat(settings['start']) >= date.fromisoformat(settings['end']):
        raise ValueError('Start must precede exclusive end date')
    x,y=transform('EPSG:4326',settings['crs'],[settings['center_lon']],[settings['center_lat']])
    scale=settings['scale_m']
    left=round((x[0]-w*scale/2)/scale)*scale
    top=round((y[0]+h*scale/2)/scale)*scale
    return dict(crs=settings['crs'],crs_transform=[scale,0,left,0,-scale,top],
                dimensions=[w,h],format='GEO_TIFF'), [left,top-h*scale,left+w*scale,top]


def download(project, output, settings=None):
    import ee
    import numpy as np
    import rasterio
    import requests

    settings=dict(DEFAULTS if settings is None else settings)
    params,bounds=grid_settings(settings)
    output=Path(output)
    if output.exists():raise ValueError('Output directory already exists')
    # Authentication is a separate interactive step; never initiate it silently.
    import os
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        import google.auth
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(credentials=credentials, project=project)
    else:
        ee.Initialize(project=project)
    ee.data.setDeadline(120_000)
    roi=ee.Geometry.Rectangle(bounds,proj=settings['crs'],geodesic=False)
    collection=(ee.ImageCollection('COPERNICUS/S1_GRD').filterBounds(roi)
        .filterDate(settings['start'],settings['end'])
        .filter(ee.Filter.eq('instrumentMode','IW'))
        .filter(ee.Filter.eq('resolution_meters',10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH'))
        .sort('system:time_start'))
    if settings.get('source_index'):
        collection=collection.filter(ee.Filter.eq('system:index',settings['source_index']))
    if settings.get('relative_orbit') is not None:
        collection=collection.filter(ee.Filter.eq('relativeOrbitNumber_start',settings['relative_orbit']))
    if settings.get('orbit_pass') is not None:
        collection=collection.filter(ee.Filter.eq('orbitProperties_pass',settings['orbit_pass']))
    if settings.get('selection_target'):
        target=ee.Date(settings['selection_target']).millis()
        collection=collection.map(lambda image: image.set('selection_distance_ms',
            ee.Number(image.get('system:time_start')).subtract(target).abs())).sort('selection_distance_ms')
    # First full-footprint candidate in the declared ordering, capped at 50.

    candidates=collection.limit(50).aggregate_array('system:index').getInfo()
    image=None
    for index in candidates:
        candidate=ee.Image('COPERNICUS/S1_GRD/'+index)
        if candidate.geometry().contains(roi,maxError=10).getInfo():
            image=candidate
            break
    if image is None:raise ValueError('No full-footprint scene among first 50 candidates; revise dates/region')
    metadata=image.toDictionary(['system:index','system:time_start','orbitProperties_pass',
        'relativeOrbitNumber_start','platform_number','instrumentMode']).getInfo()
    selected=image.select(['VV','VH']).toFloat().unmask(-9999,sameFootprint=False)
    use_pixels = bool(settings.get("source_index"))
    if not use_pixels:
        url=selected.getDownloadURL(dict(params))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.s1-download-',dir=output.parent) as temp:
        stage=Path(temp)/'complete';stage.mkdir()
        raw_path=stage/'download.tif'
        if use_pixels:
            scale_x, shear_x, left, shear_y, scale_y, top = params['crs_transform']
            grid = dict(dimensions=dict(width=settings['width'], height=settings['height']),
                        crsCode=settings['crs'], affineTransform=dict(scaleX=scale_x,
                        shearX=shear_x, translateX=left, shearY=shear_y, scaleY=scale_y, translateY=top))
            raw_path.write_bytes(ee.data.computePixels(dict(expression=selected,
                fileFormat='GEO_TIFF', grid=grid, bandIds=['VV','VH'])))
        else:
            with requests.get(url,stream=True,timeout=(30,180)) as response:
                response.raise_for_status()
                with raw_path.open('wb') as f:
                    for block in response.iter_content(1024*1024):f.write(block)
        destination=stage/'sentinel1_vv_vh_db.tif'
        with rasterio.open(raw_path) as src:
            if src.count!=2 or (src.width,src.height)!=(settings['width'],settings['height']):
                raise ValueError('Downloaded raster has unexpected dimensions/bands')
            expected=rasterio.Affine(*params['crs_transform'])
            if src.crs!=rasterio.crs.CRS.from_string(settings['crs']) or not src.transform.almost_equals(expected):
                raise ValueError('Downloaded raster grid differs from requested grid')
            raw=src.read().astype('float32')
            valid=np.isfinite(raw).all(axis=0)&(raw!=-9999).all(axis=0)
            if valid.mean()<0.99:raise ValueError('Less than 99% usable radar coverage; inspect source selection')
            profile=src.profile.copy();profile.update(driver='GTiff',dtype='float32',nodata=-9999,compress='deflate')
            raw[:,~valid]=-9999
            with rasterio.open(destination,'w',**profile) as dst:
                dst.write(raw);dst.set_band_description(1,'VV dB');dst.set_band_description(2,'VH dB')
                dst.update_tags(units='dB',source_collection='COPERNICUS/S1_GRD',source_index=metadata['system:index'])
        raw_path.unlink()
        report=dict(settings=settings,download_grid=params,source=metadata,
            source_collection='COPERNICUS/S1_GRD',units='dB',bands=['VV','VH'],
            selection=('Exact inventory source index with full-footprint check' if settings.get('source_index') else
                       'Closest full-footprint scene to declared target date among first 50 ordered candidates'
                       if settings.get('selection_target') else 'Earliest full-footprint IW dual-pol 10m scene among first 50 chronologically ordered candidates'),
            valid_fraction=float(valid.mean()),sha256=file_hash(destination),
            interpretation='Historical acquisition for engineering inspection; no confirmed flood-event label',
            status='Downloaded and grid/coverage checked; model predictions and accuracy not yet evaluated')
        (stage/'acquisition.json').write_text(json.dumps(report,indent=2)+'\n')
        if output.exists():raise ValueError('Output directory appeared during download')
        stage.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True,help='Earth Engine-enabled Cloud project ID')
    p.add_argument('--output',type=Path,default=Path('data/scenes/lagos_2022'))
    p.add_argument('--plan',action='store_true',help='Print the requested grid without contacting Earth Engine')
    a=p.parse_args()
    try:
        if a.plan:print(json.dumps(dict(settings=DEFAULTS,grid=grid_settings(DEFAULTS)[0]),indent=2))
        else:print(json.dumps(download(a.project,a.output),indent=2))
    except (ValueError,OSError) as exc:p.error(str(exc))


if __name__=='__main__':main()
