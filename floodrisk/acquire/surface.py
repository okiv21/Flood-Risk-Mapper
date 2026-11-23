"""Acquire land cover and mapped building exposure for a small radar ROI."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import rasterio
from floodrisk.deploy.tiled import file_hash


CLASSES={10:'trees',20:'shrubland',30:'grassland',40:'cropland',50:'built_up',
         60:'bare_sparse',70:'snow_ice',80:'permanent_water',90:'wetland',95:'mangroves',100:'moss_lichen'}


def acquire(project,acquisition_path,output):
    import ee
    import requests
    a=json.loads(Path(acquisition_path).read_text());grid=dict(a['download_grid'])
    output=Path(output)
    if output.exists():raise ValueError('Output directory already exists')
    t=grid['crs_transform'];w,h=grid['dimensions']
    if t[1] or t[3] or t[0]!=10 or t[4]!=-10:raise ValueError('Expected north-up 10m grid')
    if w*h*8>16*1024*1024:raise ValueError('Small-area surface download limited to 16MiB')
    ee.Initialize(project=project)
    roi=ee.Geometry.Rectangle([t[2],t[5]-h*10,t[2]+w*10,t[5]],proj=grid['crs'],geodesic=False)
    cover_collection=ee.ImageCollection('ESA/WorldCover/v200').filterBounds(roi).sort('system:index')
    cover=cover_collection.select('Map').mosaic().rename('worldcover_code')
    source='GOOGLE/Research/open-buildings/v3/polygons'
    polygons=ee.FeatureCollection(source).filterBounds(roi).filter(ee.Filter.gte('confidence',0.75))
    def cell(feature):
        center=feature.geometry().centroid(1).transform(grid['crs'],1)
        xy=center.coordinates()
        col=ee.Number(xy.get(0)).subtract(t[2]).divide(100).floor()
        row=ee.Number(t[5]).subtract(ee.Number(xy.get(1))).divide(100).floor()
        return ee.Feature(center,{'cell_key':row.format('%d').cat('_').cat(col.format('%d'))})
    centres=polygons.map(cell).filterBounds(roi)
    counts=centres.aggregate_histogram('cell_key').getInfo()
    footprint=ee.Image.constant(0).byte().paint(polygons,1).rename('building_occupancy_10m')
    stack=ee.Image.cat([cover,footprint]).toFloat().unmask(-9999,sameFootprint=False)
    url=stack.getDownloadURL(dict(grid))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.surface-',dir=output.parent) as temp:
        stage=Path(temp)/'complete';stage.mkdir();download=stage/'download.tif'
        with requests.get(url,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            with download.open('wb') as f:
                for block in response.iter_content(1024*1024):f.write(block)
        with rasterio.open(download) as src:
            if src.count!=2 or src.shape!=(h,w) or src.crs!=rasterio.crs.CRS.from_string(grid['crs']) or not src.transform.almost_equals(rasterio.Affine(*t)):
                raise ValueError('Surface grid mismatch')
            raw=src.read();profile=src.profile.copy()
        if not np.isin(raw[0],[*CLASSES,-9999]).all():raise ValueError('Unexpected land-cover code')
        if not np.isin(raw[1],[0,1,-9999]).all():raise ValueError('Unexpected occupancy value')
        profile.update(dtype='float32',nodata=-9999,compress='deflate')
        target=stage/'surface_features.tif'
        with rasterio.open(target,'w',**profile) as dst:
            dst.write(raw)
            dst.set_band_description(1,'worldcover_code');dst.set_band_description(2,'building_occupancy_10m')
            dst.update_tags(attribution='ESA WorldCover / Google Research Open Buildings',confidence_filter='>=0.75')
        download.unlink()
        (stage/'building_centroid_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
        report=dict(sources=dict(worldcover='ESA/WorldCover/v200',worldcover_images=cover_collection.aggregate_array('system:index').getInfo(),buildings=source),
            grid=grid,sha256=file_hash(target),classes=CLASSES,
            building_confidence_min=0.75,building_detections_with_centroid_in_roi=int(sum(counts.values())),
            cells_with_building_centroids=len(counts),mapped_occupied_pixels=int((raw[1]==1).sum()),
            land_cover_pixels={str(code):int((raw[0]==code).sum()) for code in CLASSES},
            land_cover_nodata_pixels=int((raw[0]==-9999).sum()),
            count_rule='Each selected detection assigned by centroid to one 100m cell within ROI; not unique occupied households',
            occupancy='10m rasterization of selected intersecting polygons; occupancy is a mapping proxy, not exact footprint area',
            temporal='WorldCover 2021 and Open Buildings inferred May 2023; inventory availability at the radar acquisition date is unverified',
            uncertainty='Detection omissions/false positives and confidence filtering affect exposure; zero mapped buildings is not proof of no buildings',
            licenses='ESA WorldCover v200 CC-BY-4.0; Open Buildings v3 CC-BY-4.0; source attribution required',
            interpretation='Land-cover and exposure features only; no confirmed flood damage, population or risk score')
        (stage/'surface.json').write_text(json.dumps(report,indent=2)+'\n')
        if output.exists():raise ValueError('Output appeared during download')
        stage.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','acquisition','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(acquire(a.project,a.acquisition,a.output),indent=2))


if __name__=='__main__':main()
