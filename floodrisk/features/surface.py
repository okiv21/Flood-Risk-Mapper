"""Join categorical land-cover shares and mapped buildings to terrain cells."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from floodrisk.acquire.surface import CLASSES


def surface_shares(cover,occupancy):
    cover=np.ma.asarray(cover);occupancy=np.ma.asarray(occupancy)
    lc=cover.compressed()
    if not np.isin(lc,list(CLASSES)).all():raise ValueError('Unexpected WorldCover category')
    occ=occupancy.compressed()
    if not np.isin(occ,[0,1]).all():raise ValueError('Invalid building occupancy')
    return dict(worldcover_valid_fraction=len(lc)/cover.size,
        **{f'cover_{name}_fraction':float((lc==code).sum()/len(lc)) if len(lc) else None for code,name in CLASSES.items()},
        building_occupancy_valid_fraction=len(occ)/occupancy.size,
        building_occupancy_fraction=float(occ.mean()) if len(occ) else None)


def augment(terrain_table,surface_raster,counts_path,output):
    counts=json.loads(Path(counts_path).read_text());output=Path(output)
    if output.exists():raise ValueError('Output table exists')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or int(v)!=v for v in counts.values()):
        raise ValueError('Invalid mapped building counts')
    with Path(terrain_table).open() as f:rows=list(csv.DictReader(f))
    seen=set()
    with rasterio.open(surface_raster) as src:
        if src.count!=2 or src.descriptions!=('worldcover_code','building_occupancy_10m'):
            raise ValueError('Expected named surface bands')
        if src.crs is None or not src.crs.is_projected or src.transform.a!=10 or src.transform.e!=-10 or src.transform.b or src.transform.d:
            raise ValueError('Expected projected north-up 10m grid')
        expected_cells=math.ceil(src.width/10)*math.ceil(src.height/10)
        for record in rows:
            row,col=int(record['cell_row']),int(record['cell_col']);key=f'{row}_{col}'
            if key in seen or row<0 or col<0 or row*10>=src.height or col*10>=src.width:
                raise ValueError('Invalid or duplicate terrain cell')
            seen.add(key)
            h,w=min(10,src.height-row*10),min(10,src.width-col*10)
            x,y=src.transform @ (col*10+w/2,row*10+h/2)
            if abs(float(record['center_x'])-x)>1e-6 or abs(float(record['center_y'])-y)>1e-6 or int(record['pixels'])!=w*h:
                raise ValueError('Terrain and surface cell grids differ')
            block=src.read(window=Window(col*10,row*10,w,h),masked=True)
            record.update(surface_shares(block[0],block[1]))
            record['mapped_building_centroid_count']=int(counts.get(key,0))
        if len(seen)!=expected_cells or set(counts)-seen:raise ValueError('Incomplete cells or unmatched building counts')
    if sum(r['mapped_building_centroid_count'] for r in rows)!=sum(counts.values()):raise ValueError('Building counts do not conserve')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return dict(cells=len(rows),fields=list(rows[0]),mapped_building_centroid_count=sum(counts.values()),
        meaning='Unlabelled engineering features; building count and occupancy are mapped-detection proxies')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['terrain','surface','counts','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(augment(a.terrain,a.surface,a.counts,a.output),indent=2))


if __name__=='__main__':main()
