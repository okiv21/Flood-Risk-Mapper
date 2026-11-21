"""Aggregate aligned terrain rasters into nominal 100m feature cells, without labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from floodrisk.acquire.terrain import BANDS
from floodrisk.deploy.tiled import file_hash


def build_cells(raster, output):
    output=Path(output)
    if output.exists():raise ValueError('Output table already exists')
    rows=[]
    with rasterio.open(raster) as src:
        if list(src.descriptions)!=BANDS or src.crs is None or not src.crs.is_projected:
            raise ValueError('Expected named terrain bands and a projected CRS')
        t=src.transform
        if t.b or t.d or t.a!=10 or t.e!=-10:raise ValueError('Expected north-up 10m projected grid')
        for row in range(0,src.height,10):
            for col in range(0,src.width,10):
                h,w=min(10,src.height-row),min(10,src.width-col)
                block=src.read(window=Window(col,row,w,h),masked=True)
                x,y=t @ (col+w/2,row+h/2)
                record=dict(cell_row=row//10,cell_col=col//10,center_x=x,center_y=y,
                            width_m=w*10,height_m=h*10,pixels=w*h)
                for name,band in zip(BANDS,block):
                    values=band.compressed();values=values[np.isfinite(values)]
                    record[name+'_valid_fraction']=len(values)/(w*h)
                    record[name+'_mean']=float(values.mean()) if len(values) else None
                    record[name+'_max']=float(values.max()) if len(values) else None
                rows.append(record)
        info=dict(crs=str(src.crs),source_sha256=file_hash(raster),cells=len(rows),
            cell_size_m=100,edge_policy='Partial edge cells retain their actual widths and heights',
            fields=list(rows[0]),labels_present=False,
            interpretation='Feature cells only; 100m grid does not make repeated 90m source values independent samples')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raster',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(build_cells(a.raster,a.output),indent=2))


if __name__=='__main__':main()
