"""Acquire retrospective CHIRPS context and an independent current forecast snapshot."""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile

from floodrisk.features.rainfall import antecedent_features,forecast_features


def acquire(project, acquisition_path, output):
    import ee
    import requests

    acquisition=json.loads(Path(acquisition_path).read_text())
    settings=acquisition['settings']
    output=Path(output)
    if output.exists():raise ValueError('Output directory already exists')
    instant=datetime.fromtimestamp(acquisition['source']['system:time_start']/1000,timezone.utc)
    end=instant.date();start=end-timedelta(days=14)
    grid=acquisition['download_grid'];a,b,left,d,e,top=grid['crs_transform']
    width,height=grid['dimensions']
    if b or d:raise ValueError('This initial rainfall ROI supports north-up acquisition grids')
    ee.Initialize(project=project)
    region=ee.Geometry.Rectangle([left,top+e*height,left+a*width,top],proj=grid['crs'],geodesic=False)
    collection=ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start.isoformat(),end.isoformat()).sort('system:time_start')
    def daily_record(obj):
        image=ee.Image(obj)
        mean=image.select('precipitation').reduceRegion(reducer=ee.Reducer.mean(),geometry=region,
                crs=image.select('precipitation').projection(),maxPixels=100000)
        return ee.Feature(None,{'date':image.date().format('YYYY-MM-dd'),'mm':mean.get('precipitation')})
    response=ee.FeatureCollection(collection.toList(14).map(daily_record)).getInfo()
    historical=[f['properties'] for f in response['features']]
    features=antecedent_features(historical,end.isoformat())
    params=dict(latitude=settings['center_lat'],longitude=settings['center_lon'],hourly='precipitation',
                forecast_days=5,timezone='UTC',precipitation_unit='mm')
    requested=datetime.now(timezone.utc)
    r=requests.get('https://api.open-meteo.com/v1/forecast',params=params,timeout=(30,120))
    r.raise_for_status();payload=r.json();retrieved=datetime.now(timezone.utc)
    forecast=forecast_features(payload,retrieved)
    report=dict(historical=dict(source='UCSB-CHG/CHIRPS/DAILY',date_start=start.isoformat(),date_end_exclusive=end.isoformat(),
                    acquisition_time_utc=instant.isoformat(),spatial_aggregation='ROI mean over available native CHIRPS cells; not a rain gauge or LGA mean',
                    native_grid_degrees=0.05,features=features,
                    availability='Final retrospective CHIRPS; does not establish which data were available at the historical issue time'),
                forecast=dict(source='Open-Meteo Forecast API',requested_at_utc=requested.isoformat(),
                    request=params,returned_coordinates={k:payload[k] for k in ['latitude','longitude']},
                    model_run_issue_time=None,**{k:v for k,v in forecast.items() if k!='periods'}),
                interpretation='Two different dates and spatial supports; current forecast must not be joined to the August 2022 target; no flood probability produced')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rainfall-',dir=output.parent) as temp:
        stage=Path(temp)/'complete';stage.mkdir()
        (stage/'historical_raw.json').write_text(json.dumps(response,indent=2)+'\n')
        (stage/'forecast_raw.json').write_text(json.dumps(payload,indent=2)+'\n')
        (stage/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        for name,rows,fields in [('historical_daily.csv',historical,['date','mm']),
                                  ('forecast_hourly.csv',forecast['periods'],['period_end_utc','mm'])]:
            with (stage/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        if output.exists():raise ValueError('Output directory appeared during acquisition')
        stage.rename(output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True);p.add_argument('--acquisition',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(acquire(a.project,a.acquisition,a.output),indent=2))


if __name__=='__main__':main()
