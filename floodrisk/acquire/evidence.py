"""Acquire an orbit-matched candidate pair for human flood-label review."""
import argparse
import json
from pathlib import Path

from floodrisk.acquire.earthengine import download


def acquire_pair(project,case_path,output):
    case=json.loads(Path(case_path).read_text());output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    common={k:case[k] for k in ['center_lon','center_lat','crs','width','height','scale_m']}
    reports={}
    for role in ['event_candidate','reference_candidate']:
        settings=dict(common,**case[role])
        if role=='reference_candidate':
            settings.update(relative_orbit=reports['event_candidate']['source']['relativeOrbitNumber_start'],
                            orbit_pass=reports['event_candidate']['source']['orbitProperties_pass'])
        dest=output/role
        if dest.exists():
            saved=json.loads((dest/'acquisition.json').read_text())
            if saved['settings']!=settings:raise ValueError('Existing acquisition settings differ')
            reports[role]=saved
        else:reports[role]=download(project,dest,settings)
        print(role,reports[role]['source']['system:index'],flush=True)
    event,reference=(reports[k] for k in ['event_candidate','reference_candidate'])
    if event['download_grid']!=reference['download_grid']:raise ValueError('Pair grids differ')
    for key in ['relativeOrbitNumber_start','orbitProperties_pass']:
        if event['source'][key]!=reference['source'][key]:raise ValueError('Pair orbits differ')
    summary=dict(case=case,acquisitions=reports,review_status='pending',flood_label=None,
                 limitation='Matched geometry/orbit reduces one acquisition confounder; seasonal, sensor and hydrological differences remain')
    (output/'pair.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['project','case','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();acquire_pair(a.project,a.case,a.output)


if __name__=='__main__':main()
