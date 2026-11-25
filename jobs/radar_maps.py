"""Process inventory scenes and checkpoint each complete map independently."""
import re
import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from floodrisk.acquire.earthengine import download
from floodrisk.deploy.tiled import file_hash, predict_scene

MODEL = Path('reports/m1-s1-kaggle-365653491/onnx_fp32/m1_s1_fp32.onnx')
CONFIG = MODEL.with_name('inference_config.json')


def build(scene, settings, folder, project):
    when = datetime.fromtimestamp(scene['system:time_start']/1000, timezone.utc)
    grid = dict(settings, start=str(when.date()), end=str((when+timedelta(days=1)).date()),
                source_index=scene['system:index'])
    grid.pop('selection_target', None)
    acquisition = download(project, folder/'input', grid)
    if acquisition['source']['system:index'] != scene['system:index']:
        raise ValueError('Downloaded scene differs from inventory')
    prediction = predict_scene(folder/'input/sentinel1_vv_vh_db.tif', MODEL, CONFIG,
                               folder/'prediction', halo=0)
    record = dict(site=scene['site'], observed_utc=when.isoformat(), source=scene,
                  model_sha256=prediction['model_sha256'], threshold=prediction['threshold'],
                  predicted_water_pixels=prediction['predicted_water_pixels'],
                  valid_pixels=prediction['valid_pixels'],
                  files={str(p.relative_to(folder)): file_hash(p) for p in folder.rglob('*') if p.is_file()})
    (folder/'map.json').write_text(json.dumps(record, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--project', required=True)
    parser.add_argument('--example', type=Path, help='Build only the newest Lagos scene locally')
    args = parser.parse_args()
    sites = {r['site']:r['settings'] for r in json.loads(Path('reports/hotspots_oct22/observations.json').read_text())}
    scenes = sorted(json.loads(args.inventory.read_text()), key=lambda s:s['system:time_start'], reverse=True)
    if args.example:
        build(next(s for s in scenes if s['site']=='Lagos'), sites['Lagos'], args.example, args.project)
        return
    completed = set(subprocess.check_output(['gh','api','--paginate','repos/{owner}/{repo}/releases?per_page=100',
        '--jq','.[] | select(.draft == false and any(.assets[]; .name == "map.zip")) | .tag_name'], text=True).splitlines())
    import shutil
    for scene in scenes:
        site = scene['site']
        tag = 'water-m1-v1-'+re.sub(r'[^a-z0-9]+','-',site.lower()).strip('-')+'-'+scene['system:index']
        if tag in completed:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'scene'
            build(scene, sites[site], folder, args.project)
            archive = shutil.make_archive(str(Path(tmp)/'map'), 'zip', folder)
            subprocess.run(['gh','release','create',tag,archive,'--prerelease','--title',site+' surface water',
                            '--notes','Experimental radar water segmentation. Acquisition time and provenance in map.json.'],check=True)


if __name__ == '__main__':
    main()
