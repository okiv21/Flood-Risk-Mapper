"""Stream a Sentinel-1 dB scene through the fixed-size ONNX model."""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from floodrisk.deploy.predict import preprocess


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tile_windows(width, height, halo=64):
    """Yield unique output cores plus 512px input windows with context margins."""
    if not isinstance(halo, int) or not 0 <= halo < 256:
        raise ValueError('halo must be an integer from 0 to 255')
    core = 512 - 2 * halo
    for row in range(0, height, core):
        for col in range(0, width, core):
            yield (Window(col, row, min(core, width-col), min(core, height-row)),
                   Window(col-halo, row-halo, 512, 512))


def predict_scene(input_path, model_path, config_path, output_dir, halo=64,
                  vv_band=1, vh_band=2, threads=4):
    import onnxruntime as ort

    input_path, model_path, config_path, output_dir = map(Path, [input_path, model_path, config_path, output_dir])
    if output_dir.exists():
        raise ValueError('Output directory already exists')
    if threads < 1 or vv_band < 1 or vh_band < 1 or vv_band == vh_band:
        raise ValueError('Require positive threads and distinct positive band indices')
    if not isinstance(halo, int) or not 0 <= halo < 256:
        raise ValueError('halo must be an integer from 0 to 255')
    config = json.loads(config_path.read_text())
    model_hash = file_hash(model_path)
    if model_hash != config['onnx_sha256']:
        raise ValueError('Model hash does not match the inference manifest')
    if config['input_shape'] != [1, 2, 512, 512] or config['output_activation'] != 'sigmoid':
        raise ValueError('Unsupported model contract')
    threshold = float(config['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid threshold')
    options = ort.SessionOptions()
    options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError('Input must have a coordinate reference system')
        if max(vv_band, vh_band) > src.count:
            raise ValueError('Requested band is absent')
        profile = dict(driver='GTiff', width=src.width, height=src.height, count=1,
                       crs=src.crs, transform=src.transform, compress='deflate',
                       tiled=True, blockxsize=256, blockysize=256, BIGTIFF='IF_SAFER')
        counts = dict(tiles=0, inferred_tiles=0, skipped_empty_tiles=0,
                      valid_pixels=0, nodata_pixels=0, predicted_water_pixels=0)
        # Temporary outputs are removed on failure; only complete runs are published.
        with tempfile.TemporaryDirectory(prefix='.flood-tiles-', dir=output_dir.parent) as temp:
            stage = Path(temp) / 'complete'
            stage.mkdir()
            with rasterio.open(stage/'water_mask.tif', 'w', dtype='uint8', nodata=255, **profile) as masks, \
                    rasterio.open(stage/'water_score.tif', 'w', dtype='float32', nodata=-9999, **profile) as scores:
                for core, context in tile_windows(src.width, src.height, halo):
                    # Read only the real source intersection and pad in array space.
                    # This also preserves exact pixel placement for rotated grids.
                    raw = np.full((2,512,512),np.nan,dtype='float32')
                    valid = np.zeros((512,512),dtype=bool)
                    left, top = max(0,int(context.col_off)), max(0,int(context.row_off))
                    right = min(src.width,int(context.col_off)+512)
                    bottom = min(src.height,int(context.row_off)+512)
                    actual = Window(left,top,right-left,bottom-top)
                    data = src.read([vv_band,vh_band],window=actual,out_dtype='float32')
                    usable_input = (src.read_masks([vv_band,vh_band],window=actual)>0).all(axis=0)
                    usable_input &= np.isfinite(data).all(axis=0)
                    y,x = top-int(context.row_off),left-int(context.col_off)
                    raw[:,y:y+data.shape[1],x:x+data.shape[2]] = data
                    valid[y:y+data.shape[1],x:x+data.shape[2]] = usable_input
                    h, w = int(core.height), int(core.width)
                    crop = (slice(halo, halo+h), slice(halo, halo+w))
                    usable = valid[crop]
                    mask = np.full((h,w),255,dtype='uint8')
                    score = np.full((h,w),-9999,dtype='float32')
                    if usable.any():
                        x = preprocess(raw, valid, config['preprocessing'])
                        logits = session.run([config['output_name']], {config['input_name']: x})[0]
                        if logits.shape != (1,1,512,512) or not np.isfinite(logits).all():
                            raise ValueError('Invalid ONNX output')
                        probability = 1/(1+np.exp(-np.clip(logits[0,0][crop],-80,80)))
                        score[usable] = probability[usable]
                        mask[usable] = (probability[usable] > threshold).astype('uint8')
                        counts['inferred_tiles'] += 1
                    else:
                        counts['skipped_empty_tiles'] += 1
                    masks.write(mask,1,window=core)
                    scores.write(score,1,window=core)
                    counts['tiles'] += 1
                    counts['valid_pixels'] += int(usable.sum())
                    counts['nodata_pixels'] += int((~usable).sum())
                    counts['predicted_water_pixels'] += int((mask==1).sum())
                for dst, description in [(masks,'0 dry; 1 water; 255 no-data'),
                                          (scores,'sigmoid water score; -9999 no-data')]:
                    dst.set_band_description(1,description)
                    dst.update_tags(model_sha256=model_hash, threshold=str(threshold), halo=str(halo),
                                    product='experimental_water_segmentation', input_units='dB')
            assert counts['valid_pixels'] + counts['nodata_pixels'] == src.width * src.height
            report = dict(input=str(input_path.resolve()), input_sha256=file_hash(input_path),
                          model_sha256=model_hash, config_sha256=file_hash(config_path),
                          width=src.width, height=src.height, crs=src.crs.to_string(),
                          transform=list(src.transform), threshold=threshold, halo=halo,
                          core_size=512-2*halo, vv_band=vv_band, vh_band=vh_band,
                          padding='normalised mean outside image and at invalid inputs',
                          assembly='Each output core written exactly once; context margins discarded; no blending',
                          scope='Water segmentation; dB units assumed; permanent water not removed', **counts)
            (stage/'prediction.json').write_text(json.dumps(report,indent=2)+'\n')
            if output_dir.exists():
                raise ValueError('Output directory appeared during prediction')
            stage.rename(output_dir)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['input','model','config','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--halo',type=int,default=64,help='Context margin in pixels; default 64 gives 384px output cores')
    p.add_argument('--vv-band',type=int,default=1)
    p.add_argument('--vh-band',type=int,default=2)
    p.add_argument('--threads',type=int,default=4)
    a=p.parse_args()
    try:
        print(json.dumps(predict_scene(a.input,a.model,a.config,a.output,a.halo,a.vv_band,a.vh_band,a.threads),indent=2))
    except (ValueError,KeyError,OSError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
