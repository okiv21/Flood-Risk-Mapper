"""Run the fixed-shape ONNX model on one georeferenced Sentinel-1 dB chip."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio


def preprocess(raw, valid, settings):
    """Use manifest normalisation; fill unusable input pixels with the mean."""
    x = raw.astype('float32').copy()
    for i, key in enumerate(['vv_clip', 'vh_clip']):
        x[i] = np.clip(x[i], *settings[key])
    mean = np.asarray(settings['mean'], dtype='float32')[:, None, None]
    std = np.asarray(settings['std'], dtype='float32')[:, None, None]
    if mean.shape != (2, 1, 1) or std.shape != (2, 1, 1) or not (std > 0).all():
        raise ValueError('Invalid two-band normalisation settings')
    x = (x - mean) / std
    x[:, ~valid] = settings['nan_fill_normalized']
    return np.ascontiguousarray(x[None])


def predict(input_path, model_path, config_path, output_dir, vv_band=1, vh_band=2, threads=4):
    import onnxruntime as ort

    input_path, model_path, config_path, output_dir = map(Path, [input_path, model_path, config_path, output_dir])
    if output_dir.exists():
        raise ValueError('Output directory already exists; choose a new directory')
    if threads < 1 or vv_band < 1 or vh_band < 1 or vv_band == vh_band:
        raise ValueError('Require positive threads and distinct positive VV/VH band indices')
    config = json.loads(config_path.read_text())
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if model_hash != config['onnx_sha256']:
        raise ValueError('Model hash does not match the inference manifest')
    if config['input_shape'] != [1, 2, 512, 512] or config['output_activation'] != 'sigmoid':
        raise ValueError('Unsupported model contract')
    threshold = float(config['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('Invalid manifest threshold')
    with rasterio.open(input_path) as src:
        if src.crs is None:
            raise ValueError('Input must have a coordinate reference system')
        if (src.height, src.width) != (512, 512):
            raise ValueError('This command requires a 512 x 512 chip; use floodrisk.deploy.tiled for larger scenes')
        if max(vv_band, vh_band) > src.count:
            raise ValueError('Requested band is absent from input')
        raw = src.read([vv_band, vh_band]).astype('float32')
        valid = (src.read_masks([vv_band, vh_band]) > 0).all(axis=0) & np.isfinite(raw).all(axis=0)
        crs, transform = src.crs, src.transform
    if not valid.any():
        raise ValueError('Input contains no valid radar pixels')
    x = preprocess(raw, valid, config['preprocessing'])
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
    logits = session.run([config['output_name']], {config['input_name']: x})[0]
    if logits.shape != (1, 1, 512, 512) or not np.isfinite(logits).all():
        raise ValueError('Invalid model output')
    probability = 1 / (1 + np.exp(-np.clip(logits[0, 0], -80, 80)))
    mask = (probability > threshold).astype('uint8')
    mask[~valid] = 255
    probability[~valid] = -9999
    output_dir.mkdir(parents=True)
    common = dict(driver='GTiff', width=512, height=512, count=1, crs=crs,
                  transform=transform, compress='deflate')
    tags = dict(model_sha256=model_hash, threshold=str(threshold),
                product='experimental_water_segmentation', input_units='dB')
    for name, array, dtype, nodata in [('water_mask.tif', mask, 'uint8', 255),
                                        ('water_score.tif', probability, 'float32', -9999)]:
        with rasterio.open(output_dir / name, 'w', dtype=dtype, nodata=nodata, **common) as dst:
            dst.write(array, 1)
            dst.update_tags(**tags)
            dst.set_band_description(1, 'water mask: 0 dry, 1 water, 255 no-data' if dtype == 'uint8'
                                     else 'sigmoid water score; not calibrated flood probability')
    report = dict(input=str(input_path.resolve()), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  model_sha256=model_hash, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  vv_band=vv_band, vh_band=vh_band, assumed_units='dB', threshold=threshold,
                  crs=crs.to_string(), transform=list(transform),
                  valid_pixels=int(valid.sum()), nodata_pixels=int((~valid).sum()),
                  predicted_water_pixels=int((mask == 1).sum()),
                  input_policy='Finite pixels unmasked in both input bands; other pixels filled with normalised mean for inference and marked no-data in outputs',
                  output_scope='Water segmentation; permanent water has not been removed')
    (output_dir / 'prediction.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True, help='512x512 GeoTIFF; VV/VH in dB')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vv-band', type=int, default=1)
    p.add_argument('--vh-band', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    try:
        report = predict(args.input, args.model, args.config, args.output, args.vv_band, args.vh_band, args.threads)
    except (ValueError, KeyError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
