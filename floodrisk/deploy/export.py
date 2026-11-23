"""Export fixed-shape FP32 ONNX, verify validation outputs, benchmark CPU inference."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--optical-cache', type=Path, help='Use RGB/NIR optical validation chips instead of the default radar inputs')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 5:
        parser.error('threads must be positive; repeats must be at least 5')
    if args.output.exists():
        parser.error('Choose a new output directory to preserve previous exports')
    config = json.loads(args.config.read_text())
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != config['checkpoint_sha256']:
        parser.error('Checkpoint does not match the selected configuration')
    if not 0 < config['threshold'] < 1:
        parser.error('Invalid threshold in selected configuration')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from floodrisk.datasets.sen1floods11 import Sen1Floods11
    from floodrisk.eval.metrics import PooledIoU
    from floodrisk.models.unet import build_unet

    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    channels = 4 if args.optical_cache else 2
    if config.get('in_channels', channels) != channels:
        raise ValueError('Checkpoint configuration and sensor channel count differ')
    input_name = 'optical' if args.optical_cache else 'sar'
    model = build_unet(pretrained=False, in_channels=channels)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    if args.optical_cache:
        from floodrisk.datasets.optical import OpticalChips
        dataset = OpticalChips(args.root, 'valid', args.optical_cache)
    else:
        dataset = Sen1Floods11(args.root, 'valid')
    if not len(dataset):
        raise ValueError('Empty validation dataset')
    split_hash = hashlib.sha256((args.root / 'splits/flood_valid_data.csv').read_bytes()).hexdigest()
    if split_hash != config['split_csv_sha256']:
        raise ValueError('Validation split differs from selected configuration')
    sample = dataset[0][0].unsqueeze(0).contiguous()
    target = args.output / ('m1_s2_fp32.onnx' if args.optical_cache else 'm1_s1_fp32.onnx')
    with torch.inference_mode():
        torch.onnx.export(model, sample, str(target), opset_version=17,
                          input_names=[input_name], output_names=['logits'],
                          do_constant_folding=True)
    onnx.checker.check_model(str(target))
    print('ONNX graph validated:', target, flush=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(target), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    # Predeclared gates: probabilities allclose at 1e-4; <= 0.01% mask disagreement.
    def probability(logits):
        return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))

    pt_metric, ort_metric = PooledIoU(), PooledIoU()
    max_probability_error = max_logit_error = 0.0
    disagreements = pixels = 0
    allclose = True
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.unsqueeze(0).contiguous()
        with torch.inference_mode():
            expected = model(x).numpy()
        actual = session.run(['logits'], {input_name: x.numpy()})[0]
        if actual.shape != expected.shape or not np.isfinite(actual).all():
            raise ValueError('Invalid ONNX output shape or non-finite output')
        pp, po = probability(expected), probability(actual)
        max_probability_error = max(max_probability_error, float(np.abs(pp-po).max()))
        max_logit_error = max(max_logit_error, float(np.abs(expected-actual).max()))
        allclose &= bool(np.allclose(pp, po, atol=1e-4, rtol=1e-4))
        a, b = pp[0, 0] > config['threshold'], po[0, 0] > config['threshold']
        disagreements += int((a != b).sum())
        pixels += a.size
        pt_metric.update(a, y.numpy())
        ort_metric.update(b, y.numpy())
        if (i+1) % 20 == 0:
            print(f'Checked {i+1}/{len(dataset)} validation chips', flush=True)
    parity = dict(chips=len(dataset), max_abs_logit_error=max_logit_error,
                  max_abs_probability_error=max_probability_error,
                  probability_allclose=allclose, atol=1e-4, rtol=1e-4,
                  mask_disagreements=disagreements, total_pixels=pixels,
                  mask_disagreement_fraction=disagreements/pixels,
                  pytorch_validation=pt_metric.compute(), onnx_validation=ort_metric.compute())
    parity['reference_validation_iou_delta'] = abs(
        parity['pytorch_validation']['iou_water'] - config['validation_selected']['iou_water'])
    parity['passed'] = (allclose and disagreements/pixels <= 1e-4
                        and parity['reference_validation_iou_delta'] <= 1e-6)
    if not parity['passed']:
        (args.output / 'parity_failure.json').write_text(json.dumps(parity, indent=2)+'\n')
        raise RuntimeError('ONNX parity gate failed; do not use this export')

    # In-memory batch 1; alternating order reduces systematic ordering bias.
    sample_np = sample.numpy()
    def run_pt():
        with torch.inference_mode():
            return model(sample)
    def run_ort():
        return session.run(['logits'], {input_name: sample_np})
    for _ in range(3):
        run_pt(); run_ort()
    timings = {'pytorch': [], 'onnxruntime': []}
    for i in range(args.repeats):
        pair = [('pytorch', run_pt), ('onnxruntime', run_ort)]
        for name, fn in pair if i % 2 == 0 else reversed(pair):
            start = time.perf_counter()
            fn()
            timings[name].append((time.perf_counter()-start)*1000)
    with (args.output / 'latency_samples.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['repeat', 'pytorch_ms', 'onnxruntime_ms'])
        writer.writerows((i+1, timings['pytorch'][i], timings['onnxruntime'][i]) for i in range(args.repeats))
    latency = {k: dict(median_ms=float(np.median(v)), p95_ms=float(np.percentile(v,95))) for k,v in timings.items()}
    try:
        cpu = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    report = dict(checkpoint_sha256=checkpoint_hash, onnx_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  split_csv_sha256=split_hash, threshold=config['threshold'],
                  input_shape=list(sample.shape), output='raw water logits', opset=17, dtype='float32',
                  platform=platform.platform(), cpu=cpu, threads=args.threads,
                  provider=session.get_providers(), warmup_runs=3, repetitions=args.repeats,
                  benchmark_chip=(dataset.items[0][0].replace('_S1Hand', '_S2Hand') if args.optical_cache else dataset.items[0][0]),
                  timing_scope='Warm in-memory forward pass only; excludes preprocessing, loading, sigmoid and mask creation',
                  environment={k:version(k) for k in ['torch','onnx','onnxruntime','numpy','segmentation-models-pytorch']},
                  bytes=dict(pytorch_checkpoint=args.checkpoint.stat().st_size,onnx_fp32=target.stat().st_size),
                  parity=parity,latency=latency)
    (args.output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    metadata = dict(config, onnx_sha256=report['onnx_sha256'], input_shape=report['input_shape'],
                    input_name=input_name, output_name='logits', output_activation='sigmoid', opset=17,
                    dtype='float32', artifact=target.name)
    (args.output/'inference_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    labels=['PyTorch FP32', 'ONNX Runtime FP32']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    bars=axes[0].bar(labels,[latency[k]['median_ms'] for k in ['pytorch','onnxruntime']],color=['#1976d2','#ef8a17'])
    axes[0].bar_label(bars,fmt='%.1f ms');axes[0].set_ylabel('Median forward-pass latency (ms)')
    sizes=[report['bytes']['pytorch_checkpoint']/1e6,report['bytes']['onnx_fp32']/1e6]
    bars=axes[1].bar(labels,sizes,color=['#1976d2','#ef8a17'])
    axes[1].bar_label(bars,fmt='%.1f MB');axes[1].set_ylabel('Artifact size (decimal MB)')
    for ax in axes: ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    fig.suptitle(f'Local Intel Mac CPU | batch 1, 512×512, {args.threads} threads, {args.repeats} timed runs')
    fig.tight_layout();fig.savefig(args.output/'cpu_export_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
