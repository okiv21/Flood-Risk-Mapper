"""Evaluate a saved M1 checkpoint on held-out Sen1Floods11 splits."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Sen1Floods11 directory')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fixed water threshold selected on validation data')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error('threshold must be between 0 and 1')
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size must be positive and num-workers nonnegative')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    for name in ['S1Hand', 'LabelHand', 'splits']:
        if not (args.root / name).is_dir():
            parser.error(f'Missing dataset directory: {args.root / name}')
    if args.output.exists():
        parser.error(f'Output already exists; choose a new run path: {args.output}')

    import torch
    from torch.utils.data import DataLoader
    from floodrisk.datasets.sen1floods11 import Sen1Floods11
    from floodrisk.models.unet import build_unet
    from floodrisk.train import evaluate

    # CPU inference supports local evaluation without a GPU or pretrained download.
    model = build_unet(pretrained=False)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    report = {
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'device': 'cpu', 'threshold': args.threshold,
        'protocol': 'Pooled pixel IoU; ignore label -1; mIoU averages water and dry IoU',
        'torch_version': str(torch.__version__),
        'environment': {name: version(name) for name in
                        ['torch', 'torchvision', 'numpy', 'rasterio',
                         'segmentation-models-pytorch']},
        'dataset_root': str(args.root.resolve()), 'splits': {},
    }
    for split in ['test', 'bolivia']:
        dataset = Sen1Floods11(args.root, split)
        if not len(dataset):
            raise ValueError(f'Empty split: {split}')
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.num_workers, shuffle=False)
        metrics = evaluate(model, loader, 'cpu', threshold=args.threshold)
        split_file = args.root / 'splits' / f'flood_{split}_data.csv'
        report['splits'][split] = {
            'chips': len(dataset),
            'split_csv_sha256': hashlib.sha256(split_file.read_bytes()).hexdigest(),
            **metrics,
        }
        print(split, report['splits'][split], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(f'Saved evaluation: {args.output}')


if __name__ == '__main__':
    main()
