"""Official Sen1Floods11 optical comparator, using RGB and near-infrared."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

import numpy as np
import rasterio
import requests
import torch
from torch.utils.data import Dataset
from floodrisk.datasets.sen1floods11 import read_split

BASE = "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand/"
BANDS = (4, 3, 2, 8)


def prepare(root, cache, splits=("train", "valid", "test", "bolivia")):
    names = sorted(
        {
            name.replace("_S1Hand", "_S2Hand")
            for split in splits
            for name, _ in read_split(root, split)
        }
    )
    cache.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        path = cache / name
        if path.exists():
            return
        for attempt in range(3):
            try:
                response = requests.get(BASE + name, timeout=(15, 120))
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2 or (
                    exc.response is not None
                    and 400 <= exc.response.status_code < 500
                    and exc.response.status_code != 429
                ):
                    raise
                sleep(2**attempt)
        stage = path.with_suffix(".tmp")
        stage.write_bytes(response.content)
        with rasterio.open(stage) as src:
            if (
                src.descriptions[3] != "B4"
                or src.descriptions[7] != "B8"
                or src.shape != (512, 512)
            ):
                raise ValueError(f"Unexpected optical source for {name}")
        stage.replace(path)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(fetch, names))


class OpticalChips(Dataset):
    def __init__(self, root, split, cache, augment=False):
        self.root, self.cache = Path(root), Path(cache)
        self.items = read_split(root, split)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sar, label = self.items[index]
        with rasterio.open(self.cache / sar.replace("_S1Hand", "_S2Hand")) as src:
            pixels = src.read(BANDS, masked=True)
            missing = np.ma.getmaskarray(pixels).any(axis=0)
            # Fixed reflectance scaling, independent of evaluation splits.
            x = np.clip(pixels.filled(0).astype("float32") / 10000, 0, 1)
            grid = (src.crs, src.transform, src.shape)
        with rasterio.open(self.root / "LabelHand" / label) as src:
            if (
                grid[0] != src.crs
                or grid[2] != src.shape
                or not (~grid[1] * src.transform).almost_equals(
                    rasterio.Affine.identity(), precision=1e-7
                )
            ):
                raise ValueError("Image/label grid mismatch")
            y = src.read(1).astype("int64")
        y[missing] = -1
        if self.augment:
            k = np.random.randint(4)
            x = np.rot90(x, k, axes=(1, 2))
            y = np.rot90(y, k)
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
