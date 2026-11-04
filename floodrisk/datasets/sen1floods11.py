import csv
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Computed on the training split (notebook 01_explore)
VV_CLIP, VH_CLIP = (-30.0, 5.0), (-42.0, -5.0)
MEAN = np.array([-10.394, -17.232], dtype="float32")
STD = np.array([4.020, 4.691], dtype="float32")

SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "valid": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}


def read_split(root, split):
    with open(Path(root) / "splits" / SPLIT_FILES[split]) as f:
        return [(r[0].strip(), r[1].strip()) for r in csv.reader(f) if r]


def preprocess_s1(x):
    """x: (2, H, W) float32 in dB -> clipped, normalised, NaN-free."""
    x = x.astype("float32").copy()
    x[0] = np.clip(x[0], *VV_CLIP)
    x[1] = np.clip(x[1], *VH_CLIP)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return np.nan_to_num(x, nan=0.0)   # NaN -> 0, i.e. the mean


class Sen1Floods11(Dataset):
    def __init__(self, root, split, augment=False):
        self.root = Path(root)
        self.items = read_split(root, split)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s1_name, label_name = self.items[i]
        with rasterio.open(self.root / "S1Hand" / s1_name) as src:
            x = preprocess_s1(src.read())
        with rasterio.open(self.root / "LabelHand" / label_name) as src:
            y = src.read(1).astype("int64")      # -1, 0, 1

        if self.augment:
            k = np.random.randint(4)
            x = np.rot90(x, k, axes=(1, 2))
            y = np.rot90(y, k)
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, ::-1]

        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())