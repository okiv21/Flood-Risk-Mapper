import numpy as np


class PooledIoU:
    """Masked IoU pooled over a whole split. Ignores label == -1."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.water_inter = self.water_union = 0
        self.dry_inter = self.dry_union = 0

    def update(self, pred, label):
        pred = np.asarray(pred)
        label = np.asarray(label)
        valid = label != -1
        p = (pred == 1) & valid
        t = (label == 1) & valid
        self.water_inter += int((p & t).sum())
        self.water_union += int((p | t).sum())
        self.dry_inter += int((~p & ~t & valid).sum())
        self.dry_union += int(((~p | ~t) & valid).sum())

    def compute(self):
        iou_water = self.water_inter / max(self.water_union, 1)
        iou_dry = self.dry_inter / max(self.dry_union, 1)
        return {"iou_water": iou_water, "miou": (iou_water + iou_dry) / 2}