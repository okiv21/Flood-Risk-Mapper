import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from floodrisk.datasets.sen1floods11 import Sen1Floods11
from floodrisk.eval.metrics import PooledIoU
from floodrisk.models.losses import masked_bce_dice
from floodrisk.models.unet import build_unet


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    metric = PooledIoU()
    for x, y in loader:
        x = x.to(device)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            logits = model(x)
        pred = (torch.sigmoid(logits.float().squeeze(1)) > threshold).cpu().numpy()
        metric.update(pred, y.numpy())
    return metric.compute()


def train(root, epochs=60, batch_size=8, lr=3e-4, seed=42, out="best.pt", num_workers=2):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_dl = DataLoader(Sen1Floods11(root, "train", augment=True), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, drop_last=True)
    valid_dl = DataLoader(Sen1Floods11(root, "valid"), batch_size=batch_size,
                          num_workers=num_workers)

    model = build_unet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                logits = model(x)
            loss = masked_bce_dice(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()

        val = evaluate(model, valid_dl, device)
        row = {"epoch": epoch, "loss": round(total / len(train_dl), 4),
               "val_iou_water": round(val["iou_water"], 4),
               "val_miou": round(val["miou"], 4), "sec": round(time.time() - t0, 1)}
        history.append(row)
        print(row)

        if val["iou_water"] > best:
            best = val["iou_water"]
            torch.save(model.state_dict(), out)

    print("best val water IoU:", round(best, 4))
    return history