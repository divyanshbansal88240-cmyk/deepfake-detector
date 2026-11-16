"""Phase 3 - train XceptionNet / ResNet deepfake classifiers (PyTorch)."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import laundering
from .config import load_config
from .data import load_bgr, make_variant
from .metrics import classification_metrics
from .models import IMAGENET_MEAN, IMAGENET_STD, build_model, save_checkpoint
from .watermark import DCTWatermarker, WatermarkParams


class FaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, size: int, train: bool, wm: DCTWatermarker | None,
                 launder_prob: float, wm_prob: float, attacks=None, seed: int = 0):
        self.df, self.size, self.train = df.reset_index(drop=True), size, train
        self.wm, self.launder_prob, self.wm_prob, self.seed = wm, launder_prob, wm_prob, seed
        self.attacks = attacks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        # per-(epoch-agnostic) worker-safe rng
        rng = np.random.default_rng((self.seed, i, int(time.time() * 1e6) % 2 ** 32) if self.train else (self.seed, i))
        img = load_bgr(r["path"])
        lab = int(r["label"])
        if self.train:
            if rng.random() < 0.5:
                img = img[:, ::-1]
            img = make_variant(np.ascontiguousarray(img), lab, rng, self.wm, self.launder_prob, self.wm_prob, self.attacks)
        img = cv2.cvtColor(cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.tensor(lab, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
        ys.append(y.numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(ps))


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="manifest.csv from `rg20 manifest`")
    ap.add_argument("--arch", default=cfg["detector"]["model"], choices=["resnet18", "resnet50", "resnet152", "xception"])
    ap.add_argument("--epochs", type=int, default=cfg["training"]["epochs"])
    ap.add_argument("--batch-size", type=int, default=cfg["training"]["batch_size"])
    ap.add_argument("--lr", type=float, default=cfg["training"]["lr"])
    ap.add_argument("--launder-prob", type=float, default=cfg["training"]["launder_prob"],
                    help="0 => clean-only (conventional passive detector)")
    ap.add_argument("--wm-prob", type=float, default=0.0,
                    help="probability that authentic training images carry the watermark (dual-stream)")
    ap.add_argument("--key", default="rg20-demo-key")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--workers", type=int, default=0 if os.name == "nt" else 2)
    ap.add_argument("--with-h264", action="store_true",
                    help="also use H.264/screen-record laundering in training (realistic but slower: spawns ffmpeg)")
    ap.add_argument("--out", default="outputs/model.pt")
    a = ap.parse_args(argv)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = cfg["detector"]["input_size"] if a.arch != "xception" else 299
    wm = DCTWatermarker(a.key, WatermarkParams.from_config(cfg)) if a.wm_prob > 0 else None
    df = pd.read_csv(a.manifest)
    attacks = None if a.with_h264 else list(laundering.FAST_ATTACKS)
    mk = lambda s, train: FaceDataset(df[df.split == s], size, train, wm, a.launder_prob, a.wm_prob, attacks)
    dl = lambda ds, sh: DataLoader(ds, batch_size=a.batch_size, shuffle=sh, num_workers=a.workers,
                                   pin_memory=device == "cuda", drop_last=sh)
    tr, va = dl(mk("train", True), True), dl(mk("val", False), False)
    pos = max(1, int(df[df.split == "train"].label.sum()))
    neg = max(1, int((df[df.split == "train"].label == 0).sum()))
    model = build_model(a.arch, pretrained=not a.no_pretrained).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / pos, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, hist = -1.0, []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, y in tqdm(tr, desc=f"epoch {ep}/{a.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = crit(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot, n = tot + loss.item() * len(y), n + len(y)
        sched.step()
        m = evaluate(model, va, device)
        m.update(epoch=ep, train_loss=tot / max(n, 1))
        hist.append(m)
        print(json.dumps(m))
        score = m["auc"] if not np.isnan(m["auc"]) else m["accuracy"]
        if score > best:
            best = score
            save_checkpoint(a.out, model, a.arch, size, {"val": m, "args": vars(a)})
    Path(a.out).with_suffix(".history.json").write_text(json.dumps(hist, indent=2))
    print(f"best val score {best:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
