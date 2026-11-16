"""Unified detector interface used by the pipeline, benchmark and web app."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .classical import ClassicalDetector


class TorchDetector:
    kind = "torch"

    def __init__(self, path: str | Path, device: str | None = None, batch_size: int = 32):
        import torch
        from .models import load_checkpoint
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.ckpt = load_checkpoint(path, self.device)
        self.size = int(self.ckpt["input_size"])
        self.batch_size = batch_size

    def _prep(self, bgr: np.ndarray) -> np.ndarray:
        from .models import IMAGENET_MEAN, IMAGENET_STD
        x = cv2.cvtColor(cv2.resize(bgr, (self.size, self.size), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
        return x.transpose(2, 0, 1)

    def predict_proba(self, images) -> np.ndarray:
        torch, out = self.torch, []
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch = np.stack([self._prep(im) for im in images[i:i + self.batch_size]])
                logits = self.model(torch.from_numpy(batch).to(self.device)).squeeze(1)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def load_detector(path: str | Path | None):
    """``.joblib`` -> classical detector, ``.pt``/``.pth`` -> CNN.  None -> None."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".joblib":
        return ClassicalDetector.load(path)
    return TorchDetector(path)
