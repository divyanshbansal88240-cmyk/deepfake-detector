"""Lightweight forensic-feature detector (no PyTorch required).

Used as (a) a fast baseline for the passive-vs-hybrid comparison, (b) the
fallback detector in the web app when no CNN checkpoint is available.
Features: radial log-power spectrum, noise-residual statistics, gradient energy,
JPEG blockiness and chroma statistics -> HistGradientBoosting.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NBINS = 32


def _radial_spectrum(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * win))
    p = np.log1p(np.abs(f) ** 2)
    yy, xx = np.indices(p.shape)
    r = np.hypot(yy - h / 2, xx - w / 2)
    r = (r / (min(h, w) / 2)).clip(0, 0.999)
    idx = (r * NBINS).astype(int)
    prof = np.bincount(idx.ravel(), p.ravel(), NBINS) / np.maximum(np.bincount(idx.ravel(), minlength=NBINS), 1)
    peak = (p * (r > 0.3)).max() / (np.median(p[r > 0.3]) + 1e-6)     # spectral peaks
    return np.concatenate([prof - prof.mean(), [prof.mean(), peak]])


def _blockiness(gray: np.ndarray) -> float:
    d = np.abs(np.diff(gray, axis=1))
    cols = np.arange(d.shape[1])
    on, off = d[:, cols % 8 == 7].mean(), d[:, cols % 8 != 7].mean()
    return float(on / (off + 1e-6))


def features(bgr: np.ndarray, size: int = 128) -> np.ndarray:
    x = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
    res = x.astype(np.float32) - cv2.GaussianBlur(x.astype(np.float32), (0, 0), 1.0)
    rs = res.reshape(-1, 3)
    mu = rs.mean(0)
    kurt = ((rs - mu) ** 4).mean(0) / (rs.var(0) ** 2 + 1e-6)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    ycc = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    # spectrum on the *native* 224/256 crop is more informative than the 128 resize
    big = cv2.cvtColor(cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return np.concatenate([
        _radial_spectrum(big), rs.std(0), kurt, np.abs(rs).mean(0),
        [lap.var(), np.hypot(gx, gy).mean(), _blockiness(g), ycc[..., 1].std(), ycc[..., 2].std()],
    ]).astype(np.float32)


class ClassicalDetector:
    kind = "classical"

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=4, random_state=0))

    def fit(self, images, labels):
        X = np.stack([features(i) for i in images])
        self.model.fit(X, np.asarray(labels))
        return self

    def predict_proba(self, images) -> np.ndarray:
        """Probability that each image is FAKE."""
        X = np.stack([features(i) for i in images])
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalDetector":
        d = cls()
        d.model = joblib.load(path)
        return d
