"""Procedural stand-in data so the whole pipeline runs without downloading FF++/DFDC.

NOTE: this is a *functional test set*, not a research dataset.  "Real" images are
procedural face-like scenes (+ a few bundled photos); "fake" images get typical
generator fingerprints: up-sampling grid artifacts, spectral peaks and
blended/pasted regions.  Use FaceForensics++ / DFDC (see README) for real results.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _fractal(rng, size, octaves=5):
    out = np.zeros((size, size), np.float32)
    for o in range(octaves):
        n = 4 * 2 ** o
        layer = rng.random((n, n)).astype(np.float32)
        out += cv2.resize(layer, (size, size), interpolation=cv2.INTER_CUBIC) / (1.7 ** o)
    out -= out.min()
    return out / (out.max() + 1e-6)


def make_real(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    base = np.stack([_fractal(rng, size) for _ in range(3)], -1)
    img = (60 + 140 * base).astype(np.float32)
    # face-like ellipses with soft shading
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * rng.uniform(.35, .65), size * rng.uniform(.35, .65)
    ax, ay = size * rng.uniform(.18, .3), size * rng.uniform(.24, .36)
    skin = np.array([rng.uniform(120, 220), rng.uniform(90, 180), rng.uniform(70, 150)], np.float32)
    d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    m = np.clip(1.2 - d, 0, 1)[..., None]
    shade = (1 - 0.35 * ((xx - cx) / ax)[..., None] * rng.uniform(-1, 1))
    img = img * (1 - m) + (skin * shade) * m
    for dx in (-0.4, 0.4):                                # eyes
        ex, ey = cx + dx * ax, cy - 0.25 * ay
        e = np.clip(1 - (((xx - ex) / (0.16 * ax)) ** 2 + ((yy - ey) / (0.09 * ay)) ** 2), 0, 1)
        img = img * (1 - e[..., None] * .8) + 30 * e[..., None] * .8
    mouth = np.clip(1 - (((xx - cx) / (0.35 * ax)) ** 2 + ((yy - (cy + .45 * ay)) / (0.07 * ay)) ** 2), 0, 1)
    img = img * (1 - mouth[..., None] * .7) + np.array([60, 40, 150], np.float32) * mouth[..., None] * .7
    img += rng.normal(0, rng.uniform(1.5, 4), img.shape)  # sensor noise
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 0.9))
    return np.clip(img, 0, 255).astype(np.uint8)


def _bundled_photos(size):
    try:
        from skimage import data
        outs = []
        for name in ("astronaut", "coffee", "chelsea", "rocket", "motorcycle_left"):
            try:
                im = cv2.cvtColor(getattr(data, name)(), cv2.COLOR_RGB2BGR)
                outs.append(im)
            except Exception:
                pass
        return outs
    except Exception:
        return []


def make_fake(real: np.ndarray, rng: np.random.Generator, other: np.ndarray | None = None) -> np.ndarray:
    s = real.shape[0]
    kind = rng.choice(["upsample", "spectral", "blend"])
    x = real.astype(np.float32)
    if kind == "upsample":                                # GAN decoder grid artifacts
        f = int(rng.choice([2, 4]))
        small = cv2.resize(x, (s // f, s // f), interpolation=cv2.INTER_AREA)
        up = cv2.resize(small, (s, s), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[0:s, 0:s]
        grid = ((xx % f == 0) ^ (yy % f == 0)).astype(np.float32) * rng.uniform(2, 5)
        x = cv2.GaussianBlur(up, (0, 0), 0.6) + grid[..., None]
    elif kind == "spectral":                              # periodic high-freq tone
        yy, xx = np.mgrid[0:s, 0:s]
        fx, fy = rng.uniform(0.18, 0.45), rng.uniform(0.18, 0.45)
        tone = np.sin(2 * np.pi * (fx * xx + fy * yy)) * rng.uniform(1.5, 4)
        x = cv2.GaussianBlur(x, (0, 0), 0.7) + tone[..., None]
    else:                                                 # face-swap style blend
        src = other if other is not None else make_real(rng, s)
        src = cv2.resize(src, (s, s)).astype(np.float32)
        r = int(s * rng.uniform(.18, .28))
        cx, cy = int(s * rng.uniform(.4, .6)), int(s * rng.uniform(.4, .6))
        mask = np.zeros((s, s), np.float32)
        cv2.ellipse(mask, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), s * 0.015)[..., None]
        patch = cv2.GaussianBlur(src, (0, 0), 1.2) * rng.uniform(.9, 1.1)
        x = x * (1 - mask) + patch * mask
    return np.clip(x, 0, 255).astype(np.uint8)


def generate_dataset(out_dir: str | Path, n_real: int = 300, n_fake: int = 300,
                     size: int = 256, seed: int = 0) -> Path:
    """Write out_dir/real/*.png and out_dir/fake/*.png.  Filenames are
    ``clip<ID>_<frame>.png`` so splits can be grouped per clip (no leakage)."""
    out = Path(out_dir)
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    photos = _bundled_photos(size)

    def real_img():
        if photos and rng.random() < 0.3:
            p = photos[int(rng.integers(len(photos)))]
            h, w = p.shape[:2]
            c = int(min(h, w) * rng.uniform(.5, 1.0))
            y, x = int(rng.integers(0, h - c + 1)), int(rng.integers(0, w - c + 1))
            return cv2.resize(p[y:y + c, x:x + c], (size, size), interpolation=cv2.INTER_AREA)
        return make_real(rng, size)

    for i in range(n_real):
        cv2.imwrite(str(out / "real" / f"clip{i // 3:04d}_{i % 3:03d}.png"), real_img())
    for i in range(n_fake):
        base = real_img()
        cv2.imwrite(str(out / "fake" / f"clip{10000 + i // 3:05d}_{i % 3:03d}.png"),
                    make_fake(base, rng, real_img()))
    return out
