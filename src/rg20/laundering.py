"""Phase 2 - media-laundering attack simulator.

Social platforms transform uploads (re-compression, down-sampling, blur, noise,
screen recording).  Each attack takes/returns a BGR uint8 image, and every attack
has three calibrated intensity levels (1=light, 2=medium, 3=heavy).
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

LEVELS = (1, 2, 3)


def jpeg(img: np.ndarray, level: int = 2) -> np.ndarray:
    q = {1: 70, 2: 40, 3: 15}[level]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def resize(img: np.ndarray, level: int = 2) -> np.ndarray:
    """Down-sample then re-up-sample (what a thumbnail / re-share does)."""
    f = {1: 0.75, 2: 0.5, 3: 0.3}[level]
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blur(img: np.ndarray, level: int = 2) -> np.ndarray:
    s = {1: 0.8, 2: 1.5, 3: 2.5}[level]
    return cv2.GaussianBlur(img, (0, 0), s)


def noise(img: np.ndarray, level: int = 2, seed: int | None = None) -> np.ndarray:
    s = {1: 4, 2: 10, 3: 20}[level]
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def h264(img: np.ndarray, level: int = 2) -> np.ndarray:
    """Re-encode as a one-frame H.264 clip with libx264 (CRF 28/35/42).

    Falls back to JPEG of matching severity when ffmpeg/libx264 is unavailable.
    """
    if shutil.which("ffmpeg") is None:
        return jpeg(img, level)
    crf = {1: 28, 2: 35, 3: 42}[level]
    h, w = img.shape[:2]
    h2, w2 = h - h % 2, w - w % 2          # yuv420 needs even sizes
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d, "in.png"), Path(d, "out.mp4")
        cv2.imwrite(str(src), img[:h2, :w2])
        cmd = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-c:v", "libx264",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-frames:v", "1", str(dst)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            return jpeg(img, level)
        cap = cv2.VideoCapture(str(dst))
        ok, frame = cap.read()
        cap.release()
    if not ok:
        return jpeg(img, level)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    return frame


def screen_record(img: np.ndarray, level: int = 2) -> np.ndarray:
    """Composite worst-case chain: resize -> blur -> noise -> H.264/JPEG."""
    x = resize(img, level)
    x = blur(x, max(1, level - 1))
    x = noise(x, max(1, level - 1))
    return h264(x, level)


ATTACKS: dict[str, Callable[..., np.ndarray]] = {
    "jpeg": jpeg, "resize": resize, "blur": blur, "noise": noise,
    "h264": h264, "screen_record": screen_record,
}


def apply(img: np.ndarray, name: str, level: int = 2) -> np.ndarray:
    if name not in ATTACKS:
        raise KeyError(f"unknown attack {name!r}; choose from {list(ATTACKS)}")
    return ATTACKS[name](img, level)


FAST_ATTACKS = ("jpeg", "resize", "blur", "noise")      # no ffmpeg subprocess


def random_launder(img: np.ndarray, rng: np.random.Generator, names=None) -> np.ndarray:
    """Random 1-3 attack chain, used as training-time augmentation."""
    names = list(names or ATTACKS)
    chain = rng.choice(names, size=min(len(names), int(rng.integers(1, 4))), replace=False)
    for n in chain:
        img = apply(img, str(n), int(rng.integers(1, 4)))
    return img
