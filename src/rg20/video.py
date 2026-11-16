"""Video helpers: frame extraction (30 FPS default), watermarking, verification."""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

import cv2
import numpy as np

from .watermark import DCTWatermarker


def iter_frames(path: str | Path, target_fps: float = 30.0, max_frames: int | None = None):
    """Yield (index, BGR frame) sampled at ``target_fps`` (never up-samples)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    step = max(1, int(round(src_fps / target_fps)))
    i = n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield n, frame
            n += 1
            if max_frames and n >= max_frames:
                break
        i += 1
    cap.release()


def extract_frames(video: str | Path, out_dir: str | Path, target_fps: float = 30.0,
                   face_extractor=None, max_frames: int | None = None) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem, count = Path(video).stem, 0
    for i, fr in iter_frames(video, target_fps, max_frames):
        if face_extractor is not None:
            fr, _ = face_extractor.crop(fr, fallback_full=False)
            if fr is None:
                continue
        cv2.imwrite(str(out / f"{stem}_{i:05d}.png"), fr)
        count += 1
    return count


def _write_video(frames_dir_or_iter, out_path: str | Path, fps: float, size, crf: int = 14):
    """Pipe BGR frames into ffmpeg/libx264 (near-lossless by default)."""
    w, h = size
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to write videos")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
           "-pix_fmt", "yuv420p", str(out_path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fr in frames_dir_or_iter:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
    p.stdin.close()
    p.wait()


def watermark_video(src: str | Path, dst: str | Path, wm: DCTWatermarker, media_id: int) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w2, h2 = w - w % 2, h - h % 2

    def gen():
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield wm.embed(fr[:h2, :w2], media_id, lsb=False)   # LSB layer is meaningless in video

    _write_video(gen(), dst, fps, (w2, h2))
    cap.release()


def verify_video(path: str | Path, wm: DCTWatermarker, every: int = 10, max_frames: int = 60) -> dict:
    """Verify every ``every``-th frame; video is authentic if most sampled frames verify."""
    results = []
    cap = cv2.VideoCapture(str(path))
    i = 0
    while len(results) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % every == 0:
            results.append(wm.verify(fr))
        i += 1
    cap.release()
    if not results:
        raise IOError("no frames decoded")
    ok_frac = float(np.mean([r.verified for r in results]))
    # temporal pooling: the same payload is in every frame, so summing soft votes
    # over frames averages out per-frame laundering noise
    pooled_ok, pooled_score, pooled_id = wm.verify_votes(np.sum([r.extra["votes"] for r in results], 0))
    ids = [r.media_id for r in results if r.verified]
    return {
        "frames_checked": len(results), "verified_fraction": ok_frac,
        "verified": ok_frac >= 0.5 or pooled_ok,
        "pooled_tag_score": pooled_score,
        "media_id": pooled_id if pooled_ok else (int(np.bincount(ids).argmax()) if ids else None),
        "mean_tag_score": float(np.mean([r.tag_score for r in results])),
        "tampered_fraction": float(np.mean([r.tampered for r in results])),
        "first": results[0],
    }
