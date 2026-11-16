"""Dataset preparation, manifests, grouped splits and the 'dual-stream' sample builder."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import laundering
from .faces import FaceExtractor
from .video import extract_frames

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


# --------------------------------------------------------------- video -> faces
def _label_for(video: Path, root: Path, dfdc_meta: dict | None) -> int | None:
    """1 = fake, 0 = real.  Uses only the path *relative to the dataset root*."""
    if dfdc_meta and video.name in dfdc_meta:
        return 1 if dfdc_meta[video.name]["label"].upper() == "FAKE" else 0
    p = "/".join(video.relative_to(root).parts).lower()
    if "original" in p or p.startswith("real") or "/real" in p:
        return 0
    if any(k in p for k in ("deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter",
                            "deepfakedetection", "manipulated", "fake")):
        return 1
    return None


def prepare_from_videos(src: str | Path, out: str | Path, frames_per_video: int = 10,
                        fps: float = 30.0, max_videos_per_class: int | None = None,
                        face_backend: str = "auto", size: int = 224,
                        exclude: tuple = (), seed: int = 0) -> pd.DataFrame:
    """Walk ``src`` for videos, extract face crops and write ``out/{real,fake}/``.

    Fake videos are sampled round-robin across manipulation folders (Deepfakes, Face2Face,
    ...) so a small ``max_videos_per_class`` still covers every method."""
    src, out = Path(src), Path(out)
    fe = FaceExtractor(face_backend, size=size)
    meta = {}
    for mp in src.rglob("metadata.json"):
        try:
            meta.update(json.loads(mp.read_text()))
        except Exception:
            pass
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, list[Path]] = {}
    for v in sorted(p for p in src.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        rel = "/".join(v.relative_to(src).parts).lower()
        if any(x.lower() in rel for x in exclude):
            continue
        lab = _label_for(v, src, meta)
        if lab is not None:
            buckets.setdefault((lab, v.parent.name), []).append(v)
    chosen = {0: [], 1: []}
    for lab in (0, 1):
        groups = [[vs[i] for i in rng.permutation(len(vs))]
                  for (l, _), vs in buckets.items() if l == lab]
        cap = max_videos_per_class or sum(len(g) for g in groups)
        while groups and len(chosen[lab]) < cap:            # round-robin over method folders
            for g in list(groups):
                if g:
                    chosen[lab].append(g.pop())
                    if len(chosen[lab]) >= cap:
                        break
                else:
                    groups.remove(g)
    rows = []
    for lab in (0, 1):
        cls = "fake" if lab else "real"
        for k, v in enumerate(chosen[lab]):
            vid = f"{re.sub(r'[^A-Za-z0-9]+', '', v.parent.name)}__{v.stem}"   # e.g. Deepfakes__000_003
            n = extract_frames_subsample(v, out / cls, vid, frames_per_video, fps, fe)
            rows.append({"video": vid, "label": lab, "frames": n})
            if (k + 1) % 25 == 0:
                print(f"[prepare] {cls}: {k + 1}/{len(chosen[lab])} videos")
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "videos.csv", index=False)
    print(f"[prepare] real videos: {len(chosen[0])}, fake videos: {len(chosen[1])}")
    return df


def extract_frames_subsample(video, out_dir, vid, n, fps, fe) -> int:
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return 0
    idxs = np.linspace(0, max(total - 1, 0), n).astype(int)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for k, i in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        face, _ = fe.crop(fr, fallback_full=False)
        if face is None:
            continue
        cv2.imwrite(str(Path(out_dir) / f"{vid}_{k:03d}.png"), face)
        saved += 1
    cap.release()
    return saved


# ------------------------------------------------------------------- manifests
def group_of(path: Path) -> str:
    """Leakage-safe group key.  FF++ files are ``Method__<target>_<source>_<frame>``: all videos
    sharing the *first* id (the same person/scene) go to the same split.  Other names fall back
    to the clip name without the frame index."""
    stem = path.stem
    if "__" in stem:
        m = re.match(r"\d+", stem.split("__", 1)[1])
        if m:
            return "src" + m.group(0)
    return re.sub(r"_\d+$", "", stem)


def build_manifest(root: str | Path, val_frac: float = 0.15, test_frac: float = 0.15,
                   seed: int = 42) -> pd.DataFrame:
    """Scan root/{real,fake}; split by source clip (no frame leakage across splits)."""
    root = Path(root)
    rows = []
    for cls, lab in (("real", 0), ("fake", 1)):
        for p in sorted((root / cls).rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"path": str(p.resolve()), "label": lab, "group": group_of(p)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no images under {root}/real and {root}/fake")
    rng = np.random.default_rng(seed)
    split = {}

    def assign(groups):
        groups = np.array(sorted(groups))
        rng.shuffle(groups)
        n_test, n_val = int(len(groups) * test_frac), int(len(groups) * val_frac)
        split.update({g: "test" for g in groups[:n_test]})
        split.update({g: "val" for g in groups[n_test:n_test + n_val]})

    g0, g1 = set(df[df.label == 0].group), set(df[df.label == 1].group)
    if g0 & g1:                       # real & fake share source ids (FF++): split jointly, no leakage
        assign(g0 | g1)
    else:                             # disjoint groups: stratify per class
        assign(g0)
        assign(g1)
    df["split"] = df["group"].map(lambda g: split.get(g, "train"))
    df.to_csv(root / "manifest.csv", index=False)
    return df


def load_bgr(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise IOError(f"cannot read {path}")
    return im


def make_variant(img: np.ndarray, label: int, rng: np.random.Generator, wm=None,
                 launder_prob: float = 0.5, wm_prob: float = 0.0, attacks=None) -> np.ndarray:
    """Dual-stream sample: clean / pristine-watermarked / laundered.

    Authentic (label 0) media is what gets watermarked before distribution, so only
    real samples receive the mark.  ``wm_prob=0`` gives a strictly *passive* baseline.
    """
    if label == 0 and wm is not None and rng.random() < wm_prob:
        img = wm.embed(img, int(rng.integers(0, 2 ** 31)))
    if rng.random() < launder_prob:
        img = laundering.random_launder(img, rng, attacks)
    return img
