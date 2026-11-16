"""Command line interface:  python -m rg20 <command> ..."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .watermark import DCTWatermarker, WatermarkParams, psnr, ssim


def _wm(args, cfg):
    if getattr(args, "profile", None):
        cfg = {**cfg, "watermark": {**cfg["watermark"], "profile": args.profile}}
    return DCTWatermarker(args.key, WatermarkParams.from_config(cfg))


def cmd_synthetic(a, cfg):
    from .synthetic import generate_dataset
    from .data import build_manifest
    generate_dataset(a.out, a.n_real, a.n_fake, seed=a.seed)
    df = build_manifest(a.out)
    print(f"wrote {len(df)} images to {a.out}; split counts:\n{df.groupby(['split', 'label']).size()}")


def cmd_prepare(a, cfg):
    from .data import prepare_from_videos, build_manifest
    prepare_from_videos(a.src, a.out, a.frames_per_video, a.fps, a.max_videos, a.face_backend,
                        exclude=tuple(a.exclude))
    df = build_manifest(a.out)
    print(df.groupby(["split", "label"]).size())


def cmd_manifest(a, cfg):
    from .data import build_manifest
    print(build_manifest(a.root).groupby(["split", "label"]).size())


def cmd_embed(a, cfg):
    wm = _wm(a, cfg)
    img = cv2.imread(a.input)
    out = wm.embed(img, a.media_id)
    cv2.imwrite(a.output, out)                              # keep .png for the lossless layer
    print(f"saved {a.output}  PSNR={psnr(img, out):.2f} dB  SSIM={ssim(img, out):.4f}")


def cmd_verify(a, cfg):
    from .pipeline import AuthenticationPipeline
    from .detectors import load_detector
    from .faces import FaceExtractor
    pipe = AuthenticationPipeline(_wm(a, cfg), load_detector(a.detector),
                                  FaceExtractor(a.face_backend) if a.detector else None,
                                  cfg["detector"]["fake_threshold"])
    img = cv2.imread(a.input)
    r = pipe.analyze(img)
    print(f"status        : {r.status}\n{r.explanation}")
    print(f"tag score     : {r.wm.tag_score:.3f} (verified={r.wm.verified})  media_id={r.wm.media_id}")
    print(f"tamper frac   : {r.wm.tamper_fraction:.3f}   LSB layer: {r.wm.extra['lsb']['state']}")
    if r.fake_prob is not None:
        print(f"fake prob     : {r.fake_prob:.3f}  ({r.detector})")
    for n in r.notes:
        print("note:", n)


def cmd_launder(a, cfg):
    from . import laundering
    img = cv2.imread(a.input)
    cv2.imwrite(a.output, laundering.apply(img, a.attack, a.level))


def cmd_video_embed(a, cfg):
    from .video import watermark_video
    watermark_video(a.input, a.output, _wm(a, cfg), a.media_id)


def cmd_video_verify(a, cfg):
    from .video import verify_video
    r = verify_video(a.input, _wm(a, cfg))
    r.pop("first")
    print(r)


def cmd_train_classical(a, cfg):
    import pandas as pd
    from .classical import ClassicalDetector
    from .data import load_bgr, make_variant
    wm = _wm(a, cfg)
    df = pd.read_csv(a.manifest)
    df = df[df.split == "train"]
    rng = np.random.default_rng(0)
    X, y = [], []
    for _, r in df.iterrows():
        img = load_bgr(r.path)
        for _ in range(a.copies):
            X.append(make_variant(img, int(r.label), rng, wm, a.launder_prob, a.wm_prob,
                                    None if a.with_h264 else list(__import__('rg20.laundering', fromlist=['x']).FAST_ATTACKS)))
            y.append(int(r.label))
    det = ClassicalDetector().fit(X, y)
    det.save(a.out)
    print(f"trained on {len(X)} samples -> {a.out}")


def cmd_benchmark(a, cfg):
    from .detectors import load_detector
    from .evaluate import run_benchmark, plot_results
    dets = {}
    for spec in a.detector:
        name, _, path = spec.partition("=")
        d = load_detector(path)
        if d is None:
            sys.exit(f"detector not found: {path}")
        dets[name] = d
    df = run_benchmark(a.manifest, dets, _wm(a, cfg), max_per_class=a.max_per_class,
                       levels=(1, 2, 3))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "benchmark.csv", index=False)
    plot_results(df, out)
    print(f"results -> {out}/benchmark.csv and plots")


def cmd_train(a, cfg):
    from . import train
    train.main(a.rest)


def build_parser():
    ap = argparse.ArgumentParser("rg20", description=__doc__)
    ap.add_argument("--key", default="rg20-demo-key", help="secret watermark key")
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", choices=["stealth", "balanced", "robust"], default=None,
                    help="watermark strength profile (default: from config.yaml)")
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("synthetic", help="generate a procedural demo dataset")
    s.add_argument("--out", default="data/synthetic"); s.add_argument("--n-real", type=int, default=300)
    s.add_argument("--n-fake", type=int, default=300); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synthetic)

    s = sp.add_parser("prepare", help="FF++/DFDC videos -> face crops + manifest")
    s.add_argument("--src", required=True); s.add_argument("--out", default="data/faces")
    s.add_argument("--frames-per-video", type=int, default=10); s.add_argument("--fps", type=float, default=30)
    s.add_argument("--max-videos", type=int, default=None, help="per class (real / fake)")
    s.add_argument("--face-backend", default="auto", choices=["auto", "mtcnn", "haar", "full"])
    s.add_argument("--exclude", nargs="*", default=[], help="folder names to skip, e.g. DeepFakeDetection")
    s.set_defaults(fn=cmd_prepare)

    s = sp.add_parser("manifest", help="(re)build manifest.csv for a real/ fake/ folder")
    s.add_argument("root"); s.set_defaults(fn=cmd_manifest)

    s = sp.add_parser("embed", help="watermark an image (save as PNG)")
    s.add_argument("input"); s.add_argument("output"); s.add_argument("--media-id", type=int, default=1)
    s.set_defaults(fn=cmd_embed)

    s = sp.add_parser("verify", help="run the full two-layer analysis on an image")
    s.add_argument("input"); s.add_argument("--detector", default=None)
    s.add_argument("--face-backend", default="auto"); s.set_defaults(fn=cmd_verify)

    s = sp.add_parser("launder", help="apply a laundering attack")
    s.add_argument("input"); s.add_argument("output")
    s.add_argument("--attack", required=True); s.add_argument("--level", type=int, default=2)
    s.set_defaults(fn=cmd_launder)

    s = sp.add_parser("video-embed"); s.add_argument("input"); s.add_argument("output")
    s.add_argument("--media-id", type=int, default=1); s.set_defaults(fn=cmd_video_embed)
    s = sp.add_parser("video-verify"); s.add_argument("input"); s.set_defaults(fn=cmd_video_verify)

    s = sp.add_parser("train-classical", help="train the sklearn forensic detector (no torch needed)")
    s.add_argument("--manifest", required=True); s.add_argument("--out", default="outputs/classical.joblib")
    s.add_argument("--launder-prob", type=float, default=0.5); s.add_argument("--wm-prob", type=float, default=0.0)
    s.add_argument("--copies", type=int, default=3)
    s.add_argument("--with-h264", action="store_true"); s.set_defaults(fn=cmd_train_classical)

    s = sp.add_parser("train", help="train a CNN (see rg20.train --help for options)")
    s.add_argument("rest", nargs=argparse.REMAINDER); s.set_defaults(fn=cmd_train)

    s = sp.add_parser("benchmark", help="laundering robustness: passive vs hybrid")
    s.add_argument("--manifest", required=True)
    s.add_argument("--detector", action="append", required=True, metavar="NAME=PATH")
    s.add_argument("--max-per-class", type=int, default=100); s.add_argument("--out", default="outputs/benchmark")
    s.set_defaults(fn=cmd_benchmark)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args, load_config(args.config))


if __name__ == "__main__":
    main()
