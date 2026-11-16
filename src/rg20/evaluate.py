"""Phase 4 - laundering-robustness benchmark: passive detector vs proactive/hybrid system.

For every (attack, intensity) condition we measure
  * passive      : detector alone on laundered real+fake faces
  * wm_survival  : % of watermarked authentic images whose watermark still verifies
  * wm_false_acc : % of fakes (never watermarked) that wrongly verify
  * hybrid       : watermark verified -> real, otherwise the detector decides
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import laundering
from .data import load_bgr
from .metrics import classification_metrics
from .watermark import DCTWatermarker

ALL_ATTACKS = list(laundering.ATTACKS)


def _prep(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA) if img.shape[0] != size else img


def run_benchmark(manifest: str | Path, detectors: dict, wm: DCTWatermarker,
                  split: str = "test", max_per_class: int = 100, attacks=None,
                  levels=(1, 2, 3), threshold: float = 0.5, size: int = 256, seed: int = 0,
                  verbose: bool = True) -> pd.DataFrame:
    attacks = attacks or ALL_ATTACKS
    df = pd.read_csv(manifest)
    df = df[df.split == split]
    rng = np.random.default_rng(seed)
    reals = df[df.label == 0].sample(frac=1, random_state=seed).head(max_per_class)
    fakes = df[df.label == 1].sample(frac=1, random_state=seed).head(max_per_class)
    real_img = [_prep(load_bgr(p), size) for p in reals.path]
    fake_img = [_prep(load_bgr(p), size) for p in fakes.path]
    # authentic media is protected BEFORE distribution
    real_wm = [wm.embed(im, int(rng.integers(0, 2 ** 31))) for im in real_img]
    y = np.array([0] * len(real_img) + [1] * len(fake_img))

    conditions = [("none", 0)] + [(a, l) for a in attacks for l in levels]
    rows = []
    for ci, (atk, lvl) in enumerate(conditions):
        f = (lambda x: x) if atk == "none" else (lambda x, a=atk, l=lvl: laundering.apply(x, a, l))
        L_real, L_realwm, L_fake = [f(x) for x in real_img], [f(x) for x in real_wm], [f(x) for x in fake_img]
        ver_real = np.array([wm.verify(x).verified for x in L_realwm])
        ver_fake = np.array([wm.verify(x).verified for x in L_fake])
        base = {"attack": atk, "level": lvl, "wm_survival": float(ver_real.mean()),
                "wm_false_accept": float(ver_fake.mean())}
        if not detectors:
            rows.append({**base, "system": "watermark_only"})
        for name, det in detectors.items():
            p_passive = det.predict_proba(L_real + L_fake)
            p_wmreal = det.predict_proba(L_realwm)
            rows.append({**base, "system": f"passive:{name}",
                         **classification_metrics(y, p_passive, threshold)})
            # hybrid: verified -> authentic (score 0); else the detector score
            p_h = np.concatenate([np.where(ver_real, 0.0, p_wmreal),
                                  np.where(ver_fake, 0.0, det.predict_proba(L_fake))])
            rows.append({**base, "system": f"hybrid:{name}",
                         **classification_metrics(y, p_h, threshold)})
        if verbose:
            print(f"[{ci + 1}/{len(conditions)}] {atk}-{lvl}: wm_survival={base['wm_survival']:.2f} "
                  f"false_accept={base['wm_false_accept']:.2f}")
    return pd.DataFrame(rows)


def plot_results(df: pd.DataFrame, out_dir: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    attacks = [a for a in df.attack.unique() if a != "none"]

    # 1) accuracy vs intensity per attack, passive vs hybrid
    if "accuracy" in df and df["accuracy"].notna().any():
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for ax, atk in zip(axes.ravel(), attacks):
            for sysname, g in df[(df.attack.isin([atk, "none"]))].groupby("system"):
                g = g.assign(level=g.level).sort_values("level")
                clean = df[(df.attack == "none") & (df.system == sysname)]
                gg = pd.concat([clean, g[g.attack == atk]]).sort_values("level")
                ax.plot(gg.level, gg.accuracy, marker="o",
                        ls="--" if sysname.startswith("passive") else "-", label=sysname)
            ax.set_title(atk)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xticklabels(["clean", "light", "medium", "heavy"])
            ax.set_ylim(0.4, 1.02)
            ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=7)
        fig.supylabel("accuracy")
        fig.suptitle("Robustness to media laundering: passive (dashed) vs hybrid watermark+detector (solid)")
        fig.tight_layout()
        fig.savefig(out / "accuracy_vs_laundering.png", dpi=150)
        plt.close(fig)

    # 2) watermark survival heat-map
    s = df.drop_duplicates(["attack", "level"])
    piv = s[s.attack != "none"].pivot(index="attack", columns="level", values="wm_survival")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(piv.shape[1]), ["light", "medium", "heavy"][: piv.shape[1]])
    ax.set_yticks(range(piv.shape[0]), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.0%}", ha="center", va="center", fontsize=9)
    ax.set_title("Watermark survival (verified %)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / "watermark_survival.png", dpi=150)
    plt.close(fig)

    # 3) AUC bars at heavy laundering
    if "auc" in df and df["auc"].notna().any():
        h = df[(df.level == 3)].pivot_table(index="attack", columns="system", values="auc")
        ax = h.plot.bar(figsize=(9, 4), rot=0)
        ax.set_ylabel("AUC-ROC")
        ax.set_ylim(0.4, 1.02)
        ax.set_title("AUC under heavy laundering")
        ax.grid(alpha=.3, axis="y")
        ax.figure.tight_layout()
        ax.figure.savefig(out / "auc_heavy.png", dpi=150)
        plt.close(ax.figure)
