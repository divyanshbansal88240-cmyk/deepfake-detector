# deepfake-detector
# RG20 — Deepfake & Tamper Detection Using Watermarking and Laundering Robustness Analysis

Major Project-I (18B19CI791), JUIT · Group RG20 · Supervisor: Dr. Ekta Gandotra

A two-layer defence: **(1)** an invisible, keyed watermark embedded in authentic media *before* distribution,
**(2)** a CNN/forensic deepfake classifier for media that fails watermark verification — plus a
laundering simulator, a robustness benchmark and a Streamlit web app.

```
            ┌────────────────────────── upload ──────────────────────────┐
            ▼                                                            │
   Phase 1: watermark verify (DCT-QIM + HMAC, tamper map, fragile LSB)   │
      │ verified, no tamper →  AUTHENTIC                               │
      │ verified, regions disagree →  TAMPERED (+ heat-map)           │
      ▼ else                                                             │
   Phase 3: face crop (MTCNN/Haar) → ResNet / XceptionNet / forensic GBM │
      │ fake prob ≥ τ →  DEEPFAKE   │ damaged mark →  TAMPERED   │ else  UNVERIFIED
   Phase 2 (attack simulator) is used to train & stress-test both layers; Phase 4 = benchmark + web app
```

## Layout
| Path | Purpose |
|---|---|
| `src/rg20/watermark.py` | Phase 1 – DCT-QIM robust watermark, HMAC tag, tamper localisation, fragile LSB layer |
| `src/rg20/laundering.py` | Phase 2 – JPEG, resize, blur, noise, H.264 (libx264), screen-record chain; 3 intensities |
| `src/rg20/faces.py`, `video.py`, `data.py` | MTCNN/Haar face crops, 30-FPS frame extraction, video watermarking, FF++/DFDC preparation, clip-grouped splits |
| `src/rg20/models.py`, `train.py`, `detectors.py` | Phase 3 – ResNet18/50/152 & XceptionNet (PyTorch), dual-stream training |
| `src/rg20/classical.py` | Torch-free forensic detector (spectrum + noise-residual features → gradient boosting) |
| `src/rg20/pipeline.py` | Two-layer decision logic |
| `src/rg20/evaluate.py` | Phase 4 – passive vs hybrid benchmark (Accuracy/Precision/Recall/F1/AUC, watermark survival) |
| `app/streamlit_app.py` | Web app: Verify · Protect · Laundering lab |
| `tests/` | 19 pytest tests (watermark, laundering, pipeline, video) |

## Setup (Windows / Linux / macOS)
```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install torch torchvision                            # NVIDIA GPU: add --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .                                         # makes `python -m rg20` work from anywhere
pytest -q
```
Install **ffmpeg** (Windows: `winget install Gyan.FFmpeg`, then reopen the terminal) for H.264 attacks and video watermarking.
Optional MTCNN: `pip install facenet-pytorch --no-deps` (otherwise OpenCV Haar faces are used).

## Quick start (no dataset needed)
```bash
bash scripts/run_demo.sh                 # (Windows: powershell -File scripts/run_demo.ps1)   synthetic data → train → benchmark → embed/launder/verify
streamlit run app/streamlit_app.py       # pick a detector in the sidebar
```
Single-image CLI:
```bash
python -m rg20 --key mykey embed photo.jpg protected.png --media-id 1001
python -m rg20 launder protected.png shared.jpg --attack jpeg --level 3
python -m rg20 --key mykey verify shared.jpg --detector outputs/classical_robust.joblib
python -m rg20 --key mykey video-embed in.mp4 out.mp4 && python -m rg20 --key mykey video-verify out.mp4
```
Always **share protected images as PNG** if you want the pixel-exact layer; the robust layer survives JPEG/H.264 anyway.

## Real experiments (FaceForensics++ / DFDC)
```bash
python -m rg20 prepare --src "D:\\FaceForensics++_C23" --out data/faces --frames-per-video 10 --max-videos 300 --exclude csv
# faces + manifest.csv; --max-videos is per class and samples every manipulation folder evenly
python -m rg20 train --manifest data/faces/manifest.csv --arch xception --launder-prob 0   --out outputs/xception_passive.pt
python -m rg20 train --manifest data/faces/manifest.csv --arch xception --launder-prob 0.6 --wm-prob 0.3 --out outputs/xception_robust.pt
python -m rg20 benchmark --manifest data/faces/manifest.csv --detector passive=outputs/xception_passive.pt --detector robust=outputs/xception_robust.pt
```
See `scripts/train_cnn_colab.md` for Colab. Splits are grouped by source clip so frames of one video never leak across train/test.
Use `--wm-prob 0` for a strictly passive baseline (otherwise the net can learn "watermark residual ⇒ real").

## How the watermark works
* 64-bit payload = 32-bit media ID ‖ 32-bit truncated HMAC-SHA256(key, ID) → unforgeable without the key; wrong key / unmarked media score ≈ chance (≈50 %) on the tag.
* Luma is resized to a canonical 128×128 grid (→ robust to rescaling), 8×8 block DCT, three low/mid coefficients per block carry **dithered QIM** bits; bit-to-block assignment and dither are key-derived; decoding sums soft votes over ~16 blocks×3 coefficients per bit.
* Verified ⇔ ≥ 28/32 tag bits match (chance false-accept ≈ 1e-5). Profiles in `watermark.PROFILES`: `stealth` ≈40 dB, `balanced` ≈37 dB, `robust` ≈34 dB PSNR.
* **Tamper map**: per-block agreement between each block's votes and the decoded payload; pasted/re-synthesised regions disagree and are flagged (see `outputs/demo_tamper.png`).
* **Fragile LSB layer**: keyed pattern in blue-channel LSBs; intact ⇒ lossless original, `localised` ⇒ pixel-exact edit regions, `absent` ⇒ re-encoded.
* Video: every frame is marked; verification pools soft votes across sampled frames (temporal averaging).

## Results produced in this repository (procedural proxy data — NOT FF++/DFDC)
`outputs/benchmark/` (CSV + plots), 80 test images, classical detector, `balanced` watermark. The synthetic "fakes" carry
generator-style artifacts, so absolute numbers say nothing about real deepfakes; they validate that the pipeline works and show the *trend*.

Watermark survival (% of protected authentic images still verified; false-accept on fakes = 0 % in every condition):

| attack | light | medium | heavy |
|---|---|---|---|
| JPEG (q70/40/15) | 100 | 100 | 100 |
| resize (0.75/0.5/0.3×) | 100 | 100 | 88 (98 robust) |
| blur (σ .8/1.5/2.5) | 100 | 98 | 78 (85 robust) |
| noise (σ 4/10/20) | 100 | 100 | 100 |
| H.264 (CRF 28/35/42) | 100 | **8** (72 robust) | 0 |
| screen-record chain | 88 (100 robust) | 0 (52 robust) | 0 |

Passive accuracy collapses under resize/blur/noise (0.94 → ~0.5) while the hybrid system stays at 0.88–1.0 for those attacks.
**Honest limits:** hybrid gains nothing where the watermark dies (H.264 ≥ CRF 35, screen recording, `balanced` profile) or where
the *fake* is the problem (JPEG: marks survive but the passive detector still misses fakes). Cropping/rotation break block alignment.
Tamper localisation can false-alarm on very dark/saturated regions under q≤30 JPEG.

## Status of this build
Verified by running here: watermark engine, tamper maps, laundering, video, classical detector, benchmark, 19/19 tests.
**Not executed here** (the build sandbox could not install PyTorch/Streamlit): `train.py`, `models.py`, `TorchDetector`, MTCNN backend
and `app/streamlit_app.py` — they compile and follow the same interfaces, but run them once (`pytest`, a 1-epoch
`rg20 train`, `streamlit run`) and expect to fix small issues.
