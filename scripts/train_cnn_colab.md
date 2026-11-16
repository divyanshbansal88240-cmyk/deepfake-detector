# Training the CNN on Google Colab (GPU)

```python
!git clone <your-repo-url> rg20 && cd rg20 && pip install -r requirements.txt
# Kaggle API token in ~/.kaggle/kaggle.json; use the FF++ / DFDC datasets you already have access to
!kaggle datasets download -d <FF++-dataset-slug-on-Kaggle> -p /content/ffpp --unzip     # FaceForensics++ (c23)
!kaggle competitions download -c deepfake-detection-challenge -p /content/dfdc  # DFDC (accept rules first)

# 1) videos -> 224px face crops (MTCNN) + manifest with clip-grouped splits
!cd rg20 && PYTHONPATH=src python -m rg20 prepare --src /content/ffpp --out data/faces --frames-per-video 10 --max-videos 300

# 2a) passive baseline (clean images only)
!cd rg20 && PYTHONPATH=src python -m rg20 train --manifest data/faces/manifest.csv --arch xception --launder-prob 0 --out outputs/xception_passive.pt
# 2b) laundering-aware, dual-stream (clean + watermarked + laundered)
!cd rg20 && PYTHONPATH=src python -m rg20 train --manifest data/faces/manifest.csv --arch xception --launder-prob 0.6 --wm-prob 0.3 --out outputs/xception_robust.pt

# 3) benchmark
!cd rg20 && PYTHONPATH=src python -m rg20 benchmark --manifest data/faces/manifest.csv \
   --detector passive=outputs/xception_passive.pt --detector robust=outputs/xception_robust.pt --max-per-class 300
```
Dataset folder names on Kaggle vary; `prepare` labels a video as *real* if its path contains
`original`/`real`, as *fake* if it contains `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `fake`, etc.,
and reads DFDC labels from `metadata.json`.
