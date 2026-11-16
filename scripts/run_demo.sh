#!/usr/bin/env bash
# End-to-end demo with the procedural dataset (no downloads, CPU only, ~6 minutes).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python -m rg20 synthetic --n-real 450 --n-fake 450
python -m rg20 train-classical --manifest data/synthetic/manifest.csv --out outputs/classical_passive.joblib --launder-prob 0 --copies 1
python -m rg20 train-classical --manifest data/synthetic/manifest.csv --out outputs/classical_robust.joblib --launder-prob 0.6 --copies 3
python -m rg20 benchmark --manifest data/synthetic/manifest.csv \
  --detector passive_trained=outputs/classical_passive.joblib \
  --detector laundering_trained=outputs/classical_robust.joblib --max-per-class 40
# watermark + verify + launder a single image
python -m rg20 embed data/synthetic/real/clip0000_000.png outputs/protected.png --media-id 1001
python -m rg20 launder outputs/protected.png outputs/protected_jpeg.jpg --attack jpeg --level 2
python -m rg20 verify outputs/protected_jpeg.jpg --detector outputs/classical_robust.joblib
echo "Now run:  streamlit run app/streamlit_app.py"
