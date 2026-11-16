# RG20 — Deepfake & Tamper Detection

**MP2 presentation documentation**

This README describes the project scope and evidence to present at MP2. It is intentionally limited to the work demonstrated for that evaluation; planned or subsequently developed extensions should not be described as MP2-completed work.

## Project overview

RG20 explores a two-stage approach to media authenticity:

1. **Media provenance / watermark verification:** embed a keyed, invisible watermark in an image and verify the watermark when the image is submitted.
2. **Deepfake screening:** when provenance cannot be verified, an available classifier can provide an additional prediction. An unverified watermark is not, by itself, proof that media is fake.

The implementation uses Python, OpenCV and NumPy for image handling and watermark processing. The repository also contains supporting modules and experiments; only features that were implemented, run and demonstrated by the MP2 evaluation should be included in the MP2 presentation and results.

## MP2 presentation scope

Use this section as the presentation checklist. Mark a feature as completed only if your team can demonstrate it on the submitted build.

- [ ] Image upload and input validation
- [ ] Image preprocessing
- [ ] Face detection / extraction (if included in the demonstrated build)
- [ ] Initial deepfake classifier inference (only if trained weights and inference were actually tested)
- [ ] Initial evaluation on a documented dataset
- [ ] Basic user workflow

Do not report model accuracy, dataset performance, watermark robustness, or frontend functionality unless the corresponding experiment was run and its outputs can be reproduced. Procedural/synthetic images are useful for software tests, but are not a substitute for FaceForensics++ or DFDC evaluation.

## Repository layout

| Path | Role |
|---|---|
| `src/rg20/` | Python package containing image, watermark, pipeline and supporting modules |
| `app/streamlit_app.py` | Streamlit interface |
| `tests/` | Automated regression tests |
| `scripts/` | Demo and experiment scripts |
| `outputs/` | Generated local experiment outputs (not ground-truth evidence unless documented) |

The presence of a source module does not imply that it was completed or evaluated by MP2. Explain only the parts your team can run and defend.

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install the dependencies listed by the project if your checkout contains its dependency file. The core test suite requires OpenCV, NumPy, SciPy, scikit-image, scikit-learn, pandas, PyYAML, joblib and pytest.

For Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scipy scikit-image scikit-learn pandas pyyaml joblib pytest
$env:PYTHONPATH = "src"
pytest -q
```

For macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scipy scikit-image scikit-learn pandas pyyaml joblib pytest
PYTHONPATH=src pytest -q
```

The optional video tests require FFmpeg. If FFmpeg is unavailable, those tests are skipped by the existing video test module.

## Run the interface

If Streamlit and all application dependencies are installed:

```bash
streamlit run app/streamlit_app.py
```

For the MP2 demonstration, use only the tabs and functions that were completed and tested by that milestone. Do not imply that an unfinished detector has been trained merely because a model architecture exists in the repository.

## Testing

Run the automated suite from the repository root:

```bash
PYTHONPATH=src pytest -v
```

The extended regression suite checks public software contracts including:

- watermark payload and image round-trips
- profile and key separation behaviour
- image geometry and laundering output contracts
- deterministic synthetic fixture generation
- face crop output shape
- classification metric ranges
- visualisation output shape
- pipeline decisions with controlled detector predictions
- invalid input handling

These are software regression tests. Passing them does not establish real-world deepfake detection accuracy or robustness against all laundering attacks. Report the actual test count and pass/skip/fail totals from the latest run rather than a planned target.

## Results and limitations

Record only measured results from reproducible runs. For each experiment, note:

- dataset and version
- train / validation / test split method
- model checkpoint and configuration
- hardware and software environment
- metric definitions and measured values

Synthetic procedural data can validate pipeline behaviour, but must not be presented as real-world deepfake benchmark performance. A classifier's output is a probability estimate, not definitive proof of manipulation. An unverified watermark means provenance could not be established; it does not automatically mean the media is fake.

## Future work

List unfinished or post-MP2 work here in the report or presentation, with its status clearly marked as planned / in progress / completed after MP2. Do not backdate later implementation or test results to the MP2 milestone.
