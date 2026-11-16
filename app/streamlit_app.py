"""RG20 - Deepfake & Tamper Detection web app.   Run:  streamlit run app/streamlit_app.py"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rg20 import laundering                                    # noqa: E402
from rg20.config import load_config                            # noqa: E402
from rg20.detectors import load_detector                       # noqa: E402
from rg20.faces import FaceExtractor                           # noqa: E402
from rg20.pipeline import (AUTHENTIC, DEEPFAKE, TAMPERED, UNVERIFIED,   # noqa: E402
                           AuthenticationPipeline)
from rg20.video import iter_frames, verify_video, watermark_video      # noqa: E402
from rg20.viz import tamper_overlay                            # noqa: E402
from rg20.watermark import PROFILES, DCTWatermarker, WatermarkParams, psnr, ssim  # noqa: E402

st.set_page_config(page_title="RG20 Deepfake & Tamper Detection", page_icon="🛡️", layout="wide")
CFG = load_config()
COLORS = {AUTHENTIC: "#1b8a3b", TAMPERED: "#d97706", DEEPFAKE: "#c62828", UNVERIFIED: "#5f6b7a"}
ICONS = {AUTHENTIC: "✅", TAMPERED: "⚠️", DEEPFAKE: "🚨", UNVERIFIED: "❔"}


@st.cache_resource
def get_detector(path: str):
    return load_detector(path) if path else None


@st.cache_resource
def get_faces():
    return FaceExtractor("auto")


def to_bgr(file) -> np.ndarray:
    arr = np.frombuffer(file.getvalue(), np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def show(img_bgr, caption=None):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


def banner(status: str, text: str):
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{COLORS[status]};color:white'>"
        f"<span style='font-size:1.5rem;font-weight:700'>{ICONS[status]} {status}</span><br>{text}</div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ RG20")
st.sidebar.caption("Watermark authentication + deepfake classification with laundering robustness analysis")
key = st.sidebar.text_input("Secret watermark key", value="rg20-demo-key", type="password",
                            help="Only the holder of this key can create or verify valid watermarks.")
profile = st.sidebar.select_slider("Watermark strength", options=list(PROFILES), value=CFG["watermark"]["profile"],
                                   help="stealth = least visible; robust = survives heavier laundering")
cands = sorted([*(ROOT / "outputs").glob("*.pt"), *(ROOT / "outputs").glob("*.joblib")])
choice = st.sidebar.selectbox("Deepfake detector", ["(none - watermark only)"] + [p.name for p in cands])
det_path = str(ROOT / "outputs" / choice) if choice in [p.name for p in cands] else None
threshold = st.sidebar.slider("Fake decision threshold", 0.05, 0.95, float(CFG["detector"]["fake_threshold"]), 0.05)

params = WatermarkParams.from_config({**CFG, "watermark": {**CFG["watermark"], "profile": profile}})
wm = DCTWatermarker(key, params)
detector = get_detector(det_path)
pipe = AuthenticationPipeline(wm, detector, get_faces() if detector else None, threshold)

st.title("Deepfake & Tamper Detection")
st.caption("Layer 1: invisible watermark verification  →  Layer 2: CNN/forensic deepfake classification")
tab_verify, tab_protect, tab_lab, tab_about = st.tabs(
    ["🔎 Verify media", "🔏 Protect (watermark)", "🧪 Laundering lab", "ℹ️ About"])

# ------------------------------------------------------------------ verify
with tab_verify:
    up = st.file_uploader("Upload an image or video", type=["png", "jpg", "jpeg", "bmp", "mp4", "avi", "mov", "webm"],
                          key="verify")
    if up is not None:
        if up.type.startswith("video") or Path(up.name).suffix.lower() in {".mp4", ".avi", ".mov", ".webm"}:
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tf:
                tf.write(up.getvalue())
            with st.spinner("Checking sampled frames…"):
                r = verify_video(tf.name, wm)
            status = AUTHENTIC if r["verified"] and r["tampered_fraction"] < 0.3 else \
                (TAMPERED if r["verified"] else UNVERIFIED)
            if status == UNVERIFIED and detector is not None:
                probs = [float(detector.predict_proba([get_faces().crop(f)[0]])[0])
                         for _, f in iter_frames(tf.name, 3, 12)]
                if probs and np.mean(probs) >= threshold:
                    status = DEEPFAKE
                st.metric("Mean fake probability (sampled frames)", f"{np.mean(probs):.2f}")
            banner(status, f"{r['frames_checked']} frames checked, {r['verified_fraction']:.0%} carry a valid watermark.")
            st.json({k: v for k, v in r.items() if k != "first"})
            st.video(up)
        else:
            img = to_bgr(up)
            rep = pipe.analyze(img)
            banner(rep.status, rep.explanation)
            c1, c2 = st.columns(2)
            with c1:
                show(img, "Uploaded media")
            with c2:
                if rep.wm.verified:
                    show(tamper_overlay(img, rep.wm.tamper_map, rep.wm.tamper_mask), "Tamper heat-map (red = altered)")
                else:
                    st.info("No verifiable watermark, so no tamper map is available.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Watermark tag match", f"{rep.wm.tag_score:.0%}", help="50% = chance level")
            m2.metric("Media ID", rep.wm.media_id if rep.wm.verified else "—")
            m3.metric("Tampered area", f"{rep.wm.tamper_fraction:.1%}" if rep.wm.verified else "—")
            m4.metric("Fake probability", f"{rep.fake_prob:.2f}" if rep.fake_prob is not None else "n/a")
            for n in rep.notes:
                st.write("•", n)
            lsb = rep.wm.extra["lsb"]
            st.caption(f"Fragile pixel-integrity layer: **{lsb['state']}** "
                       f"(intact = lossless original; absent = re-encoded or unprotected)")

# ------------------------------------------------------------------ protect
with tab_protect:
    st.write("Embed an invisible, keyed watermark **before** distributing authentic media.")
    up = st.file_uploader("Authentic image or video", type=["png", "jpg", "jpeg", "bmp", "mp4", "mov", "avi"], key="protect")
    media_id = st.number_input("Media ID (32-bit)", 0, 2 ** 32 - 1, 1)
    if up is not None and st.button("Embed watermark"):
        if Path(up.name).suffix.lower() in {".mp4", ".mov", ".avi"}:
            with tempfile.TemporaryDirectory() as d:
                src, dst = Path(d, "in" + Path(up.name).suffix), Path(d, "protected.mp4")
                src.write_bytes(up.getvalue())
                with st.spinner("Watermarking video frames…"):
                    watermark_video(src, dst, wm, int(media_id))
                st.download_button("Download protected video", dst.read_bytes(), "protected.mp4", "video/mp4")
        else:
            img = to_bgr(up)
            try:
                out = wm.embed(img, int(media_id))
            except ValueError as e:
                st.error(str(e))
            else:
                c1, c2 = st.columns(2)
                with c1:
                    show(img, "Original")
                with c2:
                    show(out, f"Protected  (PSNR {psnr(img, out):.1f} dB, SSIM {ssim(img, out):.3f})")
                ok, buf = cv2.imencode(".png", out)
                st.download_button("Download protected PNG", buf.tobytes(), "protected.png", "image/png")
                st.caption("Save/share as PNG to keep the fragile integrity layer; the robust layer also survives JPEG/H.264.")

# ------------------------------------------------------------------ laundering lab
with tab_lab:
    st.write("Simulate social-media laundering and see whether the watermark and detector survive.")
    up = st.file_uploader("Image (ideally one you protected in the previous tab)", type=["png", "jpg", "jpeg"], key="lab")
    c1, c2 = st.columns(2)
    atk = c1.selectbox("Attack", list(laundering.ATTACKS))
    lvl = c2.select_slider("Intensity", [1, 2, 3], value=2, format_func=lambda v: {1: "light", 2: "medium", 3: "heavy"}[v])
    if up is not None:
        img = to_bgr(up)
        out = laundering.apply(img, atk, lvl)
        before, after = pipe.analyze(img), pipe.analyze(out)
        a, b = st.columns(2)
        with a:
            show(img, f"Before — {before.status}")
            st.write(f"tag match {before.wm.tag_score:.0%}")
        with b:
            show(out, f"After {atk} (level {lvl}) — {after.status}")
            st.write(f"tag match {after.wm.tag_score:.0%}")
        if before.wm.verified and not after.wm.verified:
            st.warning("The watermark did not survive this attack — try the 'robust' strength or a lighter attack.")
        elif after.wm.verified:
            st.success("Watermark survived the laundering.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Pipeline**

1. **Watermark layer** – a 64-bit payload (media ID + HMAC tag) is embedded in block-DCT coefficients (QIM, keyed
   block selection, soft-vote decoding). Verified ⇒ *authentic*. Blocks that disagree with the surrounding
   watermark reveal *tampered* regions.
2. **Detector layer** – media without a valid watermark goes to the CNN (ResNet / XceptionNet) or the
   forensic-feature fallback, trained on clean, watermarked and laundered samples.
3. **Outcomes** – ✅ Authentic · ⚠️ Tampered · 🚨 Deepfake · ❔ Unverified (provenance unknown).

**Limitations** – cropping/rotation breaks block alignment; extreme laundering can erase the watermark;
an unverified result is *not* proof of a fake.
""")
