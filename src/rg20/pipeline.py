"""Two-layer authentication pipeline (Phase 1 -> Phase 3).

  1. Verify invisible watermark.  Verified (+ no localised tamper) -> AUTHENTIC.
  2. Otherwise flag and pass the face crop to the deepfake classifier.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .watermark import DCTWatermarker, VerifyResult

AUTHENTIC, TAMPERED, DEEPFAKE, UNVERIFIED = "AUTHENTIC", "TAMPERED", "DEEPFAKE", "UNVERIFIED"

EXPLANATIONS = {
    AUTHENTIC: "A valid watermark signed with your key was found and no region disagrees with it.",
    TAMPERED: "The watermark is present (or partly present) but parts of the media were altered.",
    DEEPFAKE: "No valid watermark and the deepfake classifier scored the media as synthetic.",
    UNVERIFIED: "No valid watermark was found; the classifier found no sign of manipulation, "
                "so provenance is unknown rather than proven.",
}


@dataclass
class Report:
    status: str
    explanation: str
    wm: VerifyResult
    fake_prob: float | None
    face_box: tuple | None = None
    detector: str | None = None
    notes: list[str] = field(default_factory=list)


class AuthenticationPipeline:
    def __init__(self, watermarker: DCTWatermarker, detector=None, face_extractor=None,
                 fake_threshold: float = 0.5):
        self.wm, self.detector, self.faces = watermarker, detector, face_extractor
        self.fake_threshold = fake_threshold

    def analyze(self, bgr: np.ndarray) -> Report:
        wmr = self.wm.verify(bgr)
        notes: list[str] = []
        fake_prob, box = None, None
        if self.detector is not None:
            crop, box = (self.faces.crop(bgr) if self.faces is not None else (bgr, None))
            fake_prob = float(self.detector.predict_proba([crop])[0])
        lsb = wmr.extra.get("lsb", {})

        if wmr.verified:
            if wmr.tampered:
                status = TAMPERED
                notes.append(f"{wmr.tamper_fraction:.1%} of the image disagrees with the watermark.")
            elif lsb.get("state") == "localised":
                status = TAMPERED
                notes.append("Pixel-exact integrity layer shows localised modifications.")
            else:
                status = AUTHENTIC
                if lsb.get("state") == "absent":
                    notes.append("Media was re-encoded (laundered) but the watermark survived.")
        else:
            if fake_prob is not None and fake_prob >= self.fake_threshold:
                status = DEEPFAKE
            elif wmr.suspicious:
                status = TAMPERED
                notes.append("Watermark traces were found but are too damaged to verify.")
            else:
                status = UNVERIFIED
            if self.detector is None:
                notes.append("No deepfake classifier loaded - only the watermark layer was used.")
        return Report(status, EXPLANATIONS[status], wmr, fake_prob, box,
                      getattr(self.detector, "kind", None), notes)
