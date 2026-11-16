"""Face detection / cropping.  MTCNN (facenet-pytorch) if present, else OpenCV Haar."""
from __future__ import annotations

import cv2
import numpy as np


class FaceExtractor:
    def __init__(self, backend: str = "auto", size: int = 224, margin: float = 0.25):
        self.size, self.margin = size, margin
        self.backend = None
        self._mtcnn = None
        self._haar = None
        if backend in ("auto", "mtcnn"):
            try:
                import torch
                from facenet_pytorch import MTCNN
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self._mtcnn = MTCNN(keep_all=False, device=dev, post_process=False)
                self.backend = "mtcnn"
            except Exception:
                if backend == "mtcnn":
                    raise
        if self.backend is None and backend in ("auto", "haar"):
            self._haar = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.backend = "haar"
        if self.backend is None:
            self.backend = "full"          # use the whole frame

    def detect(self, bgr: np.ndarray):
        """Return (x1, y1, x2, y2) of the largest face or None."""
        if self.backend == "mtcnn":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            boxes, probs = self._mtcnn.detect(rgb)
            if boxes is None:
                return None
            i = int(np.argmax(probs))
            return tuple(int(v) for v in boxes[i])
        if self.backend == "haar":
            g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            f = self._haar.detectMultiScale(g, 1.1, 5, minSize=(40, 40))
            if len(f) == 0:
                return None
            x, y, w, h = max(f, key=lambda b: b[2] * b[3])
            return int(x), int(y), int(x + w), int(y + h)
        h, w = bgr.shape[:2]
        return 0, 0, w, h

    def crop(self, bgr: np.ndarray, fallback_full: bool = True):
        """Return (face_crop_resized, box).  Falls back to the full frame."""
        h, w = bgr.shape[:2]
        box = self.detect(bgr)
        if box is None:
            if not fallback_full:
                return None, None
            box = (0, 0, w, h)
        x1, y1, x2, y2 = box
        mx, my = int((x2 - x1) * self.margin), int((y2 - y1) * self.margin)
        x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)
        face = bgr[y1:y2, x1:x2]
        if face.size == 0:
            face = bgr
        return cv2.resize(face, (self.size, self.size), interpolation=cv2.INTER_AREA), box
