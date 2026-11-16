from __future__ import annotations
import cv2
import numpy as np


def tamper_overlay(bgr: np.ndarray, tamper_map: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Heat-map of watermark disagreement; flagged regions outlined in red."""
    h, w = bgr.shape[:2]
    heat = cv2.resize(np.clip(tamper_map, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 255), max(2, w // 200))
    return out
