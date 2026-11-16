from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def classification_metrics(y_true, prob_fake, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob_fake, dtype=float)
    pred = (p >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "n": int(len(y)),
    }
    return {k: float(v) if k != "n" else v for k, v in out.items()}


def roc(y_true, prob_fake):
    return roc_curve(np.asarray(y_true).astype(int), np.asarray(prob_fake, dtype=float))
