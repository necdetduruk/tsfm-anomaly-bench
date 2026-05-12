"""Anomaly detection metrics for SMD-style evaluation.

We report three metrics deliberately:

1. **AUC-PR (average precision)**. Threshold-free, robust. The right metric
   for class-imbalanced anomaly detection.
2. **Best raw F1**. F1 sweeping all thresholds. The strict "did the model
   flag the right timesteps" measure.
3. **Best point-adjusted F1 (PA-F1)**. The classic SMD/OmniAnomaly metric:
   if any timestep in a contiguous true-anomaly segment is flagged, the
   whole segment is counted as detected. PA-F1 is widely reported but
   inflates results — Garg et al. 2021 showed a random scorer can score
   well under PA-F1. We report it for comparability with the literature
   and discuss the gap in the writeup.

All "best F1" metrics use threshold selection on the test set itself, which
is the SMD-paper convention. Tighter protocols (val-set threshold) are an
extension we can run later.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def best_f1_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    """Find threshold maximizing F1.

    Returns (best_f1, precision_at_best, recall_at_best, threshold).
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # PR curve has one extra trailing point with no threshold
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.0, 0.0, 0.0, 0.0
    best_idx = int(np.nanargmax(f1))
    return (
        float(f1[best_idx]),
        float(precision[best_idx]),
        float(recall[best_idx]),
        float(thresholds[best_idx]),
    )


def apply_point_adjustment(
    predictions: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    """If any timestep in a contiguous true-anomaly segment is predicted positive,
    mark the entire segment as predicted positive. Standard SMD trick.
    """
    pred = predictions.copy().astype(np.int64)
    in_seg = False
    seg_start = 0
    for i, lab in enumerate(labels):
        if lab == 1 and not in_seg:
            seg_start = i
            in_seg = True
        elif lab == 0 and in_seg:
            if pred[seg_start:i].any():
                pred[seg_start:i] = 1
            in_seg = False
    if in_seg and pred[seg_start:].any():
        pred[seg_start:] = 1
    return pred


def best_point_adjusted_f1(
    scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 400
) -> tuple[float, float, float, float]:
    """Sweep thresholds, apply point-adjustment, return best-F1 stats."""
    qs = np.linspace(0.0, 1.0, n_thresholds + 1)[1:-1]
    thresholds = np.unique(np.quantile(scores, qs))

    best_f1 = 0.0
    best_p, best_r, best_t = 0.0, 0.0, 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int64)
        pred_pa = apply_point_adjustment(pred, labels)
        tp = int(((pred_pa == 1) & (labels == 1)).sum())
        fp = int(((pred_pa == 1) & (labels == 0)).sum())
        fn = int(((pred_pa == 0) & (labels == 1)).sum())
        if tp == 0:
            continue
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r + 1e-12)
        if f1 > best_f1:
            best_f1, best_p, best_r, best_t = f1, p, r, float(t)
    return best_f1, best_p, best_r, best_t


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute all three metric families. Returns flat dict for logging."""
    if labels.sum() == 0:
        # No anomalies in test — degenerate, skip
        return {
            "auc_pr": float("nan"),
            "best_f1": float("nan"),
            "best_f1_precision": float("nan"),
            "best_f1_recall": float("nan"),
            "best_f1_threshold": float("nan"),
            "pa_best_f1": float("nan"),
            "pa_best_f1_precision": float("nan"),
            "pa_best_f1_recall": float("nan"),
            "pa_best_f1_threshold": float("nan"),
            "anomaly_rate": 0.0,
        }

    auc_pr = float(average_precision_score(labels, scores))
    f1, p, r, t = best_f1_threshold(scores, labels)
    pa_f1, pa_p, pa_r, pa_t = best_point_adjusted_f1(scores, labels)

    return {
        "auc_pr": auc_pr,
        "best_f1": f1,
        "best_f1_precision": p,
        "best_f1_recall": r,
        "best_f1_threshold": t,
        "pa_best_f1": pa_f1,
        "pa_best_f1_precision": pa_p,
        "pa_best_f1_recall": pa_r,
        "pa_best_f1_threshold": pa_t,
        "anomaly_rate": float(labels.mean()),
    }
