"""Per-label threshold optimization on validation probabilities.

For each label we search a 1-D grid for the cutoff that maximises a
chosen objective (F1 by default). The motivation, in full:

* Hamming Loss with imbalanced labels has a degenerate per-label
  optimum at threshold=1.0 for the rarest labels (predict-nothing
  beats predict-anything in the worst case). Optimising F1 per label
  and then evaluating Hamming Loss on the full prediction matrix
  avoids this degeneracy while still producing a thresholded matrix
  with lower Hamming Loss than the global-0.5 baseline.
* The search grid is small (13 values) and per-label, so the total
  cost is 13 × n_labels F1 computations on the validation matrix —
  cheap.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    grid=(0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60),
    objective: str = "f1",
) -> np.ndarray:
    """Return one threshold per label.

    Parameters
    ----------
    y_true, y_prob : arrays of shape (n_samples, n_labels)
    grid : iterable of candidate thresholds.
    objective : "f1" (default), "youden", or "accuracy".

    Notes
    -----
    A label with zero positives in ``y_true`` returns 0.5 — we have no
    signal to tune on, so we don't pretend otherwise. The notebook
    surfaces these labels in the audit table.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n_labels = y_true.shape[1]
    thr = np.full(n_labels, 0.5, dtype=float)
    grid_arr = np.asarray(list(grid), dtype=float)

    for j in range(n_labels):
        yt = y_true[:, j]
        if yt.sum() == 0:
            continue
        yp = y_prob[:, j]
        best_score, best_t = -1.0, 0.5
        for t in grid_arr:
            pred = (yp >= t).astype(int)
            if objective == "f1":
                s = f1_score(yt, pred, zero_division=0)
            elif objective == "accuracy":
                s = float((pred == yt).mean())
            elif objective == "youden":
                tp = ((pred == 1) & (yt == 1)).sum()
                tn = ((pred == 0) & (yt == 0)).sum()
                fp = ((pred == 1) & (yt == 0)).sum()
                fn = ((pred == 0) & (yt == 1)).sum()
                tpr = tp / max(tp + fn, 1)
                fpr = fp / max(fp + tn, 1)
                s = tpr - fpr
            else:
                raise ValueError(f"Unknown objective: {objective!r}")
            if s > best_score:
                best_score, best_t = s, float(t)
        thr[j] = best_t
    return thr
