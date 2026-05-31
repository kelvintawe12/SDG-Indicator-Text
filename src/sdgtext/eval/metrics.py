"""Evaluation metrics for multi-label classification.

The assignment is evaluated on Hamming Loss (lower is better). We
report Hamming Loss as the primary metric and a battery of secondary
metrics that triangulate model behaviour:

* micro-F1 — pools TP/FP/FN across all labels; dominated by head labels.
* macro-F1 — unweighted mean F1 over labels; sensitive to tail labels.
* samples-F1 — per-document F1, averaged across documents.
* subset accuracy — strict exact-set match; brutal but cited often.
* LRAP (label ranking average precision) — uses probabilities, not
  thresholded predictions; isolates ranking quality from threshold
  choice.

All functions accept either thresholded ``y_pred`` (0/1) or raw
probabilities ``y_prob`` and apply the supplied thresholds when needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    label_ranking_average_precision_score,
    precision_score,
    recall_score,
)


@dataclass
class MetricBundle:
    hamming_loss: float
    micro_f1: float
    macro_f1: float
    samples_f1: float
    subset_accuracy: float
    lrap: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "hamming_loss": self.hamming_loss,
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "samples_f1": self.samples_f1,
            "subset_accuracy": self.subset_accuracy,
            "lrap": self.lrap,
        }


def apply_thresholds(
    y_prob: np.ndarray, thresholds: float | np.ndarray, min_labels_per_doc: int = 0
) -> np.ndarray:
    """Binarize a probability matrix.

    If ``min_labels_per_doc`` is set, any row whose max probability
    falls below its highest threshold still emits its top-1 label. The
    rationale: every Devex document is by construction an SDG-3 text,
    so an empty prediction is structurally implausible.
    """
    y_prob = np.asarray(y_prob)
    if np.ndim(thresholds) == 0:
        thr = np.full(y_prob.shape[1], float(thresholds))
    else:
        thr = np.asarray(thresholds, dtype=float)
        if thr.shape[0] != y_prob.shape[1]:
            raise ValueError(
                f"thresholds shape {thr.shape} doesn't match n_labels {y_prob.shape[1]}"
            )
    y_pred = (y_prob >= thr[None, :]).astype(int)
    if min_labels_per_doc > 0:
        empty_rows = y_pred.sum(axis=1) < min_labels_per_doc
        if empty_rows.any():
            # Force the row's top-k labels on by probability (no
            # thresholding) — this preserves ranking-quality signal
            # for nearly-empty rows.
            for i in np.where(empty_rows)[0]:
                top = np.argsort(-y_prob[i])[:min_labels_per_doc]
                y_pred[i, top] = 1
    return y_pred


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, thresholds, min_labels_per_doc: int = 0) -> MetricBundle:
    """Compute the full metric bundle."""
    y_pred = apply_thresholds(y_prob, thresholds, min_labels_per_doc)
    return MetricBundle(
        hamming_loss=float(hamming_loss(y_true, y_pred)),
        micro_f1=float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        samples_f1=float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        subset_accuracy=float((y_true == y_pred).all(axis=1).mean()),
        lrap=float(label_ranking_average_precision_score(y_true, y_prob))
        if y_true.sum() > 0
        else None,
    )


def per_label_report(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> dict:
    """Per-label precision/recall/F1/support. Used to populate the
    discussion section's tail-vs-head analysis."""
    out = {}
    for j, name in enumerate(label_names):
        yt, yp = y_true[:, j], y_pred[:, j]
        out[name] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
        }
    return out
