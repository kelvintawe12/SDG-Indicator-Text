"""Multi-label classifier heads.

Three principled choices for this dataset size and feature shape:

* ``logreg_ovr`` — L2 logistic regression in a One-vs-Rest wrapper.
  Calibrated probabilities out of the box, strong with sparse text.
* ``linsvc_ovr`` — Linear SVC, calibrated via sigmoid for probabilities.
  Slightly stronger on margins but slower to calibrate with CV.
* ``complement_nb`` — Complement Naive Bayes. Built for imbalanced
  text. We wrap in OvR for multi-label support.

All three expose ``.fit`` / ``.predict_proba`` so the eval and ensemble
layers don't care which is plugged in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC


@dataclass
class HeadSpec:
    name: str
    params: dict[str, Any]


def build_head(spec: str, params: dict[str, Any] | None = None, *, seed: int = 42):
    """Factory returning an unfitted multi-label classifier.

    Parameters
    ----------
    spec : str
        One of {"logreg_ovr", "linsvc_ovr", "complement_nb", "sgd_logloss"}.
    params : dict
        Per-head hyperparameter overrides from the YAML config.
    seed : int
        Forwarded to every estimator that accepts ``random_state``.
    """
    params = params or {}
    if spec == "logreg_ovr":
        base = LogisticRegression(
            C=params.get("C", 4.0),
            solver=params.get("solver", "liblinear"),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 2000),
            random_state=seed,
        )
        return OneVsRestClassifier(base, n_jobs=params.get("n_jobs", -1))

    if spec == "linsvc_ovr":
        # CalibratedClassifierCV wraps LinearSVC so we get predict_proba.
        # Inner CV is small (3 folds) since we calibrate per-label and
        # the data is modest.
        base = LinearSVC(
            C=params.get("C", 1.0),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 3000),
            random_state=seed,
        )
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        return OneVsRestClassifier(calibrated, n_jobs=params.get("n_jobs", -1))

    if spec == "complement_nb":
        base = ComplementNB(alpha=params.get("alpha", 0.3), norm=params.get("norm", False))
        return OneVsRestClassifier(base, n_jobs=params.get("n_jobs", -1))

    if spec == "sgd_logloss":
        # Faster alternative for very large feature dims; same loss as
        # logistic regression but with online updates.
        base = SGDClassifier(
            loss="log_loss",
            alpha=params.get("alpha", 1e-5),
            class_weight=params.get("class_weight", "balanced"),
            max_iter=params.get("max_iter", 50),
            random_state=seed,
        )
        return OneVsRestClassifier(base, n_jobs=params.get("n_jobs", -1))

    raise ValueError(f"Unknown head spec: {spec!r}")


def predict_proba_safe(model, X) -> np.ndarray:
    """Always return an (n_samples, n_labels) probability matrix.

    Some scikit-learn estimators (e.g. uncalibrated LinearSVC) only have
    ``decision_function``. We fall back to a sigmoid of the decision
    scores in that case so the downstream threshold-tuning code can be
    classifier-agnostic.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        # Per-column min-max → sigmoid keeps relative ordering and
        # produces well-behaved [0,1] outputs for thresholding.
        return 1.0 / (1.0 + np.exp(-scores))
    raise AttributeError(f"{type(model).__name__} has neither predict_proba nor decision_function.")
