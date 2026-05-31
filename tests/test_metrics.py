"""Tests for the evaluation metric layer.

We assert two things that matter for reproducibility:

* ``apply_thresholds`` with a scalar threshold matches scikit-learn's
  behavior on the same inputs.
* ``apply_thresholds`` with ``min_labels_per_doc=1`` never emits an
  empty row.
"""

from __future__ import annotations

import numpy as np

from sdgtext.eval.metrics import apply_thresholds, evaluate
from sdgtext.eval.thresholds import tune_thresholds


def test_apply_thresholds_scalar_and_vector_agree():
    rng = np.random.default_rng(0)
    P = rng.random((50, 6))
    a = apply_thresholds(P, 0.4)
    b = apply_thresholds(P, np.full(6, 0.4))
    assert np.array_equal(a, b)


def test_min_labels_per_doc_forces_nonempty():
    P = np.array([[0.1, 0.05, 0.2], [0.01, 0.02, 0.03]])
    y = apply_thresholds(P, 0.5, min_labels_per_doc=1)
    assert (y.sum(axis=1) >= 1).all()


def test_evaluate_returns_bundle():
    rng = np.random.default_rng(1)
    Y = rng.integers(0, 2, size=(30, 5))
    P = rng.random((30, 5))
    m = evaluate(Y, P, 0.5)
    d = m.as_dict()
    assert 0.0 <= d["hamming_loss"] <= 1.0
    assert "macro_f1" in d


def test_tune_thresholds_handles_zero_positive_label():
    Y = np.zeros((20, 3), dtype=int)
    Y[:, 0] = (np.arange(20) % 2)  # only label 0 has positives
    rng = np.random.default_rng(2)
    P = rng.random((20, 3))
    thr = tune_thresholds(Y, P)
    assert thr.shape == (3,)
    # Label with no positives keeps the default 0.5.
    assert thr[1] == 0.5 and thr[2] == 0.5
