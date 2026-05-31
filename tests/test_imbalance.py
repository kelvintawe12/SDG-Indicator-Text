"""Smoke tests for the imbalance strategies."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sdgtext.models.imbalance import STRATEGIES


def _toy():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.random((40, 10)))
    Y = np.zeros((40, 4), dtype=int)
    Y[:30, 0] = 1               # head label
    Y[:8, 1] = 1                # mid label
    Y[:3, 2] = 1                # tail label (rare)
    Y[:1, 3] = 1                # extreme tail
    return X, Y


def test_inverse_freq_weights_upweight_rare():
    X, Y = _toy()
    _, _, w = STRATEGIES["inverse_freq_sample_weight"](X, Y)
    assert w is not None and w.shape == (40,)
    # Doc 0 carries all four labels; doc 30 carries only the head label.
    assert w[0] > w[30]


def test_mlsmote_grows_sample_count_or_noops():
    X, Y = _toy()
    Xo, Yo, _ = STRATEGIES["mlsmote"](X, Y, n_synthetic_ratio=0.5, seed=0)
    assert Yo.shape[0] >= Y.shape[0]
    assert Xo.shape[0] == Yo.shape[0]
