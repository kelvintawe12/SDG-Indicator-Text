"""Class-imbalance remedies for multi-label text.

This is Experiment 5's machinery. We expose three strategies that all
take the (X, Y) training arrays and return a (X', Y', sample_weight)
triple consumable by any sklearn ``fit`` call.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def class_weight_only(X, Y):
    """Control: relies on the head's own class_weight='balanced'."""
    return X, Y, None


def inverse_freq_sample_weight(X, Y, *, eps: float = 1.0):
    """One scalar weight per sample = sum over labels of inverse positive frequency.

    Documents that carry rare labels get up-weighted; documents that
    carry only head labels stay near 1. This is a cheap alternative to
    oversampling: no synthetic rows, no extra memory.
    """
    Y = np.asarray(Y)
    pos = Y.sum(axis=0).astype(float) + eps
    inv = (Y.shape[0] / pos)
    inv /= inv.mean()  # center around 1 so loss scale is comparable
    w = (Y * inv).sum(axis=1)
    # Documents with no positive labels (shouldn't happen on Devex but
    # we defend anyway) get neutral weight.
    w = np.where(w == 0, 1.0, w)
    return X, Y, w


def mlsmote(X, Y, *, k: int = 5, n_synthetic_ratio: float = 0.3, seed: int = 42):
    """A pragmatic, dependency-free MLSMOTE variant.

    The original paper (Charte et al., 2015) operates on dense feature
    vectors. With TF-IDF we either densify (memory-prohibitive) or
    interpolate in sparse space. We do the latter: for each minority
    sample, pick a random one of its k nearest neighbours **among the
    minority pool** and interpolate sparse feature vectors with a
    uniform random weight. The synthetic label vector is the union of
    the two parents' labels.
    """
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    Y = np.asarray(Y)
    n, n_labels = Y.shape
    pos_per_label = Y.sum(axis=0)
    median_pos = float(np.median(pos_per_label[pos_per_label > 0])) if pos_per_label.any() else 0.0
    minority_labels = np.where((pos_per_label < median_pos) & (pos_per_label > 0))[0]
    if minority_labels.size == 0:
        return X, Y, None

    # Sample pool: docs that touch at least one minority label.
    pool_mask = Y[:, minority_labels].any(axis=1)
    pool_idx = np.where(pool_mask)[0]
    if pool_idx.size < 2:
        return X, Y, None

    X_pool = X[pool_idx]
    nn = NearestNeighbors(n_neighbors=min(k + 1, pool_idx.size), metric="cosine")
    nn.fit(X_pool)
    _, neigh = nn.kneighbors(X_pool)

    n_new = int(n_synthetic_ratio * pool_idx.size)
    if n_new == 0:
        return X, Y, None

    parents_a = rng.integers(0, pool_idx.size, size=n_new)
    # Drop self (column 0) when picking a neighbour.
    parents_b_local = neigh[parents_a, rng.integers(1, neigh.shape[1], size=n_new)]
    weights = rng.uniform(0.0, 1.0, size=n_new).astype(np.float32)

    Xa = X_pool[parents_a]
    Xb = X_pool[parents_b_local]
    if sp.issparse(Xa):
        # Sparse element-wise interp: w*Xa + (1-w)*Xb. Diag-mul to scale rows.
        Wa = sp.diags(weights)
        Wb = sp.diags(1.0 - weights)
        X_syn = Wa @ Xa + Wb @ Xb
        X_out = sp.vstack([X, X_syn], format="csr")
    else:
        X_syn = weights[:, None] * np.asarray(Xa) + (1 - weights)[:, None] * np.asarray(Xb)
        X_out = np.vstack([X, X_syn])

    Y_syn = (Y[pool_idx[parents_a]] | Y[pool_idx[parents_b_local]]).astype(Y.dtype)
    Y_out = np.vstack([Y, Y_syn])
    return X_out, Y_out, None


STRATEGIES = {
    "class_weight_only": class_weight_only,
    "inverse_freq_sample_weight": inverse_freq_sample_weight,
    "mlsmote": mlsmote,
}
