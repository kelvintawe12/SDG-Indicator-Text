"""Multi-label-aware data splitting.

A plain random split tends to leave rare SDG-3 indicators (e.g. 3.b,
3.d) entirely absent from one fold, which both inflates apparent
performance and makes per-label threshold tuning impossible. We use
``MultilabelStratifiedKFold`` / ``MultilabelStratifiedShuffleSplit``
from the iterative-stratification package, which preserves per-label
positive ratios across folds.
"""

from __future__ import annotations

import numpy as np

try:
    from iterstrat.ml_stratifiers import (
        MultilabelStratifiedKFold,
        MultilabelStratifiedShuffleSplit,
    )
    _HAS_ITERSTRAT = True
except ImportError:  # pragma: no cover
    from sklearn.model_selection import KFold, ShuffleSplit
    _HAS_ITERSTRAT = False


def make_kfold(n_splits: int, seed: int):
    """K-fold splitter; multi-label-stratified when possible.

    Returns an object with the sklearn splitter interface (``split``).
    """
    if _HAS_ITERSTRAT:
        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def make_holdout(val_fraction: float, seed: int):
    if _HAS_ITERSTRAT:
        return MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=seed
        )
    return ShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)


def single_holdout_indices(Y: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return (train_idx, val_idx) for a single hold-out."""
    splitter = make_holdout(val_fraction, seed)
    X_dummy = np.zeros((Y.shape[0], 1))
    train_idx, val_idx = next(splitter.split(X_dummy, Y))
    return train_idx, val_idx
