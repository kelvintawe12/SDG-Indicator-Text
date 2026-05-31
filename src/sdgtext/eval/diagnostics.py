"""Confusion-matrix grids and learning-curve plots.

These visualizations are called out explicitly by the assignment
rubric ("confusion matrices or classification visualizations" +
"learning curves where applicable"). We keep them as pure functions
operating on already-computed predictions / probabilities so the
notebook and ``scripts/run_all_experiments.py`` can drop them in
without re-fitting any model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import hamming_loss, multilabel_confusion_matrix


def plot_confusion_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    *,
    n_cols: int = 6,
    out_path: str | Path | None = None,
    title: str = "Per-label confusion matrices",
) -> plt.Figure:
    """Render a grid of per-label confusion matrices.

    Each cell is a 2x2 confusion for one SDG-3 indicator. We annotate
    raw counts (not row-normalized) because rare-label TPs are the
    interesting signal — normalising would hide them.
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    n = len(label_names)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.0))
    axes = np.array(axes).reshape(-1)

    for j, (cm, name) in enumerate(zip(mcm, label_names)):
        ax = axes[j]
        sns.heatmap(
            cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
            xticklabels=["pred 0", "pred 1"], yticklabels=["true 0", "true 1"],
            annot_kws={"size": 8},
        )
        pretty = name.replace("lbl__", "")
        support = int(cm[1].sum())
        ax.set_title(f"{pretty} (n+={support})", fontsize=9)
        ax.tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13, y=1.005)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def compute_learning_curve(
    fit_and_predict_fn,
    X,
    Y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed: int = 42,
) -> dict:
    """Compute a Hamming-Loss learning curve.

    Parameters
    ----------
    fit_and_predict_fn : callable
        Takes (X_train, Y_train, X_val) and returns Y_val probabilities.
        Caller controls the model / features.
    X : feature matrix (n_total, n_feat) — sparse or dense.
    Y : (n_total, n_labels) integer multi-hot.
    train_idx, val_idx : index arrays for the split to subsample.
    fractions : training-set fractions to evaluate at.
    seed : RNG seed for the subsample shuffle.

    Returns
    -------
    dict with keys ``fractions``, ``train_sizes``, ``hamming_train``,
    ``hamming_val``. Train Hamming is computed on the *subsampled*
    training set; val Hamming is computed on the full val_idx.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_total = len(shuffled)

    rec = {"fractions": [], "train_sizes": [], "hamming_train": [], "hamming_val": []}
    for frac in fractions:
        k = max(2, int(frac * n_total))
        sub = shuffled[:k]
        P_tr = fit_and_predict_fn(X[sub], Y[sub], X[sub])
        P_va = fit_and_predict_fn(X[sub], Y[sub], X[val_idx])
        # 0.5 threshold here is appropriate — we're measuring fit
        # capacity, not the final decision layer.
        yhat_tr = (P_tr >= 0.5).astype(int)
        yhat_va = (P_va >= 0.5).astype(int)
        rec["fractions"].append(float(frac))
        rec["train_sizes"].append(int(k))
        rec["hamming_train"].append(float(hamming_loss(Y[sub], yhat_tr)))
        rec["hamming_val"].append(float(hamming_loss(Y[val_idx], yhat_va)))
    return rec


def plot_learning_curve(rec: dict, *, out_path: str | Path | None = None, title: str = "Learning curve — Hamming Loss") -> plt.Figure:
    """Plot the (train_size → Hamming) curve produced by
    :func:`compute_learning_curve`."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rec["train_sizes"], rec["hamming_train"], marker="o", label="Train Hamming Loss")
    ax.plot(rec["train_sizes"], rec["hamming_val"], marker="s", label="Validation Hamming Loss")
    ax.set_xlabel("Training documents")
    ax.set_ylabel("Hamming Loss (↓)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig
