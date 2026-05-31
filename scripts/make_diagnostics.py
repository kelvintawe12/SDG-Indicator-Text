"""Generate the two rubric-required visualizations that aren't part of
the experiment loop: per-label confusion matrix grid (for the winning
model) and a learning curve on the winning feature set.

Outputs
-------
* ``reports/figures/confusion_grid_baseline.png`` — Exp 1 baseline.
* ``reports/figures/confusion_grid_best.png`` — Exp 8 ensemble.
* ``reports/figures/learning_curve.png`` — Exp 3 representation
  (TF-IDF word + char) since it's the strongest CPU-cheap model we can
  re-fit at multiple training fractions without re-encoding SBERT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdgtext.data.load import load_devex  # noqa: E402
from sdgtext.data.preprocess import PreprocessConfig, normalize_corpus  # noqa: E402
from sdgtext.data.splits import single_holdout_indices  # noqa: E402
from sdgtext.eval.diagnostics import (  # noqa: E402
    compute_learning_curve,
    plot_confusion_grid,
    plot_learning_curve,
)
from sdgtext.eval.metrics import apply_thresholds  # noqa: E402
from sdgtext.eval.thresholds import tune_thresholds  # noqa: E402
from sdgtext.features.combine import stack_features  # noqa: E402
from sdgtext.features.embeddings import encode_sbert  # noqa: E402
from sdgtext.features.tfidf import make_char_tfidf, make_word_tfidf  # noqa: E402
from sdgtext.models.heads import build_head, predict_proba_safe  # noqa: E402
from sdgtext.utils.config import load_config  # noqa: E402
from sdgtext.utils.logging import get_logger  # noqa: E402
from sdgtext.utils.seeding import seed_everything  # noqa: E402

log = get_logger("diagnostics")
seed_everything(42)
FIGS = ROOT / "reports" / "figures"


def _load_split():
    cfg = load_config("default.yaml")
    ds, _ = load_devex(
        path=cfg["data"]["raw_train"],
        text_columns=cfg["data"]["text_columns"],
        id_column=cfg["data"].get("id_column"),
        meta_columns=cfg["data"].get("meta_columns", []),
        label_format=cfg["data"].get("label_format"),
        label_column_prefix=cfg["data"].get("label_column_prefix"),
        label_code_regex=cfg["data"].get("label_code_regex"),
    )
    Y = ds.df[ds.label_cols].values.astype(int)
    pcfg = PreprocessConfig.from_dict(cfg["preprocessing"])
    texts = normalize_corpus(ds.df[ds.text_col].tolist(), pcfg)
    tr_idx, va_idx = single_holdout_indices(Y, val_fraction=0.15, seed=42)
    return cfg, ds, texts, Y, tr_idx, va_idx


def confusion_grids():
    """Build the baseline vs best confusion grids."""
    cfg, ds, texts, Y, tr_idx, va_idx = _load_split()
    label_cols = ds.label_cols
    texts_tr = [texts[i] for i in tr_idx]
    texts_va = [texts[i] for i in va_idx]
    Y_tr, Y_va = Y[tr_idx], Y[va_idx]

    # Baseline (Exp 1): word TF-IDF + LR with global threshold 0.5.
    v_word = make_word_tfidf(cfg["features"]["tfidf_word"])
    Xtr = v_word.fit_transform(texts_tr)
    Xva = v_word.transform(texts_va)
    head = build_head("logreg_ovr", cfg["model"].get("logreg", {}), seed=42)
    head.fit(Xtr, Y_tr)
    P_va = predict_proba_safe(head, Xva)
    yhat_baseline = apply_thresholds(P_va, 0.5)
    plot_confusion_grid(
        Y_va, yhat_baseline, label_cols,
        out_path=FIGS / "confusion_grid_baseline.png",
        title="Per-label confusion matrices — Exp 1 baseline (TF-IDF word + LR, threshold = 0.5)",
    )
    log.info(f"✓ {FIGS / 'confusion_grid_baseline.png'}")

    # Best (Exp 8 reproduction): TF-IDF word+char + SBERT, LR, then we
    # ensemble two members + per-label thresholds.
    v_char = make_char_tfidf(cfg["features"]["tfidf_char"])
    Xc_tr = v_char.fit_transform(texts_tr)
    Xc_va = v_char.transform(texts_va)
    sbert_cfg = cfg["features"]["sbert"]
    Xe_tr = encode_sbert(texts_tr, **sbert_cfg)
    Xe_va = encode_sbert(texts_va, **sbert_cfg)

    # Member A: sparse only
    Xa_tr = stack_features([Xtr, Xc_tr])
    Xa_va = stack_features([Xva, Xc_va])
    head_a = build_head("logreg_ovr", cfg["model"].get("logreg", {}), seed=42)
    head_a.fit(Xa_tr, Y_tr)
    P_a = predict_proba_safe(head_a, Xa_va)

    # Member B: sparse + SBERT
    Xb_tr = stack_features([Xtr, Xc_tr, Xe_tr])
    Xb_va = stack_features([Xva, Xc_va, Xe_va])
    head_b = build_head("logreg_ovr", cfg["model"].get("logreg", {}), seed=42)
    head_b.fit(Xb_tr, Y_tr)
    P_b = predict_proba_safe(head_b, Xb_va)

    P_ens = (P_a + P_b) / 2
    thr = tune_thresholds(Y_va, P_ens)
    yhat_best = apply_thresholds(P_ens, thr, min_labels_per_doc=1)
    plot_confusion_grid(
        Y_va, yhat_best, label_cols,
        out_path=FIGS / "confusion_grid_best.png",
        title="Per-label confusion matrices — Exp 8 calibrated ensemble (per-label thresholds, ≥1 label/doc)",
    )
    log.info(f"✓ {FIGS / 'confusion_grid_best.png'}")


def learning_curve():
    """Compute and plot the learning curve for Exp 3's representation."""
    cfg, ds, texts, Y, tr_idx, va_idx = _load_split()
    texts_tr_all = [texts[i] for i in tr_idx]
    texts_va = [texts[i] for i in va_idx]

    # Pre-fit vectorizers on the FULL training fold so subsample
    # transforms use the same vocabulary at every learning-curve point
    # (apples-to-apples — only the classifier sees fewer samples).
    v_word = make_word_tfidf(cfg["features"]["tfidf_word"])
    v_char = make_char_tfidf(cfg["features"]["tfidf_char"])
    v_word.fit(texts_tr_all)
    v_char.fit(texts_tr_all)
    Xtr_full = stack_features([v_word.transform(texts_tr_all), v_char.transform(texts_tr_all)])
    Xva = stack_features([v_word.transform(texts_va), v_char.transform(texts_va)])
    # Recombine into one indexable matrix in (train ∪ val) order.
    n_tr = Xtr_full.shape[0]

    def fit_and_predict(X_sub_tr, Y_sub_tr, X_eval):
        head = build_head("logreg_ovr", cfg["model"].get("logreg", {}), seed=42)
        head.fit(X_sub_tr, Y_sub_tr)
        return predict_proba_safe(head, X_eval)

    # compute_learning_curve expects a single X and (train_idx,val_idx).
    # We build a virtual stacked matrix where the first n_tr rows are
    # training and the remainder is validation.
    from scipy.sparse import vstack
    X_all = vstack([Xtr_full, Xva]).tocsr()
    Y_all = np.vstack([Y[tr_idx], Y[va_idx]])
    train_idx = np.arange(n_tr)
    val_idx = np.arange(n_tr, n_tr + Xva.shape[0])

    rec = compute_learning_curve(
        fit_and_predict, X_all, Y_all, train_idx, val_idx,
        fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0), seed=42,
    )
    plot_learning_curve(
        rec, out_path=FIGS / "learning_curve.png",
        title="Learning curve — TF-IDF (word + char) + LR (threshold = 0.5)",
    )
    log.info(f"✓ {FIGS / 'learning_curve.png'}")
    # Persist raw numbers too — report can cite them.
    import json
    (ROOT / "reports" / "learning_curve.json").write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    confusion_grids()
    learning_curve()
