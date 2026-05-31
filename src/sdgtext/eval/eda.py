"""Exploratory data analysis helpers.

Pure functions returning pandas DataFrames or matplotlib axes — kept
out of the notebook so the notebook can stay narrative-heavy and
re-runs cheaply. Each function takes a :class:`LoadedDataset` and
returns something figureable.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def label_frequency_table(df: pd.DataFrame, label_cols: Iterable[str]) -> pd.DataFrame:
    """Per-label positive count, prevalence, and rank.

    Used to build the imbalance bar chart and the head/tail split for
    Experiment 5's motivation.
    """
    rows = []
    n = len(df)
    for c in label_cols:
        pos = int(df[c].sum())
        rows.append({"label": c, "positives": pos, "prevalence": pos / max(n, 1)})
    out = pd.DataFrame(rows).sort_values("positives", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    return out


def cardinality_stats(df: pd.DataFrame, label_cols: Iterable[str]) -> dict:
    """Multi-label cardinality and density.

    * Label cardinality = average labels per document.
    * Label density = cardinality / n_labels.
    These are the canonical Tsoumakas & Katakis (2007) descriptors and
    we cite them in the report.
    """
    Y = df[list(label_cols)].values.astype(int)
    per_doc = Y.sum(axis=1)
    return {
        "n_samples": int(Y.shape[0]),
        "n_labels": int(Y.shape[1]),
        "cardinality": float(per_doc.mean()),
        "density": float(per_doc.mean() / max(Y.shape[1], 1)),
        "max_per_doc": int(per_doc.max()),
        "zero_label_docs": int((per_doc == 0).sum()),
    }


def cooccurrence_matrix(df: pd.DataFrame, label_cols: Iterable[str]) -> pd.DataFrame:
    """Pairwise label co-occurrence counts.

    Diagonal is the per-label positive count; off-diagonals are joint
    positives. We surface this in the report to motivate the
    classifier-chain question (Exp 4 follow-up): if labels co-occur
    strongly, sharing structure helps.
    """
    Y = df[list(label_cols)].values.astype(int)
    co = Y.T @ Y
    return pd.DataFrame(co, index=list(label_cols), columns=list(label_cols))


def document_length_stats(texts: Iterable[str]) -> dict:
    """Distribution of document lengths (characters and word tokens).

    Used to justify the ``max_text_chars`` cap and the choice of
    sentence-transformer with a 128/256 max_seq_len."""
    chars, words = [], []
    for t in texts:
        chars.append(len(t))
        words.append(len(t.split()))
    chars_arr, words_arr = np.array(chars), np.array(words)
    pct = lambda a, q: float(np.percentile(a, q))
    return {
        "chars": {
            "mean": float(chars_arr.mean()),
            "p50": pct(chars_arr, 50),
            "p90": pct(chars_arr, 90),
            "p99": pct(chars_arr, 99),
            "max": int(chars_arr.max()),
        },
        "words": {
            "mean": float(words_arr.mean()),
            "p50": pct(words_arr, 50),
            "p90": pct(words_arr, 90),
            "p99": pct(words_arr, 99),
            "max": int(words_arr.max()),
        },
    }


def top_terms_per_label(
    vectorizer, X, Y: np.ndarray, label_names: list[str], top_k: int = 15
) -> dict[str, list[tuple[str, float]]]:
    """Mean TF-IDF weight per term within each label's positive subset.

    A quick, model-free way to surface which terms drive each indicator
    — useful for the qualitative section of the report. ``vectorizer``
    must be a fitted TF-IDF estimator and ``X`` its transform.
    """
    vocab = np.array(vectorizer.get_feature_names_out())
    out: dict[str, list[tuple[str, float]]] = {}
    for j, name in enumerate(label_names):
        idx = np.where(Y[:, j] == 1)[0]
        if idx.size == 0:
            out[name] = []
            continue
        mean_weights = np.asarray(X[idx].mean(axis=0)).ravel()
        top = np.argsort(-mean_weights)[:top_k]
        out[name] = [(vocab[i], float(mean_weights[i])) for i in top]
    return out


def vocabulary_overview(texts: Iterable[str], top_k: int = 30) -> pd.DataFrame:
    """Most-frequent unigrams across the corpus (pre-vectorizer)."""
    counter: Counter[str] = Counter()
    for t in texts:
        counter.update(t.split())
    rows = [{"term": w, "count": c} for w, c in counter.most_common(top_k)]
    return pd.DataFrame(rows)
