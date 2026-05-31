"""TF-IDF builders — word-level and character n-gram.

We expose two factory functions instead of a single FeatureUnion so the
notebook can fit and inspect each vectorizer independently (for the EDA
section we want top features per label, which is much easier with named
vectorizers).
"""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer


def make_word_tfidf(cfg: dict[str, Any]) -> TfidfVectorizer:
    """Word-level TF-IDF with 1-2 grams (sublinear TF, English stopwords).

    Sublinear TF dampens the effect of frequent terms — empirically a
    consistent small win on log-spaced term distributions like ours.
    """
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
        sublinear_tf=cfg.get("sublinear_tf", True),
        max_features=cfg.get("max_features", 60000),
        stop_words=cfg.get("stop_words", "english"),
        strip_accents="unicode",
        lowercase=False,  # preprocessing module already controls casing
    )


def make_char_tfidf(cfg: dict[str, Any]) -> TfidfVectorizer:
    """Character n-gram TF-IDF with the ``char_wb`` analyzer.

    ``char_wb`` is character n-grams from inside word boundaries — it
    captures morphology without crossing whitespace, which gives cleaner
    features than plain ``char`` on multi-word documents.
    """
    return TfidfVectorizer(
        analyzer=cfg.get("analyzer", "char_wb"),
        ngram_range=tuple(cfg.get("ngram_range", [3, 5])),
        min_df=cfg.get("min_df", 2),
        max_df=cfg.get("max_df", 0.95),
        sublinear_tf=cfg.get("sublinear_tf", True),
        max_features=cfg.get("max_features", 60000),
        strip_accents="unicode",
        lowercase=False,
    )
