"""Text normalization pipeline.

Each step is opt-in via the config. The motivation for the granularity
is Experiment 2 (preprocessing ablation): we want to measure each step's
contribution to Hamming Loss independently, not bundle them together.

We deliberately keep the pipeline pure-Python + regex + NLTK so it runs
unchanged on Colab CPU and on a grader's laptop. No spaCy dependency.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import regex as re2  # supports \p{} unicode classes

try:
    import nltk
    from nltk.corpus import stopwords as _stopwords
    from nltk.stem import WordNetLemmatizer
except ImportError:  # pragma: no cover
    nltk = None

URL_RE = re2.compile(r"https?://\S+|www\.\S+", flags=re2.IGNORECASE)
HTML_RE = re2.compile(r"<[^>]+>")
WHITESPACE_RE = re2.compile(r"\s+")
NUMBER_RE = re2.compile(r"\b\d[\d,\.]*\b")
PUNCT_RE = re2.compile(r"[\p{P}\p{S}]+")  # unicode punctuation + symbols

# Common contractions; small list keeps the function fast and predictable.
_CONTRACTIONS = {
    "ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "could've": "could have", "couldn't": "could not", "didn't": "did not",
    "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
    "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
    "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'll": "it will", "it's": "it is", "let's": "let us", "ma'am": "madam",
    "might've": "might have", "must've": "must have", "needn't": "need not",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have", "weren't": "were not",
    "what's": "what is", "where's": "where is", "who's": "who is", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will", "you're": "you are",
    "you've": "you have",
}
_CONTRACTION_RE = re2.compile(
    r"\b(" + "|".join(map(re.escape, _CONTRACTIONS.keys())) + r")\b", flags=re2.IGNORECASE
)


def _expand_contractions(text: str) -> str:
    return _CONTRACTION_RE.sub(lambda m: _CONTRACTIONS[m.group(0).lower()], text)


@lru_cache(maxsize=1)
def _ensure_nltk_resources() -> tuple[set[str], "WordNetLemmatizer | None"]:
    """Download the small NLTK resources we actually use, once.

    Lemmatization is optional (cfg.preprocessing.lemmatize) but the
    stopword list is also useful for downstream EDA, so we fetch both.
    """
    if nltk is None:
        return set(), None
    for pkg in ("stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass
    try:
        sw = set(_stopwords.words("english"))
    except Exception:
        sw = set()
    try:
        lem = WordNetLemmatizer()
    except Exception:
        lem = None
    return sw, lem


@dataclass(frozen=True)
class PreprocessConfig:
    """Frozen view of cfg.preprocessing so it hashes for lru_cache use."""

    lowercase: bool = True
    strip_html: bool = True
    strip_urls: bool = True
    strip_numbers: bool = False
    expand_contractions: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = False
    lemmatize: bool = False
    min_token_len: int = 2
    max_text_chars: int = 20000

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessConfig":
        # Only consume known keys, so unrelated config noise can't break us.
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def normalize_text(text: str, cfg: PreprocessConfig) -> str:
    """Apply the configured normalization steps to a single document.

    Order matters: we strip HTML and URLs *before* lowercasing or
    expanding contractions, because URL hosts like 'WHO.int' would
    otherwise look like the word 'who' and disappear under stopword
    removal.
    """
    if not isinstance(text, str) or not text:
        return ""
    if len(text) > cfg.max_text_chars:
        text = text[: cfg.max_text_chars]

    # 1. Unicode + HTML entities first; these are structural.
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    if cfg.strip_html:
        text = HTML_RE.sub(" ", text)
    if cfg.strip_urls:
        text = URL_RE.sub(" ", text)

    # 2. Casing + contractions before tokenization.
    if cfg.lowercase:
        text = text.lower()
    if cfg.expand_contractions:
        text = _expand_contractions(text)

    # 3. Numbers and punctuation.
    if cfg.strip_numbers:
        text = NUMBER_RE.sub(" ", text)
    if cfg.remove_punctuation:
        text = PUNCT_RE.sub(" ", text)

    # 4. Token-level filters (stopwords, lemma, min-length).
    if cfg.remove_stopwords or cfg.lemmatize or cfg.min_token_len > 1:
        sw, lem = _ensure_nltk_resources()
        tokens = text.split()
        out_tokens: list[str] = []
        for t in tokens:
            if cfg.min_token_len and len(t) < cfg.min_token_len:
                continue
            if cfg.remove_stopwords and t in sw:
                continue
            if cfg.lemmatize and lem is not None:
                t = lem.lemmatize(t)
            out_tokens.append(t)
        text = " ".join(out_tokens)

    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_corpus(texts, cfg: PreprocessConfig) -> list[str]:
    """Vectorized wrapper. Kept simple: per-doc preprocessing is fast."""
    p = PreprocessConfig(**cfg.__dict__) if not isinstance(cfg, PreprocessConfig) else cfg
    return [normalize_text(t, p) for t in texts]
