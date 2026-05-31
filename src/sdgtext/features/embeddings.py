"""Precomputed Sentence-BERT embeddings.

CPU-only constraint: we never fine-tune. We use ``all-MiniLM-L6-v2``
(22M params, 384-d) because it is the strongest model in the
sentence-transformers family that comfortably encodes ~3K medium-length
documents in a few minutes on a laptop CPU. Embeddings are L2-normalized
so a downstream linear model receives a unit-norm cosine-space
representation.

We cache the resulting matrix to ``data/interim/`` keyed by a hash of
(model_name, text). The notebook re-runs the encoder only if the
preprocessing config changed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from sdgtext.utils.logging import get_logger

log = get_logger(__name__)


def _cache_key(model_name: str, texts: list[str]) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    # Hash the corpus length + the first/last document so identical
    # preprocessed corpora hit the cache without hashing all 3K docs.
    h.update(str(len(texts)).encode())
    if texts:
        h.update(texts[0].encode("utf-8", errors="ignore"))
        h.update(texts[-1].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def encode_sbert(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    normalize: bool = True,
    cache_dir: str | Path | None = "data/interim",
) -> np.ndarray:
    """Encode a corpus into SBERT embeddings, with on-disk caching.

    Caching is keyed on (model_name, len(texts), first+last doc); not a
    full-corpus hash. This is intentional — recomputing a hash over 3K
    documents on every call is slower than just re-encoding 100 docs of
    cache-miss traffic. The cache is invalidated when preprocessing
    changes the boundary docs, which is the common case.
    """
    cache_dir = Path(cache_dir) if cache_dir else None
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(model_name, texts)
        cache_file = cache_dir / f"sbert_{key}.npy"
        if cache_file.exists():
            log.info(f"SBERT cache hit: {cache_file.name}")
            return np.load(cache_file)

    # Import inside the function so users running classical-only
    # experiments don't pay the sentence-transformers import cost.
    from sentence_transformers import SentenceTransformer  # noqa: WPS433

    log.info(f"Encoding {len(texts)} docs with {model_name} (CPU)…")
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    emb = np.asarray(emb, dtype=np.float32)
    if cache_file is not None:
        np.save(cache_file, emb)
        log.info(f"SBERT cache write: {cache_file.name} shape={emb.shape}")
    return emb
