"""Sanity tests for the preprocessing pipeline.

These aren't research tests — they protect the contract that downstream
feature extractors depend on (deterministic output, NaN-safe, URL/HTML
removed, casing preserved when configured).
"""

from __future__ import annotations

from sdgtext.data.preprocess import PreprocessConfig, normalize_text


def test_strip_url_and_html():
    cfg = PreprocessConfig()
    txt = 'Visit <a href="http://who.int">WHO</a> for stats https://example.com'
    out = normalize_text(txt, cfg)
    assert "http" not in out
    assert "<a" not in out
    assert "who" in out  # the visible anchor text survives


def test_lowercase_and_contractions():
    cfg = PreprocessConfig(lowercase=True, expand_contractions=True, remove_punctuation=False)
    out = normalize_text("They're WAITING — won't deliver vaccines.", cfg)
    assert "they are" in out
    assert "will not" in out


def test_min_token_len():
    cfg = PreprocessConfig(min_token_len=3, remove_punctuation=True)
    out = normalize_text("a ab abc abcd", cfg)
    tokens = out.split()
    assert all(len(t) >= 3 for t in tokens)


def test_empty_and_nan_safe():
    cfg = PreprocessConfig()
    assert normalize_text("", cfg) == ""
    assert normalize_text(None, cfg) == ""  # type: ignore[arg-type]
