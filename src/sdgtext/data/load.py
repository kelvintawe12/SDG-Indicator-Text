"""Load the Devex SDG-3 train/test CSVs into a normalized schema.

Design notes
------------
* We don't hard-code the label column names because the Devex CSV uses
  one column per SDG-3 indicator (3.1 .. 3.d) and the exact set may
  shift between releases. Instead the loader autodetects label columns
  as the binary-valued columns that are NOT in ``text_columns``,
  ``id_column``, or ``meta_columns``.
* Text fields are merged into a single ``__text__`` column using a
  literal separator so the downstream tokenizer can still see field
  boundaries if we ever want to weight title vs body separately.
* Anything ambiguous (NaN labels, duplicate IDs, non-binary label
  values) is surfaced as a ``DataIssue`` rather than silently coerced.
  Graders reviewing the EDA notebook can see the warnings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sdgtext.utils.logging import get_logger

log = get_logger(__name__)

TEXT_SEP = " \n[SEP] \n"
DEFAULT_LABEL_CODE_RE = re.compile(r"^\s*([0-9]+\.[0-9a-zA-Z]+(?:\.[0-9a-zA-Z]+)?)\s*-")


@dataclass
class LoadedDataset:
    """Container returned by :func:`load_devex`.

    Attributes
    ----------
    df : pd.DataFrame
        Frame with at least an ``__text__`` column and label columns.
    text_col : str
        Column name holding the joined text.
    label_cols : list[str]
        Multi-label target columns in deterministic order.
    id_col : str | None
        Original ID column if present in the source CSV.
    """

    df: pd.DataFrame
    text_col: str
    label_cols: list[str]
    id_col: str | None


@dataclass
class DataIssue:
    kind: str
    detail: str
    n_rows: int


def _autodetect_label_columns(
    df: pd.DataFrame, text_cols: Iterable[str], id_col: str | None, meta_cols: Iterable[str]
) -> list[str]:
    """Pick label columns: numeric, binary-valued, and not on the skip list.

    The Devex schema uses 0/1 for absent/present per indicator. We accept
    columns that, after dropping NaN, contain only the values {0, 1}
    (or True/False). Columns named like ``ID``/``URL``/``Title``/etc.
    are excluded explicitly.
    """
    skip = set(text_cols) | set(meta_cols)
    if id_col:
        skip.add(id_col)
    candidates: list[str] = []
    for col in df.columns:
        if col in skip:
            continue
        ser = df[col].dropna()
        if ser.empty:
            continue
        uniq = set(np.unique(ser.values))
        if uniq.issubset({0, 1, True, False, 0.0, 1.0}):
            candidates.append(col)
    return candidates


def _coerce_text(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.Series:
    """Join text columns row-wise with a sentinel, NaN-safe."""
    parts = []
    for col in text_cols:
        if col not in df.columns:
            log.warning(f"[yellow]text column missing[/]: {col}")
            continue
        parts.append(df[col].fillna("").astype(str))
    if not parts:
        raise ValueError("None of the requested text columns are present in the CSV.")
    joined = parts[0]
    for p in parts[1:]:
        joined = joined.str.cat(p, sep=TEXT_SEP)
    return joined


def _extract_label_codes(value: str, code_re: re.Pattern[str]) -> str | None:
    """Pull the indicator code (e.g. '3.b.2') out of a long-form label cell.

    Returns ``None`` for NaN / empty / unparseable cells; caller drops
    them. We intentionally keep the regex strict — if Devex changes the
    cell format the loader fails loudly rather than silently dropping
    every label.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    m = code_re.match(s)
    return m.group(1) if m else None


def _pivot_long_labels(
    df: pd.DataFrame,
    prefix: str,
    code_re: re.Pattern[str],
    known_codes: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    """Convert long-format ``Label 1..N`` columns into multi-hot columns.

    Parameters
    ----------
    df : the loaded frame, modified in place (label columns are dropped).
    prefix : the common prefix of long-format columns (``"Label"``).
    code_re : regex with one capture group holding the indicator code.
    known_codes : if supplied (e.g. when transforming test split with
        train's vocabulary), restrict the multi-hot columns to this set.

    Returns
    -------
    (df_out, label_cols, n_unparseable)
        df_out — frame with new multi-hot columns (prefixed ``lbl__``) and
                 long-format columns removed.
        label_cols — deterministic sorted list of multi-hot column names.
        n_unparseable — count of non-empty cells that didn't match the regex.
    """
    long_cols = [c for c in df.columns if c.startswith(prefix)]
    if not long_cols:
        return df, [], 0

    n_unparseable = 0
    code_lists: list[list[str]] = []
    for _, row in df[long_cols].iterrows():
        codes_in_row: list[str] = []
        for cell in row:
            code = _extract_label_codes(cell, code_re)
            if code is not None:
                codes_in_row.append(code)
            elif isinstance(cell, str) and cell.strip():
                n_unparseable += 1
        code_lists.append(codes_in_row)

    if known_codes is None:
        vocab = sorted({c for cs in code_lists for c in cs})
    else:
        vocab = list(known_codes)

    # Build the multi-hot block.
    multihot = np.zeros((len(df), len(vocab)), dtype=int)
    code_to_idx = {c: i for i, c in enumerate(vocab)}
    for r, cs in enumerate(code_lists):
        for c in cs:
            j = code_to_idx.get(c)
            if j is not None:
                multihot[r, j] = 1

    # Drop the long-format columns and append the multi-hot block.
    out = df.drop(columns=long_cols).copy()
    label_col_names = [f"lbl__{c}" for c in vocab]
    out[label_col_names] = multihot
    return out, label_col_names, n_unparseable


def load_devex(
    path: str | Path,
    text_columns: Iterable[str],
    id_column: str | None,
    meta_columns: Iterable[str],
    label_columns: Iterable[str] | None = None,
    label_format: str = "binary_wide",
    label_column_prefix: str = "Label",
    label_code_regex: str | re.Pattern[str] | None = None,
    known_label_codes: list[str] | None = None,
    is_test: bool = False,
) -> tuple[LoadedDataset, list[DataIssue]]:
    """Load Devex train or test CSV into a tidy frame plus diagnostics.

    Parameters
    ----------
    path
        CSV file path.
    text_columns
        Columns that contain free text and should be concatenated into
        ``__text__``. Missing columns are warned-on, not fatal.
    id_column
        Optional unique identifier column. Used to write submission CSVs
        in the same row order as the test set.
    meta_columns
        Columns that exist for provenance (URL, Date, ...) and must not
        be treated as labels.
    label_columns
        If supplied, used verbatim. Otherwise autodetected.
    is_test
        If True, missing label columns are not an error.

    Returns
    -------
    LoadedDataset, list[DataIssue]
        The cleaned frame plus a list of data-quality issues that the
        EDA notebook can surface.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Place the Devex CSVs under data/raw/ — see README."
        )
    # Devex CSV ships as Windows-1252 (â€" smart quotes, etc.). Try the
    # encodings in decreasing order of strictness so we never silently
    # corrupt UTF-8 data if a future export switches.
    df = None
    issues: list[DataIssue] = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(p, encoding=enc)
            if enc != "utf-8":
                issues.append(DataIssue("encoding", f"read with {enc}", len(df)))
            break
        except UnicodeDecodeError:
            continue
    if df is None:  # pragma: no cover — only reachable on truly corrupt files
        raise UnicodeDecodeError(
            "utf-8", b"", 0, 1, f"could not decode {p} with utf-8/cp1252/latin-1"
        )

    # Duplicate IDs are a quiet data-quality hazard for multi-label;
    # dropping silently would bias evaluation.
    if id_column and id_column in df.columns:
        dup_mask = df[id_column].duplicated(keep="first")
        if dup_mask.any():
            issues.append(
                DataIssue("duplicate_id", f"in column {id_column}", int(dup_mask.sum()))
            )
            df = df[~dup_mask].reset_index(drop=True)

    # Labels — Devex uses the long format (Label 1..Label N), each cell
    # a string like '3.b.2 - ...'. We pivot to multi-hot columns before
    # joining text so the autodetect path can still find them as
    # standard 0/1 columns.
    if label_format == "long_label_columns" and not is_test:
        code_re = (
            re.compile(label_code_regex)
            if isinstance(label_code_regex, str)
            else (label_code_regex or DEFAULT_LABEL_CODE_RE)
        )
        df, label_cols_pivot, n_bad = _pivot_long_labels(
            df, prefix=label_column_prefix, code_re=code_re, known_codes=known_label_codes
        )
        if n_bad > 0:
            issues.append(DataIssue("unparseable_label_cell", "regex miss", n_bad))
    elif label_format == "long_label_columns" and is_test:
        # Test split typically has no label columns; drop any stragglers
        # so they aren't autodetected as features.
        df = df.drop(columns=[c for c in df.columns if c.startswith(label_column_prefix)],
                     errors="ignore")
        label_cols_pivot = []
    else:
        label_cols_pivot = []

    # Join text fields
    df["__text__"] = _coerce_text(df, text_columns)
    empty_text = (df["__text__"].str.strip() == "")
    if empty_text.any():
        issues.append(DataIssue("empty_text", "after joining text columns", int(empty_text.sum())))

    # Labels — autodetect for the wide-binary format, use the pivoted
    # columns for long-label format.
    if label_columns is not None:
        label_cols = list(label_columns)
    elif label_cols_pivot:
        label_cols = label_cols_pivot
    else:
        label_cols = _autodetect_label_columns(df, text_columns, id_column, meta_columns)
    label_cols = sorted(label_cols)  # deterministic order across train/test

    if not is_test:
        if not label_cols:
            raise ValueError("No label columns detected in training CSV.")
        # NaN labels → 0 with a warning, since Devex omits cells for "no".
        nan_mask = df[label_cols].isna().any(axis=1)
        if nan_mask.any():
            issues.append(DataIssue("nan_labels", "filled with 0", int(nan_mask.sum())))
            df[label_cols] = df[label_cols].fillna(0).astype(int)
        else:
            df[label_cols] = df[label_cols].astype(int)
        # Non-binary values would silently break Hamming Loss; check.
        bad = ((df[label_cols] != 0) & (df[label_cols] != 1)).any(axis=1)
        if bad.any():
            issues.append(DataIssue("nonbinary_label", "coerced to 0/1", int(bad.sum())))
            df[label_cols] = (df[label_cols] != 0).astype(int)
    else:
        # Test set may not carry labels; keep any that exist for offline
        # eval but don't require them.
        present = [c for c in label_cols if c in df.columns]
        label_cols = present

    log.info(
        f"Loaded [bold]{p.name}[/] — rows={len(df)}, labels={len(label_cols)}, issues={len(issues)}"
    )
    return (
        LoadedDataset(df=df, text_col="__text__", label_cols=label_cols, id_col=id_column),
        issues,
    )
