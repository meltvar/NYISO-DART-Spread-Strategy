"""
Date-based train / validation / test masks.

The split is locked by calendar date in nyiso_dart.config:
    Train      2015-01-01 .. 2019-12-31
    Validation 2020-01-01 .. 2021-12-31
    Test       2022-01-01 .. 2025-12-31

This module is intentionally tiny and dependency-free so every other module
imports the same masks. Any attempt to redraw the split happens here and only
here.
"""
from __future__ import annotations

import pandas as pd

from nyiso_dart.config import (
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
)


def _to_local(idx) -> pd.DatetimeIndex:
    """Coerce an index/Series of timestamps to tz-aware Eastern. The feature
    matrix index is already tz-aware; this just unwraps it consistently."""
    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    return idx


def _between(idx: pd.DatetimeIndex, start: str, end: str) -> pd.Series:
    """Inclusive-both-ends date mask. Compares on calendar date in the local
    timezone of the index (assumed Eastern)."""
    s = pd.Timestamp(start, tz=idx.tz)
    # end-of-day for inclusive end
    e = pd.Timestamp(end, tz=idx.tz) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return pd.Series((idx >= s) & (idx <= e), index=idx)


def train_mask(idx) -> pd.Series:
    return _between(_to_local(idx), TRAIN_START, TRAIN_END)


def val_mask(idx) -> pd.Series:
    return _between(_to_local(idx), VAL_START, VAL_END)


def test_mask(idx) -> pd.Series:
    return _between(_to_local(idx), TEST_START, TEST_END)


def assert_disjoint(idx) -> None:
    """Belt-and-braces: the three masks must be pairwise disjoint."""
    tr = train_mask(idx).values
    va = val_mask(idx).values
    te = test_mask(idx).values
    if (tr & va).any() or (tr & te).any() or (va & te).any():
        raise RuntimeError("Train/Val/Test masks overlap. Check config date bounds.")
