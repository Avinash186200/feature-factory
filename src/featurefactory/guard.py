"""The leakage guard: a feature on row ``t`` may use data up to ``t-1`` only.

Two layers:

1. **Construction.** ``compute`` first shifts every column by its publication lag (so a
   value appears on the row where it was actually published), runs the feature function,
   then shifts the result by one row. The value on row ``t`` is therefore built from rows
   ``<= t-1`` as published by then.
2. **Verification.** ``check_no_lookahead`` does not trust the construction. It perturbs
   every input that was *not yet available* at the close of ``t-1`` and checks that the
   feature on row ``t`` does not move. A feature function with a hidden centred window, a
   ``shift(-k)``, or a full-sample statistic fails this check and is rejected by the screen.
   So does a feature that is NaN on every probe row, since nothing could be compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

FeatureFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class Feature:
    """A candidate feature: a pure function of the data plus metadata for the catalogue."""

    name: str
    family: str
    fn: FeatureFn
    description: str = ""
    params: dict = field(default_factory=dict)


def published_view(df: pd.DataFrame, pub_lags: Mapping[str, int]) -> pd.DataFrame:
    """Move each column to the row on which it became known.

    ``macro`` with a lag of 2 means the value for reference date ``d`` is first usable on
    row ``d + 2``.
    """
    out = df.copy()
    for col, lag in pub_lags.items():
        if col in out.columns and lag:
            out[col] = out[col].shift(lag)
    return out


def compute(
    feature: Feature, df: pd.DataFrame, pub_lags: Mapping[str, int], guard: bool = True
) -> pd.Series:
    """Compute a feature with the guard on (default) or off (only for the breakage study).

    With ``guard=True`` the value on row ``t`` is built from data published by the close of
    ``t-1``. With ``guard=False`` neither publication lags nor the one-row shift are
    applied, which reproduces the two classic mistakes.
    """
    if not guard:
        return feature.fn(df).rename(feature.name)
    return feature.fn(published_view(df, pub_lags)).shift(1).rename(feature.name)


def lookahead_report(
    feature: Feature,
    df: pd.DataFrame,
    pub_lags: Mapping[str, int],
    probes: int = 4,
    seed: int = 0,
    guard: bool = True,
    compute_lags: Mapping[str, int] | None = None,
) -> tuple[bool, str]:
    """Run the look-ahead check and return ``(passed, reason)``.

    For a few probe rows ``t``, every column ``c`` is scrambled from row ``t - lag_c`` onward
    (rows that had not been published by ``t-1``), the feature is recomputed, and the value on
    row ``t`` must be unchanged (NaN-equal).

    A feature that is NaN on every probe row in both runs fails: nothing was compared, so a
    pass would prove nothing (for example, a window longer than the data hiding a
    ``shift(-k)``).

    ``pub_lags`` are the true publication lags used for scrambling. ``compute_lags`` are the
    lags the feature is computed with (default: the same). Passing ``compute_lags={}`` tests
    a pipeline that wrongly assumes every value is known on its reference date.
    """
    lags_used = pub_lags if compute_lags is None else compute_lags
    rng = np.random.default_rng(seed)
    base = compute(feature, df, lags_used, guard=guard)
    n = len(df)
    lo, hi = max(300, n // 4), n - 5
    compared = 0
    for t in rng.integers(lo, hi, size=probes):
        scrambled = df.copy()
        for j, col in enumerate(df.columns):
            start = max(0, int(t) - int(pub_lags.get(col, 0)))
            vals = scrambled[col].to_numpy(copy=True)
            vals[start:] = vals[start:] * (1.5 + rng.random(n - start)) + 1.0
            scrambled[col] = vals
        new = compute(feature, scrambled, lags_used, guard=guard)
        a, b = base.iloc[int(t)], new.iloc[int(t)]
        if np.isnan(a) and np.isnan(b):
            continue
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12):
            return False, f"value on row {int(t)} changed when later data was scrambled"
        compared += 1
    if compared == 0:
        return False, "the feature is NaN on every probe row, so nothing could be checked"
    return True, f"unchanged on {compared} of {probes} probe rows"


def check_no_lookahead(
    feature: Feature,
    df: pd.DataFrame,
    pub_lags: Mapping[str, int],
    probes: int = 4,
    seed: int = 0,
    guard: bool = True,
    compute_lags: Mapping[str, int] | None = None,
) -> bool:
    """Return True if the feature on row ``t`` ignores everything unknown at the close of ``t-1``.

    See ``lookahead_report`` for the method and for the reason behind a failure.
    """
    return lookahead_report(feature, df, pub_lags, probes, seed, guard, compute_lags)[0]
