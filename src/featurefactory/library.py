"""The feature generator: a pre-registered grid of transforms x inputs x windows.

Every candidate is generated up front, before any screening, so the number of tests is
known in advance and the multiple-testing correction covers all of them. Functions here
are written as if the row they produce is "today"; ``guard.compute`` handles the shift to
``t-1`` and the publication lags.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .guard import Feature


def _logret(s: pd.Series) -> pd.Series:
    return np.log(s / s.shift(1))


def _volume_z(df: pd.DataFrame) -> pd.Series:
    lv = np.log(df["volume"])
    return (lv - lv.rolling(63, min_periods=20).mean()) / lv.rolling(63, min_periods=20).std()


def _clv(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng


def _make(name: str, family: str, fn, description: str, **params) -> Feature:
    return Feature(name=name, family=family, fn=fn, description=description, params=params)


def base_features() -> list[Feature]:
    """Single-input transforms over a fixed set of windows."""
    feats: list[Feature] = []
    for w in (5, 10, 21, 63):
        feats.append(_make(f"mom_{w}", "momentum", lambda d, w=w: np.log(d["close"] / d["close"].shift(w)),
                           f"{w}-day log return", w=w))
    for w in (5, 21, 63):
        feats.append(_make(f"vol_{w}", "volatility", lambda d, w=w: _logret(d["close"]).rolling(w).std(),
                           f"{w}-day realised volatility", w=w))
    for a, b in ((5, 21), (21, 63)):
        feats.append(_make(f"volratio_{a}_{b}", "volatility",
                           lambda d, a=a, b=b: _logret(d["close"]).rolling(a).std()
                           / _logret(d["close"]).rolling(b).std(),
                           f"{a}-day over {b}-day volatility", a=a, b=b))
    for w in (1, 5, 10, 21):
        feats.append(_make(f"volume_z_{w}", "volume_z", lambda d, w=w: _volume_z(d).rolling(w).mean(),
                           f"{w}-day mean of log-volume z-score (63-day trailing baseline)", w=w))
    for w in (5, 21):
        feats.append(_make(f"range_{w}", "range", lambda d, w=w: np.log(d["high"] / d["low"]).rolling(w).mean(),
                           f"{w}-day mean high-low range", w=w))
    for w in (1, 5):
        feats.append(_make(f"clv_{w}", "close_location", lambda d, w=w: _clv(d).rolling(w).mean(),
                           f"{w}-day mean close location in the day's range", w=w))
    for w in (10, 50, 200):
        feats.append(_make(f"dist_ma_{w}", "trend", lambda d, w=w: d["close"] / d["close"].rolling(w).mean() - 1,
                           f"distance from the {w}-day moving average", w=w))
    for w in (21, 63):
        feats.append(_make(f"xcorr_{w}", "cross_asset",
                           lambda d, w=w: _logret(d["close"]).rolling(w).corr(_logret(d["close_b"])),
                           f"{w}-day correlation with the second asset", w=w))
        feats.append(_make(f"rel_mom_{w}", "cross_asset",
                           lambda d, w=w: np.log(d["close"] / d["close"].shift(w))
                           - np.log(d["close_b"] / d["close_b"].shift(w)),
                           f"{w}-day momentum relative to the second asset", w=w))
    for w in (1, 5, 21):
        feats.append(_make(f"macro_chg_{w}", "macro", lambda d, w=w: d["macro"].diff(w),
                           f"{w}-period change in the macro indicator (as published)", w=w))
    feats.append(_make("macro_z_252", "macro",
                       lambda d: (d["macro"] - d["macro"].rolling(252, min_periods=60).mean())
                       / d["macro"].rolling(252, min_periods=60).std(),
                       "macro level z-score against its trailing year"))
    return feats


INTERACTION_INPUTS = ("mom_21", "volume_z_5", "vol_21", "dist_ma_50")


def interaction_features(base: list[Feature]) -> list[Feature]:
    """Pairwise products of a small, fixed set of base features (chosen before screening)."""
    by_name = {f.name: f for f in base}
    out = []
    for a, b in combinations(INTERACTION_INPUTS, 2):
        fa, fb = by_name[a], by_name[b]
        out.append(_make(f"{a}_x_{b}", "interaction", lambda d, fa=fa, fb=fb: fa.fn(d) * fb.fn(d),
                         f"product of {a} and {b}", a=a, b=b))
    return out


def generate_candidates() -> list[Feature]:
    """The full candidate set: base grid plus interactions."""
    base = base_features()
    return base + interaction_features(base)
