"""Synthetic daily market data with known, planted structure.

Columns (indexed by business date):

- ``open, high, low, close, volume``: the traded asset
- ``close_b``: a second, correlated asset (for cross-asset features)
- ``macro``: a macro indicator indexed by its *reference* date. It is only published
  ``MACRO_PUB_LAG`` business days later, so on day ``d`` the latest usable value is the one
  for ``d - MACRO_PUB_LAG``.

What is planted, so the screen can be checked against the truth:

- **A real signal**: the return on day ``t`` loads on the 5-day volume z-score known at the
  close of ``t-1`` (strength ``signal``). Features in the ``volume_z`` family should be found.
- **A look-ahead trap**: the return on day ``t`` also loads on the macro change for
  reference date ``t-1``, which is not published until ``t-1+MACRO_PUB_LAG``. A feature that
  ignores the publication lag looks predictive; a correctly lagged one does not.
- **A same-bar trap**: the close location in the day's range is mechanically tied to that
  day's return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MACRO_PUB_LAG = 2
#: Publication lag in rows for each column. Everything else is known at its own close.
PUB_LAGS = {"macro": MACRO_PUB_LAG}
TRUE_FAMILY = "volume_z"


def make_synthetic_market(
    n_days: int = 2500,
    seed: int = 7,
    signal: float = 0.15,
    macro_trap: float = 0.25,
    start: str = "2014-01-01",
) -> pd.DataFrame:
    """Simulate the panel described in the module docstring.

    Args:
        n_days: number of business days.
        seed: RNG seed.
        signal: loading of next-day return (in units of daily vol) on the lagged 5-day
            volume z-score. ``0`` gives a market with no usable signal at all.
        macro_trap: loading on the *unpublished* macro change (the look-ahead trap).
        start: first business date.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days, name="date")

    # log volume: AR(1) around a level; its 5-day standardised mean is the planted driver
    lv = np.empty(n_days)
    lv[0] = 0.0
    for t in range(1, n_days):
        lv[t] = 0.9 * lv[t - 1] + 0.3 * rng.standard_normal()
    lv_s = pd.Series(lv, index=idx)
    vz = ((lv_s - lv_s.rolling(63, min_periods=20).mean()) / lv_s.rolling(63, min_periods=20).std())
    driver = vz.rolling(5, min_periods=1).mean().fillna(0.0).to_numpy()

    # macro: random walk by reference date
    dm = rng.standard_normal(n_days)
    macro = 100 + np.cumsum(dm)

    # daily vol with clustering: log-vol AR(1) around 1% a day
    lsig = np.empty(n_days)
    lsig[0] = 0.0
    for t in range(1, n_days):
        lsig[t] = 0.97 * lsig[t - 1] + 0.08 * rng.standard_normal()
    sig = 0.01 * np.exp(lsig)

    eps = rng.standard_normal(n_days)
    ret = np.zeros(n_days)
    ret[1:] = sig[1:] * (signal * driver[:-1] + macro_trap * dm[:-1] + eps[1:])

    close = 100 * np.exp(np.cumsum(ret))
    gap = rng.normal(0, 0.2, n_days) * sig
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(gap)
    span = np.abs(rng.normal(0, 0.6, n_days)) * sig
    high = np.maximum(open_, close) * np.exp(span)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.6, n_days)) * sig)

    ret_b = 0.6 * ret + 0.8 * sig * rng.standard_normal(n_days)
    close_b = 50 * np.exp(np.cumsum(ret_b))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.exp(13 + lv),
            "close_b": close_b,
            "macro": macro,
        },
        index=idx,
    )


def target_returns(df: pd.DataFrame, vol_window: int = 21) -> pd.Series:
    """The target on row ``t``: day ``t``'s log return divided by trailing volatility known at ``t-1``.

    Scaling matters for the screen. Raw returns are heteroskedastic, and any feature that
    tracks volatility lines up with the target's variance. That alignment widens the real
    sampling spread of the IC but is broken by the circular-shift null, so raw returns give
    p-values that are too small. Dividing by volatility up to ``t-1`` removes most of it
    without using day ``t``.
    """
    r = np.log(df["close"] / df["close"].shift(1))
    scale = r.rolling(vol_window).std().shift(1)
    return (r / scale).rename("target")
