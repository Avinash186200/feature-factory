"""Phase 6 of the brief: make the classic mistakes on purpose and measure the inflation.

Each experiment computes one feature twice, the wrong way and the right way, and reports
the partial rank IC of both on the same rows plus whether the leakage guard flags the wrong
version.
"""
from __future__ import annotations

import pandas as pd

from .guard import Feature, check_no_lookahead, compute
from .library import _clv, _logret
from .screen import partial_ic


def _ic(s: pd.Series, target: pd.Series) -> float:
    return float(partial_ic(s.to_frame("x"), target).iloc[0])


def run_breakage(df: pd.DataFrame, target: pd.Series, pub_lags: dict) -> pd.DataFrame:
    """Return one row per mistake: IC wrong, IC right, inflation, and the guard's verdict."""
    rows = []

    # 1. same-bar: close location of day t used to "predict" the return of day t
    clv = Feature("clv_1", "close_location", _clv)
    wrong = compute(clv, df, pub_lags, guard=False)
    right = compute(clv, df, pub_lags)
    rows.append(("same-bar data (day t's bar for day t's return)", wrong, right,
                 check_no_lookahead(clv, df, pub_lags, guard=False)))

    # 2. publication lag ignored: macro change matched to its reference date
    macro = Feature("macro_chg_1", "macro", lambda d: d["macro"].diff(1))
    wrong = compute(macro, df, {})  # shifted one row, but as if published on its reference date
    right = compute(macro, df, pub_lags)
    rows.append(("macro used on reference date, not release date", wrong, right,
                 check_no_lookahead(macro, df, pub_lags, compute_lags={})))

    # 3. centred rolling window instead of a trailing one
    centred = Feature("ret_mean_5c", "momentum", lambda d: _logret(d["close"]).rolling(5, center=True).mean())
    trailing = Feature("ret_mean_5", "momentum", lambda d: _logret(d["close"]).rolling(5).mean())
    rows.append(("centred 5-day window", compute(centred, df, pub_lags), compute(trailing, df, pub_lags),
                 check_no_lookahead(centred, df, pub_lags)))

    # 4. full-sample normalisation: a z-score using the whole sample's mean and sd
    full = Feature("dist_ma_50_fullz", "trend",
                   lambda d: _z_full(d["close"] / d["close"].rolling(50).mean() - 1))
    roll = Feature("dist_ma_50_rollz", "trend",
                   lambda d: _z_roll(d["close"] / d["close"].rolling(50).mean() - 1))
    rows.append(("full-sample z-score instead of trailing", compute(full, df, pub_lags),
                 compute(roll, df, pub_lags), check_no_lookahead(full, df, pub_lags)))

    out = []
    for label, w, r, passed in rows:
        rows_ok = w.notna() & r.notna()  # score both versions on the same days
        ic_w, ic_r = _ic(w[rows_ok], target), _ic(r[rows_ok], target)
        out.append({"mistake": label, "ic_wrong": ic_w, "ic_right": ic_r,
                    "abs_inflation": abs(ic_w) - abs(ic_r), "guard_flags_it": not passed})
    return pd.DataFrame(out).set_index("mistake")


def _z_full(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def _z_roll(s: pd.Series) -> pd.Series:
    return (s - s.rolling(252, min_periods=60).mean()) / s.rolling(252, min_periods=60).std()
