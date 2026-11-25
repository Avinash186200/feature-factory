"""Fractional differentiation with a fixed-width window (FFD).

A price level remembers everything (it is not stationary). Its first difference, the
return, is stationary but has thrown most of that memory away. Fractional differentiation
sits in between: ``(1 - B)^d`` with ``0 < d < 1``, where ``B`` is the backshift operator.
The aim is the smallest ``d`` that makes the series stationary, so the feature keeps as
much memory of the level as possible.

The binomial expansion gives weights ``w_0 = 1`` and

    w_k = -w_{k-1} * (d - k + 1) / k

and the transformed value is ``x~_t = sum_k w_k * x_{t-k}``. The weights never reach zero
for non-integer ``d``, so FFD keeps only the weights with ``|w_k| >= threshold``. Every
output then uses the same fixed window of past values, and the value on row ``t`` reads
rows ``t - width + 1 .. t`` only (no future rows). The first ``width - 1`` rows are NaN.
For integer ``d`` the expansion is exact: ``d = 0`` returns the series and ``d = 1`` the
first difference.

Stationarity is judged with an augmented Dickey-Fuller (ADF) test. statsmodels is not a
dependency of this repo, so ``adf_test`` is a small numpy version: OLS of the change on a
constant, the lagged level and lagged changes, the lag count chosen by AIC (as statsmodels
does with ``autolag="AIC"``), and the critical values from the response surface in
MacKinnon (2010), "Critical values for cointegration tests", constant-only case.

Point-in-time use: wrap the transform with ``fracdiff_feature`` so the guard shifts it to
``t-1`` and checks it for look-ahead, and choose ``d`` with ``min_ffd_d`` on the discovery
period only. Choosing ``d`` on the whole sample would let the hold-out influence a setting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .guard import Feature

#: MacKinnon (2010) response-surface coefficients, ADF with a constant, one variable:
#: critical value = b0 + b1 / T + b2 / T^2 + b3 / T^3.
_MACKINNON_C = {
    "1%": (-3.43035, -6.5393, -16.786, -79.433),
    "5%": (-2.86154, -2.8903, -4.234, -40.040),
    "10%": (-2.56677, -1.5384, -2.809, 0.0),
}


def ffd_weights(d: float, threshold: float = 1e-4, max_width: int = 10_000) -> np.ndarray:
    """FFD weights ``[w_0, w_1, ...]`` (newest lag first), cut where ``|w_k| < threshold``.

    Uses the recursion ``w_k = -w_{k-1} * (d - k + 1) / k``. ``max_width`` caps the window
    for very small ``d``, where the weights decay slowly.
    """
    if d < 0:
        raise ValueError("d must be >= 0")
    w = [1.0]
    k = 1
    while k < max_width:
        nxt = -w[-1] * (d - k + 1) / k
        if abs(nxt) < threshold:
            break
        w.append(nxt)
        k += 1
    return np.asarray(w)


def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-4, max_width: int = 10_000) -> pd.Series:
    """Apply FFD to ``series``. The value on row ``t`` uses rows ``t - width + 1 .. t`` only.

    Rows without a full window of non-missing values are NaN.
    """
    w = ffd_weights(d, threshold, max_width)
    width = len(w)
    x = series.to_numpy(dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= width:
        # window[i] holds x[i .. i + width - 1]; reverse the weights so w_0 meets the newest value
        windows = np.lib.stride_tricks.sliding_window_view(x, width)
        out[width - 1:] = windows @ w[::-1]
    return pd.Series(out, index=series.index, name=series.name)


def _ols_tstat(y: np.ndarray, X: np.ndarray, col: int) -> tuple[float, float]:
    """t-statistic of coefficient ``col`` and the residual sum of squares."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ssr = float(resid @ resid)
    dof = len(y) - X.shape[1]
    sigma2 = ssr / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    return float(coef[col] / np.sqrt(cov[col, col])), ssr


def _adf_design(x: np.ndarray, lags: int, maxlag: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows for ``dx_t = a + g * x_{t-1} + sum_i b_i dx_{t-i}``, starting after ``maxlag`` lags."""
    dx = np.diff(x)
    rows = np.arange(maxlag, len(dx))
    cols = [np.ones(len(rows)), x[rows]]          # x[rows] is x_{t-1} for dx[rows] = x_t - x_{t-1}
    cols += [dx[rows - i] for i in range(1, lags + 1)]
    return dx[rows], np.column_stack(cols)


def adf_test(series: pd.Series | np.ndarray, maxlag: int | None = None) -> dict:
    """Augmented Dickey-Fuller test with a constant (numpy only).

    The lag count is chosen by AIC over ``0..maxlag`` on a common sample, then the chosen
    model is refitted on all the rows it can use. ``maxlag`` defaults to
    ``12 * (n / 100) ** 0.25``. Returns the statistic, the lag used, the number of
    observations and the 1%/5%/10% critical values. Reject a unit root (call the series
    stationary) when ``stat < crit["5%"]``.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(x)
    if n < 20:
        raise ValueError("need at least 20 observations")
    if maxlag is None:
        maxlag = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
    maxlag = max(0, min(maxlag, n // 2 - 2))

    best_lag, best_aic = 0, np.inf
    for p in range(maxlag + 1):
        y, X = _adf_design(x, p, maxlag)
        _, ssr = _ols_tstat(y, X, 1)
        aic = len(y) * np.log(ssr / len(y)) + 2 * X.shape[1]
        if aic < best_aic:
            best_lag, best_aic = p, aic
    y, X = _adf_design(x, best_lag, best_lag)
    stat, _ = _ols_tstat(y, X, 1)
    nobs = len(y)
    crit = {k: b0 + b1 / nobs + b2 / nobs ** 2 + b3 / nobs ** 3 for k, (b0, b1, b2, b3) in _MACKINNON_C.items()}
    return {"stat": stat, "lag": best_lag, "nobs": nobs, "crit": crit}


def min_ffd_d(
    series: pd.Series,
    d_grid: np.ndarray | None = None,
    threshold: float = 1e-4,
    max_width: int = 10_000,
    level: str = "5%",
    maxlag: int | None = None,
) -> tuple[float | None, pd.DataFrame]:
    """Find the smallest ``d`` on the grid from which the FFD series passes the ADF test.

    For each ``d`` the table holds the window width, the sum of the kept weights, the ADF
    statistic, the critical value at ``level``, whether it passes, and the correlation of the
    FFD series with the original (on rows where both exist).

    The sum of the weights matters. The full (infinite-window) expansion sums to zero for any
    ``d > 0``, but the truncated window does not, so the output still carries ``sum(w)`` times
    the level of the series. A random walk theoretically needs ``d > 0.5`` before the full
    transform is stationary. When the search returns a smaller ``d`` for a random walk, the
    leftover level is being missed by a low-power ADF test on a truncated, shorter sample, not
    kept as useful memory.

    The chosen ``d`` is the smallest grid value that passes *and* after which every larger
    grid value passes too. Near the boundary the ADF statistic is noisy, so a single lucky
    pass below a failing value is not taken. Correlation with the original falls as ``d``
    rises, so this ``d`` keeps the most memory: among the values in the passing block it has
    the highest correlation. Returns ``(None, table)`` if the largest grid value fails.

    Pass the discovery period only; the choice of ``d`` is a fitted setting.
    """
    if d_grid is None:
        d_grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    rows = []
    for d in d_grid:
        fd = frac_diff_ffd(series, float(d), threshold, max_width)
        valid = fd.notna() & series.notna()
        w = ffd_weights(float(d), threshold, max_width)
        width, sum_w = len(w), float(w.sum())
        if valid.sum() < 20:
            rows.append({"d": float(d), "width": width, "sum_w": sum_w, "adf_stat": np.nan, "crit": np.nan,
                         "stationary": False, "corr": np.nan})
            continue
        res = adf_test(fd[valid], maxlag=maxlag)
        rows.append({"d": float(d), "width": width, "sum_w": sum_w, "adf_stat": res["stat"], "crit": res["crit"][level],
                     "stationary": bool(res["stat"] < res["crit"][level]),
                     "corr": float(np.corrcoef(fd[valid], series[valid])[0, 1])})
    table = pd.DataFrame(rows).set_index("d").sort_index()
    return choose_d(table), table


def choose_d(table: pd.DataFrame) -> float | None:
    """The smallest ``d`` (index) from which every larger ``d`` has ``stationary`` True."""
    table = table.sort_index()
    chosen = None
    for d, ok in zip(table.index[::-1], table["stationary"].to_numpy()[::-1]):
        if not ok:
            break
        chosen = float(d)
    return chosen


def fracdiff_feature(column: str = "close", d: float = 0.4, threshold: float = 1e-4, log: bool = True,
                     max_width: int = 10_000) -> Feature:
    """A guarded ``Feature``: FFD of (log) ``column`` with order ``d``.

    The function only reads past rows (a trailing window), and ``guard.compute`` shifts it
    to ``t-1`` and applies publication lags like any other feature.
    """
    def fn(df: pd.DataFrame) -> pd.Series:
        s = np.log(df[column]) if log else df[column]
        return frac_diff_ffd(s, d, threshold, max_width)

    width = len(ffd_weights(d, threshold, max_width))
    return Feature(name=f"ffd_{column}_{d:g}", family="memory", fn=fn,
                   description=f"fractionally differenced {'log ' if log else ''}{column}, d={d:g}, "
                               f"{width}-row window",
                   params={"column": column, "d": d, "threshold": threshold, "log": log})
