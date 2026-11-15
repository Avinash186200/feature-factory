"""Screen every candidate the same way, with the multiple-testing count fixed in advance.

Steps, on the **discovery** period only:

1. Rank-transform each feature and the target, and remove the persistence baseline
   (yesterday's return) from both by OLS. What remains is *incremental* information.
2. Partial IC = correlation of the two residuals.
3. **Circular-shift permutation test**: shift each feature's residual against the target by
   a random offset many times. Shifting keeps each feature's own autocorrelation, which an
   i.i.d. shuffle would destroy, so the null is honest for slow-moving features.
4. **Multiple testing**: BH step-up FDR q-values across all candidates (``selected``:
   a candidate list at false discovery rate ``q``), plus a family-wise max-|IC|
   permutation p-value, with the same shifts applied to all features at once so their
   dependence is respected (``strict``: also family-wise significant).
5. **Stability**: the share of discovery sub-periods in which the IC has the same sign.

The hold-out period is used once, at the end, to report the IC of the selected features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .guard import Feature, check_no_lookahead, compute


def _rank(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()


def _residualise(y: np.ndarray, base: np.ndarray) -> np.ndarray:
    """OLS residual of ``y`` on ``[1, base]`` (column-wise if ``y`` is 2-D)."""
    B = np.column_stack([np.ones(len(base)), base])
    coef, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ coef


def _standardise(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - x.mean(axis=0)) / sd


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """BH step-up false-discovery-rate q-values (monotone)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


@dataclass(frozen=True)
class ScreenConfig:
    n_perm: int = 500
    q: float = 0.10
    min_shift: int = 63
    n_blocks: int = 4
    min_stability: float = 0.75
    fwer: float = 0.05
    holdout_frac: float = 0.30
    seed: int = 7


def build_matrix(
    features: list[Feature], df: pd.DataFrame, pub_lags: dict, guard: bool = True
) -> pd.DataFrame:
    """Compute every candidate (guarded by default) into one frame."""
    return pd.concat([compute(f, df, pub_lags, guard=guard) for f in features], axis=1)


def screen_matrix(
    F: pd.DataFrame, target: pd.Series, families: dict[str, str], cfg: ScreenConfig = ScreenConfig()
) -> pd.DataFrame:
    """Run steps 1 to 5 on a discovery-period feature matrix and target.

    Rows with any missing value are dropped, so every feature is tested on exactly the
    same days.
    """
    base = target.shift(1).rename("persistence")
    data = pd.concat([F, target.rename("y"), base], axis=1, sort=True).dropna()
    n = len(data)
    Xr = np.column_stack([_rank(data[c].to_numpy()) for c in F.columns])
    yr = _rank(data["y"].to_numpy())
    br = _rank(data["persistence"].to_numpy())
    Xres = _standardise(_residualise(Xr, br))
    yres = _standardise(_residualise(yr, br))
    ic = Xres.T @ yres / n

    rng = np.random.default_rng(cfg.seed)
    shifts = rng.integers(cfg.min_shift, n - cfg.min_shift, size=cfg.n_perm)
    null = np.empty((cfg.n_perm, Xres.shape[1]))
    for i, s in enumerate(shifts):
        null[i] = np.roll(Xres, s, axis=0).T @ yres / n
    abs_ic = np.abs(ic)
    p_perm = (1 + (np.abs(null) >= abs_ic).sum(axis=0)) / (cfg.n_perm + 1)
    max_null = np.abs(null).max(axis=1)
    p_fwer = (1 + (max_null[:, None] >= abs_ic).sum(axis=0)) / (cfg.n_perm + 1)
    q_bh = bh_qvalues(p_perm)

    edges = np.linspace(0, n, cfg.n_blocks + 1).astype(int)
    block_ic = np.array(
        [Xres[a:b].T @ yres[a:b] / (b - a) for a, b in zip(edges[:-1], edges[1:])]
    )
    stability = (np.sign(block_ic) == np.sign(ic)).mean(axis=0)

    out = pd.DataFrame(
        {
            "family": [families[c] for c in F.columns],
            "partial_ic": ic,
            "p_perm": p_perm,
            "q_bh": q_bh,
            "p_fwer": p_fwer,
            "stability": stability,
        },
        index=F.columns,
    )
    out["naive_p05"] = out["p_perm"] < 0.05
    out["selected"] = (out["q_bh"] <= cfg.q) & (out["stability"] >= cfg.min_stability)
    out["strict"] = out["selected"] & (out["p_fwer"] <= cfg.fwer)
    out.attrs["n_obs"] = n
    return out


def partial_ic(F: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Partial rank IC (persistence removed) of each column of ``F`` against ``target``."""
    base = target.shift(1)
    data = pd.concat([F, target.rename("y"), base.rename("b")], axis=1, sort=True).dropna()
    br = _rank(data["b"].to_numpy())
    yres = _standardise(_residualise(_rank(data["y"].to_numpy()), br))
    res = {}
    for c in F.columns:
        x = _standardise(_residualise(_rank(data[c].to_numpy()), br))
        res[c] = float(x @ yres / len(x))
    return pd.Series(res, name="partial_ic")


def run_screen(
    features: list[Feature],
    df: pd.DataFrame,
    target: pd.Series,
    pub_lags: dict,
    cfg: ScreenConfig = ScreenConfig(),
) -> pd.DataFrame:
    """Full pipeline: leakage check, discovery screen, then one look at the hold-out.

    Candidates that fail ``check_no_lookahead`` are reported and excluded before any
    statistic is computed.
    """
    passed = [f for f in features if check_no_lookahead(f, df, pub_lags)]
    rejected = sorted({f.name for f in features} - {f.name for f in passed})
    F = build_matrix(passed, df, pub_lags)
    split = df.index[int(len(df) * (1 - cfg.holdout_frac))]
    disc = F.index < split
    families = {f.name: f.family for f in passed}
    table = screen_matrix(F[disc], target[disc], families, cfg)
    table["holdout_ic"] = partial_ic(F[~disc], target[~disc]).reindex(table.index)
    table.attrs["rejected_by_guard"] = rejected
    table.attrs["split"] = split
    return table.sort_values("p_perm")
