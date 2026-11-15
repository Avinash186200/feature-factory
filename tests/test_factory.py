import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurefactory.breakage import run_breakage  # noqa: E402
from featurefactory.data import PUB_LAGS, TRUE_FAMILY, make_synthetic_market, target_returns  # noqa: E402
from featurefactory.guard import Feature, check_no_lookahead, compute  # noqa: E402
from featurefactory.library import generate_candidates  # noqa: E402
from featurefactory.screen import ScreenConfig, bh_qvalues, run_screen  # noqa: E402

CFG = ScreenConfig(n_perm=200, seed=1)


@pytest.fixture(scope="module")
def market():
    return make_synthetic_market(n_days=2000, seed=5)


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates()


def test_candidate_names_are_unique(candidates):
    names = [f.name for f in candidates]
    assert len(names) == len(set(names))


def test_every_generated_feature_passes_the_guard(market, candidates):
    for f in candidates:
        assert check_no_lookahead(f, market, PUB_LAGS), f.name


def test_guarded_value_at_t_is_raw_value_at_t_minus_1(market, candidates):
    f = next(c for c in candidates if c.name == "mom_21")
    raw = f.fn(market)
    guarded = compute(f, market, PUB_LAGS)
    pd.testing.assert_series_equal(guarded.iloc[1:], raw.shift(1).iloc[1:].rename("mom_21"))


@pytest.mark.parametrize(
    "fn",
    [
        lambda d: d["close"].pct_change().rolling(5, center=True).mean(),  # centred window
        lambda d: d["close"].pct_change().shift(-2),  # explicit future value
        lambda d: (d["close"] - d["close"].mean()) / d["close"].std(),  # full-sample stats
    ],
)
def test_guard_catches_hidden_lookahead(market, fn):
    assert not check_no_lookahead(Feature("bad", "bad", fn), market, PUB_LAGS)


def test_guard_catches_ignored_publication_lag(market):
    macro = Feature("macro_chg_1", "macro", lambda d: d["macro"].diff(1))
    assert check_no_lookahead(macro, market, PUB_LAGS)
    assert not check_no_lookahead(macro, market, PUB_LAGS, compute_lags={})


def test_macro_feature_ignores_unpublished_values(market):
    macro = Feature("macro_chg_1", "macro", lambda d: d["macro"].diff(1))
    t = 900
    base = compute(macro, market, PUB_LAGS)
    altered = market.copy()
    altered.iloc[t - PUB_LAGS["macro"] :, altered.columns.get_loc("macro")] += 50.0
    assert compute(macro, altered, PUB_LAGS).iloc[t] == base.iloc[t]


def test_screen_rejects_leaky_candidate(market, candidates):
    leaky = Feature("leaky", "bad", lambda d: d["close"].pct_change().shift(-1))
    table = run_screen(candidates + [leaky], market, target_returns(market), PUB_LAGS, CFG)
    assert table.attrs["rejected_by_guard"] == ["leaky"]
    assert "leaky" not in table.index


def test_bh_qvalues_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(bh_qvalues(p), [0.04, 0.16 / 3, 0.16 / 3, 0.20])
    assert (bh_qvalues(p) >= p).all()


def test_discovery_statistics_ignore_the_hold_out(market, candidates):
    y = target_returns(market)
    a = run_screen(candidates, market, y, PUB_LAGS, CFG)
    split = a.attrs["split"]
    altered = market.copy()
    after = altered.index >= split
    altered.loc[after, ["close", "close_b"]] *= np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.05, after.sum())))[:, None]
    b = run_screen(candidates, altered, target_returns(altered), PUB_LAGS, CFG)
    cols = ["partial_ic", "p_perm", "q_bh", "stability", "selected"]
    pd.testing.assert_frame_equal(a[cols].sort_index(), b[cols].sort_index())


def test_noise_selects_nothing(candidates):
    df = make_synthetic_market(n_days=2000, seed=3, signal=0.0, macro_trap=0.0)
    table = run_screen(candidates, df, target_returns(df), PUB_LAGS, CFG)
    assert table["selected"].sum() == 0
    assert table["strict"].sum() == 0


def test_planted_signal_is_found_and_holds_out(market, candidates):
    table = run_screen(candidates, market, target_returns(market), PUB_LAGS, CFG)
    sel = table[table["selected"]]
    assert (sel["family"] == TRUE_FAMILY).any()
    assert sel.loc[sel["family"] == TRUE_FAMILY, "holdout_ic"].mean() > 0


def test_breakage_mistakes_inflate_and_are_flagged(market):
    brk = run_breakage(market, target_returns(market), PUB_LAGS)
    assert brk["guard_flags_it"].all()
    inflating = brk.iloc[:3]  # same-bar, publication lag, centred window
    assert (inflating["abs_inflation"] > 0.1).all()
