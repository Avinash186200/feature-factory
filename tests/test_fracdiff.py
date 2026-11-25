import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurefactory.data import PUB_LAGS, make_synthetic_market  # noqa: E402
from featurefactory.fracdiff import (adf_test, choose_d, ffd_weights, frac_diff_ffd, fracdiff_feature,  # noqa: E402
                                     min_ffd_d)
from featurefactory.guard import Feature, check_no_lookahead, compute, lookahead_report  # noqa: E402


def _walk(seed: int) -> pd.Series:
    idx = pd.bdate_range("2015-01-01", periods=1500)
    return pd.Series(np.cumsum(np.random.default_rng(seed).standard_normal(1500)), index=idx, name="x")


@pytest.fixture(scope="module")
def walk():
    # At a 5% level the ADF test wrongly calls about 1 random walk in 20 stationary; seed 0 is not one of them.
    return _walk(0)


@pytest.mark.parametrize("d", [0.1, 0.35, 0.5, 0.8, 1.7])
def test_weights_follow_the_recursion(d):
    w = ffd_weights(d, threshold=1e-5)
    assert w[0] == 1.0
    for k in range(1, len(w)):
        assert w[k] == pytest.approx(-w[k - 1] * (d - k + 1) / k, rel=1e-12)
    # the generalised binomial coefficient, written as a product: w_k = prod_{i<k} (i - d) / (i + 1)
    k = min(6, len(w) - 1)
    assert w[k] == pytest.approx(np.prod([(i - d) / (i + 1) for i in range(k)]), rel=1e-12)
    assert (np.abs(w) >= 1e-5).all()


def test_weights_known_values_and_window_width():
    np.testing.assert_allclose(ffd_weights(0.5, threshold=0.03), [1.0, -0.5, -0.125, -0.0625, -0.0390625])
    widths = [len(ffd_weights(d, 1e-4)) for d in (0.4, 0.6, 0.8)]
    assert widths == sorted(widths, reverse=True)  # larger d, faster decay, shorter window


def test_d0_returns_the_series_and_d1_the_first_difference(walk):
    np.testing.assert_allclose(ffd_weights(0.0), [1.0])
    np.testing.assert_allclose(ffd_weights(1.0), [1.0, -1.0])
    pd.testing.assert_series_equal(frac_diff_ffd(walk, 0.0), walk)
    d1 = frac_diff_ffd(walk, 1.0)
    assert np.isnan(d1.iloc[0])
    np.testing.assert_allclose(d1.iloc[1:], walk.diff().iloc[1:], atol=1e-12)


def test_d_close_to_one_is_close_to_the_first_difference(walk):
    near = frac_diff_ffd(walk, 0.99, threshold=1e-3)
    width = len(ffd_weights(0.99, threshold=1e-3))
    diff = walk.diff()
    valid = near.notna()
    # the remaining weights sum to about 0.01 per lag, so the gap is small next to the step size
    assert np.corrcoef(near[valid], diff[valid])[0, 1] > 0.99
    assert near.iloc[: width - 1].isna().all() and near.iloc[width - 1:].notna().all()


def test_transform_never_reads_future_values(walk):
    t = 900
    base = frac_diff_ffd(walk, 0.4)
    altered = walk.copy()
    altered.iloc[t + 1:] = altered.iloc[t + 1:] * 3.0 + 100.0
    new = frac_diff_ffd(altered, 0.4)
    pd.testing.assert_series_equal(new.iloc[: t + 1], base.iloc[: t + 1])
    assert not np.isclose(new.iloc[t + 1], base.iloc[t + 1])


def test_guarded_feature_is_point_in_time():
    df = make_synthetic_market(n_days=1500, seed=5)
    feat = fracdiff_feature("close", d=0.45)
    assert check_no_lookahead(feat, df, PUB_LAGS)
    t = 800
    base = compute(feat, df, PUB_LAGS)
    altered = df.copy()
    altered.iloc[t:, altered.columns.get_loc("close")] *= 1.5  # day t's close and later
    new = compute(feat, altered, PUB_LAGS)
    pd.testing.assert_series_equal(new.iloc[: t + 1], base.iloc[: t + 1])
    # the guard shifts by one row: row t uses the transform up to t-1
    raw = frac_diff_ffd(np.log(df["close"]), 0.45)
    np.testing.assert_allclose(base.iloc[t], raw.iloc[t - 1])


def test_adf_separates_a_random_walk_from_noise(walk):
    rw = adf_test(walk)
    noise = adf_test(pd.Series(np.random.default_rng(3).standard_normal(1500)))
    assert rw["stat"] > rw["crit"]["5%"]
    assert noise["stat"] < noise["crit"]["5%"]
    assert rw["crit"]["1%"] < rw["crit"]["5%"] < rw["crit"]["10%"]
    assert rw["crit"]["5%"] == pytest.approx(-2.863, abs=0.002)  # MacKinnon (2010), constant, n near 1500


@pytest.mark.parametrize("seed", range(8))
def test_min_d_search_on_random_walks(seed):
    d, table = min_ffd_d(_walk(seed))
    assert d is not None and 0 < d < 1
    assert not table.loc[0.0, "stationary"]
    assert table.loc[table.index >= d, "stationary"].all()
    block = table[table.index >= d]
    assert block["corr"].idxmax() == d  # the highest correlation among the passing values
    assert table.loc[d, "corr"] > table.loc[1.0, "corr"]
    # the truncated window leaves sum(w) of the level in the output; it is zero only at d = 1
    np.testing.assert_allclose(table["sum_w"], [ffd_weights(x).sum() for x in table.index])
    assert table.loc[d, "sum_w"] > 0 and table.loc[1.0, "sum_w"] == 0


def test_guard_rejects_a_feature_that_is_nan_on_every_probe():
    df = make_synthetic_market(n_days=800, seed=5)
    w = ffd_weights(0.05, threshold=1e-7, max_width=5000)
    assert len(w) > len(df)  # window longer than the data: the transform is all NaN

    def leaky(d):
        return frac_diff_ffd(np.log(d["close"]), 0.05, threshold=1e-7, max_width=5000).shift(-5)

    feat = Feature("ffd_leaky", "memory", leaky)
    assert compute(feat, df, PUB_LAGS).isna().all()
    ok, reason = lookahead_report(feat, df, PUB_LAGS)
    assert not ok and "NaN" in reason
    assert not check_no_lookahead(feat, df, PUB_LAGS)


def test_adf_matches_statsmodels_when_installed(walk):
    sm = pytest.importorskip("statsmodels.tsa.stattools")
    for x in (walk, pd.Series(np.random.default_rng(3).standard_normal(1500))):
        ours = adf_test(x)
        ref = sm.adfuller(x.to_numpy(), regression="c", autolag="AIC")
        assert ours["stat"] == pytest.approx(ref[0], abs=1e-8)
        assert ours["lag"] == ref[2] and ours["nobs"] == ref[3]
        for k in ("1%", "5%", "10%"):
            assert ours["crit"][k] == pytest.approx(ref[4][k], abs=1e-8)


def test_choose_d_skips_an_isolated_pass():
    idx = pd.Index([0.0, 0.25, 0.5, 0.75, 1.0], name="d")
    assert choose_d(pd.DataFrame({"stationary": [False, True, False, True, True]}, index=idx)) == 0.75
    assert choose_d(pd.DataFrame({"stationary": [True] * 5}, index=idx)) == 0.0
    assert choose_d(pd.DataFrame({"stationary": [True, True, True, True, False]}, index=idx)) is None
