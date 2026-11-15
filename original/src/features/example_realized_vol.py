"""
Worked example: a realized-volatility feature computed from daily close prices.

This is the pattern every new feature in this repo should follow:
  1. pull the data it needs
  2. compute the raw signal
  3. state explicitly when each value would have been knowable
  4. wrap it as a FeatureResult so the screening pipeline can test it

Copy this file's structure for your own features rather than starting blank.
"""
import numpy as np
import pandas as pd

from .base import FeatureResult, make_feature


def realized_vol_21d(close: pd.Series) -> FeatureResult:
    """21-trading-day realized volatility of daily log returns, annualized."""
    log_ret = np.log(close / close.shift(1))
    rv = log_ret.rolling(21).std() * np.sqrt(252)

    # A 21-day trailing window ending at t is knowable the moment t's close
    # price is in -- i.e. at t itself. A centered window would NOT be knowable
    # at t, which is exactly the kind of mistake Phase 5 in the project brief
    # asks you to go find and quantify.
    knowable_at = rv.index.to_series()

    return make_feature(
        values=rv,
        knowable_at=knowable_at,
        name="realized_vol_21d",
        description="21-day rolling realized volatility of daily log returns, annualized.",
    )
