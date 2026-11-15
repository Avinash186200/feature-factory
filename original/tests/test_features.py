import numpy as np
import pandas as pd

from src.features.example_realized_vol import realized_vol_21d


def test_realized_vol_shape_and_index():
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.normal(0, 1, size=100)), index=idx)

    result = realized_vol_21d(close)

    assert result.values.index.equals(close.index)
    assert result.knowable_at.equals(result.values.index.to_series())
    # first 20 points can't have a full 21-day window yet
    assert result.values.iloc[:20].isna().all()
