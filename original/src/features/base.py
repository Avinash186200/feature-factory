"""
Contract every feature in this repo follows.

A feature is a function that takes market data and returns a value at each
timestamp, plus an explicit statement of when that value would actually have
been *knowable*. Point-in-time correctness is the property that matters most
here: if you can't say when a value became available, don't ship the feature.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class FeatureResult:
    values: pd.Series
    knowable_at: pd.Series
    name: str
    description: str


def make_feature(
    values: pd.Series,
    knowable_at: pd.Series,
    name: str,
    description: str,
) -> FeatureResult:
    if not values.index.equals(knowable_at.index):
        raise ValueError("values and knowable_at must share the same index")
    return FeatureResult(values=values, knowable_at=knowable_at, name=name, description=description)
