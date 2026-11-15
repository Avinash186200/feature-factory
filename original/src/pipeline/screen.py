"""
Minimal feature-screening harness.

This is deliberately small -- extending it is part of the project, not a
prerequisite to finish before starting. See docs/project_brief.md Phase 4 for
what a fuller version should cover: decorrelation against a persistence
baseline, proper walk-forward splitting, stability diagnostics, and a
significance test that accounts for overlapping/autocorrelated data rather
than a naive t-stat.
"""
import pandas as pd
from scipy.stats import spearmanr


def information_coefficient(feature: pd.Series, forward_target: pd.Series) -> float:
    """Spearman rank correlation between a feature and a forward-looking target,
    aligned and NA-dropped. A starting point, not a final metric -- it doesn't
    yet correct for persistence, overlapping windows, or multiple testing."""
    aligned = pd.concat([feature, forward_target], axis=1).dropna()
    if len(aligned) < 30:
        return float("nan")
    ic, _ = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return ic
