# Methodology and bias audit

## Background

Models get replaced every year or two, and a well-measured feature can outlive several of them. The brief I wrote for this project asked for a process, not a pile: generate candidates systematically, screen every candidate the same way (incremental information, stability, point-in-time integrity), and then deliberately break the process to measure how much each classic mistake inflates the result.

## Data contract

| Column | Meaning | Publication lag |
|---|---|---|
| `open, high, low, close, volume` | Traded asset, daily | 0 (known at that day's close) |
| `close_b` | A second, correlated asset | 0 |
| `macro` | Macro indicator indexed by **reference** date | 2 business days |

The synthetic generator plants three things:

- **A real signal**: day t's return loads on the 5-day mean of the log-volume z-score known at t-1 (`signal`, default 0.15 daily-vol units).
- **A publication-lag trap**: day t's return also loads on the macro change for reference date t-1, which is not published until t+1. Only a pipeline that ignores publication lags can see it.
- **A same-bar trap**: the close location within a day's range is mechanically tied to that day's return.

## Timing and the guard

Convention: the target on row t is day t's log return divided by the 21-day volatility up to t-1. A feature on row t must be built from data published by the close of t-1.

`compute(feature, df, pub_lags)`:

1. Shifts each column down by its publication lag (`published_view`), so a value sits on the row where it became known.
2. Calls the feature function, which is written as if its last row were "today".
3. Shifts the result one row.

`check_no_lookahead` verifies this independently. For four probe rows t, it scrambles every column from row `t - lag_c` onward, recomputes, and requires the value at t to be unchanged. It catches hidden centred windows, negative shifts and full-sample statistics (all tested). It also catches a pipeline that assumes the wrong publication lag: `compute_lags={}` with the true lags used for scrambling fails, and that is tested too. The screen runs this check on every candidate and excludes failures before computing any statistic.

## Candidate generation

34 candidates, all generated before screening:

| Family | Windows / variants |
|---|---|
| momentum | 5, 10, 21, 63-day log return |
| volatility | 5, 21, 63-day realised vol; 5/21 and 21/63 ratios |
| volume_z | 1, 5, 10, 21-day mean of the log-volume z-score (63-day trailing baseline) |
| range | 5, 21-day mean log(high/low) |
| close_location | 1, 5-day mean close location |
| trend | distance from the 10, 50 and 200-day moving average |
| cross_asset | 21, 63-day correlation and relative momentum vs `close_b` |
| macro | 1, 5, 21-period change; 252-day z-score (all as published) |
| interaction | pairwise products of `mom_21`, `volume_z_5`, `vol_21`, `dist_ma_50` |

The interaction inputs were fixed in code before any screening. Choosing them after looking at the single-feature results would make the multiple-testing count wrong.

## Screen

Run on the first 70% of the rows (discovery). All candidates are tested on exactly the same days (rows with any missing value are dropped).

1. **Partial rank IC.** Rank the feature, the target and yesterday's target. Regress the feature ranks and the target ranks on the persistence ranks, then correlate the residuals.
2. **Permutation p-value.** 500 circular shifts of the residualised feature matrix against the residualised target, with shifts of at least 63 days. A circular shift keeps each feature's autocorrelation, so slow-moving features are not given an artificially narrow null. The p-value is two-sided, with the usual +1 correction.
3. **BH step-up FDR** q-values over all candidates. BH controls the false discovery rate under independence or positive dependence. Many of these features are positively correlated by construction (shared windows), which fits that assumption.
4. **Family-wise max-T.** For each shift, take the maximum |IC| across all features. A feature's family-wise p-value is the share of shifts whose maximum beats its |IC|. Because the same shift is applied to every feature, their dependence is respected exactly (the Westfall-Young idea).
5. **Stability.** Split discovery into 4 blocks. The share of blocks whose IC has the same sign as the full-period IC must be at least 0.75.
6. `selected` = q <= 0.10 and stable. `strict` = selected and family-wise p <= 0.05.
7. **Hold-out.** The partial IC of every candidate on the last 30%, reported next to the discovery statistics and never used for selection. A test changes prices after the split and checks that every discovery statistic stays bit-identical.

## Why the target is vol-scaled

With raw returns, the target's variance moves with market volatility. Volatility-like features (and the magnitude of momentum) are large exactly when the target is noisy, so the real sampling spread of their IC is wider than under the circular-shift null, which breaks that alignment. Dividing by volatility known at t-1 removes most of the effect without touching day t. Measured on 20 null seeds (2,000 days, 200 shifts each): BH falsely selected at least one feature in 4 of 20 runs with raw returns, and in 3 of 20 with scaled returns. The family-wise test had a false selection in 1 of 20 either way. The effect is modest on this generator, so I report it as such.

## Breakage study (Phase 6 of the brief)

`breakage.py` computes a feature the wrong way and the right way and scores both on the same rows:

| Mistake | How it is reproduced |
|---|---|
| Same-bar data | `clv_1` with the guard off, so day t's bar is used for day t's return |
| Ignored publication lag | `macro_chg_1` computed as if `macro` were published on its reference date |
| Centred window | 5-day mean return with `center=True`, against trailing |
| Full-sample normalisation | `dist_ma_50` z-scored with whole-sample mean and sd, against a trailing 252-day z-score |

The first three inflate the partial rank IC from about 0 to 0.22-0.78. The fourth barely changes a rank IC, because a full-sample z-score is nearly a monotone transform. It inflates results when a model or a fixed threshold uses the raw values. The guard flags all four.

## Bias audit of the starter code

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | The feature contract declared `knowable_at`, but nothing enforced it and the target alignment was not defined | Same-bar leakage was one careless join away | Enforced shift to t-1 plus a perturbation check on every candidate |
| 2 | No publication-lag handling | Reference-date matching of macro data | Per-column lags applied before computation |
| 3 | A plain Spearman IC over the full sample | Multiple testing ignored, no hold-out | Permutation p-values, BH, family-wise max-T, stability, discovery/hold-out |
| 4 | No persistence baseline | Autocorrelation passed off as signal | Partial IC controlling for yesterday's return |
| 5 | (In my own build) a raw-return target | Permutation null too narrow for vol-like features | Vol-scaled target |
| 6 | Unseeded test data | Flaky, non-reproducible results | Seeds everywhere |

## Not done

- **Intraday and order-book (L2) features** (order-flow imbalance, depth imbalance, microprice, trade signing). The brief made this a separate track. It needs a recorded L2 dataset to prototype offline before any real-time work, and I did not want to commit vendor data. The `Feature` interface works unchanged on intraday bars.
- **LLM-assisted feature proposal.** Also a separate track in the brief. The right way to judge it is to feed its proposals through this same screen with the multiple-testing count including every proposal. That harness is what this repo provides.
- **Real data.** I don't publish any real-data screen here.
