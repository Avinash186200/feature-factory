# Feature Factory

[![CI](https://github.com/Avinash186200/feature-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/Avinash186200/feature-factory/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11-blue) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, repeatable way to generate many candidate predictive features, prove each one only uses information that was available, and screen them without fooling yourself through multiple testing.

This is a research project I designed and mentored. I wrote the brief and the starter code. The brief had two goals: new candidate features, and, more important, a pipeline that proposes and screens candidates the same way every time instead of adding them by hand. This repo is my own working implementation of the pipeline. It runs end to end on a **synthetic market with known, planted structure**, so you can see what the screen finds, what it wrongly lets through, and how the hold-out catches it.

## At a glance

| | |
|---|---|
| **Input** | Daily `open, high, low, close, volume`, a second asset `close_b`, and a `macro` series published with a lag (synthetic, same shape as a real daily panel) |
| **Method** | A pre-registered grid of 34 candidates (transforms x inputs x windows, plus interactions), a leakage guard, partial rank IC against a persistence baseline, a circular-shift permutation test, BH step-up FDR control plus family-wise max-T, a stability filter, and one look at a hold-out |
| **Output** | A feature catalogue (IC, p, q, family-wise p, stability, hold-out IC, selected or not) and a breakage study that measures how much each classic mistake inflates the result |
| **Leakage controls** | Every feature at t is built from data published by t-1, and a perturbation check rejects any feature that reads later data. Publication lags per column. Discovery and hold-out split. Noise test in CI |
| **Status** | Framework complete and tested on synthetic data. No real-data claims |

## How it works

```
data ──► published view ──► candidate grid ──► guard (shift + perturbation check) ──► screen (discovery) ──► hold-out
data.py   guard.py           library.py          guard.py                               screen.py               screen.py
```

1. **Generate.** `library.py` builds every candidate up front: momentum, volatility and vol ratios, volume z-scores, range, close location, distance from moving averages, cross-asset correlation and relative momentum, macro changes and level, and six pairwise interactions. The full count is known before any test is run, so the multiple-testing correction covers all of them.
2. **Guard.** `compute` moves each column to the row on which it was published (`macro` arrives 2 days after its reference date), runs the feature function, and shifts the result by one row. The feature on row t therefore uses data up to t-1. `check_no_lookahead` then verifies this rather than trusting it: it scrambles every input not yet published at t-1 and checks that the value at t does not move. A hidden centred window, `shift(-k)` or full-sample statistic fails, and the candidate is rejected before screening.
3. **Target.** Day t's return divided by trailing volatility known at t-1.
4. **Screen** (first 70% of the sample only):
   - **Partial rank IC**: rank-transform each feature and the target, and remove yesterday's return from both, so the IC measures information beyond simple persistence.
   - **Circular-shift permutation p-value**, 500 shifts. Shifting keeps each feature's own autocorrelation, which a plain shuffle would destroy.
   - **BH step-up FDR q-value** across all 34 candidates, and a **family-wise max-|IC| p-value** from the same shifts applied to all features at once.
   - **Stability**: the IC has the same sign in at least 3 of 4 sub-periods.
   - `selected` = q <= 0.10 and stable. `strict` = also family-wise p <= 0.05.
5. **Hold-out** (last 30%): the partial IC of every candidate, looked at once, after selection.

## Results on synthetic data

`python scripts/run_screen.py` (seed 7, 2,500 days, planted signal 0.15). The planted driver is the 5-day volume z-score.

| Rule | Selected | Contain the planted driver | False | Hold-out IC (planted / false) |
|---|---|---|---|---|
| Naive p < 0.05 | 10 | 5 | 5 | n/a |
| BH q <= 0.10 + stable | 8 | 5 | 3 | 0.137 / -0.005 |
| Also family-wise p <= 0.05 | 4 | 4 | 0 | 0.142 / none |

What this shows:

- The naive rule is half wrong. With 34 candidates, a 5% threshold is expected to pass one or two by luck. Correlated features cluster, so in practice it passed five.
- FDR control is a candidate list, not a verdict. At q = 0.10 the three false picks (`clv_1`, `volratio_21_63`, `macro_chg_21`) got through. FDR is a rate in expectation, so one run can be worse. All three collapse to about zero on the hold-out, while the planted features keep an IC of 0.11 to 0.16. I checked `clv_1` on a 20,000-day sample, and its true IC there is about zero.
- The family-wise max-T rule picked only true features here, at the price of missing `volume_z_21`.
- **Noise check**: with the signal and the macro trap switched off, over 20 seeds, BH selected something in 3 runs and max-T in 1, close to their 10% and 5% levels. A CI test checks that a null seed selects nothing.

**Breakage study**: each classic mistake made on purpose, with partial rank IC over the whole sample:

| Mistake | IC wrong | IC right | Guard flags it |
|---|---|---|---|
| Same-bar data (day t's bar for day t's return) | 0.782 | -0.055 | yes |
| Macro value used on its reference date, not its release date | 0.221 | 0.002 | yes |
| Centred 5-day window instead of trailing | 0.474 | 0.045 | yes |
| Full-sample z-score instead of trailing | 0.025 | 0.008 | yes |

The first three mistakes turn noise into an apparently excellent feature. The full-sample z-score barely moves a **rank** IC, because it is close to a monotone transform. It matters once a model or a fixed threshold consumes the values, and the guard flags it either way.

## Leakage and bias audit

| Issue | Effect | Fix |
|---|---|---|
| The starter feature contract declared `knowable_at` but nothing enforced it, and the target alignment was undefined | A feature knowable at t's close could be used to predict day t's own return (same-bar) | `compute` shifts to t-1, and `check_no_lookahead` perturbation-tests every candidate; the screen rejects failures |
| No handling of publication lags | Macro data matched to its reference date: IC inflated from 0.00 to 0.22 in the breakage study | Per-column `pub_lags`, applied before any feature is computed; tested |
| The starter screen was a plain Spearman IC over the whole sample, with no correction | With 34 candidates, a naive 5% rule passed 10 here, 5 of them false | Partial IC, permutation p-values, BH and family-wise control, a stability filter, a discovery/hold-out split |
| The IC was not measured against a persistence baseline | Features that only restate yesterday's return look predictive | Yesterday's return is removed from both feature and target ranks |
| A naive i.i.d. significance test on autocorrelated features | p-values too small for slow-moving features | Circular-shift permutation keeps each feature's autocorrelation |
| (Found while building this) a raw-return target is heteroskedastic, and vol-like features line up with its variance | The shift null gets narrower than the true spread. Over 20 null seeds, BH falsely selected something in 4 with raw returns, against 3 with vol-scaled returns. A modest effect, but in the wrong direction | Target is the return scaled by trailing (t-1) volatility |
| The starter test used unseeded random data | Non-reproducible | Every generator and test is seeded |

## Repo layout

```
src/featurefactory/   data.py · guard.py · library.py · screen.py · breakage.py
scripts/              run_screen.py
tests/                guard (good and bad features, publication lag), BH, hold-out isolation, noise, signal, breakage
docs/                 methodology.md
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_screen.py                 # planted signal
python scripts/run_screen.py --signal 0      # pure noise
```

To add a feature, write a function of the data frame as if its last row were "today", and wrap it in `Feature(name, family, fn)`. The guard handles the shift and publication lags, and rejects it if it peeks.

## Original research code

[`original/`](original/README.md) holds the code as I first wrote it for this work, sanitised and nothing else:
client, company and person names replaced, credentials and local paths removed, data files and notebook outputs
dropped. It is kept so the real work can be read alongside the maintained version. Some of it contains the issues
described above; comparing the two shows each fix.

## License

MIT

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [CHANGELOG](CHANGELOG.md).
