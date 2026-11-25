# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.3.0] - 2026-11-25

### Added
- `src/featurefactory/fracdiff.py`: fixed-width-window fractional differentiation (FFD) weights and transform,
  a numpy augmented Dickey-Fuller test (AIC lag choice, MacKinnon 2010 critical values), a search for the
  smallest d that passes the test, and `fracdiff_feature`, which runs the transform through the existing
  point-in-time guard.
- `scripts/run_fracdiff.py`: chooses d on the discovery period of the synthetic market and on seeded random walks.
- `tests/test_fracdiff.py`: weight recursion, d = 0 and d = 1, no look-ahead (raw and guarded), the all-NaN
  guard case, ADF sanity checks, an ADF cross-check against statsmodels (skipped when it is not installed), and the
  min-d search on random walks (22 tests).
- `guard.lookahead_report`, which returns the look-ahead check result with a reason.
- README and methodology sections on memory vs stationarity, with the script's output. CI runs the new script.

### Changed
- `featurefactory.__version__` now matches the changelog (it still read 0.1.0).

### Fixed
- `check_no_lookahead` passed a feature that was NaN on every probe row, because nothing was compared, so a window
  longer than the data could hide a `shift(-k)`. It now fails such a feature. All 34 screen candidates still pass
  and the screen results are unchanged.

## [1.2.0] - 2026-11-15

### Changed
- Docs: training-grade README and walkthrough. Added what problem this solves, what you will learn, key concepts,
  a mermaid diagram of the pipeline, an accurate repo layout, and a clearer synthetic-data label on the results.

### Fixed
- `--signal 0` was documented as "pure noise", but the macro look-ahead trap stayed on, so at seed 7 BH still
  selected `macro_chg_21`. Added `--macro-trap` to `scripts/run_screen.py`; the pure-noise command is now
  `--signal 0 --macro-trap 0`, which selects nothing under BH or max-T.

## [1.0.0] - 2026-11-15

### Added
- First public release: the maintained code, documentation, tests and CI.
- `original/`: the original research code (read-only).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.
