# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
- `original/`: the original research code, sanitised (names, credentials, data and notebook outputs removed).
- `CHANGELOG.md`, `CONTRIBUTING.md` and a documentation index in `docs/README.md`.
