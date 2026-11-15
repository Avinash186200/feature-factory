# Project Brief: The Feature Factory

## Motivation

Whatever the strategy — equities, commodities, FX, volatility, anything else — the single most
durable asset behind it is never the model. It's the features: the set of things someone decided
were worth measuring, measured correctly, and available when the model needs them. Models get
swapped out every year or two; a good feature keeps paying off for a decade.

We already have a base set of features. This project isn't about rebuilding that — it's about
building the *next* ones, and building a repeatable process for doing that, instead of adding
features to the pile one at a time by hand whenever someone has an idea.

This is your internship project, and it's real work: the features and pipeline you build here are
meant to be used in our own experiments and in the products we build on top of them, not filed away
once the internship ends.

## What you're actually building

Two things, not one:

1. **New candidate features.** Genuinely new predictors, not restatements of what's already in the
   base library.
2. **A pipeline that generates and screens candidates.** The bigger prize. If it works, it should
   be able to propose plausible new features semi-automatically, not just evaluate ones a human
   already thought of.

## Phase 1 — Pick a proving ground

Choose one or two strategies or markets to work against. Something we already run is easiest to get
data for, but a market you're curious about is fine too — the pipeline you build should generalize
regardless.

## Phase 2 — Baseline features

Before anything exotic, build a small set of standard features as your baseline and scaffolding for
the screening harness: price/volume-based measures, rolling statistics, simple cross-asset
comparisons. `src/features/example_realized_vol.py` is a worked example of the pattern — copy it.

## Phase 3 — Intraday and order-book (L2) features

Everything in the base library today is daily / end-of-day. This phase is about extending into
intraday, using **Level 2 (L2)** market data — and it's genuinely new ground, not an extension of
something we already have.

**What L2 data actually is.** Level 1 is just the top of book — best bid, best ask, last trade
price. Level 2 shows the *full depth* of the order book: multiple price levels on both the bid and
ask side, with the size resting at each level, updated continuously as orders are placed, changed,
or cancelled. It's a much richer picture of supply and demand than a single bid/ask snapshot.

**What can actually be built from it:**
- **Order flow imbalance** — the net change in bid depth vs. ask depth at the top of the book.
  Simple to compute, and one of the more robust known relationships to short-term price moves.
- **Depth / liquidity imbalance** — is one side of the book meaningfully thicker than the other,
  and how deep does real size sit versus how much is right at the touch.
- **Spread dynamics** — how the bid-ask spread widens and tightens, especially around volatility or
  news.
- **Microprice** — a size-weighted version of the midpoint (weighting toward whichever side is
  thinner), which tends to track the next trade price better than a naive midpoint.
- **Trade signing** — classifying trades as buyer- or seller-initiated (e.g. a tick-test style
  rule) to build a signed, directional flow feature rather than raw volume.

**Getting it on a real-time basis.** This is a different problem from the daily/EOD loaders
elsewhere in this repo:
- It needs a live/streaming feed rather than a once-a-day pull — either a direct exchange feed or a
  vendor that provides book depth (cost and licensing vary a lot here; scope the actual vendor with
  the team rather than assuming).
- The processing model is different too: instead of a batch job that recomputes a feature once a
  day, you maintain a running order-book state in memory and recompute features incrementally on
  each update (or on a fixed short cadence), since updates can arrive many times a second.
- **Prototype offline first.** Prove the feature has signal on a recorded/historical L2 dataset
  before building any real-time streaming infrastructure — there's no reason to solve the
  real-time/latency problem for a feature that turns out not to matter.

## Phase 4 — Push into new territory with LLMs

Use LLMs and agentic tooling (e.g. MCP-connected pipelines) to turn unstructured or real-time
information — news, filings, sentiment, sector commentary — into structured, testable features. The
genuinely open research question here is:

> **Can an LLM-assisted process propose useful candidate features faster or better than a person
> doing it by hand?**

Not "can it fetch data" — that part's easy. The hard part is whether what it proposes actually
survives the same scrutiny as a hand-built feature.

## Phase 5 — Screen every candidate the same way

Every candidate, regardless of where it came from, has to clear the same bar:

- **Incremental information.** Does it add anything on top of a naive persistence baseline (e.g.
  "tomorrow looks like today")? A lot of things look predictive purely because the target is
  autocorrelated — that's not a finding.
- **Stability.** Does it hold up across different time periods, or did it only work in one unusual
  stretch?
- **Point-in-time integrity.** Would this value actually have been knowable, in this form, at this
  timestamp? This is the single most common way research like this quietly cheats — using
  information (a revised number, a same-day close) that wasn't really available yet.

`src/pipeline/screen.py` has a minimal starting point (a plain information-coefficient calculation)
— it's intentionally incomplete. Extending it (decorrelating against the persistence baseline,
proper train/test splitting, stability diagnostics) is part of the project, not a prerequisite to
finish before starting.

## Phase 6 — Deliberately break it

Introduce a few classic mistakes on purpose and measure how much they inflate your results:

- Using a value that wasn't actually available yet (a common one: matching a data point to a
  timestamp when it hadn't been published)
- Fitting any normalization/scaling on the full sample instead of only the training fold
- A centered rolling window instead of a trailing one

Quantifying how much each mistake inflates the apparent result is one of the most useful things you
can hand back — it's a number people can use to sanity-check every other project, not just yours.

## Deliverables

- A feature catalogue: for each feature, what it measures, why it should matter, the evidence for
  and against it, and whether it earned its place.
- The screening pipeline itself, extended beyond the starting stub.
- A short write-up on the intraday/L2 track specifically: what you tried, what showed signal, and
  what a real-time version would need (data source, refresh cadence, compute cost).
- A short write-up on whether the LLM-assisted generation approach earned its extra complexity
  versus just doing it by hand.

## Data

There's an internal Polygon-based data pipeline elsewhere in the org (the `Data-Fetch` repo) for
higher-frequency equity data — ask the team if your feature needs it. `src/data/loaders.py` pulls
free public data (via `yfinance`) so you can start without waiting on that access.
