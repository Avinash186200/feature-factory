"""Find the smallest fractional-differencing order d that makes a series stationary.

Runs on the synthetic market (discovery period only, the first 70% of rows) and on a plain
seeded random walk, prints the ADF statistic and the correlation with the original series
for each d, and checks the resulting feature with the look-ahead guard.

Examples:
    python scripts/run_fracdiff.py
    python scripts/run_fracdiff.py --threshold 1e-3     # shorter windows, more rows kept
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurefactory.data import PUB_LAGS, make_synthetic_market  # noqa: E402
from featurefactory.fracdiff import ffd_weights, fracdiff_feature, min_ffd_d  # noqa: E402
from featurefactory.guard import lookahead_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-days", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=1e-4, help="drop FFD weights smaller than this")
    ap.add_argument("--discovery-frac", type=float, default=0.70)
    ap.add_argument("--rw-seeds", type=int, default=20, help="extra random walks to summarise (0 = skip)")
    args = ap.parse_args()

    df = make_synthetic_market(n_days=args.n_days, seed=args.seed)
    n_disc = int(len(df) * args.discovery_frac)
    disc = df.iloc[:n_disc]
    rw = pd.Series(np.cumsum(np.random.default_rng(args.seed).standard_normal(n_disc)), name="random_walk")

    series = {"log close": np.log(disc["close"]), "macro (as published)": disc["macro"].shift(PUB_LAGS["macro"]),
              "random walk": rw}
    pd.set_option("display.width", 140)
    print(f"Synthetic market: seed {args.seed}, {len(df)} days; d is chosen on the first {n_disc} rows only")
    print(f"FFD weight threshold {args.threshold:g}\n")

    chosen = {}
    for label, s in series.items():
        d, table = min_ffd_d(s.dropna(), threshold=args.threshold)
        chosen[label] = d
        print(f"--- {label}: min d = {'none' if d is None else d}")
        print(table.round(3).to_string())
        print()

    print("Summary (sum(w) = share of the level left in the output by the truncated window):")
    for label, d in chosen.items():
        if d is None:
            print(f"   {label:22s} d* = none (the largest d on the grid fails the test)")
            continue
        _, table = min_ffd_d(series[label].dropna(), d_grid=np.array([d, 1.0]), threshold=args.threshold)
        print(f"   {label:22s} d* = {d:.2f}  sum(w) = {table.loc[d, 'sum_w']:.3f}  "
              f"corr at d* = {table.loc[d, 'corr']:.3f}  "
              f"corr at d = 1 (plain difference) = {table.loc[1.0, 'corr']:.3f}")

    if args.rw_seeds:
        ds = [min_ffd_d(pd.Series(np.cumsum(np.random.default_rng(s).standard_normal(n_disc))),
                        threshold=args.threshold)[0] for s in range(args.rw_seeds)]
        ds = np.array([np.nan if d is None else d for d in ds])
        med = float(np.nanmedian(ds))
        print(f"\nRandom walks, seeds 0-{args.rw_seeds - 1}: min d median {med:.2f} "
              f"(sum(w) {ffd_weights(med, args.threshold).sum():.3f}), "
              f"range {np.nanmin(ds):.2f}-{np.nanmax(ds):.2f}, no d found in {int(np.isnan(ds).sum())}")
        print("   A random walk needs d > 0.5 in theory; a smaller d here means the ADF test missed "
              "the level left by the truncated window.")

    if chosen["log close"] is None:
        print("\nNo d found for log close; guard check skipped")
        return
    feat = fracdiff_feature("close", d=chosen["log close"], threshold=args.threshold)
    ok, reason = lookahead_report(feat, df, PUB_LAGS)
    print(f"\nGuard check for {feat.name} ({feat.description}): "
          f"{'passes, uses only data up to t-1' if ok else 'FAILED'} ({reason})")


if __name__ == "__main__":
    main()
