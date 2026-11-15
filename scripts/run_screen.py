"""Generate the candidate features, screen them, and run the breakage study.

Examples:
    python scripts/run_screen.py                 # synthetic data with a planted signal
    python scripts/run_screen.py --signal 0      # pure noise: the screen should select nothing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from featurefactory.breakage import run_breakage  # noqa: E402
from featurefactory.data import PUB_LAGS, TRUE_FAMILY, make_synthetic_market, target_returns  # noqa: E402
from featurefactory.library import generate_candidates  # noqa: E402
from featurefactory.screen import ScreenConfig, run_screen  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-days", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--signal", type=float, default=0.15, help="planted signal strength (0 = none)")
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--q", type=float, default=0.10, help="false discovery rate target")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = make_synthetic_market(n_days=args.n_days, seed=args.seed, signal=args.signal)
    y = target_returns(df)
    cands = generate_candidates()
    cfg = ScreenConfig(n_perm=args.n_perm, q=args.q, seed=args.seed)
    table = run_screen(cands, df, y, PUB_LAGS, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "feature_catalogue.csv")
    pd.set_option("display.width", 140)

    print(f"Synthetic market: {len(df)} days, seed {args.seed}, planted signal {args.signal}")
    print(f"{len(cands)} candidates generated, {len(table)} passed the leakage guard, "
          f"rejected: {table.attrs['rejected_by_guard'] or 'none'}")
    print(f"Discovery period ends {table.attrs['split'].date()}; hold-out used once at the end\n")
    print(table.head(12).round(4).to_string())

    planted = table.index.str.contains(TRUE_FAMILY)  # the planted driver, alone or in an interaction
    print(f"\nnaive p < 0.05: {int(table['naive_p05'].sum())} features, "
          f"{int((table['naive_p05'] & planted).sum())} involve the planted driver")
    for col, label in (("selected", f"BH q <= {args.q} + stable"), ("strict", "also family-wise p <= 0.05")):
        sel = table[table[col]]
        is_true = sel.index.str.contains(TRUE_FAMILY)
        print(f"{label}: {len(sel)} selected, {int(is_true.sum())} involve the planted driver, "
              f"{int((~is_true).sum())} false")
        if len(sel):
            print(f"   {', '.join(sel.index)}")
            msg = f"   hold-out partial IC: planted mean {sel.loc[is_true, 'holdout_ic'].mean():.4f}"
            if (~is_true).any():
                msg += f", false mean {sel.loc[~is_true, 'holdout_ic'].mean():.4f}"
            print(msg)

    brk = run_breakage(df, y, PUB_LAGS)
    brk.to_csv(out / "breakage.csv")
    print("\nBreakage study (partial rank IC, whole sample):")
    print(brk.round(4).to_string())


if __name__ == "__main__":
    main()
