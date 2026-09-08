#!/usr/bin/env python3
"""Did changing DENSE_DECODE_BACKEND actually remove the penalty?

    python scripts/compare_decode_eras.py \
        --before results/stage5/phases.parquet \
        --after  results/stage5_flashdecode/phases.parquet

The banked Stage 5 run measured the sparse arms decoding through `sdpa_math`
(54.94-55.92 ms/token at 8192, against dense `sdpa_flash` at 34.16). If the
2026-09-08 change works, the sparse arms' decode step should now land on top
of the dense one, and the 23-64% penalty should be gone rather than reduced.

This turns a MODELLED penalty into a MEASURED one. The analytical correction
in claims.md subtracted `n * (sparse_decode - dense_decode)` using phases
from a single era; this compares two eras directly.

Prints the residual gap after the change. A gap that does not close is the
interesting outcome -- it would mean the decode difference was never purely
the kernel, and the correction rests on a premise that does not hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.accuracy.grid_configs import (                         # noqa: E402
    DENSE_DECODE_BACKEND_HISTORY)

CLOSE_ENOUGH_PCT = 5.0   # what counts as "landed on top of dense"

# The era names come from the history constant, not from literals here. Two
# scripts disagreeing about which kernel "before" means is the same
# single-source-of-truth failure this whole change was about.
ERA_BEFORE = DENSE_DECODE_BACKEND_HISTORY[0][0]
ERA_AFTER = DENSE_DECODE_BACKEND_HISTORY[-1][0]


def decode_table(df: pd.DataFrame) -> dict:
    out = {}
    for r in df[df.phase == "decode_step"].itertuples():
        sp = None if pd.isna(r.sparsity) else float(r.sparsity)
        out[(int(r.context_length), sp)] = float(r.ms_mean)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="results/stage5/phases.parquet")
    ap.add_argument("--after", default="results/stage5_flashdecode/phases.parquet")
    args = ap.parse_args()

    before = decode_table(pd.read_parquet(args.before))
    after_df = pd.read_parquet(args.after)
    after = decode_table(after_df)

    print("decode step, ms/token -- sparse arms before and after the fallback change")
    print(f"{'band':>6}{'sp':>6}{'dense':>9}"
          f"{'was(' + ERA_BEFORE.replace('sdpa_', '') + ')':>11}"
          f"{'now(' + ERA_AFTER.replace('sdpa_', '') + ')':>12}"
          f"{'penalty was':>13}{'penalty now':>13}")
    residual = []
    for band in sorted({b for b, _ in after}):
        d_now = after[(band, None)]
        for sp in sorted({s for b, s in after if b == band and s is not None}):
            was = before.get((band, sp))
            now = after[(band, sp)]
            d_was = before[(band, None)]
            pw = 100 * (was - d_was) / d_was if was else float("nan")
            pn = 100 * (now - d_now) / d_now
            residual.append(abs(pn))
            print(f"{band:>6}{sp:>6g}{d_now:>9.2f}{was:>11.2f}{now:>12.2f}"
                  f"{pw:>12.1f}%{pn:>12.1f}%")

    worst = max(residual)
    print()
    if worst <= CLOSE_ENOUGH_PCT:
        print(f"PENALTY REMOVED -- worst residual {worst:.1f}% <= "
              f"{CLOSE_ENOUGH_PCT}%. The sparse arms now decode through the "
              f"same kernel as dense, and the correction's premise holds: "
              f"the gap WAS the kernel.")
        rc = 0
    else:
        print(f"PENALTY NOT REMOVED -- worst residual {worst:.1f}% > "
              f"{CLOSE_ENOUGH_PCT}%. Part of the decode difference was never "
              f"the kernel choice. The analytical correction in claims.md "
              f"attributes ALL of it to the kernel and would be overstated "
              f"by whatever remains. Do not fold this away -- it changes "
              f"what the correction licenses.")
        rc = 1

    locked = sorted(after_df.clocks_locked.unique())
    print(f"\nclocks_locked on the new rows: {locked}"
          + ("" if locked == [True] else "   <-- treat the split as indicative"))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
