#!/usr/bin/env python3
"""Precondition 2: canary drift against the previous session.

Ordering note, because the plan's wording is easy to misread. It says to run
the canary "before any sweep cell runs", but the canary cells ARE sweep cells
(seq_len 1024 and 4096) -- there is nothing to compare until this session has
measured them. The workable reading, and what this supports:

    sweep --max-seq-len 4096      # the cheap band: minutes, and it contains
                                  # every canary cell
    run_canary_check.py           # <- here, before committing to 8192+
    sweep --min-seq-len 8192      # the expensive part

which still satisfies the intent: a drifting environment is caught before it
has produced a sweep's worth of unjoinable numbers, at the cost of the
cheapest band rather than of nothing.

    python3 scripts/run_canary_check.py \
        --reference results/stage2/segment_20260903_seg1/segment.parquet \
        --observed  results/stage2/<this segment>/segment.parquet \
        --rebased flex,naive

Exit 0 to proceed, non-zero to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                        # noqa: E402

from attnbench.analysis import canary as C                 # noqa: E402
from attnbench.analysis.cross_arch import CrossArchError   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--observed", type=Path, required=True)
    ap.add_argument("--rebased", default="",
                    help="comma-separated backends whose timed region changed "
                         "on purpose. Each must have a canary.BaselineChange "
                         "entry or this refuses.")
    ap.add_argument("--tolerance", type=float, default=C.DRIFT_TOLERANCE)
    args = ap.parse_args()

    rebased = frozenset(b for b in args.rebased.split(",") if b.strip())
    ref = pd.read_parquet(args.reference)
    obs = pd.read_parquet(args.observed)

    print(f"reference: {args.reference}  ({len(ref)} rows)")
    print(f"observed : {args.observed}  ({len(obs)} rows)")
    print(f"canary seq_lens {C.CANARY_SEQ_LENS} | reference backend "
          f"{C.CANARY_REFERENCE_BACKEND} | tolerance {args.tolerance:.0%}")
    if rebased:
        print("\nexcluded as deliberate baseline changes:")
        for b in sorted(rebased):
            print(f"  {b}: {C._rebase_reason(b) or '(NO ENTRY -- will refuse)'}")
    print()

    try:
        ref_r = C.canary_ratios(ref)
        obs_r = C.canary_ratios(obs)
    except Exception as e:
        print(f"STOP -- could not compute canary ratios: {e}")
        return 2

    shared = sorted(set(ref_r) & set(obs_r))
    compared = [k for k in shared if k[1] not in rebased]
    print(f"keys in both frames: {len(shared)} | compared after exclusions: "
          f"{len(compared)}")
    if not compared:
        # An empty comparison reports success. Where a check can be vacuous,
        # assert it is not.
        print("STOP -- nothing left to compare. A canary that compares zero "
              "keys reports 'no drift' while verifying nothing.")
        return 2

    for key in compared:
        gpu, backend, cfg = key
        r, o = ref_r[key], obs_r[key]
        change = abs(o - r) / max(abs(r), 1e-12)
        flag = "DRIFT" if change > args.tolerance else "ok"
        print(f"  {backend:<16} {cfg:<14} ref={r:8.3f} obs={o:8.3f} "
              f"{change:6.1%}  {flag}")

    try:
        C.assert_no_canary_drift(ref, obs, tolerance=args.tolerance,
                                 rebased_backends=rebased)
    except CrossArchError as e:
        print(f"\nSTOP -- do not run the expensive band.\n{e}")
        return 1

    print(f"\nNo drift beyond {args.tolerance:.0%} across {len(compared)} "
          f"compared keys. Proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
