#!/usr/bin/env python3
"""Gate segment 2 on Stage 1 reproducing the 2026-09-04 flex diagnostic.

Run immediately after segment 2's Stage 1 probe and BEFORE its sweep. If the
pipeline path does not reproduce what the diagnostic measured, that is worth
knowing at the cost of one comparison rather than 504 cells.

    python3 scripts/check_stage1_against_diagnostic.py \
        results/stage2/segment_.../probe/correctness.parquet

Exit 0 to proceed, non-zero to stop. A non-zero exit naming EAGER_SIGNATURE
means torch.compile fell back again and no flex row in the run is trustworthy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench.analysis import diagnostic_agreement as DA   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("correctness", type=Path,
                    help="Stage 1 correctness.parquet from the new run")
    ap.add_argument("--compiled-ref", type=Path, default=DA.DEFAULT_COMPILED_REF)
    ap.add_argument("--eager-ref", type=Path, default=DA.DEFAULT_EAGER_REF)
    ap.add_argument("--rel-tol", type=float, default=DA.REL_TOL)
    args = ap.parse_args()

    compiled = DA.load_compiled_reference(args.compiled_ref)
    if not compiled:
        print(f"REFUSING: {args.compiled_ref} yielded no reference cells. An "
              f"empty reference would compare nothing and report success.")
        return 2

    eager = {}
    if args.eager_ref.exists():
        eager = DA.load_eager_reference(args.eager_ref)

    observed = DA.observed_from_correctness(args.correctness)
    rows = DA.compare(observed, compiled, eager, rel_tol=args.rel_tol)

    print(f"reference cells: {len(compiled)} | observed flex block_sparse "
          f"rows: {len(observed)} | rel_tol {args.rel_tol:.0%}\n")
    print(f"  {'config_key':<14}{'observed':>11}{'compiled':>11}"
          f"{'eager':>11}  verdict")
    for r in rows:
        obs = f"{r.observed:.6f}" if r.observed is not None else "-"
        eag = f"{r.eager_ref:.6f}" if r.eager_ref is not None else "-"
        print(f"  {r.config_key:<14}{obs:>11}{r.compiled_ref:>11.6f}"
              f"{eag:>11}  {r.verdict}")
        if r.verdict != "AGREES":
            print(f"      {r.detail}")

    try:
        DA.assert_agreement(rows)
    except DA.DiagnosticAgreementError as e:
        print(f"\nSTOP -- do not run the sweep.\n{e}")
        return 1

    print(f"\nAll {len(rows)} cells reproduce the diagnostic. Sweep may proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
