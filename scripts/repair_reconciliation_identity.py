#!/usr/bin/env python3
"""Recompute banked `reconciliation.parquet` files under the `(n-1)` identity.

Why a repair rather than a re-run
---------------------------------
`phase_timing.reconcile` used `prefill + n_generated * decode_step` until
`eeccf97` (2026-09-08). Three of the four banked reconciliation files were
written before that and carry the retired identity:

    results/stage5/reconciliation.parquet             24 rows, 24+/0-
    results/stage5_flashdecode/reconciliation.parquet 24 rows, 14+/10-
    results/stage5_ols/reconciliation.parquet         28 rows, 19+/9-
    results/stage5_32768/reconciliation.parquet        4 rows  -- already correct

Every input the identity needs is in the file -- `prefill_ms`,
`decode_step_ms`, `n_generated`, `observed_total_ms`. Only the three derived
columns are wrong. So this is arithmetic on banked measurements, not a
re-measurement, and it needs no GPU. Nothing measured is touched: the script
asserts the four input columns are byte-identical before and after, and
recomputes the derived ones by calling `phase_timing.reconcile` itself rather
than restating the formula -- a second copy of the identity in a repair
script is how the project got four copies of it in the first place.

The 24-of-24 same-sign residuals in `results/stage5/` are the finding
recorded as silent_failure_patterns #27, and they were what the OLS intercept
check caught. Repairing the file does not retract that: the residuals under
the corrected identity are what the file should have said, and the history is
in the pattern entry and in git.

Preconditions, all asserted:
  1. the file exists and has the expected columns
  2. the four input columns survive the rewrite unchanged
  3. every row's stored `implied_total_ms` matches the OLD identity, so the
     file really is pre-fix and this is not a second application
  4. the row count is unchanged
  5. the recomputed `implied_total_ms` matches the NEW identity
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                                # noqa: E402
from attnbench.accuracy.phase_timing import reconcile           # noqa: E402

INPUTS = ["context_length", "backend", "sparsity", "prefill_ms",
          "decode_step_ms", "n_generated", "observed_total_ms"]
TOL = 1e-9


def repair(path: Path, *, apply: bool) -> bool:
    df = pd.read_parquet(path)
    missing = [c for c in INPUTS if c not in df.columns]
    assert not missing, f"{path}: missing input columns {missing}"

    old_implied = df.prefill_ms + df.n_generated * df.decode_step_ms
    new_implied = df.prefill_ms + (df.n_generated - 1) * df.decode_step_ms
    if (df.implied_total_ms - new_implied).abs().max() < TOL:
        if "analysis_git_commit" in df.columns:
            print(f"{path}: already on the (n-1) identity and stamped")
            return False
        # Correct arithmetic, no stamp. Nothing derived changes; the file
        # gains only the record of what produced it.
        print(f"{path}: already on the (n-1) identity, adding the stamp")
        if apply:
            provenance.stamp_analysis(
                df, "scripts/repair_reconciliation_identity.py")
            df.to_parquet(path, index=False)
            print("  -> stamped")
        return True
    assert (df.implied_total_ms - old_implied).abs().max() < TOL, (
        f"{path}: stored implied_total_ms matches NEITHER identity -- refusing "
        f"to guess what produced it")

    rows = [reconcile(context_length=int(r.context_length), backend=r.backend,
                      sparsity=None if pd.isna(r.sparsity) else float(r.sparsity),
                      prefill_ms=float(r.prefill_ms),
                      decode_step_ms=float(r.decode_step_ms),
                      n_generated=float(r.n_generated),
                      observed_total_ms=float(r.observed_total_ms)).to_dict()
            for r in df.itertuples()]
    out = pd.DataFrame(rows)

    assert len(out) == len(df), "row count changed"
    for c in INPUTS:
        a, b = df[c].reset_index(drop=True), out[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a):
            assert ((a - b).abs() < TOL) .all() or a.equals(b), f"{c} changed"
        else:
            assert a.equals(b), f"{c} changed"
    assert (out.implied_total_ms - new_implied.reset_index(drop=True)
            ).abs().max() < TOL

    pos = int((out.residual_ms > 0).sum())
    print(f"{path}: {len(out)} rows, residual signs {pos}+/{len(out) - pos}-, "
          f"closes {int(out.closes.sum())}/{len(out)}  "
          f"(was {int((df.residual_ms > 0).sum())}+/"
          f"{len(df) - int((df.residual_ms > 0).sum())}-, "
          f"closes {int(df.closes.sum())}/{len(df)})")

    if apply:
        provenance.stamp_analysis(out, "scripts/repair_reconciliation_identity.py")
        out.to_parquet(path, index=False)
        print(f"  -> rewritten")
    return True


def main() -> int:
    apply = "--apply" in sys.argv
    paths = [Path(a) for a in sys.argv[1:] if not a.startswith("-")]
    if not paths:
        paths = sorted(Path("results").glob("stage5*/reconciliation.parquet"))
    changed = [p for p in paths if repair(p, apply=apply)]
    if not apply and changed:
        print(f"\n(dry run -- {len(changed)} file(s) would be rewritten; "
              f"pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
