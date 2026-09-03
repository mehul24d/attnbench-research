#!/usr/bin/env python
"""Stage 2 runner: kernel microbenchmarks.

    python scripts/run_sweep.py --dry-run
    python scripts/run_sweep.py --out results/sweep

--dry-run applies both preconditions (Stage 1 pass table, GPU exclusivity)
and prints what would run/skip/reject, without calling measure() or
touching the GPU. Run this first in any rented session -- if the Stage 1
table is stale or exclusivity misbehaves, this finds out in seconds instead
of minutes into a real sweep.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench.backends import all_backends            # noqa: E402
from attnbench.backends.impls import SDPABackend       # noqa: E402
from attnbench.config import SweepGrid                 # noqa: E402
from attnbench.sweep import build_cells, run_sweep      # noqa: E402
from attnbench.timing import measure                    # noqa: E402


def instantiate():
    out = []
    for name, cls in all_backends(available_only=True).items():
        if name == "sdpa":
            out += [SDPABackend(k) for k in ("efficient", "math", "flash", "cudnn")]
        else:
            out.append(cls())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/sweep")
    ap.add_argument("--stage1", default="results/probe/correctness.parquet")
    ap.add_argument("--mask-source", default=None, choices=[None, "random"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    backends = instantiate()
    print(f"backends : {', '.join(b.name for b in backends)}")

    grid = SweepGrid()
    cells = build_cells(grid, backends, mask_source=args.mask_source)
    lookup = {b.name: b for b in backends}

    report = run_sweep(
        cells, out_dir=Path(args.out), stage1_path=Path(args.stage1),
        backend_lookup=lookup, measure_fn=measure, dry_run=args.dry_run,
    )

    print(f"\ntotal cells    : {report.total}")
    print(f"would run      : {report.run}")
    print(f"skip (done)    : {report.skip_done}")
    print(f"reject stage1  : {report.reject_stage1}")
    print(f"reject exclus. : {report.reject_exclusivity}")
    if args.dry_run:
        print("\n[dry run -- nothing executed, nothing written]")
    else:
        print(f"\nwritten to {args.out}/sweep.parquet")


if __name__ == "__main__":
    main()
