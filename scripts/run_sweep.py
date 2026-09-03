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
    ap.add_argument("--max-seq-len", type=int, default=None,
                    help="run only cells at or below this length. Stage 2 is "
                         "split across sessions shortest-first (see "
                         "docs/stage2_plan.md): cost is O(S^2), so short "
                         "lengths bank the most completed cells per hour and "
                         "an interrupted session loses least.")
    ap.add_argument("--min-seq-len", type=int, default=None,
                    help="run only cells at or above this length, so a later "
                         "session picks up where the previous one stopped")
    ap.add_argument("--at-commit", default=None,
                    help="require Stage 1 passes recorded at this commit. "
                         "Pass 'auto' to use the current one.")
    args = ap.parse_args()

    if args.at_commit == "auto":
        from attnbench import provenance as _prov
        args.at_commit = _prov.capture().git_commit
        if args.at_commit is None:
            raise SystemExit(
                "--at-commit auto: provenance reports no usable git commit. "
                "A Stage 1 pass cannot be tied to code that has no commit; "
                "see docs/stage2_plan.md.")
        print(f"stage1 gate : passes must be recorded at {args.at_commit[:12]}")

    backends = instantiate()
    print(f"backends : {', '.join(b.name for b in backends)}")

    grid = SweepGrid()
    cells = build_cells(grid, backends, mask_source=args.mask_source)

    # Length filtering happens BEFORE shuffling, not after: shuffled() exists
    # so thermal drift cannot correlate with backend identity, and that
    # property must hold within whatever subset this session actually runs.
    before = len(cells)
    if args.max_seq_len is not None:
        cells = [c for c in cells if c.cfg.seq_len <= args.max_seq_len]
    if args.min_seq_len is not None:
        cells = [c for c in cells if c.cfg.seq_len >= args.min_seq_len]
    if len(cells) != before:
        lo = args.min_seq_len if args.min_seq_len is not None else 0
        hi = args.max_seq_len if args.max_seq_len is not None else "inf"
        print(f"seq_len band: [{lo}, {hi}] -- {len(cells)} of {before} cells")

    lookup = {b.name: b for b in backends}

    report = run_sweep(
        cells, out_dir=Path(args.out), stage1_path=Path(args.stage1),
        backend_lookup=lookup, measure_fn=measure, dry_run=args.dry_run,
        stage1_at_commit=args.at_commit,
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
