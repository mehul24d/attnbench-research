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
from dataclasses import replace
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench import compile_guard                    # noqa: E402
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
    # Raise the dynamo recompile ceiling before anything runs. Past the
    # default of 8, torch.compile silently runs eagerly -- which is how
    # Stage 1 certified flex block-sparse 72/72 for cells Stage 2 could
    # not lower at all. compile_guard.guard() catches it if it happens
    # anyway; this makes it not happen. See attnbench/compile_guard.py.
    compile_guard.configure()
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
    ap.add_argument("--pass-kind", default=None, choices=[None, "fwd", "fwd_bwd"],
                    help="run only cells of this pass kind. Stage 1 currently "
                         "validates forward only, so fwd_bwd cells have no "
                         "correctness pass and would be rejected; filtering "
                         "makes that intentional rather than 432 silent "
                         "rejections in the report.")
    ap.add_argument("--seq-lens", default=None,
                    help="comma-separated seq_lens to run, INSTEAD of the "
                         "grid's full ladder. --min/--max-seq-len express a "
                         "contiguous band, which is the right shape for "
                         "splitting Stage 2 across sessions shortest-first. "
                         "A representative SLICE is a different shape: three "
                         "lengths spanning the range, not three adjacent "
                         "ones. Filtering, never widening -- a value not in "
                         "the grid is a caller error and raises, because "
                         "silently measuring a length the grid does not "
                         "contain would produce cells no Stage 1 pass covers.")
    ap.add_argument("--batches", default=None,
                    help="comma-separated batch sizes to run, INSTEAD of the "
                         "grid's. Same filtering-only contract as --seq-lens.")
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

    def _slice(spec, field, name):
        """Narrow one grid axis. Refuses values the grid does not contain."""
        if spec is None:
            return getattr(grid, field)
        want = tuple(int(x) for x in spec.split(",") if x.strip())
        have = set(getattr(grid, field))
        unknown = [w for w in want if w not in have]
        if unknown:
            raise SystemExit(
                f"--{name} names {unknown}, which the grid does not contain "
                f"({sorted(have)}). This filter narrows the grid; it cannot "
                f"add to it. A cell outside the grid has no Stage 1 pass and "
                f"would be rejected after it had already been measured.")
        return want

    grid = replace(grid,
                   seq_lens=_slice(args.seq_lens, "seq_lens", "seq-lens"),
                   batches=_slice(args.batches, "batches", "batches"))
    if args.seq_lens or args.batches:
        print(f"grid slice  : seq_lens={grid.seq_lens} batches={grid.batches}")
    cells = build_cells(grid, backends, mask_source=args.mask_source)

    # Length filtering happens BEFORE shuffling, not after: shuffled() exists
    # so thermal drift cannot correlate with backend identity, and that
    # property must hold within whatever subset this session actually runs.
    before = len(cells)
    if args.max_seq_len is not None:
        cells = [c for c in cells if c.cfg.seq_len <= args.max_seq_len]
    if args.min_seq_len is not None:
        cells = [c for c in cells if c.cfg.seq_len >= args.min_seq_len]
    if args.pass_kind is not None:
        cells = [c for c in cells if c.cfg.pass_kind == args.pass_kind]
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
