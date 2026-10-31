#!/usr/bin/env python3
"""Launch-queue depth Q for one pairing, as one provenance-stamped row (T2.6).

A long spin kernel, then short kernels launched behind it and timed on the
host: the first launch that waits for the blocker marks a full queue. Method
and the Q it reports: `attnbench/components/launch_queue.py`.

Usage, on a CUDA instance from the repo root:
    python3 scripts/measure_launch_queue.py --out OUT_DIR
Dry run on CPU (a simulated queue of depth --dry-run-depth; the row says
dry_run=True and `queue_depth` refuses it):
    python3 scripts/measure_launch_queue.py --out OUT_DIR --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                     # noqa: E402
from attnbench.components import launch_queue as lq  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default=",".join(lq.KINDS))
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--launches", type=int, default=16384,
                    help="short kernels per trial; must exceed Q for Q to be measured")
    ap.add_argument("--blocker-s", type=float, default=2.0,
                    help="seconds the blocker spins; must outlast the fast launches")
    ap.add_argument("--blocked-fraction", type=float, default=0.25)
    ap.add_argument("--warm", action="store_true",
                    help="one completed launch of each kind before its blocker (default: cold)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-run-depth", type=int, default=1024)
    args = ap.parse_args()

    out = Path(args.out)
    path = out / "launch_queue.parquet"
    if path.exists():
        print(f"{path} exists; one queue row per session directory", file=sys.stderr)
        return 2

    if args.dry_run:
        sim = lq.SimulatedQueue(args.dry_run_depth)
        ops, clock = sim.ops(), sim.clock
    else:
        import time
        ops, clock = lq.cuda_queue_ops(), time.perf_counter_ns
    row = lq.measure(ops, kinds=[k for k in args.kinds.split(",") if k],
                     trials=args.trials, launches=args.launches,
                     blocker_s=args.blocker_s,
                     blocked_fraction=args.blocked_fraction,
                     dry_run=args.dry_run, clock=clock, warm=args.warm)

    import pandas as pd
    df = pd.DataFrame([row])
    provenance.stamp_onto(df, provenance.capture().to_dict())
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"wrote {path}{'  (DRY RUN: simulated queue)' if args.dry_run else ''}")
    for kind in args.kinds.split(","):
        if not kind:
            continue
        q = row[f"{kind}_q"]
        print(f"  {kind:7s} Q {'not agreed' if q is None else q}  "
              f"(trials {row[f'{kind}_q_trials']}; lower bound "
              f"{row[f'{kind}_q_lower_bound']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
