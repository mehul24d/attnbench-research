#!/usr/bin/env python3
"""Link microbenchmarks for one pairing, as one provenance-stamped row (T2.5).

s_fixed for every synchronisation mechanism at one byte with the device idle;
pinned and pageable copy bandwidth both ways; blocking device-to-host
bandwidth for the two copy mechanisms; the same under a CPU load at each
context length; PCIe generation and width; NUMA placement. What each figure
is, and why it is measured that way: `attnbench/components/link.py`.

Run it with the process pinned to the GPU's NUMA node (plan Section 5.6);
the row records the affinity and whether it lies within that node.

Usage, on a CUDA instance from the repo root:
    python3 scripts/measure_link.py --out OUT_DIR
Dry run on CPU (a device that does nothing; the row says dry_run=True and
the readers refuse it):
    python3 scripts/measure_link.py --out OUT_DIR --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                     # noqa: E402
from attnbench.components import link                # noqa: E402


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sync-reps", type=int, default=2000)
    ap.add_argument("--copy-reps", type=int, default=50)
    ap.add_argument("--sizes", default=",".join(map(str, link.DEFAULT_SIZES)),
                    help="copy payloads in bytes; the largest gives the headline bandwidth")
    ap.add_argument("--context-lengths",
                    default=",".join(map(str, link.DEFAULT_CONTEXT_LENGTHS)),
                    help="one loaded measurement per length")
    ap.add_argument("--load", default="reference_builder", choices=sorted(link.LOADS))
    ap.add_argument("--load-density", type=float, default=0.5,
                    help="density the load's mask build keeps")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    path = out / "link.parquet"
    if path.exists():
        print(f"{path} exists; one link row per session directory", file=sys.stderr)
        return 2

    if args.dry_run:
        ops, pcie, numa = link.dry_run_ops(), None, None
    else:
        ops = link.cuda_ops()
        pcie, numa = link.query_pcie, link.numa_placement(link.gpu_bus_id())

    row = link.measure(ops, sync_reps=args.sync_reps, copy_reps=args.copy_reps,
                       sizes=_ints(args.sizes),
                       context_lengths=_ints(args.context_lengths),
                       load=args.load, load_density=args.load_density,
                       pcie=pcie, numa=numa, dry_run=args.dry_run)

    import pandas as pd
    df = pd.DataFrame([row])
    provenance.stamp_onto(df, provenance.capture().to_dict())
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    print(f"wrote {path}{'  (DRY RUN: fake device)' if args.dry_run else ''}")
    for m in link.SYNC_MECHANISMS:
        print(f"  s_fixed {m:24s} p50 {row[f's_fixed_{m}_median'] * 1e6:9.3f} us")
    for key in ("h2d_pinned", "h2d_pageable", "d2h_pinned", "d2h_pageable",
                "blocking_d2h_pinned", "blocking_d2h_pageable"):
        bw = row[f"bw_{key}_Bps"]
        print(f"  {key:24s} {'--' if bw is None else f'{bw / 1e9:8.2f} GB/s'}")
    print(f"  PCIe gen {row['pcie_gen_current_under_copy']} (max {row['pcie_gen_max_under_copy']})"
          f" x{row['pcie_width_current_under_copy']}; GPU NUMA node {row['gpu_numa_node']},"
          f" affinity on node: {row['affinity_on_gpu_node']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
