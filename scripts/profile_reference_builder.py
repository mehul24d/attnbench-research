#!/usr/bin/env python3
"""Profile the reference mask builder on this host and bank the profile (T2.2).

Re-measures the Python-overhead share of `masks.importance_block_mask` that
the plan states as about 94% from an unbanked laptop profile. One row per
sequence length, provenance-stamped. What is measured, how, and what the
profiler does to the number: `attnbench/components/builder_profile.py`.

The builder runs on the CPU, so run this on the target host (the L4 session's
host for T2.27 step 1, the A100's for the figure the plan compares with), with
the process pinned as the other host components are (plan Section 5.6).

Usage, from the repo root:
    python3 scripts/profile_reference_builder.py --out OUT_DIR
    python3 scripts/profile_reference_builder.py --out OUT_DIR --seq-lens 1024 --reps 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                       # noqa: E402
from attnbench.components import builder_profile as BP  # noqa: E402

DEFAULT_LENGTHS = "2048,4096,8192,16384,32768"           # the plan's grid (Section 5.3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq-lens", default=DEFAULT_LENGTHS)
    ap.add_argument("--sparsity", type=float, default=0.5,
                    help="fraction of candidate blocks dropped (density = 1 - this)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--block-size", type=int, default=BP.BLOCK_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    path = out / "builder_profile.parquet"
    if path.exists():
        print(f"{path} exists; one builder profile per session directory", file=sys.stderr)
        return 2
    rows = [BP.profile_reference_builder(int(n), sparsity=args.sparsity, reps=args.reps,
                                         block_size=args.block_size, seed=args.seed)
            for n in args.seq_lens.split(",") if n]

    import pandas as pd
    df = pd.DataFrame(rows)
    df["unprofiled_s_per_call_samples"] = df["unprofiled_s_per_call_samples"].map(list)
    provenance.stamp_onto(df, provenance.capture().to_dict())
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    print(f"wrote {path}")
    print("python share by instrument (they answer different questions; see "
          "attnbench/components/builder_profile.py):")
    print(f"{'seq_len':>8} {'plain s/call':>13} {'aten calls':>11} {'torch.prof':>11} "
          f"{'lower bound':>12} {'cProfile':>9}")
    for r in rows:
        print(f"{r['seq_len']:>8} {r['unprofiled_s_per_call_median']:>13.4f} "
              f"{r['aten_calls_per_call']:>11.0f} {r['python_share']:>11.3f} "
              f"{r['python_share_vs_unprofiled']:>12.3f} {r['cprofile_python_share']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
