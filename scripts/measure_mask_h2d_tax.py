#!/usr/bin/env python3
"""Measure the host-to-device copy that `block_sparse` pays per kernel call.

**What this settles.** Audit item S11 bounds the timed-region tax that
`BlockSparseMask.to_block_sparse_attn_mask()` adds to every `block_sparse`
call. The transferred object is a bool grid of (n_blocks x n_blocks) bytes --
0.06 KB at 1024/128, 4 KB at 8192, 16 KB at 16384 -- and the bound rests on
one number that was never measured: an ASSUMED 15-30 us per pageable copy.
Everything else in S11 is arithmetic over banked data. This measures the
constant.

**The structural claim being tested, not just the magnitude.** S11 argues the
tax is "roughly constant per call" because at these sizes a pageable H2D copy
is dominated by fixed launch cost rather than bandwidth. That is falsifiable:
if the 16 KB copy costs materially more than the 0.06 KB one, the cost is
bandwidth-sensitive, "constant per call" is wrong, and the tax at the long
bands is understated relative to the short ones. Both the magnitude and the
shape are reported.

**What would falsify what, stated before the run:**

  - measured p50 outside 15-30 us      -> the assumed range was wrong; S11's
                                          percentages must be recomputed, and
                                          the range must NOT be widened to fit
  - cost rises with size               -> "roughly constant per call" is false
  - cost falls with size               -> measurement error, not a result
  - sync overhead ~ copy cost          -> the timing method cannot resolve
                                          this quantity; report and stop

**What cannot be falsified here, and why that is fine.** The bound's
DIRECTION does not depend on this constant. S14 established that `sdpa_math`
and `sdpa_flash` rebuild nothing per call (`per_call = [0, 0, 0]`), so the tax
is one-sided: it inflates the numerator of every block_sparse-over-dense
ratio and never the denominator. Every reported speedup is arithmetically a
floor whatever this number turns out to be. This tightens the magnitude; it
cannot change the sign.

Usage (on a CUDA instance, from the repo root):
    python3 scripts/measure_mask_h2d_tax.py --out results/s11_mask_h2d
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attnbench import provenance                                  # noqa: E402
from attnbench.masks import BlockSparseMask                       # noqa: E402

# (band, block_size) -> the three sizes S11's bound names, plus the two
# intermediate bands so the size/cost relationship has more than three points.
BANDS = (1024, 2048, 4096, 8192, 16384)
BLOCK_SIZE = 128


def _grid(band: int, block_size: int) -> torch.Tensor:
    """A realistic `active` grid: causal-lower-triangular over blocks, which
    is the shape every mask in this study has. Content does not affect copy
    cost, but constructing it the real way keeps the object identical in
    dtype, layout and contiguity to what the backend transfers."""
    n = -(-band // block_size)
    return torch.tril(torch.ones(n, n, dtype=torch.bool))


def _time_calls(fn, reps: int) -> list[float]:
    """Per-call host-blocking time in microseconds.

    Synchronised around EVERY call rather than once around the loop. A loop
    with one sync at the end measures throughput; the timed region pays
    latency, once per layer per forward, and those differ whenever the copies
    can overlap. Pageable H2D is host-synchronous anyway, so the two should
    agree -- both are reported, and a disagreement is itself information.
    """
    out = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        out.append((time.perf_counter_ns() - t0) / 1000.0)
    return out


def _summary(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {
        "n": len(xs),
        "min_us": round(xs[0], 3),
        "p50_us": round(statistics.median(xs), 3),
        "p90_us": round(xs[int(0.90 * len(xs))], 3),
        "p99_us": round(xs[int(0.99 * len(xs))], 3),
        "mean_us": round(statistics.fmean(xs), 3),
        "stdev_us": round(statistics.stdev(xs), 3) if len(xs) > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s11_mask_h2d")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=28,
                    help="head count; only affects the expand(), which is a "
                         "view and must therefore cost nothing")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this measurement needs a GPU; nothing to do on CPU")

    dev = torch.device("cuda")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")
    print(f"capability : {torch.cuda.get_device_capability(0)}")
    print(f"torch      : {torch.__version__}")
    print(f"reps       : {args.reps} (after {args.warmup} warmup)\n", flush=True)

    # Context init and allocator warmup, discarded. The first CUDA call in a
    # process costs milliseconds and would dominate a microsecond median.
    warm = _grid(8192, BLOCK_SIZE)
    for _ in range(args.warmup):
        warm.to(device=dev, dtype=torch.bool)
    torch.cuda.synchronize()

    # The floor: what the timing method itself costs, so a per-call figure
    # near this number is known to be unresolvable rather than "fast".
    one = torch.zeros(1, dtype=torch.bool)
    floor = _summary(_time_calls(lambda: one.to(device=dev, dtype=torch.bool),
                                 args.reps))
    sync_only = _summary(_time_calls(lambda: None, min(args.reps, 500)))
    print(f"timing floor (1-byte copy) : p50 {floor['p50_us']:.2f} us")
    print(f"sync-only overhead         : p50 {sync_only['p50_us']:.2f} us\n",
          flush=True)

    rows = []
    for band in BANDS:
        active = _grid(band, BLOCK_SIZE)
        n = active.shape[0]
        kb = active.numel() / 1024.0
        mask = BlockSparseMask(seq_len=band, block_size=BLOCK_SIZE,
                               active=active, seed=0, source="random",
                               causal=True)

        copy = _summary(_time_calls(
            lambda: active.to(device=dev, dtype=torch.bool), args.reps))
        full = _summary(_time_calls(
            lambda: mask.to_block_sparse_attn_mask(args.batch, args.heads,
                                                   device=dev), args.reps))

        # Throughput form: one sync around N calls. Reported so the
        # per-call latency figure can be checked against it.
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(args.reps):
            active.to(device=dev, dtype=torch.bool)
        torch.cuda.synchronize()
        batched_us = (time.perf_counter_ns() - t0) / 1000.0 / args.reps

        row = {
            "band": band, "block_size": BLOCK_SIZE, "n_blocks": n,
            "bytes": active.numel(), "kb": round(kb, 3),
            "copy_p50_us": copy["p50_us"], "copy_mean_us": copy["mean_us"],
            "copy_min_us": copy["min_us"], "copy_p90_us": copy["p90_us"],
            "copy_stdev_us": copy["stdev_us"],
            "full_call_p50_us": full["p50_us"],
            "batched_per_call_us": round(batched_us, 3),
            "floor_p50_us": floor["p50_us"],
        }
        rows.append(row)
        print(f"{band:6d}  n_blocks={n:4d}  {kb:8.2f} KB   "
              f"copy p50 {copy['p50_us']:7.2f} us   "
              f"full call p50 {full['p50_us']:7.2f} us   "
              f"batched {batched_us:7.2f} us", flush=True)

    # The structural question, answered numerically rather than by eye.
    smallest, largest = rows[0], rows[-1]
    ratio_cost = largest["copy_p50_us"] / smallest["copy_p50_us"]
    ratio_size = largest["bytes"] / smallest["bytes"]
    print(f"\nsize     {smallest['kb']:.2f} KB -> {largest['kb']:.2f} KB "
          f"({ratio_size:.0f}x)")
    print(f"cost     {smallest['copy_p50_us']:.2f} us -> "
          f"{largest['copy_p50_us']:.2f} us ({ratio_cost:.2f}x)")
    verdict = ("launch-dominated (cost roughly constant)" if ratio_cost < 1.5
               else "bandwidth-sensitive: S11's 'roughly constant per call' "
                    "does NOT hold")
    print(f"verdict  {verdict}")

    assumed_lo, assumed_hi = 15.0, 30.0
    p50s = [r["copy_p50_us"] for r in rows]
    inside = [assumed_lo <= p for p in p50s] and [p <= assumed_hi for p in p50s]
    print(f"\nassumed range was {assumed_lo:.0f}-{assumed_hi:.0f} us; "
          f"measured p50 spans {min(p50s):.2f}-{max(p50s):.2f} us")
    if min(p50s) < assumed_lo or max(p50s) > assumed_hi:
        print("  *** OUTSIDE the assumed range -- S11's percentages must be "
              "recomputed from the measured figure, and the assumed range "
              "reported as wrong rather than widened to fit. ***")
    else:
        print("  within the assumed range")

    import pandas as pd
    df = pd.DataFrame(rows)
    stamp = provenance.capture().to_dict()
    provenance.stamp_onto(df, stamp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "mask_h2d.parquet", index=False)
    (out / "summary.json").write_text(json.dumps({
        "rows": rows, "floor": floor, "sync_only": sync_only,
        "ratio_cost": round(ratio_cost, 3), "ratio_size": ratio_size,
        "gpu": torch.cuda.get_device_name(0),
        "assumed_range_us": [assumed_lo, assumed_hi],
        "measured_p50_span_us": [min(p50s), max(p50s)],
    }, indent=2))
    print(f"\nwrote {out}/mask_h2d.parquet and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
