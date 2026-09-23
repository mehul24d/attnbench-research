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

# A copy whose cost over the harness floor is below this fraction of the floor
# is not resolved by this method. 0.10 is deliberately generous: at 2026-09-21's
# A100 floor of 29.443 us it flags anything under 2.9 us of marginal cost, which
# caught bands 1024-4096 (0.266 / 0.255 / 0.436 us). The point is to make the
# stop condition fire, not to tune it.
RESOLUTION_FLOOR_FRACTION = 0.10

# The range S11's bound originally assumed, kept as a constant so every
# candidate figure can be checked against it rather than one of them.
ASSUMED_LO, ASSUMED_HI = 15.0, 30.0


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


def _batched(fn, reps: int) -> float:
    """Mean per-call time with ONE sync around `reps` calls -- the throughput
    form. `reps` is a parameter rather than read from `args`: this began as a
    closure inside `main()` and `tests/test_cli_arg_scope.py` refused it, which
    is the guard working. A nested function reading a CLI namespace it happens
    to be able to see is how #11 got into this repository.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - t0) / 1000.0 / reps


def analyse(rows: list[dict], floor_p50: float) -> dict:
    """Every derived quantity S11 is entitled to quote, from the banked rows
    alone. A pure function so it can be replayed on CPU against banked
    parquets -- `tests/test_mask_h2d_tax_analysis.py` does exactly that.

    It exists because the 2026-09-21 run's conclusions were written from the
    figures this script PRINTED rather than from the columns it BANKED, and the
    printing had no floor comparison in it. Now the reasoning is a function,
    the banked data is its input, and a test asserts what it says about that
    data.
    """
    smallest, largest = rows[0], rows[-1]
    over = {r["band"]: r["copy_p50_us"] - floor_p50 for r in rows}

    unresolvable = [r["band"] for r in rows
                    if over[r["band"]] < RESOLUTION_FLOOR_FRACTION * floor_p50]

    raw_ratio = largest["copy_p50_us"] / smallest["copy_p50_us"]
    marg_lo, marg_hi = over[smallest["band"]], over[largest["band"]]
    marg_ratio = (marg_hi / marg_lo) if marg_lo > 0 else None
    ratio_size = largest["bytes"] / smallest["bytes"]

    if marg_ratio is None:
        verdict = ("marginal cost is at or below zero -- unresolvable, "
                   "report and stop")
        launch_dominated = None
    elif marg_ratio < 1.5:
        verdict = "launch-dominated (marginal cost roughly constant)"
        launch_dominated = True
    else:
        verdict = (f"SIZE-SENSITIVE: the marginal cost rises {marg_ratio:.1f}x "
                   f"across a {ratio_size:.0f}x size range, so 'roughly "
                   f"constant per call' does NOT hold. The raw p50 looks flat "
                   f"only because the floor dominates it.")
        launch_dominated = False

    candidates = {}
    for label, key in (("copy_latency", "copy_p50_us"),
                       ("copy_throughput", "batched_per_call_us"),
                       ("full_call_latency", "full_call_p50_us"),
                       ("full_call_throughput", "batched_full_per_call_us")):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            continue
        candidates[label] = {
            "min": min(vals), "max": max(vals),
            "bands_inside": sum(1 for v in vals
                                if ASSUMED_LO <= v <= ASSUMED_HI),
            "n_bands": len(vals),
            "wholly_inside": all(ASSUMED_LO <= v <= ASSUMED_HI for v in vals),
        }
    return {
        "floor_p50_us": floor_p50,
        "over_floor_us": over,
        "floor_share": {r["band"]: floor_p50 / r["copy_p50_us"] for r in rows},
        "unresolvable_bands": unresolvable,
        "raw_ratio": raw_ratio,
        "marginal_ratio": marg_ratio,
        "ratio_size": ratio_size,
        "verdict": verdict,
        "launch_dominated": launch_dominated,
        "candidates": candidates,
    }


def print_analysis(rows: list[dict], a: dict) -> None:
    fl = a["floor_p50_us"]
    print(f"\ntiming floor (1-byte copy) p50 {fl:.3f} us")
    print("band      copy p50   over floor   % of figure that is floor")
    for r in rows:
        flag = ("  <-- UNRESOLVABLE" if r["band"] in a["unresolvable_bands"]
                else "")
        print(f"{r['band']:6d}  {r['copy_p50_us']:9.3f}  "
              f"{a['over_floor_us'][r['band']]:11.3f}  "
              f"{100 * a['floor_share'][r['band']]:22.1f}%{flag}")
    if a["unresolvable_bands"]:
        print(f"\n  *** STOP CONDITION MET at bands "
              f"{a['unresolvable_bands']}: the copy sits within "
              f"{100 * RESOLUTION_FLOOR_FRACTION:.0f}% of the harness floor, "
              f"so at these bands this method does not resolve the copy at "
              f"all. Do NOT report `copy_p50_us` as a measured constant "
              f"here. ***")

    print(f"\nsize     {rows[0]['kb']:.2f} KB -> {rows[-1]['kb']:.2f} KB "
          f"({a['ratio_size']:.0f}x)")
    print(f"raw p50  {rows[0]['copy_p50_us']:.2f} -> "
          f"{rows[-1]['copy_p50_us']:.2f} us ({a['raw_ratio']:.2f}x)  "
          f"<- includes the floor; not the structural answer")
    if a["marginal_ratio"] is not None:
        print(f"marginal {a['over_floor_us'][rows[0]['band']]:.3f} -> "
              f"{a['over_floor_us'][rows[-1]['band']]:.3f} us "
              f"({a['marginal_ratio']:.1f}x)  <- the grid's own cost")
    print(f"verdict  {a['verdict']}")

    print("\nlatency form vs throughput form (the disagreement the header "
          "calls informative):")
    print("band      copy lat   copy thru    full lat   full thru   lat/thru")
    for r in rows:
        # `batched_full_per_call_us` was added 2026-09-23. Rows banked before
        # that lack it, and this function has to read them: the whole reason it
        # is a pure function is so the 2026-09-21 parquets can be replayed
        # through it. Print a dash rather than crashing on its absence.
        bf = r.get("batched_full_per_call_us")
        bf_s = f"{bf:10.2f}" if bf is not None else f"{'--':>10}"
        print(f"{r['band']:6d}  {r['copy_p50_us']:9.2f}  "
              f"{r['batched_per_call_us']:10.2f}  "
              f"{r['full_call_p50_us']:10.2f}  {bf_s}  "
              f"{r['copy_p50_us'] / r['batched_per_call_us']:9.2f}x")
    print("  Stage 2 times N calls under ONE sync, so the THROUGHPUT column is "
          "the like-for-like figure;\n  the latency column is the conservative "
          "upper bound. Report the bracket, not one of them.")

    print(f"\nassumed range was {ASSUMED_LO:.0f}-{ASSUMED_HI:.0f} us. Every "
          f"candidate this run measured, against it:")
    for label, c in a["candidates"].items():
        print(f"  {label:22s}  {c['min']:6.2f}-{c['max']:6.2f} us   "
              f"{c['bands_inside']}/{c['n_bands']} bands inside   "
              f"{'WHOLLY INSIDE' if c['wholly_inside'] else 'not wholly inside'}")
    print("  A claim that the range is contradicted must say WHICH candidate "
          "contradicts it, and at how\n  many bands. 'Contradicted at every "
          "band' is a stronger claim than 'max exceeds the top'.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s11_mask_h2d")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=28,
                    help="head count; affects only the expand(), which is a "
                         "view -- so this should cost nothing, and MEASURABLY "
                         "DOES NOT: full_call minus copy came in at 12.4-13.5 "
                         "us on both cards in 2026-09-21, consistently across "
                         "bands. Banked as `views_us`; unexplained.")
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

        # Throughput form: one sync around N calls. Reported so the per-call
        # latency figure can be checked against it -- and measured for BOTH
        # quantities, because the latency form carries a fixed ~29 us of
        # double-sync harness that Stage 2's timed region does not pay per
        # call. Measuring only the copy here (which is all this did until
        # 2026-09-23) left the quantity S11 actually asks about with no
        # throughput form to be checked against at all.
        batched_us = _batched(
            lambda: active.to(device=dev, dtype=torch.bool), args.reps)
        batched_full_us = _batched(
            lambda: mask.to_block_sparse_attn_mask(args.batch, args.heads,
                                                   device=dev), args.reps)

        row = {
            "band": band, "block_size": BLOCK_SIZE, "n_blocks": n,
            "bytes": active.numel(), "kb": round(kb, 3),
            "copy_p50_us": copy["p50_us"], "copy_mean_us": copy["mean_us"],
            "copy_min_us": copy["min_us"], "copy_p90_us": copy["p90_us"],
            "copy_stdev_us": copy["stdev_us"],
            "full_call_p50_us": full["p50_us"],
            "full_call_mean_us": full["mean_us"],
            "batched_per_call_us": round(batched_us, 3),
            "batched_full_per_call_us": round(batched_full_us, 3),
            "floor_p50_us": floor["p50_us"],
            # The copy net of the timing harness's own floor. THIS is the cost
            # attributable to the grid; `copy_p50_us` is that plus a constant
            # the harness imposes and Stage 2 does not.
            "copy_over_floor_us": round(copy["p50_us"] - floor["p50_us"], 3),
            # What the backend actually calls, net of the bare copy: the
            # unsqueeze/unsqueeze/expand that `--heads` help claims is a view
            # and must therefore cost nothing.
            "views_us": round(full["p50_us"] - copy["p50_us"], 3),
        }
        rows.append(row)
        print(f"{band:6d}  n_blocks={n:4d}  {kb:8.2f} KB   "
              f"copy p50 {copy['p50_us']:7.2f} us "
              f"(over floor {row['copy_over_floor_us']:6.3f})   "
              f"full call p50 {full['p50_us']:7.2f} us   "
              f"batched copy {batched_us:6.2f} / full {batched_full_us:6.2f} us",
              flush=True)

    a = analyse(rows, floor["p50_us"])
    print_analysis(rows, a)
    raw_ratio = a["raw_ratio"]
    ratio_size = a["ratio_size"]
    unresolvable = a["unresolvable_bands"]
    verdict = a["verdict"]
    marg_lo = a["over_floor_us"][rows[0]["band"]]
    marg_hi = a["over_floor_us"][rows[-1]["band"]]

    import pandas as pd
    df = pd.DataFrame(rows)
    stamp = provenance.capture().to_dict()
    provenance.stamp_onto(df, stamp)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "mask_h2d.parquet", index=False)
    (out / "summary.json").write_text(json.dumps({
        "rows": rows, "floor": floor, "sync_only": sync_only,
        "ratio_cost": round(raw_ratio, 3), "ratio_size": ratio_size,
        "marginal_ratio": (round(marg_hi / marg_lo, 3) if marg_lo > 0 else None),
        "unresolvable_bands": unresolvable,
        "verdict": verdict,
        "gpu": torch.cuda.get_device_name(0),
        "assumed_range_us": [ASSUMED_LO, ASSUMED_HI],
        "analysis": {k: v for k, v in a.items() if k != "over_floor_us"},
        "measured_p50_span_us": [min(p50s), max(p50s)],
    }, indent=2))
    print(f"\nwrote {out}/mask_h2d.parquet and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
