#!/usr/bin/env python3
"""The whole 2026-09-04 diagnostic session, in one scripted pass.

Three questions, in order, and the session stops at the first that fails:

  1. Do 64x64 tiles let flex block-sparse compile on this card at all?
     (scripts/flex_kernel_options_probe.py -- the ladder lives there.)
  2. Does it then AGREE with the float64 oracle, so it is a measurement and
     not just a kernel that ran?
  3. Did hoisting mask construction out of the timed region actually move
     flex's dense throughput on real hardware -- not merely compile?

Question 3 is the one that distinguishes "the fix is correct" from "the fix
type-checks". Segment 1 measured flex dense at a latency floor pinned near
2.0 ms at every shape, which is the signature of a constant addend rather than
a slow kernel; if that diagnosis is right, removing the addend must move the
short-sequence cells a lot and the long ones less. If throughput does NOT
move, the 2.0 ms floor was something else and the diagnosis was wrong.

THIS IS A DIAGNOSTIC, NOT A STAGE 2 SEGMENT. It bypasses the Stage 1 pass
table deliberately, because re-running the full gate would not fit the
session's 30-minute cap. Its output must never be joined into segment data:
it is written to results/diagnostics/, not results/stage2/.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                                              # noqa: E402

from attnbench import compile_guard, provenance           # noqa: E402
from attnbench.backends.impls import FlexAttentionBackend  # noqa: E402
from attnbench.config import AttnConfig                   # noqa: E402
from attnbench.gates import check_for_family              # noqa: E402
from attnbench.masks import mask_for                      # noqa: E402
from attnbench.timing import measure                      # noqa: E402
from scripts.flex_kernel_options_probe import (CANDIDATES,  # noqa: E402
                                               try_one)

# Segment 1's measured flex dense rows, verbatim. `config_key` is carried so
# the re-run can PROVE it is measuring the same cell rather than something
# that merely resembles it -- cfg.key() hashes every config field, so a
# mismatch means the comparison would have been meaningless.
BASELINE = [
    # seq_len, batch, n_heads_kv, config_key, p50_ms, useful_tflops, peak_mb
    (1024, 1, 32, "91ed6b4d948a", 2.034688, 4.22, 581.04),
    (1024, 16, 32, "e31ea76b3308", 2.899968, 47.39, 1354.90),
    (2048, 1, 32, "b6ee101ba9b0", 2.093056, 16.42, 637.67),
    (4096, 1, 32, "c93f88df3b1c", 3.702784, 37.12, 612.51),
]

SPARSE_CELLS = [(1024, 64, 0.9), (1024, 128, 0.9)]


def dense_cfg(seq_len: int, batch: int, n_heads_kv: int) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=batch, n_heads_q=32,
                      n_heads_kv=n_heads_kv, head_dim=128, dtype="bfloat16",
                      mask="causal", pass_kind="fwd", regime="prefill")


def sparse_cfg(seq_len: int, block_size: int, sparsity: float) -> AttnConfig:
    return AttnConfig(seq_len=seq_len, batch=1, n_heads_q=32, n_heads_kv=32,
                      head_dim=128, dtype="bfloat16", mask="block_sparse",
                      block_size=block_size, sparsity=sparsity,
                      mask_source="random", pass_kind="fwd", regime="prefill")


def main() -> int:
    compile_guard.configure()
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 2

    outdir = Path("results/diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)
    prov = provenance.capture()
    rows: list[dict] = []
    t0 = time.time()

    print(f"device: {prov.gpu_name} cc{prov.compute_capability} | "
          f"torch {prov.torch} | commit {prov.git_commit}")
    print(f"clean start: {compile_guard.process_has_fallen_back()=}\n")

    # ---- Q1/Q2: does block-sparse compile, and does it agree? -------------
    print("=" * 68)
    print("Q1/Q2  block-sparse: compile + oracle agreement")
    print("=" * 68)
    winners: dict[int, str] = {}
    for seq_len, block_size, sparsity in SPARSE_CELLS:
        cfg = sparse_cfg(seq_len, block_size, sparsity)
        print(f"\nseq_len={seq_len} block_size={block_size} sparsity={sparsity}")
        for label, opts in CANDIDATES:
            verdict, detail = try_one(cfg, opts)
            print(f"    {label:<12} {verdict:<13} {detail}")
            rows.append(dict(phase="kernel_options", config_key=cfg.key(),
                             block_size=block_size, candidate=label,
                             verdict=verdict, detail=detail))
            if verdict == "OK":
                winners[block_size] = label
                break
        else:
            print(f"    -> NOTHING WORKED at block_size={block_size}")

    if set(winners) != {64, 128}:
        print(f"\nSTOP. Working configurations: {winners or 'none'}.")
        print("Not improvising a fix on billed hardware -- tearing down and "
              "reporting, per the session rule.")
        (outdir / "flex_recheck.json").write_text(
            json.dumps({"prov": prov.to_dict(), "rows": rows}, indent=1))
        return 1

    print(f"\nBoth block sizes compile and agree: {winners}")

    # ---- Q2b: timed, through the real backend path ------------------------
    print("\n" + "=" * 68)
    print("Q2b  block-sparse: timed through FlexAttentionBackend")
    print("=" * 68)
    be = FlexAttentionBackend()
    for seq_len, block_size, sparsity in SPARSE_CELLS:
        cfg = sparse_cfg(seq_len, block_size, sparsity)
        mask = mask_for(cfg)
        m = measure(be, cfg, mask=mask)
        print(f"  bs={block_size:<4} {m.status:<18} "
              f"p50={m.latency_ms_p50}  useful={m.useful_tflops} TFLOPS  "
              f"peak={m.peak_memory_mb} MB  {m.detail[:60]}")
        rows.append(dict(phase="sparse_timing", config_key=cfg.key(),
                         block_size=block_size, **m.to_dict()))

    # ---- Q3: did the mask-hoisting fix move dense throughput? -------------
    print("\n" + "=" * 68)
    print("Q3  dense: does removing per-call mask construction move it?")
    print("=" * 68)
    print(f"  {'cell':<20} {'seg1 p50':>9} {'now p50':>9} "
          f"{'seg1 TF':>8} {'now TF':>8} {'speedup':>8}")
    for seq_len, batch, kv, key, base_p50, base_tf, base_mb in BASELINE:
        cfg = dense_cfg(seq_len, batch, kv)
        if cfg.key() != key:
            # A silent mismatch here would compare two different cells and
            # report the difference as an effect of the fix.
            print(f"  !! config_key mismatch for {seq_len}/{batch}: "
                  f"built {cfg.key()}, segment 1 recorded {key} -- SKIPPING, "
                  f"the comparison would be meaningless")
            rows.append(dict(phase="dense_timing", config_key=cfg.key(),
                             status="key_mismatch", detail=f"expected {key}"))
            continue

        m = measure(be, cfg)
        if not m.ok:
            print(f"  {f'{seq_len}/b{batch}':<20} {m.status}  {m.detail[:60]}")
        else:
            sp = base_p50 / m.latency_ms_p50
            print(f"  {f'{seq_len}/b{batch}':<20} {base_p50:>9.3f} "
                  f"{m.latency_ms_p50:>9.3f} {base_tf:>8.2f} "
                  f"{m.useful_tflops:>8.2f} {sp:>7.2f}x")
        rows.append(dict(phase="dense_timing", config_key=cfg.key(),
                         seq_len=seq_len, batch=batch, n_heads_kv=kv,
                         seg1_p50=base_p50, seg1_tflops=base_tf,
                         seg1_peak_mb=base_mb, **m.to_dict()))

    print(f"\nfell back at any point: {compile_guard.process_has_fallen_back()}")
    print(f"elapsed: {time.time() - t0:.0f}s")

    payload = {"prov": prov.to_dict(), "winners": winners, "rows": rows}
    (outdir / "flex_recheck.json").write_text(json.dumps(payload, indent=1,
                                                         default=str))
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(outdir / "flex_recheck.parquet")
    except Exception as e:
        print(f"(parquet write skipped: {e})")
    print(f"wrote {outdir}/flex_recheck.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
