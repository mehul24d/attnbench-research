#!/usr/bin/env python
"""Stage 0 + Stage 1 runner: capability matrix and correctness gate.

    python scripts/run_probe.py --out results/probe

Writes three artefacts:
    versions.json      provenance stamp for this machine and session
    probe.parquet      per (backend, config) support status
    correctness.parquet  per (backend, config) numerical error vs float64

Safe to run on free-tier hardware. Nothing here needs locked clocks, because
none of it is a timing measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attnbench import provenance                      # noqa: E402
from attnbench.config import AttnConfig               # noqa: E402
from attnbench.backends import all_backends           # noqa: E402
from attnbench.backends.impls import SDPABackend      # noqa: E402
from attnbench.gates import probe, check_for_family  # noqa: E402


def probe_configs(max_seq: int):
    """EXACTLY the configs Stage 2 will run.

    This used to be a small independent grid (batch 2, 8 heads, seq_len
    512-2048) while Stage 2 runs batch 1, 32 heads, seq_len 1024-32768. The
    pass table is keyed on (backend, config_key), so the two sets shared ZERO
    keys and `plan()` would have rejected all 504 cells -- Stage 2 would have
    measured nothing.

    Deriving from SweepGrid keeps the original rule intact ("a cell whose
    (backend, config) has no recorded pass is skipped") rather than weakening
    the key to make stale evidence match. What varies by length is the KIND of
    check, not whether one happened: exact float64 agreement where the oracle
    fits, cross-backend agreement above it. See gates.check_for_family.
    """
    from attnbench.config import SweepGrid
    from attnbench.sweep import build_cells

    grid = SweepGrid()
    seen = set()
    for cfg in list(grid.dense_configs()) + list(
            grid.sparse_configs(mask_source="random")):
        if cfg.seq_len > max_seq or cfg.key() in seen:
            continue
        seen.add(cfg.key())
        yield cfg


def _legacy_probe_configs(max_seq: int):
    """The old independent grid, kept only for reference in git history."""
    for s in (512, 1024, 2048):
        if s > max_seq:
            continue
        for hq, hkv in ((8, 8), (8, 2)):          # MHA, GQA
            for pk in ("fwd", "fwd_bwd"):
                for mask in ("causal", "full"):
                    yield AttnConfig(seq_len=s, batch=2, n_heads_q=hq,
                                     n_heads_kv=hkv, head_dim=128,
                                     mask=mask, pass_kind=pk)

        # block_sparse configs. Without these, a sparse backend has NO
        # supported config, therefore no correctness row, therefore no Stage 1
        # pass -- and Stage 2 rejects every one of its cells while reporting
        # nothing unusual. That is exactly what happened on 2026-09-03: the
        # sweep would have measured only dense backends and looked complete.
        #
        # block_size 128 is the only size BSA supports (hardcoded inside
        # block_sparse_attn_func); 64 is included because flex covers it and
        # it is half of Stage 2's block-size axis.
        for block_size in (64, 128):
            for sparsity in (0.5, 0.9):
                # mask_source="random" matches what Stage 2 times. Leaving
                # it None makes mask_for() unable to build a mask, so the
                # sparse correctness check compares against nothing.
                yield AttnConfig(seq_len=s, batch=2, n_heads_q=8,
                                 n_heads_kv=8, head_dim=128,
                                 mask="block_sparse", pass_kind="fwd",
                                 block_size=block_size, sparsity=sparsity,
                                 mask_source="random")


def instantiate():
    """One instance per backend, plus the SDPA variants we pin explicitly."""
    out = []
    for name, cls in all_backends(available_only=True).items():
        if name == "sdpa":
            out += [SDPABackend("efficient"), SDPABackend("math"),
                    SDPABackend("flash"), SDPABackend("cudnn")]
        else:
            out.append(cls())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--max-seq", type=int, default=32768,
                    help="probe only configs at or below this length. Stage 2 "
                         "cells above it will have no pass and be rejected.")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    prov = provenance.write(outdir / "versions.json")
    print(f"gpu      : {prov.gpu_name} (sm{prov.compute_capability}, "
          f"{prov.gpu_memory_gb} GB)")
    print(f"torch    : {prov.torch} / cuda {prov.torch_cuda} / "
          f"triton {prov.triton}")

    if not torch.cuda.is_available():
        print("\nno CUDA device; nothing to probe")
        return

    backends = instantiate()
    print(f"backends : {', '.join(b.name for b in backends)}")

    unavailable = [n for n, c in all_backends().items() if not c.is_available()]
    if unavailable:
        print(f"missing  : {', '.join(unavailable)}")

    configs = list(probe_configs(args.max_seq))
    print(f"\nprobing {len(backends)} backends x {len(configs)} configs\n")

    rows = []
    for b in backends:
        counts = {}
        for cfg in configs:
            r = probe(b, cfg)
            counts[r.actual] = counts.get(r.actual, 0) + 1
            # Full provenance on every row, not just gpu/compute_capability.
            # The README rule is that no result row is written without a
            # provenance stamp, and this file was violating it: correctness
            # rows carried no git_commit at all, so the Stage 2 commit gate
            # could never be satisfied by ANY table this script produced --
            # a gate that can never pass, which is the "HEAD" bug again.
            rows.append({**r.to_dict(), **cfg.to_dict(), **prov.to_dict()})
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        mism = sum(1 for r in rows[-len(configs):] if r["claim_mismatch"])
        flag = f"   [{mism} claim mismatches]" if mism else ""
        print(f"  {b.name:<16} {summary}{flag}")

    df = pd.DataFrame(rows)
    df.to_parquet(outdir / "probe.parquet", index=False)

    if not args.skip_correctness:
        print("\ncorrectness gate (vs float64 naive)\n")
        crows = []
        supported = df[df.actual == "supported"]
        for b in backends:
            keys = set(supported[supported.backend == b.name].config_key)
            for cfg in configs:
                if cfg.key() not in keys or cfg.pass_kind != "fwd":
                    continue
                # Dispatch on family: exact for dense, masked-exact for
                # sparse, structural for linear. Grading every family against
                # an exact softmax oracle made gla fail 6/6 by construction.
                # references: independent dense implementations, used only
                # where the float64 oracle cannot be allocated (>4096).
                refs = [o for o in backends
                        if o.name != b.name
                        and o.capability.family in ("dense_exact", "reference")]
                r = check_for_family(b, cfg, references=refs)
                crows.append({**r.to_dict(), **cfg.to_dict(), **prov.to_dict()})
            mine = [r for r in crows if r["backend"] == b.name]
            if mine:
                worst = max((r["max_abs_err"] or 0) for r in mine)
                failed = sum(1 for r in mine if not r["passed"])
                kinds = sorted({r["check_kind"] for r in mine})
                print(f"  {b.name:<16} max_abs_err={worst:.2e}  "
                      f"failed={failed}/{len(mine)}  [{','.join(kinds)}]")
        pd.DataFrame(crows).to_parquet(
            outdir / "correctness.parquet", index=False)

    print(f"\nwritten to {outdir}/")


if __name__ == "__main__":
    main()
