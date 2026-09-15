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

from attnbench import checkpoint                      # noqa: E402
from attnbench import compile_guard                   # noqa: E402
from attnbench import provenance                      # noqa: E402
from attnbench.config import AttnConfig               # noqa: E402
from attnbench.backends import all_backends           # noqa: E402
from attnbench.backends.impls import SDPABackend      # noqa: E402
from attnbench.gates import probe, check_for_family  # noqa: E402


def probe_configs(max_seq: int, min_seq: int = 0):
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
        if not (min_seq <= cfg.seq_len <= max_seq) or cfg.key() in seen:
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


def band_plan(configs, backend_names, done):
    """(band, backend, configs-still-to-run), shortest band first.

    Pure, so the two properties that matter can be tested without a GPU:
    bands are visited in ascending seq_len, and anything already banked is
    skipped. Both were bought on 2026-09-04, when the probe wrote only at the
    end and an Xid 31 MMU fault at 32768 destroyed every shorter band with it.
    """
    for band in sorted({c.seq_len for c in configs}):
        band_cfgs = [c for c in configs if c.seq_len == band]
        for name in backend_names:
            yield band, name, [c for c in band_cfgs
                               if (name, c.key()) not in done]


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
    # Raise the dynamo recompile ceiling before anything runs. Past the
    # default of 8, torch.compile silently runs eagerly -- which is how
    # Stage 1 certified flex block-sparse 72/72 for cells Stage 2 could
    # not lower at all. compile_guard.guard() catches it if it happens
    # anyway; this makes it not happen. See attnbench/compile_guard.py.
    compile_guard.configure()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/probe")
    ap.add_argument("--max-seq", type=int, default=32768,
                    help="probe only configs at or below this length. Stage 2 "
                         "cells above it will have no pass and be rejected.")
    ap.add_argument("--min-seq", type=int, default=0,
                    help="probe only configs at or above this length, so a "
                         "band that faulted can be retried on its own without "
                         "redoing the bands that already succeeded")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-probe rows already present in the checkpoint "
                         "instead of skipping them")
    ap.add_argument("--exclude-backends", default="",
                    help="comma-separated backend names to leave out of this "
                         "run entirely. Exists because a backend that faults "
                         "with an illegal memory access does not fail alone: "
                         "it poisons the CUDA context, so the NEXT band dies "
                         "in torch.cuda.empty_cache() before probing anything. "
                         "On 2026-09-16 sdpa_cudnn logged "
                         "illegal_memory_access on all 84 configs at "
                         "seq_len=16384 and took the 32768 band down with it. "
                         "Excluding it here and probing it last, in its own "
                         "process, is the standing 'cuDNN last' rule made "
                         "enforceable rather than remembered.")
    ap.add_argument("--only-backends", default="",
                    help="comma-separated backend names to probe, to the "
                         "exclusion of all others. The inverse of "
                         "--exclude-backends, and the instrument for isolating "
                         "an ASYNCHRONOUS fault: a kernel that faults without "
                         "raising is reported at whatever CUDA call comes "
                         "next, which may be a different backend in a "
                         "different band. One backend per process, with "
                         "CUDA_LAUNCH_BLOCKING=1, is the only configuration in "
                         "which the error location names the cause.")
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
    only = {n.strip() for n in args.only_backends.split(",") if n.strip()}
    if only:
        known = {b.name for b in backends}
        unknown = only - known
        if unknown:
            raise SystemExit(
                f"--only-backends names unknown backend(s): {sorted(unknown)}. "
                f"Available: {sorted(known)}")
        backends = [b for b in backends if b.name in only]
        print(f"only     : {', '.join(sorted(only))}")
    excluded = {n.strip() for n in args.exclude_backends.split(",") if n.strip()}
    if excluded:
        known = {b.name for b in backends}
        unknown = excluded - known
        if unknown:
            # A typo here silently probes the backend you meant to exclude,
            # and the whole point of the flag is that that backend takes the
            # run down with it. Fail loudly instead.
            raise SystemExit(
                f"--exclude-backends names unknown backend(s): "
                f"{sorted(unknown)}. Available: {sorted(known)}")
        backends = [b for b in backends if b.name not in excluded]
        print(f"excluded : {', '.join(sorted(excluded))}")
    print(f"backends : {', '.join(b.name for b in backends)}")

    unavailable = [n for n, c in all_backends().items() if not c.is_available()]
    if unavailable:
        print(f"missing  : {', '.join(unavailable)}")

    configs = list(probe_configs(args.max_seq, args.min_seq))
    bands = sorted({c.seq_len for c in configs})
    probe_path = outdir / "probe.parquet"
    corr_path = outdir / "correctness.parquet"

    # Shortest band first, and every band fully written before the next one
    # starts. Two things follow, and both were paid for on 2026-09-04:
    #
    #   * A crash loses at most the current band. The first attempt lost 24
    #     minutes of work and the second 29, both complete, because this
    #     script wrote only at the very end.
    #   * A fault at 32768 cannot destroy 8192's results. The second attempt
    #     died to an Xid 31 MMU fault whose last logged configs were 32768,
    #     and took every shorter band down with it. That coupling has no
    #     reason to exist: the bands are independent measurements.
    #
    # Resume is keyed on (backend, config_key), so re-running after a fault
    # skips what is already banked and costs only the band that failed.
    done_probe = set() if args.no_resume else checkpoint.done_keys(
        probe_path, "backend", "config_key")
    done_corr = set() if args.no_resume else checkpoint.done_keys(
        corr_path, "backend", "config_key")
    if done_probe or done_corr:
        print(f"resuming: {len(done_probe)} probe rows and {len(done_corr)} "
              f"correctness rows already banked")

    print(f"\nprobing {len(backends)} backends x {len(configs)} configs "
          f"in {len(bands)} bands: {bands}\n")

    rows = []
    by_name = {b.name: b for b in backends}
    seen_band = None
    for band, name, todo in band_plan(configs, list(by_name), done_probe):
        if band != seen_band:
            seen_band = band
            n = sum(1 for c in configs if c.seq_len == band)
            print(f"[stage 0] seq_len={band}  ({n} configs)")
        b = by_name[name]
        counts, pending = {}, []
        for cfg in todo:
            r = probe(b, cfg)
            counts[r.actual] = counts.get(r.actual, 0) + 1
            # Full provenance on every row, not just gpu/compute_capability.
            # The README rule is that no result row is written without a
            # provenance stamp, and this file was violating it: correctness
            # rows carried no git_commit at all, so the Stage 2 commit gate
            # could never be satisfied by ANY table this script produced --
            # a gate that can never pass, which is the "HEAD" bug again.
            pending.append({**r.to_dict(), **cfg.to_dict(), **prov.to_dict()})
        if pending:
            checkpoint.append_checkpoint(probe_path, pending)
            rows += pending
        summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        mism = sum(1 for r in pending if r["claim_mismatch"])
        flag = f"   [{mism} claim mismatches]" if mism else ""
        print(f"    {b.name:<16} {summary or 'all resumed'}{flag}", flush=True)

    df = pd.read_parquet(probe_path)

    if not args.skip_correctness:
        print("\ncorrectness gate (vs float64 naive)\n")
        crows = []
        supported = df[df.actual == "supported"]
        for band in bands:
            band_cfgs = [c for c in configs if c.seq_len == band]
            print(f"[stage 1] seq_len={band}")
            for b in backends:
                keys = set(supported[supported.backend == b.name].config_key)
                pending = []
                for cfg in band_cfgs:
                    if cfg.key() not in keys or cfg.pass_kind != "fwd":
                        continue
                    if (b.name, cfg.key()) in done_corr:
                        continue
                    # Respect the CLAIM, not just whether it happened to run.
                    # A dense backend handed a block_sparse config ignores the
                    # mask argument entirely and computes plain attention,
                    # which "succeeds" -- so probe() marks it supported and the
                    # correctness check then compares it against an oracle that
                    # DID apply the mask. That produced max_abs_err=4.81 and
                    # 114/126 failures for fa2/sdpa_flash/sdpa_cudnn: not a
                    # numerical defect, a config they never agreed to run.
                    # Stage 2's build_cells filters on claims_support too, so
                    # this keeps probe and sweep looking at the same cells.
                    claimed, _ = b.claims_support(cfg)
                    if not claimed:
                        continue
                    # Dispatch on family: exact for dense, masked-exact for
                    # sparse, structural for linear. Grading every family
                    # against an exact softmax oracle made gla fail 6/6 by
                    # construction. references: independent dense
                    # implementations, used only where the float64 oracle
                    # cannot be allocated (>4096).
                    refs = [o for o in backends
                            if o.name != b.name
                            and o.capability.family in ("dense_exact",
                                                        "reference")]
                    r = check_for_family(b, cfg, references=refs)
                    pending.append({**r.to_dict(), **cfg.to_dict(),
                                    **prov.to_dict()})
                if pending:
                    checkpoint.append_checkpoint(corr_path, pending)
                    crows += pending
                if pending:
                    worst = max((r["max_abs_err"] or 0) for r in pending)
                    failed = sum(1 for r in pending if not r["passed"])
                    kinds = sorted({r["check_kind"] for r in pending})
                    print(f"    {b.name:<16} max_abs_err={worst:.2e}  "
                          f"failed={failed}/{len(pending)}  "
                          f"[{','.join(kinds)}]", flush=True)

    print(f"\nwritten to {outdir}/")


if __name__ == "__main__":
    main()
