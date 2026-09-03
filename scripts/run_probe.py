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
from attnbench.gates import probe, check_correctness  # noqa: E402


def probe_configs(max_seq: int):
    """Small grid. The probe only needs to establish which cells exist, so we
    keep sequence lengths short and let Stage 2 handle the real sweep."""
    for s in (512, 1024, 2048):
        if s > max_seq:
            continue
        for hq, hkv in ((8, 8), (8, 2)):          # MHA, GQA
            for pk in ("fwd", "fwd_bwd"):
                for mask in ("causal", "full"):
                    yield AttnConfig(seq_len=s, batch=2, n_heads_q=hq,
                                     n_heads_kv=hkv, head_dim=128,
                                     mask=mask, pass_kind=pk)


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
    ap.add_argument("--max-seq", type=int, default=2048)
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
            rows.append({**r.to_dict(), **cfg.to_dict(),
                         "gpu": prov.gpu_name,
                         "compute_capability": prov.compute_capability})
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
                r = check_correctness(b, cfg)
                crows.append({**r.to_dict(), **cfg.to_dict()})
            mine = [r for r in crows if r["backend"] == b.name]
            if mine:
                worst = max((r["max_abs_err"] or 0) for r in mine)
                failed = sum(1 for r in mine if not r["passed"])
                print(f"  {b.name:<16} max_abs_err={worst:.2e}  "
                      f"failed={failed}/{len(mine)}")
        pd.DataFrame(crows).to_parquet(
            outdir / "correctness.parquet", index=False)

    print(f"\nwritten to {outdir}/")


if __name__ == "__main__":
    main()
