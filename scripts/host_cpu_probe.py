#!/usr/bin/env python3
"""Host CPU identity, clock and steal -- and the mask builders timed on it.

**Why this exists.** No session in this study recorded its host CPU.
`provenance.capture()` stamps the GPU, driver and software stack, and the
2026-10-01 recovery (`limitations.md`, "Host CPU provenance") had to rebuild
CPU identity from serial-console boot lines and the GCP audit log. What that
recovery found is the reason this script measures and does not just log: the
L4 and A100 hosts are the same SKU (Xeon Platinum 8273CL) running identical
builder code, yet the L4's Stage 5 rows bound its construction cost at
8192/0.50 below half the A100 host's. An identity string cannot explain that.
A clock trace, a steal counter and the builder timed on the same host can.

Three subcommands:

  snapshot --out DIR --tag TAG   lscpu, /proc/cpuinfo, /proc/stat, loadavg
  sample   --out FILE            per-second MHz and steal, until SIGTERM
  builders --out DIR             both builders, per band x sparsity, on this CPU

`builders` times `masks.importance_block_mask` (the reference, whose cost is
one Python-level assignment per kept block, so it scales with 1 - sparsity)
and the vectorised builder on random fp32 scores. The reference's cost does
not depend on score values, only on how many blocks are kept, so random scores
time the same work the model's scores would. Every row carries the clock and
steal observed across its own timing loop.

CPU-only: no CUDA needed, so it runs on a laptop for a reference point.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

N_LAYERS = 28   # Qwen2.5-1.5B; the builder runs once per layer per forward


# --- identity and counters ---------------------------------------------------

def _read(path: str) -> str | None:
    try:
        return Path(path).read_text()
    except OSError:
        return None


def _sh(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_identity() -> dict:
    """Model string, family/model/stepping, microcode, vCPU count. Linux reads
    /proc/cpuinfo; macOS falls back to sysctl so a laptop run still says what
    it ran on."""
    info = _read("/proc/cpuinfo")
    out = {"nproc": os.cpu_count()}
    if info:
        first = info.split("\n\n")[0]
        kv = dict(line.split(":", 1) for line in first.splitlines() if ":" in line)
        kv = {k.strip(): v.strip() for k, v in kv.items()}
        for k in ("model name", "cpu family", "model", "stepping", "microcode"):
            out[k.replace(" ", "_")] = kv.get(k)
    else:
        out["model_name"] = (_sh(["sysctl", "-n", "machdep.cpu.brand_string"])
                             or platform.processor() or "").strip()
    return out


def cpu_mhz() -> list[float]:
    """Per-vCPU current MHz as the guest sees it. On GCE this is the KVM-
    reported value and moves with turbo; empty on macOS."""
    info = _read("/proc/cpuinfo") or ""
    return [float(line.split(":")[1]) for line in info.splitlines()
            if line.startswith("cpu MHz")]


def cpu_ticks() -> dict | None:
    """Aggregate /proc/stat counters. `steal` is time the hypervisor ran
    something else while this guest wanted the CPU -- the co-tenant signal."""
    stat = _read("/proc/stat")
    if not stat:
        return None
    f = [int(x) for x in stat.splitlines()[0].split()[1:]]
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    return dict(zip(names, f))


def steal_fraction(a: dict | None, b: dict | None) -> float | None:
    if not a or not b:
        return None
    total = sum(b.values()) - sum(a.values())
    return (b["steal"] - a["steal"]) / total if total > 0 else None


# --- subcommands ---------------------------------------------------------------

def cmd_snapshot(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    snap = {
        "tag": args.tag,
        "timestamp": time.time(),
        "host": platform.node(),
        "identity": cpu_identity(),
        "mhz": cpu_mhz(),
        "ticks": cpu_ticks(),
        "loadavg": _read("/proc/loadavg"),
        "lscpu": _sh(["lscpu"]),
        "cpuinfo": _read("/proc/cpuinfo"),
    }
    path = out / f"cpu_snapshot_{args.tag}.json"
    path.write_text(json.dumps(snap, indent=1))
    ident = snap["identity"]
    print(f"{path}: {ident.get('model_name')} "
          f"(family {ident.get('cpu_family')} model {ident.get('model')} "
          f"stepping {ident.get('stepping')}), {ident['nproc']} vCPU, "
          f"MHz {min(snap['mhz'], default=0):.0f}-{max(snap['mhz'], default=0):.0f}")
    return 0


def cmd_sample(args) -> int:
    """Append one CSV row per interval until SIGTERM/SIGINT. Started in the
    background around a GPU phase, so the phase's own clock and steal are on
    record next to its timings."""
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = cpu_ticks()
    with path.open("a") as fh:
        if fh.tell() == 0:
            fh.write("timestamp,mhz_min,mhz_median,mhz_max,steal_frac,load1\n")
        while not stop:
            time.sleep(args.interval)
            now = cpu_ticks()
            mhz = cpu_mhz() or [float("nan")]
            load = (_read("/proc/loadavg") or "nan").split()[0]
            sf = steal_fraction(prev, now)
            fh.write(f"{time.time():.3f},{min(mhz):.1f},{statistics.median(mhz):.1f},"
                     f"{max(mhz):.1f},{'' if sf is None else f'{sf:.5f}'},{load}\n")
            fh.flush()
            prev = now
    return 0


def cmd_builders(args) -> int:
    import pandas as pd
    import torch

    from attnbench import masks, provenance
    from _vec_mask_for_measurement import importance_block_mask_vectorised

    if args.threads:
        torch.set_num_threads(args.threads)
    bands = [int(x) for x in args.bands.split(",")]
    sparsities = [float(x) for x in args.sparsities.split(",")]
    ident = cpu_identity()
    block_size = args.block_size
    rows = []
    for band in bands:
        n = masks._n_blocks(band, block_size)
        g = torch.Generator().manual_seed(band)
        scores = torch.rand(n, n, generator=g, dtype=torch.float32)
        reps = args.reps if band < 32768 else max(3, args.reps // 2)
        for sparsity in sparsities:
            for builder in ("reference", "vectorised"):
                def call():
                    if builder == "reference":
                        return masks.importance_block_mask(
                            band, block_size, sparsity, scores,
                            causal=True, identity_seed=f"hostcpu-{band}")
                    return importance_block_mask_vectorised(
                        n, sparsity, scores, causal=True)

                for _ in range(args.warmup):
                    call()
                t0_ticks, mhz_seen, times = cpu_ticks(), [], []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    call()
                    times.append((time.perf_counter() - t0) * 1e3)
                    mhz_seen.extend(cpu_mhz())
                p50 = statistics.median(times)
                rows.append({
                    "band": band, "n_blocks": n, "sparsity": sparsity,
                    "builder": builder, "reps": reps,
                    "ms_per_call_p50": p50,
                    "ms_per_call_min": min(times),
                    "ms_per_call_max": max(times),
                    "ms_per_forward_p50": p50 * N_LAYERS,
                    "torch_threads": torch.get_num_threads(),
                    "cpu_model_name": ident.get("model_name"),
                    "cpu_family": ident.get("cpu_family"),
                    "cpu_model": ident.get("model"),
                    "cpu_stepping": ident.get("stepping"),
                    "cpu_microcode": ident.get("microcode"),
                    "nproc": ident["nproc"],
                    "mhz_median": statistics.median(mhz_seen) if mhz_seen else None,
                    "mhz_min": min(mhz_seen) if mhz_seen else None,
                    "mhz_max": max(mhz_seen) if mhz_seen else None,
                    "steal_frac": steal_fraction(t0_ticks, cpu_ticks()),
                })
                print(f"{band:6d} s={sparsity:.2f} {builder:10s} "
                      f"{p50:9.3f} ms/call  x{N_LAYERS} = {p50 * N_LAYERS:9.1f} ms/forward",
                      flush=True)

    df = pd.DataFrame(rows)
    provenance.stamp_onto(df, provenance.capture().to_dict())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "builders.parquet", index=False)
    print(f"wrote {out / 'builders.parquet'} ({len(df)} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--out", required=True)
    s.add_argument("--tag", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--out", required=True)
    s.add_argument("--interval", type=float, default=1.0)
    s = sub.add_parser("builders")
    s.add_argument("--out", required=True)
    s.add_argument("--bands", default="4096,8192,16384,32768")
    s.add_argument("--sparsities", default="0.5,0.75,0.9")
    s.add_argument("--block-size", type=int, default=128)
    s.add_argument("--reps", type=int, default=10)
    s.add_argument("--warmup", type=int, default=2)
    s.add_argument("--threads", type=int, default=0,
                   help="torch threads; 0 keeps the default")
    args = ap.parse_args()
    return {"snapshot": cmd_snapshot, "sample": cmd_sample,
            "builders": cmd_builders}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
