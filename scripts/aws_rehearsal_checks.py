#!/usr/bin/env python3
"""Checks 1 and 2 of the g6.xlarge rehearsal (docs/launch_cards/aws_rehearsal_g6.md;
adviser's task of 23 Nov 2026, second, item 2). Run on the instance, from the
deployed repository's root, by `scripts/aws_rehearsal_instance.sh`. EXPLORATORY:
nothing here enters training, the floors, the prediction or any claim.

    python3 scripts/aws_rehearsal_checks.py gpu OUT_DIR
    python3 scripts/aws_rehearsal_checks.py fa2 OUT_DIR [--config FILE]

`gpu` (check 1): `nvidia-smi` lists exactly one NVIDIA L4, the driver's CUDA
version is at least 12.9 (torch 2.9.1+cu129), torch sees CUDA at compute
capability (8, 9), and one bf16 matmul runs. Then the GPU libraries import
untimed (flash-attn at its pinned version, block-sparse-attention, x-attention's
`xattn.src.Xattention`:
plan default, flagged on the card), and `cuobjdump --list-elf` lists the
architectures in flash-attn's library. flash-attn was built for `80;90`, so a
kernel that runs on this (8, 9) card runs sm_80 code.

`fa2` (check 2): the dense selection's untimed screen (`t31_steps.dense_select_smoke`:
each L4 candidate's capability check, then its correctness check against the
others) at 2048 and 8192 tokens, batch 1. Passes when `fa2` is `correct` at both.

Each writes `OUT_DIR/<check>.json` and exits 0 on a pass, 1 on a failure.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

GPU_NAME = "NVIDIA L4"
MIN_DRIVER_CUDA = (12, 9)
CAPABILITY = (8, 9)
FA2_VERSION = "2.8.3.post1"
FA2_LENGTHS = (2048, 8192)
FA2 = "fa2"


def driver_cuda(smi_text: str):
    """The driver's CUDA version from `nvidia-smi`'s banner, or None."""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi_text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def elf_archs(listing: str) -> list[str]:
    """The sm_NN architectures `cuobjdump --list-elf` names, sorted."""
    return sorted(set(re.findall(r"\bsm_\d+[a-z]?\b", listing)))


def gpu_problems(rec: dict) -> list[str]:
    """Check 1's verdict: every failed condition, empty on a pass."""
    p = []
    if rec.get("gpus") != [GPU_NAME]:
        p.append(f"nvidia-smi lists {rec.get('gpus')!r}, not exactly one {GPU_NAME}")
    dc = rec.get("driver_cuda")
    if dc is None or tuple(dc) < MIN_DRIVER_CUDA:
        p.append(f"driver CUDA {dc!r} is below {MIN_DRIVER_CUDA}")
    if not rec.get("torch_cuda_available"):
        p.append("torch.cuda.is_available() is False")
    if tuple(rec.get("capability") or ()) != CAPABILITY:
        p.append(f"compute capability {rec.get('capability')!r}, not {CAPABILITY}")
    if not rec.get("bf16_matmul_finite"):
        p.append("the bf16 matmul did not run or was not finite")
    if rec.get("imports", {}).get("flash_attn") != FA2_VERSION:
        p.append(f"flash_attn imports as {rec.get('imports', {}).get('flash_attn')!r}, "
                 f"not {FA2_VERSION}")
    for lib in ("block_sparse_attn", "xattn.src.Xattention"):
        if not str(rec.get("imports", {}).get(lib, "")).startswith("ok"):
            p.append(f"{lib} does not import: {rec.get('imports', {}).get(lib)!r}")
    if (rec.get("fa2_elf_archs") or []) != ["sm_80", "sm_90"]:
        p.append(f"flash-attn's library lists {rec.get('fa2_elf_archs')!r}, not sm_80 and "
                 f"sm_90 (the build's 80;90): no proof of which code the L4 runs")
    return p


def fa2_problems(rec: dict) -> list[str]:
    """Check 2's verdict: fa2 correct at every length, empty on a pass."""
    p = []
    for L in FA2_LENGTHS:
        cell = rec.get("cells", {}).get(f"L4_{L}_1") or {}
        st = (cell.get("results", {}).get(FA2) or {}).get("status")
        if st != "correct":
            p.append(f"fa2 at {L} tokens, batch 1: {st!r}, not 'correct'")
    return p


def _import(name: str) -> str:
    import importlib
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "ok")
    except Exception as e:      # recorded, then judged by gpu_problems
        return f"FAILED {type(e).__name__}: {e}"[:300]


def run_gpu(out: Path) -> dict:
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    names = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True).stdout.strip().splitlines()
    rec = {"gpus": [n.split(",")[0].strip() for n in names if n.strip()],
           "driver_version": names[0].split(",")[1].strip() if names and "," in names[0] else None,
           "driver_cuda": driver_cuda(smi)}
    import torch
    rec["torch"] = torch.__version__
    rec["torch_cuda_available"] = torch.cuda.is_available()
    if rec["torch_cuda_available"]:
        rec["capability"] = list(torch.cuda.get_device_capability(0))
        a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
        c = a @ a
        torch.cuda.synchronize()
        rec["bf16_matmul_finite"] = bool(torch.isfinite(c).all().item())
    imports = {"flash_attn": _import("flash_attn")}
    bsa = _import("block_sparse_attn")
    imports["block_sparse_attn"] = "ok" if not bsa.startswith("FAILED") else bsa
    xa = _import("xattn.src.Xattention")
    imports["xattn.src.Xattention"] = "ok" if not xa.startswith("FAILED") else xa
    rec["imports"] = imports
    rec["fa2_elf_archs"] = None
    try:
        import flash_attn_2_cuda
        lib = flash_attn_2_cuda.__file__
        cuobj = shutil.which("cuobjdump") or next(
            (str(p) for p in sorted(Path("/usr/local").glob("cuda*/bin/cuobjdump"))), None)
        rec["fa2_library"], rec["cuobjdump"] = lib, cuobj
        if cuobj:
            lst = subprocess.run([cuobj, "--list-elf", lib], capture_output=True, text=True).stdout
            rec["fa2_elf_archs"] = elf_archs(lst)
    except Exception as e:
        rec["fa2_library"] = f"FAILED {type(e).__name__}: {e}"[:300]
    return rec


def run_fa2(out: Path, config: str | None) -> dict:
    import t31_steps as S
    from attnbench.components.shakedown_arms import Harness
    cfg = S.load_config(config or str(ROOT / "configs" / "aws" / "rehearsal_g6.json"),
                        dry_run=False)
    cfg = dict(cfg, lengths=list(FA2_LENGTHS), batches={})
    h = Harness(device="cuda", dry_run=False)
    try:
        return S.dense_select_smoke(h, cfg)
    finally:
        h.unwrap()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=("gpu", "fa2"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        rec = run_gpu(a.out) if a.check == "gpu" else run_fa2(a.out, a.config)
        problems = gpu_problems(rec) if a.check == "gpu" else fa2_problems(rec)
    except Exception as e:
        rec, problems = {}, [f"{type(e).__name__}: {e}"[:600]]
    rec.update(check=a.check, seconds=round(time.time() - t0, 1), problems=problems,
               status="passed" if not problems else "failed")
    (a.out / f"{a.check}.json").write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(json.dumps({k: rec[k] for k in ("check", "status", "seconds", "problems")}))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
