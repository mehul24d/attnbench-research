#!/usr/bin/env python3
"""P6's image and dense-backend check, block 1's abort gate (the researcher's
task of 26 Nov 2026 after P3's push, item 2c). Runs on the p5.4xlarge before
anything else in block 1; exit 0 lets the block go on, anything else stops it,
and the Mac's driver then pulls what there is and tears the session down.

The image (`ami-096c5dc30364c9041`, the environment build of 22 Nov 2026) was
built and rehearsed on the L4 (sm_89) only. Its build record says FA2
`2.8.3.post1` and block-sparse-attn were compiled for sm_80 and sm_90, and
that FA3 `3.0.0b1` was installed from the restricted wheel (sha256
`1676ef0b…`, built 28 Oct 2026 for sm_90a). Nothing has run it on an H100.
Checked here, in this order, every failure named:

  1. the GPU: an H100, compute capability 9.0;
  2. the distributions: flash_attn 2.8.3.post1, flash_attn_3 3.0.0b1 installed
     from the wheel whose sha256 its `direct_url.json` records as the built
     one, block_sparse_attn present;
  3. the compiled architectures (`cuobjdump --list-elf`) of FA2's and
     block-sparse-attn's extensions include sm_90, where cuobjdump exists
     (recorded either way; the forwards below are the proof);
  4. K2's planned candidates on the H100 (`dense_selection.planned_candidates`:
     the SDPA kernels, FA2 and FA3) all supplied by the repository's backends
     (`registry_candidates`): a missing one would make every dense record
     incomplete, which `check_control` refuses at ground truth, hours later;
  5. K2's own correctness screen (`dense_selection.screen`, the preflight
     smoke check's) at 2048 tokens, batch 1, on the primary model's geometry:
     every candidate P4's H100 found correct there (`REQUIRED_CORRECT`: the
     SDPA flash and cuDNN kernels, FA2, FA3) correct here; SDPA's
     memory-efficient kernel, "incorrect" on P4 (no available kernel), is
     recorded, not gated.

The prediction (`docs/p6_prediction/`, P4's committed values) priced
`sdpa_flash` for every dense cell (`shakedown_arms.ARMS["dense"]`); ground
truth runs `dense_select`'s per-cell choice. The record names both, so the
scoring wrapper can split matched from mismatched cells.

    python3 scripts/p6_backend_check.py --out ~/p3/b1_backends.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import asdict
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WANT = {"flash_attn": "2.8.3.post1", "flash_attn_3": "3.0.0b1"}
FA3_WHEEL_SHA256 = "1676ef0b86c377ae911a7de238c2a1f6b58a7666f1a6eef919d4dc7b0973b64a"
PRICED_DENSE = "sdpa_flash"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"          # configs/accuracy/stage3_grid.yaml, model.primary
SCREEN_LENGTH, SCREEN_BATCH = 2048, 1
EXTENSIONS = ("flash_attn_2_cuda", "block_sparse_attn_cuda")
# The candidates P4's H100 found correct at 2048 x 1 (its dense record of 16 Nov
# 2026, t64_p4_20261116T113724Z_b3: sdpa_flash, sdpa_cudnn, fa2 and fa3 correct,
# fa3 chosen at every dense cell; sdpa_efficient "incorrect": no available
# kernel). Each must be correct here too; sdpa_efficient's status is recorded,
# not gated, as it was not on P4.
REQUIRED_CORRECT = ("sdpa_flash", "sdpa_cudnn", "fa2", "fa3")


def _dist(name: str):
    from importlib import metadata
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None


def check_gpu(rep: dict, problems: list) -> None:
    import torch
    if not torch.cuda.is_available():
        problems.append("no CUDA device")
        return
    name = torch.cuda.get_device_name(0)
    cap = list(torch.cuda.get_device_capability(0))
    rep["gpu"] = {"name": name, "capability": cap, "torch": torch.__version__,
                  "torch_arch_list": torch.cuda.get_arch_list()}
    if "H100" not in name:
        problems.append(f"the GPU is {name!r}, not an H100")
    if cap != [9, 0]:
        problems.append(f"compute capability {cap}, not [9, 0]")


def wheel_sha256(direct_url) -> str | None:
    """The sha256 pip recorded for the archive a distribution came from
    (`direct_url.json`: archive_info.hashes, archive_info.hash, or the URL's
    `#sha256=` fragment)."""
    if not direct_url:
        return None
    d = json.loads(direct_url)
    a = d.get("archive_info") or {}
    if (a.get("hashes") or {}).get("sha256"):
        return a["hashes"]["sha256"]
    for s in (a.get("hash") or "", (d.get("url") or "").partition("#")[2]):
        if s.startswith("sha256="):
            return s[len("sha256="):]
    return None


def check_distributions(rep: dict, problems: list) -> None:
    out = {}
    for name, want in WANT.items():
        d = _dist(name)
        out[name] = None if d is None else d.version
        if d is None:
            problems.append(f"{name} is not installed")
        elif d.version != want:
            problems.append(f"{name} is {d.version}, not {want}")
    d3 = _dist("flash_attn_3")
    if d3 is not None:
        got = wheel_sha256(d3.read_text("direct_url.json"))
        out["flash_attn_3_wheel_sha256"] = got
        if got != FA3_WHEEL_SHA256:
            problems.append(f"flash_attn_3 was installed from a wheel with sha256 {got or 'unrecorded'}, "
                            f"not the built {FA3_WHEEL_SHA256[:12]}…")
    bsa = _dist("block-sparse-attn") or _dist("block_sparse_attn")
    out["block_sparse_attn"] = None if bsa is None else bsa.version
    if bsa is None:
        problems.append("block_sparse_attn is not installed")
    rep["distributions"] = out


def _cuobjdump() -> str | None:
    found = shutil.which("cuobjdump")
    if found:
        return found
    hits = sorted(glob.glob("/usr/local/cuda*/bin/cuobjdump"))
    return hits[-1] if hits else None


def check_archs(rep: dict, problems: list) -> None:
    import importlib.util
    tool = _cuobjdump()
    out = {"cuobjdump": tool}
    for ext in EXTENSIONS:
        spec = importlib.util.find_spec(ext)
        if spec is None or not spec.origin:
            problems.append(f"the extension {ext} is not found")
            continue
        if tool is None:
            out[ext] = "not read (no cuobjdump)"
            continue
        p = subprocess.run([tool, "--list-elf", spec.origin], capture_output=True, text=True)
        archs = sorted(set(re.findall(r"sm_[0-9]+a?", p.stdout)))
        out[ext] = archs
        if not any(a.startswith("sm_90") for a in archs):
            problems.append(f"{ext} has no sm_90 code ({archs or p.stderr.strip()[:120]})")
    rep["archs"] = out


def geometry():
    from transformers import AutoConfig
    from attnbench.accuracy.generation import ModelGeometry
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    return ModelGeometry.from_config(cfg, dtype="bfloat16")


def check_candidates(rep: dict, problems: list) -> None:
    from attnbench import dense_selection as ds
    from attnbench.config import AttnConfig
    planned = ds.planned_candidates("H100")
    cands = ds.registry_candidates("H100")
    rep["planned"] = list(planned)
    rep["supplied"] = {n: cands.get(n) is not None for n in planned}
    missing = [n for n in planned if cands.get(n) is None]
    if missing:
        problems.append(f"K2 candidates not supplied on the H100: {missing}")
    if ds.FA3 not in planned:
        problems.append("FA3 is not a planned candidate on the H100")
    g = geometry()
    cfg = AttnConfig(seq_len=SCREEN_LENGTH, batch=SCREEN_BATCH, n_heads_q=g.n_heads_q,
                     n_heads_kv=g.n_heads_kv, head_dim=g.head_dim, dtype=g.dtype, mask="causal")
    rep["screen_config"] = {"seq_len": cfg.seq_len, "batch": cfg.batch, "n_heads_q": cfg.n_heads_q,
                            "n_heads_kv": cfg.n_heads_kv, "head_dim": cfg.head_dim, "dtype": cfg.dtype}
    results, live = ds.screen(cfg, planned, cands, device="cuda")
    rep["screen"] = {n: asdict(r) for n, r in results.items()}
    rep["required_correct"] = list(REQUIRED_CORRECT)
    bad = {n: f"{results[n].status}: {results[n].reason}"[:160] if n in results else "not screened"
           for n in REQUIRED_CORRECT if n not in results or results[n].status != ds.CORRECT}
    if bad:
        problems.append(f"K2's screen at {SCREEN_LENGTH}x{SCREEN_BATCH}: not correct: {bad}")


CHECKS = (("gpu", check_gpu), ("distributions", check_distributions), ("archs", check_archs),
          ("candidates", check_candidates))


def run(checks=CHECKS) -> tuple[dict, list]:
    rep, problems = {"priced_dense_kernel": PRICED_DENSE, "fa3_wheel_sha256": FA3_WHEEL_SHA256}, []
    for name, fn in checks:
        try:
            fn(rep, problems)
        except Exception as e:  # a check that cannot run is a failure, named
            problems.append(f"{name} check could not run: {type(e).__name__}: {str(e)[:200]}")
        if name == "gpu" and problems:
            break      # not an H100: nothing after this means anything
    rep["problems"] = problems
    rep["passed"] = not problems
    return rep, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    rep, problems = run()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    if problems:
        print("P6_BACKEND_CHECK FAILED: " + "; ".join(problems), file=sys.stderr)
        return 1
    print(f"P6_BACKEND_CHECK PASSED: {', '.join(rep['planned'])} correct on {rep['gpu']['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
