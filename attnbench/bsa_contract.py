"""Block-Sparse-Attention's boundary read, the adapter's fix, and the
kernel-contract check (researcher's ruling, 2 Nov 2026, after the XAttention
crash diagnostic, `docs/t31_xa_diag_card.md`).

**The read.** At the pinned commit (`BSA_COMMIT`) the library turns the
boolean block mask into int32 rows (`convert_blockmask_row_reverse`): each
row lists its selected key blocks in descending order, then -1 to the row's
end. The forward kernel walks a row (`flash_blockmask.h`,
`fwdBlockmask::mask_val`) until it reads -1 or its index reaches the row's
block count, but `mask_val` reads the entry at that index before the bound
stops the walk (its guard is `>`, not `>=`). A row with no -1, every key
block selected, is therefore read one entry past its end: the next row's
first entry, or, for the last row of the last head, one entry past the
buffer. The value read there is discarded (`is_last_block` is set from the
bound in the same statement, and nothing after it uses the value), so the
output does not depend on it; the read itself faults when the address past
the buffer is unmapped. The L4 diagnostic saw exactly that read under
memcheck (grid block (30, 0, 11), 4 bytes past a 12,288-byte buffer).

**The fix, in this adapter; the library is unchanged.** `install` replaces
the interface module's `convert_blockmask_row_reverse` with `row_blockmask`:
the same rows in a buffer that has `TAIL_ROWS` rows of -1 after it, so every
row, the last included, has a -1 to read after it inside the allocation. It
costs one fill kernel of one row per call. Every kernel call site installs
it (`backends/block_sparse._call`, `kernels.bsa_prefill`). Its identity
(`ROW_MASK_FIX`) and the library's (`BSA_COMMIT`, `INTERFACE_SHA256`) are in
every row's provenance and in the code identity of each block-sparse arm.

**The check** (`contract_check`, T3.1's preflight) refuses the session on any
of: a library other than the pinned one; a conversion that differs from the
library's; the kernel's output against a reference masked attention outside
`TOLERANCE`, on synthetic masks with every head's last row fully selected; a
tolerance that cannot see a one-block change; any output difference made by
the value past a row (`tail_value_effect`) or by the padding itself
(`padded_vs_unpadded`, on a mask whose stray reads stay inside the buffer, as
in the results banked before the fix); a memcheck error with the fix; or no
memcheck error without it (a check that cannot see the read proves nothing).
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

BSA_COMMIT = "49d6c39e4dc0303442cda3bb758b3925d4399c49"   # pyproject.toml's `kernels` pin
BSA_VERSION = "0.0.2"
# sha256 of block_sparse_attn/block_sparse_attn_interface.py at BSA_COMMIT. The
# images build the library from a local clone, so pip records no commit; the
# file the fix patches is identified by its bytes instead.
INTERFACE_SHA256 = "0d28e66dc3519573956d190aea3bb757bdde057f8f97e93edb8a39d05ac7b5de"
INTERFACE_MODULE = "block_sparse_attn.block_sparse_attn_interface"
CONVERT_NAME = "convert_blockmask_row_reverse"
SHIPPED_ATTR = "_attnbench_shipped_convert"
ROW_MASK_FIX = "row-mask-tail-1"      # change with `row_blockmask` or `TAIL_ROWS`
TAIL_ROWS = 1

TOLERANCE = 2e-2          # max |kernel - reference|, bf16 output against fp32
QK_SCALE = 2.0            # peaked attention, so one block more or less is visible
CHECK_LENGTHS = (1951, 4096)   # 1951: the crash's prompt length (uneven tiles)
HEADS, HEAD_DIM = 12, 128      # Qwen2.5-1.5B, the L4 model
SANITIZER_LENGTH = 1951
SANITIZER_TIMEOUT_S = 600
BLOCK = 128


class ContractError(RuntimeError):
    """The kernel's contract does not hold on this machine."""


# --- the fix ------------------------------------------------------------------------

def row_blockmask(base_blockmask: torch.Tensor, causal: bool = False, *,
                  tail_rows: Optional[int] = None, tail_value: int = -1) -> torch.Tensor:
    """The library's row form of a (B, H, R, C) boolean mask, in a buffer
    with `tail_rows` rows of `tail_value` after it (`TAIL_ROWS` of -1 unless
    told otherwise; the others are for the check). `causal` is the library's
    argument, unused there too."""
    tail_rows = TAIL_ROWS if tail_rows is None else tail_rows
    m = base_blockmask.to(dtype=torch.uint8)
    vals, idx = m.sort(dim=-1, stable=True, descending=False)
    idx[vals == 0] = -1
    idx = torch.flip(idx, dims=[-1])
    n, c = idx.numel(), idx.shape[-1]
    buf = torch.empty(n + tail_rows * c, dtype=torch.int32, device=idx.device)
    if tail_rows:
        buf[n:].fill_(tail_value)
    out = buf[:n].view(idx.shape)
    out.copy_(idx)
    return out


def _interface():
    """The library's interface module, or None for a test's stand-in for the
    library (a plain module, not a package, with no interface: no buffer)."""
    pkg = sys.modules.get("block_sparse_attn") or importlib.import_module("block_sparse_attn")
    iface = sys.modules.get(INTERFACE_MODULE) or getattr(pkg, "block_sparse_attn_interface", None)
    if iface is None and not hasattr(pkg, "__path__"):
        return None
    return iface or importlib.import_module(INTERFACE_MODULE)


def install():
    """Every later kernel call converts its mask with `row_blockmask`. The
    library's own conversion is kept on the module (`SHIPPED_ATTR`) for the
    check. Idempotent. Returns the interface module (None for a stand-in)."""
    iface = _interface()
    if iface is None:
        return None
    if getattr(iface, CONVERT_NAME) is not row_blockmask:
        if not hasattr(iface, SHIPPED_ATTR):
            setattr(iface, SHIPPED_ATTR, getattr(iface, CONVERT_NAME))
        setattr(iface, CONVERT_NAME, row_blockmask)
    return iface


def installed() -> bool:
    iface = sys.modules.get(INTERFACE_MODULE)
    return iface is not None and getattr(iface, CONVERT_NAME, None) is row_blockmask


@contextlib.contextmanager
def _conversion(fn):
    iface = install()
    setattr(iface, CONVERT_NAME, fn)
    try:
        yield
    finally:
        setattr(iface, CONVERT_NAME, row_blockmask)


# --- identity ------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _interface_path() -> Optional[Path]:
    """The installed interface file, found without importing the library."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("block_sparse_attn")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    p = Path(spec.origin).parent / "block_sparse_attn_interface.py"
    return p if p.exists() else None


def _version() -> Optional[str]:
    import importlib.metadata as md
    for name in ("block_sparse_attn", "block-sparse-attn"):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            continue
    mod = sys.modules.get("block_sparse_attn")
    return getattr(mod, "__version__", None)


def identity() -> dict:
    """What a row records: the library's version, the sha256 of its
    interface file, and the fix this checkout applies at every kernel call."""
    p = _interface_path()
    return {"block_sparse_attn": _version(),
            "bsa_interface_sha256": _sha256(p) if p else None,
            "bsa_row_mask_fix": ROW_MASK_FIX}


def require_pinned(ident: dict) -> None:
    if ident.get("block_sparse_attn") != BSA_VERSION:
        raise ContractError(f"block_sparse_attn version {ident.get('block_sparse_attn')!r}, "
                            f"not {BSA_VERSION}")
    if ident.get("bsa_interface_sha256") != INTERFACE_SHA256:
        raise ContractError(f"the installed interface file is not {BSA_COMMIT[:7]}'s "
                            f"(sha256 {ident.get('bsa_interface_sha256')})")


# --- the kernel's walk, on the host ---------------------------------------------------

def storage_elements(rows: torch.Tensor) -> int:
    """Elements of `rows`' allocation from its first element on."""
    return rows.untyped_storage().nbytes() // rows.element_size() - rows.storage_offset()


def past_row_reads(rows: torch.Tensor) -> int:
    """Rows the kernel reads one entry past: those with no -1."""
    return int((rows != -1).all(dim=-1).sum())


def reads_past_storage(rows: torch.Tensor) -> bool:
    """Whether the walk of some full row reads outside the allocation."""
    c = rows.shape[-1]
    full = (rows != -1).all(dim=-1).reshape(-1).nonzero()
    if full.numel() == 0:
        return False
    return (int(full.max()) + 1) * c >= storage_elements(rows)


def synthetic_mask(heads: int, nb: int, *, full_last, seed: int = 0) -> torch.Tensor:
    """(1, heads, nb, nb) bool: random causal blocks, every diagonal and
    block 0 kept, row nb // 2 fully selected (future blocks too, as
    XAttention's masks have), and the last row fully selected on the heads
    in `full_last`, one block short on the others."""
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(1, heads, nb, nb, generator=g) < 0.5) & torch.ones(nb, nb).tril().bool()
    i = torch.arange(nb)
    m[..., i, i] = True
    m[..., :, 0] = True
    m[..., nb // 2, :] = True
    full = set(full_last)
    for h in range(heads):
        m[0, h, -1, :] = True
        if h not in full and nb > 2:
            m[0, h, -1, 1] = False
    return m.contiguous()


def walk_check(*, lengths=CHECK_LENGTHS, heads: int = HEADS) -> dict:
    """The positive control on the host: with the library's layout the walk
    of a full last row leaves the buffer; with the fix it does not."""
    rep = {"shipped_reads_past_buffer": True, "fixed_reads_past_buffer": False,
           "past_row_reads": []}
    for L in lengths:
        m = synthetic_mask(heads, math.ceil(L / BLOCK), full_last=range(heads))
        shipped, fixed = row_blockmask(m, tail_rows=0), row_blockmask(m)
        if not reads_past_storage(shipped):
            raise ContractError("positive control: the library's layout shows no read past "
                                "the buffer, so the walk emulation cannot see it")
        if reads_past_storage(fixed):
            rep["fixed_reads_past_buffer"] = True
            raise ContractError(f"the fixed buffer is still read past its end at {L}")
        rep["past_row_reads"].append(past_row_reads(fixed))
    return rep


# --- the kernel against a reference ---------------------------------------------------

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _kernel():
    from block_sparse_attn import block_sparse_attn_func
    return block_sparse_attn_func


def _inputs(heads, L, d, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(1, heads, L, d, generator=g) for _ in range(3))
    q, k = q * QK_SCALE, k * QK_SCALE
    return [t.to(device=device, dtype=torch.bfloat16) for t in (q, k, v)]


def output_check(*, lengths=CHECK_LENGTHS, heads: int = HEADS, head_dim: int = HEAD_DIM,
                 device: str = "cuda") -> dict:
    from . import kernels
    iface = install()
    shipped = getattr(iface, SHIPPED_ATTR)
    run = functools.partial(kernels.bsa_prefill, kernel=_kernel())
    rep = {k: [] for k in ("vs_reference", "tolerance_separates", "tail_value_effect",
                           "padded_vs_unpadded", "conversion_equal")}
    for i, L in enumerate(lengths):
        nb = math.ceil(L / BLOCK)
        q, k, v = _inputs(heads, L, head_dim, device, seed=i)
        full = synthetic_mask(heads, nb, full_last=range(heads), seed=i).to(device)
        inner = synthetic_mask(heads, nb, full_last=range(heads - 1), seed=i).to(device)
        rep["conversion_equal"].append(all(
            torch.equal(shipped(m), row_blockmask(m)) for m in (full, inner)))
        out = run(q, k, v, full)
        ref = kernels.reference_prefill(q, k, v, full)
        rep["vs_reference"].append(max_abs_diff(out, ref))
        one_off = full.clone()
        one_off[0, 0, -1, 1] = False
        rep["tolerance_separates"].append(
            max_abs_diff(ref, kernels.reference_prefill(q, k, v, one_off)))
        effects = []
        for tv in (0, nb - 1):
            with _conversion(functools.partial(row_blockmask, tail_value=tv)):
                effects.append(max_abs_diff(run(q, k, v, full), out))
        rep["tail_value_effect"].append(max(effects))
        fixed_inner = run(q, k, v, inner)
        with _conversion(shipped):
            shipped_inner = run(q, k, v, inner)
        rep["padded_vs_unpadded"].append(max_abs_diff(shipped_inner, fixed_inner))
    return rep


def judge_outputs(rep: dict) -> None:
    if not all(rep.get("conversion_equal", [True])):
        raise ContractError("conversion_equal: the fix's rows differ from the library's")
    if max(rep["vs_reference"]) > TOLERANCE:
        raise ContractError(f"vs_reference: kernel against the reference "
                            f"{max(rep['vs_reference']):.3g} > {TOLERANCE}")
    if min(rep["tolerance_separates"]) <= TOLERANCE:
        raise ContractError(f"tolerance_separates: one block changes the reference by only "
                            f"{min(rep['tolerance_separates']):.3g}, within {TOLERANCE}")
    for key in ("tail_value_effect", "padded_vs_unpadded"):
        if max(rep[key]) != 0.0:
            raise ContractError(f"{key}: the output moved by {max(rep[key]):.3g}; the value "
                                f"read past a row is used")


# --- memcheck ---------------------------------------------------------------------------

def memcheck_errors(log: str) -> Optional[int]:
    import re
    m = re.search(r"ERROR SUMMARY: (\d+) error", log)
    return int(m.group(1)) if m else None


def sanitizer_command(tool: str, variant: str, log: Path) -> tuple[list, dict]:
    cmd = [tool, "--tool", "memcheck", "--print-limit", "20", "--error-exitcode", "99",
           "--log-file", str(log), sys.executable, "-m", "attnbench.bsa_contract",
           "sanitize-case", "--variant", variant]
    # Each tensor its own allocation, so a read past one is past an allocation.
    return cmd, dict(os.environ, PYTORCH_NO_CUDA_MEMORY_CACHING="1")


def run_sanitizer(tool: str, out: Path, *, timeout_s: float = SANITIZER_TIMEOUT_S) -> dict:
    res = {}
    root = Path(__file__).resolve().parents[1]
    for variant in ("shipped", "fixed"):
        log = out / f"memcheck_{variant}.log"
        cmd, env = sanitizer_command(tool, variant, log)
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, env=env, cwd=root, timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        text = log.read_text(errors="replace") if log.exists() else ""
        res[variant] = {"rc": rc, "errors": memcheck_errors(text),
                        "seconds": round(time.time() - t0, 1), "log": log.name}
    return res


def judge_sanitizer(res: dict) -> None:
    for variant in ("shipped", "fixed"):
        if res[variant]["errors"] is None:
            raise ContractError(f"memcheck {variant}: no summary (rc {res[variant]['rc']})")
    if res["fixed"]["errors"]:
        raise ContractError(f"memcheck fixed: {res['fixed']['errors']} errors with the fix")
    if not res["shipped"]["errors"]:
        raise ContractError("memcheck positive control: no error with the library's own "
                            "buffer, so memcheck cannot see the read here")


def sanitize_case(variant: str) -> None:
    """One kernel call at `SANITIZER_LENGTH` with every head's last row full,
    with the library's buffer (`shipped`) or the fix's."""
    from . import kernels
    iface = install()
    fn = getattr(iface, SHIPPED_ATTR) if variant == "shipped" else row_blockmask
    q, k, v = _inputs(HEADS, SANITIZER_LENGTH, HEAD_DIM, "cuda", seed=0)
    mask = synthetic_mask(HEADS, math.ceil(SANITIZER_LENGTH / BLOCK),
                          full_last=range(HEADS)).cuda()
    with _conversion(fn):
        kernels.bsa_prefill(q, k, v, mask, kernel=_kernel())
    torch.cuda.synchronize()


# --- the check ------------------------------------------------------------------------

def contract_check(out: Path, *, sanitizer: Optional[str], lengths=CHECK_LENGTHS,
                   heads: int = HEADS, head_dim: int = HEAD_DIM) -> dict:
    """Run every part, write `kernel_contract.json` into `out`, and raise
    `ContractError` on the first refusal. Behind the CPU boundary the kernel
    is its stub, which ignores the row buffer: the identity and the sanitizer
    are not checked there, and the output parts only exercise the code."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    rep = {"status": "failed", "identity": identity(), "fix": ROW_MASK_FIX,
           "bsa_commit": BSA_COMMIT, "tolerance": TOLERANCE, "lengths": list(lengths),
           "heads": heads}
    try:
        if cuda:
            require_pinned(rep["identity"])
        install()
        if not installed():
            raise ContractError("the fix is not installed: no interface module to patch")
        rep["walk"] = walk_check(lengths=lengths, heads=heads)
        rep["outputs"] = output_check(lengths=lengths, heads=heads, head_dim=head_dim,
                                      device="cuda" if cuda else "cpu")
        judge_outputs(rep["outputs"])
        if not cuda:
            rep["sanitizer"] = {"status": "not run: no CUDA device"}
        elif sanitizer is None:
            raise ContractError("compute-sanitizer not found; the memcheck case is required")
        else:
            rep["sanitizer"] = {"status": "ran", **run_sanitizer(sanitizer, out)}
            judge_sanitizer(rep["sanitizer"])
        rep["status"] = "passed"
        return rep
    except ContractError as e:
        rep["reason"] = str(e)
        raise
    finally:
        (out / "kernel_contract.json").write_text(json.dumps(rep, indent=1, default=str))


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True)
    c.add_argument("--sanitizer", default=None)
    s = sub.add_parser("sanitize-case")
    s.add_argument("--variant", choices=("shipped", "fixed"), required=True)
    args = ap.parse_args(argv)
    if args.cmd == "sanitize-case":
        sanitize_case(args.variant)
        return 0
    try:
        contract_check(Path(args.out), sanitizer=args.sanitizer)
    except ContractError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
