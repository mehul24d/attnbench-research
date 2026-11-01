"""The GPU boundary, stubbed, so the T3.1 steps run the real arms on a CPU
(researcher's request, 31 Oct 2026, after Shakedown 3).

Shakedown 3 stopped on five wiring defects that a CPU test could have
caught, and each had passed its tests because the tests used stand-ins
for the arms. A CPU dry run of the rehearsal (steps 1 to 8) therefore runs
the real `Harness`, the real arms and backends, the real recorder, null
pass, replay and pricing, on the two-layer random-weight Qwen2
(`shakedown_arms.tiny_model`), and replaces only what needs a GPU. Everything
replaced is listed here, so nothing else can be stubbed by accident:

  - **The block-sparse kernel.** The `block_sparse_attn` package (a CUDA
    extension) is a module whose `block_sparse_attn_func` takes the real
    signature and calls one custom operator, `attnbench_cpu::
    block_sparse_attn_forward`, a blockwise reference on the CPU. It is one
    operator so it is one launch in the trace and one call for the null
    pass, as the pinned build's `flash_attn::_block_sparse_attn_forward`
    is. The library's own Python around the kernel
    (`replace_ones_with_count`, `convert_blockmask_row_reverse`, and their
    synchronisations) is not reproduced: the dry run's sequences have none
    of those waits. `block_sparse_attn_cuda.fwd_block` exists so
    `host_profile.block_sparse_entry` has an extension call to null.
  - **Triton.** `triton` and `triton.language` are import-only modules
    while the authors' `xattn.src.kernels` is imported (and its assert-free
    copy, `assert_strip`'s, for `xattention_noassert`), and are removed
    again at once, so no other library takes them for Triton. No Triton
    kernel runs: the
    stub device is named "NVIDIA L4 (CPU stub)", and the authors' own test sends a
    device whose name lacks "100" down the torch path, as on the L4.
  - **The x-attention checkout** is put on `sys.path` from
    `ATTNBENCH_XATTN_DIR` (a clone outside the repo, never copied in; rule
    8) and refused at any commit but `XATTN_COMMIT`.
  - **The CUDA device.** `torch.cuda.get_device_properties` and
    `current_device` answer for the stub device, `torch.cuda.synchronize`
    returns at once, and a tensor sent or created on "cuda" (the authors'
    estimator hard-codes `.to("cuda")` and `device="cuda"`) stays on the
    CPU (`CudaToCpu`). The tape (`sync_tape`), whose mode sees a call before
    `CudaToCpu` does, reads "cuda" as the CPU too (`_as_device`), so it
    classifies `.to("cuda")` on a device tensor as the L4's tape does, a call
    that keeps the device, and not as a copy to the host (found by the null
    pass on XAttention, 1 Nov 2026).
  - **The compiler's back end.** A compiled variant is traced by Dynamo as
    on the L4, and its graphs run under Dynamo's "eager" back end
    (`COMPILE_KWARGS`): inductor's Triton code and CUDA graphs are the GPU's.
  - **The profiler's device activity.** On a CPU the profiler records host
    operators only. `synthesize_device` rewrites an exported trace into the
    form CUPTI gives: one `cudaLaunchKernel` and one kernel per launching
    operator (a leaf operator that is not a view, an allocation, a
    metadata read or wholly CompositeImplicitAutograd, or a custom kernel
    operator whatever its children), and
    for a scalar read (`_local_scalar_dense`) a device-to-host copy and a
    stream synchronisation, with correlation ids and the stub device. Durations are the CPU's; nothing timed here is a result.
  - **The CUDA graph.** `GraphLikeBackend` keeps the one property of a graph
    the replay's checks rest on: a capture runs the arm's Python and writes
    no tensor (every aten call returns an unwritten output, as a captured
    kernel does until a replay), and a replay runs the prefill for real.
    `replay.EagerBackend`, which wrote at the "capture", hid Shakedown 3's
    unwritten live masks (ruling, 1 Nov 2026).

What a CPU dry run cannot test, and the L4 must: the CUDA graph's own
replay (the stand-in reruns the Python), the device's own timings, CUPTI's records
(memsets with no device record), the library's Python around the kernel,
and which operators a CUDA build lowers to (on a CPU `embedding` reaches
`index_select`, on the L4 `gather`).
"""

from __future__ import annotations

import contextlib
import functools
import json
import math
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Iterator, Optional, Union

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from .components import sync_tape

STUB_DEVICE_NAME = "NVIDIA L4 (CPU stub)"   # lacks "100", as the L4 does
# The CPU's kernel operator for the embedding lookup, where the L4's is
# `aten::gather` (`host_profile.ISOLATED_OUTSIDE_LAYERS`).
EMBEDDING_KERNEL_OP = "aten::index_select"
# `CompiledBackend`'s keywords behind the boundary (module docstring).
COMPILE_KWARGS = {"mode": None, "dynamo_backend": "eager"}


def compile_kwargs() -> dict:
    return dict(COMPILE_KWARGS) if active() else {}
XATTN_DIR_ENV = "ATTNBENCH_XATTN_DIR"
KERNEL_OP = "attnbench_cpu::block_sparse_attn_forward"
BLOCK = 128

# --- the block-sparse kernel -------------------------------------------------------


@torch.library.custom_op(KERNEL_OP, mutates_args=())
def _bsa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 base_blockmask: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int,
                 softmax_scale: float, is_causal: bool) -> torch.Tensor:
    """Blockwise reference: (batch*s, h, d) in, the same out. Every sequence
    in a call has one length (the study's cells), so the layout is read from
    the shapes, never from `cu_seqlens` values (no host read inside)."""
    b = base_blockmask.shape[0]
    s, sk = max_seqlen_q, max_seqlen_k
    h, d = q.shape[1], q.shape[2]
    qb = q.reshape(b, s, h, d).transpose(1, 2).float()
    kb = k.reshape(b, sk, h, d).transpose(1, 2).float()
    vb = v.reshape(b, sk, h, d).transpose(1, 2).float()
    out = torch.empty_like(qb)
    nbk = math.ceil(sk / BLOCK)
    keep = base_blockmask[..., :nbk].repeat_interleave(BLOCK, dim=-1)[..., :sk]   # (b,h,nbq,sk)
    kpos = torch.arange(sk)
    for i in range(math.ceil(s / BLOCK)):
        lo, hi = i * BLOCK, min((i + 1) * BLOCK, s)
        scores = qb[:, :, lo:hi] @ kb.transpose(-1, -2) * softmax_scale        # (b,h,r,sk)
        allowed = keep[:, :, i:i + 1, :].expand(-1, -1, hi - lo, -1)
        if is_causal:
            qpos = torch.arange(lo, hi).unsqueeze(1) + (sk - s)
            allowed = allowed & (kpos.unsqueeze(0) <= qpos)
        scores = scores.masked_fill(~allowed, float("-inf"))
        p = torch.softmax(scores, dim=-1).nan_to_num(0.0)
        out[:, :, lo:hi] = p @ vb
    return out.transpose(1, 2).reshape(b * s, h, d).to(q.dtype)


@_bsa_forward.register_fake
def _(q, k, v, base_blockmask, max_seqlen_q, max_seqlen_k, softmax_scale, is_causal):
    return torch.empty_like(q)


def block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
                           streaming_info, base_blockmask, max_seqlen_q_, max_seqlen_k_,
                           p_dropout, deterministic=False, softmax_scale=None,
                           is_causal=False, exact_streaming=False,
                           return_attn_probs=False):
    """`block_sparse_attn.block_sparse_attn_func`'s signature, on the CPU."""
    if p_dropout:
        raise ValueError("the CPU stub has no dropout")
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    return _bsa_forward(q, k, v, base_blockmask, int(max_seqlen_q_), int(max_seqlen_k_),
                        float(scale), bool(is_causal))


def _fwd_block(q, *args, **kwargs):
    raise RuntimeError("the CPU stub's extension call is for the null pass to replace")


def _modules() -> dict:
    bsa = types.ModuleType("block_sparse_attn")
    bsa.__version__ = "cpu-stub"
    bsa.block_sparse_attn_func = block_sparse_attn_func
    ext = types.ModuleType("block_sparse_attn_cuda")
    ext.fwd_block = _fwd_block
    return {"block_sparse_attn": bsa, "block_sparse_attn_cuda": ext}


def _import_xattn() -> None:
    """Import the authors' estimator with import-only Triton modules present,
    then take them away again (module docstring)."""
    triton = types.ModuleType("triton")
    triton.__version__ = "cpu-stub"
    lang = types.ModuleType("triton.language")
    lang.constexpr = type("constexpr", (), {})

    def jit(fn=None, **_kw):
        return fn if fn is not None else (lambda f: f)
    triton.jit = jit
    triton.cdiv = lambda a, b: -(-a // b)
    triton.language = lang
    stubs = {"triton": triton, "triton.language": lang}
    if any(n in sys.modules for n in stubs):
        raise RuntimeError("Triton is imported in this process; the CPU boundary needs it absent")
    sys.modules.update(stubs)
    try:
        import xattn.src.Xattention  # noqa: F401
        from . import assert_strip         # the assert-free copy (T2.17) imports Triton too
        assert_strip.install()
        import attnbench_xattn_noassert.src.Xattention  # noqa: F401
    finally:
        for n in stubs:
            sys.modules.pop(n, None)


# --- the x-attention checkout ----------------------------------------------------------

def xattn_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """The clone `xattn` imports from, at the pinned commit only."""
    from .backends.xattention import XATTN_COMMIT
    p = Path(path or os.environ.get(XATTN_DIR_ENV, "") or "/nonexistent")
    if not (p / "xattn" / "src" / "Xattention.py").exists():
        raise RuntimeError(f"no x-attention checkout; clone it outside the repo and set "
                           f"{XATTN_DIR_ENV}")
    head = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != XATTN_COMMIT:
        raise RuntimeError(f"{p} is at {head or 'no commit'}, not {XATTN_COMMIT}")
    return p


# --- the CUDA device -------------------------------------------------------------------

def _is_cuda(dev) -> bool:
    return (isinstance(dev, str) and dev.startswith("cuda")) or \
        (isinstance(dev, torch.device) and dev.type == "cuda")


class CudaToCpu(TorchFunctionMode):
    """A tensor sent to or created on "cuda" stays on the CPU."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        if _is_cuda(kwargs.get("device")):
            kwargs["device"] = "cpu"
        if func in (torch.Tensor.to, torch.Tensor.cuda):
            if func is torch.Tensor.cuda:
                return args[0]
            args = tuple("cpu" if _is_cuda(a) else a for a in args)
        return func(*args, **kwargs)


# --- the profiler's device activity ----------------------------------------------------

# Operators that issue no device work on a CUDA build: views, allocation,
# metadata. A leaf operator not here, and every custom kernel operator,
# launches one kernel.
NO_LAUNCH = frozenset({
    "aten::empty", "aten::empty_strided", "aten::empty_like", "aten::empty_permuted",
    "aten::view", "aten::_unsafe_view", "aten::reshape", "aten::_reshape_alias",
    "aten::as_strided", "aten::expand", "aten::expand_as", "aten::transpose", "aten::t",
    "aten::permute", "aten::slice", "aten::select", "aten::unsqueeze", "aten::squeeze",
    "aten::detach", "aten::alias", "aten::lift_fresh", "aten::lift", "aten::resize_",
    "aten::narrow", "aten::split", "aten::split_with_sizes", "aten::chunk", "aten::unbind",
    "aten::unflatten", "aten::flatten", "aten::view_as", "aten::contiguous", "aten::to",
    "aten::_to_copy", "aten::item", "aten::is_nonzero", "aten::size", "aten::stride",
    "aten::numel", "aten::dim", "aten::result_type", "aten::set_", "aten::_has_compatible_shallow_copy_type",
    "aten::is_same_size", "aten::_local_scalar_dense", "aten::scalar_tensor",
    "aten::unfold", "aten::movedim", "aten::swapaxes", "aten::diagonal",
    "aten::resolve_conj", "aten::resolve_neg", "aten::conj",
    # Operators that make no tensor (found by the null pass, 1 Nov 2026):
    "aten::promote_types", "aten::can_cast", "aten::is_floating_point", "aten::is_complex",
    "aten::detach_", "detach_", "detach",           # autograd metadata, as `aten::detach`
})
SYNC_OPS = frozenset({"aten::_local_scalar_dense"})


@functools.lru_cache(maxsize=None)
def composite_implicit(name: str) -> bool:
    """Whether every overload of an aten operator is CompositeImplicitAutograd:
    it has no kernel of its own, so on a CUDA build its device work is its
    children's, and as a leaf it launched nothing (`aten::dropout` at p=0 in
    eval returns its input; found by the null pass on XAttention, 1 Nov 2026).
    TorchScript-only overloads, which the dispatcher does not hold, are
    skipped."""
    if not name.startswith("aten::"):
        return False
    base = name.split("::", 1)[1]
    packet = getattr(torch.ops.aten, base, None)
    if packet is None:
        return False
    found = []
    for o in packet.overloads():
        full = f"aten::{base}" + ("" if o == "default" else f".{o}")
        try:
            found.append(torch._C._dispatch_has_kernel_for_dispatch_key(
                full, "CompositeImplicitAutograd"))
        except RuntimeError:
            continue
    return bool(found) and all(found)


def _may_launch(name: str) -> bool:
    return name not in NO_LAUNCH and not composite_implicit(name)
CUSTOM_KERNELS = (KERNEL_OP,)
_STREAM = 7


def _device_properties() -> list:
    return [{"id": 0, "name": STUB_DEVICE_NAME, "totalGlobalMem": 23659151360,
             "computeMajor": 8, "computeMinor": 9}]


def synthesize_device(trace: Union[str, Path, dict]) -> dict:
    """The CPU trace with the device activity a CUDA build would record
    (module docstring). Writes the file back when given a path."""
    path = None if isinstance(trace, dict) else Path(trace)
    doc = trace if path is None else json.loads(path.read_text())
    events = doc.get("traceEvents", [])
    ops = sorted((e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"),
                 key=lambda e: (e["pid"], e["tid"], e["ts"], -e.get("dur", 0)))
    # Nesting per thread: parent pointers by a stack sweep.
    parent: dict = {}
    children: dict = {}
    stack: list = []
    for e in ops:
        while stack and (stack[-1]["pid"], stack[-1]["tid"]) != (e["pid"], e["tid"]):
            stack.pop()
        while stack and stack[-1]["ts"] + stack[-1].get("dur", 0) < e["ts"] + e.get("dur", 0):
            stack.pop()
        if stack:
            parent[id(e)] = stack[-1]
            children.setdefault(id(stack[-1]), []).append(e)
        stack.append(e)

    def inside_custom(e) -> bool:
        p = parent.get(id(e))
        while p is not None:
            if p["name"] in CUSTOM_KERNELS:
                return True
            p = parent.get(id(p))
        return False

    # An operator launches when it is not a view, allocation or metadata
    # read and nothing below it launches (an `index_select` whose only
    # children allocate its output is still the kernel).
    below: dict = {}

    def subtree(e) -> bool:
        if id(e) not in below:
            kids = [subtree(c) for c in children.get(id(e), ())]
            below[id(e)] = any(kids) or (_may_launch(e["name"]) and not any(kids))
        return below[id(e)]

    def launches(e) -> bool:
        return _may_launch(e["name"]) and \
            not any(subtree(c) for c in children.get(id(e), ()))

    def gap(e) -> tuple[float, float]:
        """(start, length) of the widest stretch of `e` no child covers, so
        a synthesized call is attributed to `e` itself, not to a child."""
        lo, end = float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0.0))
        best = (lo, 0.0)
        for c in sorted(children.get(id(e), ()), key=lambda c: c["ts"]):
            if c["ts"] - lo > best[1]:
                best = (lo, c["ts"] - lo)
            lo = max(lo, c["ts"] + c.get("dur", 0.0))
        if end - lo > best[1]:
            best = (lo, end - lo)
        return best

    corr = 10_000_000
    added = []
    for e in ops:
        name = e["name"]
        if inside_custom(e):
            continue
        custom = name in CUSTOM_KERNELS
        sync = name in SYNC_OPS
        if not (custom or sync or launches(e)):
            continue
        ts, dur = gap(e)
        host = {"ph": "X", "pid": e["pid"], "tid": e["tid"]}
        dev = {"ph": "X", "pid": 0, "tid": _STREAM}
        if sync:
            corr += 1
            added.append({**host, "cat": "cuda_runtime", "name": "cudaMemcpyAsync",
                          "ts": ts + dur * 0.1, "dur": dur * 0.3,
                          "args": {"correlation": corr}})
            added.append({**dev, "cat": "gpu_memcpy", "name": "Memcpy DtoH (Device -> Pageable)",
                          "ts": ts + dur * 0.2, "dur": max(dur * 0.1, 0.001),
                          "args": {"correlation": corr, "device": 0, "stream": _STREAM,
                                   "bytes": 8}})
            corr += 1
            added.append({**host, "cat": "cuda_runtime", "name": "cudaStreamSynchronize",
                          "ts": ts + dur * 0.5, "dur": dur * 0.4,
                          "args": {"correlation": corr}})
            continue
        corr += 1
        added.append({**host, "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                      "ts": ts + dur * 0.4, "dur": dur * 0.2, "args": {"correlation": corr}})
        added.append({**dev, "cat": "kernel", "name": f"cpu_stub::{name}",
                      "ts": ts + dur * 0.5, "dur": max(dur * 0.5, 0.001),
                      "args": {"correlation": corr, "device": 0, "stream": _STREAM}})
    doc["traceEvents"] = list(events) + added
    doc["deviceProperties"] = _device_properties()
    if path is not None:
        path.write_text(json.dumps(doc))
    return doc


# --- the whole boundary -------------------------------------------------------------------

_installed = False


class _NoKernels(TorchDispatchMode):
    """Every aten call returns its outputs allocated and unwritten (an
    in-place call or `out=` returns its input untouched)."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from attnbench.components.host_profile import _null_outputs
        return _null_outputs(func, args, kwargs or {}, "cpu")


class GraphLikeBackend:
    """`replay.CudaGraphBackend` behind the boundary (module docstring)."""

    def __init__(self):
        self.replays = 0

    def capture(self, body):
        with _NoKernels():
            body()
        return body

    def warmup(self, body) -> None:
        body()

    def replay(self, body) -> None:
        self.replays += 1
        body()

    def synchronize(self) -> None:
        pass

    def time_once(self, body) -> float:
        import time
        t0 = time.perf_counter()
        self.replay(body)
        return time.perf_counter() - t0


def active() -> bool:
    return _installed


def have_xattn() -> bool:
    try:
        xattn_dir()
        return True
    except RuntimeError:
        return False


@contextlib.contextmanager
def installed(xattn_path: Optional[Union[str, Path]] = None, *,
              xattn: bool = True) -> Iterator[None]:
    """Every stub above, for the duration (module docstring). Refuses on a
    machine with CUDA: there the real boundary is the one to test. With
    `xattn=False` the XAttention arms are left out (tests of the others)."""
    global _installed
    if torch.cuda.is_available():
        raise RuntimeError("a CUDA device is present; the CPU boundary is for a CPU")
    if _installed:
        raise RuntimeError("the CPU boundary is already installed")
    root = xattn_dir(xattn_path) if xattn else None
    saved_mods = {n: sys.modules.get(n) for n in _modules()}
    saved = {
        "props": torch.cuda.get_device_properties, "current": torch.cuda.current_device,
        "sync": torch.cuda.synchronize, "name": torch.cuda.get_device_name,
        "export": torch.profiler.profile.export_chrome_trace,
        "as_device": sync_tape._as_device,
    }
    props = types.SimpleNamespace(name=STUB_DEVICE_NAME, major=8, minor=9,
                                  total_memory=23659151360, L2_cache_size=48 * 1024 * 1024,
                                  multi_processor_count=58)

    def export(self, path):
        saved["export"](self, path)
        synthesize_device(path)

    def as_device(d):
        if _is_cuda(d) or (isinstance(d, int) and not isinstance(d, bool)):
            return torch.device("cpu")
        return saved["as_device"](d)

    mode = CudaToCpu()
    if root is not None:
        sys.path.insert(0, str(root))
    try:
        sys.modules.update(_modules())
        torch.cuda.get_device_properties = lambda *a, **k: props
        torch.cuda.current_device = lambda: 0
        torch.cuda.synchronize = lambda *a, **k: None
        torch.cuda.get_device_name = lambda *a, **k: STUB_DEVICE_NAME
        torch.profiler.profile.export_chrome_trace = export
        sync_tape._as_device = as_device
        if root is not None:
            _import_xattn()
        _installed = True
        with mode:
            yield
    finally:
        _installed = False
        torch.cuda.get_device_properties = saved["props"]
        torch.cuda.current_device = saved["current"]
        torch.cuda.synchronize = saved["sync"]
        torch.cuda.get_device_name = saved["name"]
        torch.profiler.profile.export_chrome_trace = saved["export"]
        sync_tape._as_device = saved["as_device"]
        for n, m in saved_mods.items():
            if m is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = m
        if root is not None:
            from . import assert_strip
            assert_strip.uninstall()
            for n in [n for n in sys.modules if n.split(".")[0] in ("xattn", assert_strip.ALIAS)]:
                sys.modules.pop(n, None)
            if str(root) in sys.path:
                sys.path.remove(str(root))
