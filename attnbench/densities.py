"""Mask density and executed density on every timed row (T2.20).

Plan Sections 5.3 and 5.5 put both on every row; the glossary says the
executed density "can differ from mask density after unions". The
definitions (researcher's decision, 25 Oct 2026):

  - **mask density**: the arm's mask as the arm produced it, forced sink and
    diagonal included. This is the `realised_density` rows already carry,
    which stays as it is so banked results keep reading the same way.
  - **executed density**: measured from the tensor actually handed to the
    block-sparse kernel, never recomputed from a formula, so it can differ
    from the mask density whenever something between the two (a group
    union, tile rounding) changes the blocks.
  - Both over one denominator: causally valid blocks times query heads, the
    convention of `realised_density`. Blocks above the causal diagonal are
    masked again inside the kernel, so they are not executed and not
    counted.
  - **Dense rows**: executed density 1.0 and mask density None (no mask).

`KernelInputRecorder` takes the executed density at the kernel boundary. All
three call sites (`kernels.bsa_prefill`, `backends/xattention.py`,
`backends/block_sparse.py`) import `block_sparse_attn_func` from its module
at call time, so replacing the module attribute for the duration of a block
sees every call without editing any of them (the XAttention backend's call
is the authors' and is not edited, rule 8).

**The density rule (researcher's decision, 31 Oct 2026).** Shakedown 2
found the oracle and mean-pool arms keeping no per-layer mask density, and
the recorder, under `compiled_default`, traced into the compiled graph,
where it read a CUDA-graph output that a later replay had overwritten. So:

  - densities come from an untimed twin pass over the executed masks
    (`ground_truth.kernel_densities`), or from a validated bank manifest
    (the precomputed arms' record);
  - both recorders copy what they see (`KernelInputRecorder` at the kernel,
    `MaskTap` where the model wrapper calls the backend) and compute only
    after the pass;
  - a compiled arm's densities come from its own compiled twin pass, the
    code path's executed masks. Two compiled twin passes must agree
    exactly (`check_repeatable`): density is a function of the input. The
    difference from the eager twin is reported and flagged beyond
    `DENSITY_FLAG_TOLERANCE` (`versus_eager`), not refused: a compiled run
    that rounds differently can threshold a slightly different mask
    (researcher's ruling, 31 Oct 2026, after Shakedown 2's XAttention gave
    0.66688 eager and 0.66582 compiled). The H6 bitwise mask gate on the
    compiled path is separate and unchanged;
  - nothing records inside a timed block (`timed_region`, entered by the
    harness's two timed sites) or inside a compiled region: either is
    refused (`RecordingRefused`), never worked around.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Sequence

import torch

KERNEL_MODULE = "block_sparse_attn"
KERNEL_NAME = "block_sparse_attn_func"
# `block_sparse_attn_func(q, k, v, cu_seqlens_q, cu_seqlens_k, head_mask_type,
# streaming_info, base_blockmask, ...)`, as `kernels.py` records it.
_HEAD_MASK_TYPE_ARG = 5
_BASE_BLOCKMASK_ARG = 7


def block_density(mask: torch.Tensor, *, causal: bool = True) -> torch.Tensor:
    """Fraction of valid blocks set, over every leading dimension, as a 0-dim
    tensor on the mask's device (no host sync).

    `mask` is bool, (..., n_q_blocks, n_k_blocks). Under `causal` the valid
    blocks are those on or below the diagonal aligned to the last key block,
    `tril(diagonal=n_k - n_q)`, as `backends.xattention.density_tensor` has
    it; that also covers a query chunk shorter than its keys (Section 5.8).
    """
    if mask.dtype != torch.bool or mask.dim() < 2:
        raise ValueError(f"a block mask is bool with at least 2 dims; got "
                         f"{mask.dtype} {tuple(mask.shape)}")
    nq, nk = mask.shape[-2:]
    if causal and nq > nk:
        raise ValueError(f"a causal mask cannot have more query blocks ({nq}) "
                         f"than key blocks ({nk})")
    valid = torch.ones(nq, nk, dtype=torch.bool, device=mask.device)
    if causal:
        valid = valid.tril(diagonal=nk - nq)
    lead = mask.numel() // (nq * nk)
    return (mask & valid).sum() / (valid.sum() * lead)


class RecordingRefused(RuntimeError):
    """A density recorder ran where the rule forbids it: inside a timed block
    or traced into a compiled region."""


class DensityMismatch(RuntimeError):
    """Two compiled twin passes of one arm on one input gave different
    densities."""


# Absolute mask-density difference, compiled minus eager, beyond which a
# compiled row is flagged. A T4.1 lock value; 0.01 is the working value
# (researcher's ruling, 31 Oct 2026).
DENSITY_FLAG_TOLERANCE = 0.01


_TIMED_DEPTH = 0


@contextmanager
def timed_region():
    """Marks a timed block; the harness enters it around every clocked forward."""
    global _TIMED_DEPTH
    _TIMED_DEPTH += 1
    try:
        yield
    finally:
        _TIMED_DEPTH -= 1


def in_timed_region() -> bool:
    return _TIMED_DEPTH > 0


def _refuse_here(what: str) -> None:
    if _TIMED_DEPTH > 0:
        raise RecordingRefused(f"{what} ran inside a timed block; densities are measured "
                               f"in the untimed twin pass only")
    if torch.compiler.is_compiling():
        raise RecordingRefused(f"{what} was traced into a compiled region; it must run "
                               f"outside torch.compile (the twin pass is eager)")


def _untraced(fn):
    try:
        import torch._dynamo
        return torch._dynamo.disable(fn)
    except Exception:              # no dynamo: nothing can trace it anyway
        return fn


def _copy(x):
    return x.detach().clone() if isinstance(x, torch.Tensor) else x


class MaskTap:
    """Wraps a backend where the model wrapper calls it, `forward(q, k, v,
    cfg, mask=)`, which is outside any compiled region. Copies each mask it
    is handed (the oracle and mean-pool arms build theirs before the
    backend) and, for a backend that builds its own and keeps a per-layer
    record (`last_layer_density`: XAttention, and the precomputed arms from
    their bank's manifest), copies the record's new entry after each call.
    Every other attribute is the backend's."""

    def __init__(self, backend):
        self._inner = backend
        self.masks: list = []          # (bool block mask copy, causal)
        self.records: list = []        # copies of the backend's own per-layer entries

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def forward(self, q, k, v, cfg, mask=None):
        _refuse_here("the mask tap")
        if mask is not None and getattr(mask, "active", None) is not None:
            self.masks.append((_copy(mask.active), bool(mask.causal)))   # as handed in
        record = getattr(self._inner, "last_layer_density", None)
        before = len(record) if record is not None else 0
        out = self._inner.forward(q, k, v, cfg, mask=mask)
        record = getattr(self._inner, "last_layer_density", None)
        if record is not None and len(record) > before:
            self.records.append(_copy(record[-1]))
        return out

    def per_layer_density(self, arm: str) -> list:
        """One float per layer: the masks handed in if there were any, else
        the backend's own record. Neither is refused."""
        if self.masks:
            return [float(block_density(m, causal=c)) for m, c in self.masks]
        if self.records:
            return [float(r) for r in self.records]
        raise RuntimeError(f"{arm}'s backend was handed no mask and keeps no per-layer mask "
                           f"density (last_layer_density), so its rows would carry none")


def refuse_nothing_compared(what: str, **sides: Sequence) -> None:
    """A per-layer comparison with no layer on either side passes or fails by
    comparing nothing, so it is refused (researcher's ruling, 1 Nov 2026)."""
    empty = [name for name, v in sides.items() if len(v) == 0]
    if empty:
        raise DensityMismatch(f"{what}: {' and '.join(empty)} has no layers, so the "
                              f"comparison would compare nothing")


def same_per_layer(what: str, source: Sequence[float], other: Sequence[float], *,
                   tol: float = 1e-9, names=("source", "other")) -> bool:
    """Layer-by-layer equality within `tol`; refused over zero layers."""
    refuse_nothing_compared(what, **{names[0]: source, names[1]: other})
    return len(source) == len(other) and all(abs(x - y) < tol for x, y in zip(source, other))


def check_repeatable(arm: str, first: list, second: list) -> None:
    """Two compiled twin passes on one input, call by call (or layer by
    layer): equal, or `DensityMismatch` naming the calls."""
    refuse_nothing_compared(f"{arm}'s compiled twin passes", first=first, second=second)
    if len(first) != len(second):
        raise DensityMismatch(f"{arm}: {len(first)} kernel calls in one compiled twin pass "
                              f"and {len(second)} in the next")
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(first, second)) if a != b]
    if bad:
        shown = ", ".join(f"call {i}: {a:.6f} vs {b:.6f}" for i, a, b in bad[:5])
        raise DensityMismatch(f"{arm}: two compiled twin passes on one input disagree on "
                              f"{len(bad)} of {len(first)} calls ({shown}); density is a "
                              f"function of the input")


def versus_eager(compiled_mask_density: float, eager_mask_density: float, *,
                 tolerance: float = DENSITY_FLAG_TOLERANCE) -> dict:
    """The compiled row's report against its eager twin: reported always,
    flagged beyond `tolerance`, never refused."""
    diff = float(compiled_mask_density) - float(eager_mask_density)
    return {"eager_mask_density": float(eager_mask_density),
            "density_vs_eager": diff,
            "density_vs_eager_flagged": abs(diff) > tolerance}


class KernelInputRecorder:
    """Records the density of every mask handed to the block-sparse kernel.

        with KernelInputRecorder() as rec:
            model(prompt)                 # an untimed pass; refused in a timed one
        row.update(rec.fields(mask_density=realised))

    Only `head_mask_type` all ones is accepted, the one value this study
    passes (`kernels.py`, sec. 4.4 of the archived pre-registration): each
    head then runs its own mask. Any other value is refused rather than
    interpreted, because what a 0 (dense) or -1 (streaming) head executes is
    not tested here.
    """

    def __init__(self, module=None):
        self._module = module
        self._original = None
        self.per_call: list[torch.Tensor] = []

    def __enter__(self) -> "KernelInputRecorder":
        _refuse_here("the kernel-input recorder")
        mod = self._module or sys.modules.get(KERNEL_MODULE)
        if mod is None:
            import importlib
            mod = importlib.import_module(KERNEL_MODULE)
        self._module = mod
        self._original = getattr(mod, KERNEL_NAME)
        original, per_call = self._original, self.per_call

        def recording(*args, **kwargs):     # never traced: see `_untraced` below
            mask = (kwargs["base_blockmask"] if "base_blockmask" in kwargs
                    else args[_BASE_BLOCKMASK_ARG])
            hmt = (kwargs["head_mask_type"] if "head_mask_type" in kwargs
                   else args[_HEAD_MASK_TYPE_ARG])
            # Copies, read after the pass: a reference could be overwritten
            # (a CUDA-graph replay) before it is read (the density rule).
            _refuse_here("the kernel-input recorder")
            per_call.append((_copy(mask), _copy(hmt), bool(kwargs.get("is_causal", False))))
            return original(*args, **kwargs)

        # Dynamo must not trace the recorder into a compiled arm's graph
        # (Shakedown 2, compiled_default): disabled, it runs eagerly between
        # the graph's pieces and copies the mask before any replay can
        # overwrite it. This is what lets the untimed compiled pass be
        # compared with its eager twin.
        setattr(mod, KERNEL_NAME, _untraced(recording))
        return self

    def __exit__(self, *exc) -> None:
        setattr(self._module, KERNEL_NAME, self._original)

    def per_call_density(self) -> list:
        """One float per recorded kernel call, from the copies."""
        return [float(block_density(m, causal=c)) for m, _, c in self.per_call]

    def executed_density(self) -> float:
        """Mean over kernel calls (one per layer per forward). Reads the
        device, so call it after the timed region."""
        if not self.per_call:
            raise RuntimeError(
                "no block-sparse kernel call was recorded, so no executed "
                "density exists for this row")
        if any(bool((hmt != 1).any()) for _, hmt, _ in self.per_call):
            raise ValueError(
                "head_mask_type was not all ones on a recorded call; what such "
                "a head executes is not modelled here")
        return float(torch.stack([block_density(m, causal=c).float()
                                  for m, _, c in self.per_call]).mean())

    def fields(self, *, mask_density: float) -> dict:
        return {"mask_density": float(mask_density),
                "executed_density": self.executed_density()}


DENSE_FIELDS = {"mask_density": None, "executed_density": 1.0}


def density_problems(row, *, dense: bool) -> list[str]:
    """Why a row's density fields are unusable; empty means usable."""
    def get(k):
        try:
            return row[k]
        except (KeyError, IndexError, TypeError):
            return None

    def missing(v) -> bool:
        return v is None or (isinstance(v, float) and v != v)

    problems = []
    md, ed = get("mask_density"), get("executed_density")
    if dense:
        if not missing(md):
            problems.append(f"a dense row has no mask, but mask_density={md!r}")
        if ed != 1.0:
            problems.append(f"a dense row executes every block, but "
                            f"executed_density={ed!r}")
        return problems
    for name, v in (("mask_density", md), ("executed_density", ed)):
        if missing(v):
            problems.append(f"{name} is missing on a sparse row")
        elif not 0.0 < float(v) <= 1.0:
            problems.append(f"{name}={v!r} is outside (0, 1]")
    return problems
