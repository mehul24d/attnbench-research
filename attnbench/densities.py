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
is the authors' and is not edited, rule 8). Each call's density stays a
0-dim tensor on the device, so recording adds no synchronisation inside the
timed region; it is read once the timer has stopped.
"""

from __future__ import annotations

import sys

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


class KernelInputRecorder:
    """Records the density of every mask handed to the block-sparse kernel.

        with KernelInputRecorder() as rec:
            model(prompt)                 # timed region
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
        mod = self._module or sys.modules.get(KERNEL_MODULE)
        if mod is None:
            import importlib
            mod = importlib.import_module(KERNEL_MODULE)
        self._module = mod
        self._original = getattr(mod, KERNEL_NAME)
        original, per_call = self._original, self.per_call

        def recording(*args, **kwargs):
            mask = (kwargs["base_blockmask"] if "base_blockmask" in kwargs
                    else args[_BASE_BLOCKMASK_ARG])
            hmt = (kwargs["head_mask_type"] if "head_mask_type" in kwargs
                   else args[_HEAD_MASK_TYPE_ARG])
            # One small device read per call would be a sync; the shape and
            # dtype checks are host-side, and the all-ones check is deferred
            # to `executed_density`, which runs after the timer.
            per_call.append((block_density(mask, causal=bool(kwargs.get("is_causal", False))),
                             (hmt != 1).any()))
            return original(*args, **kwargs)

        setattr(mod, KERNEL_NAME, recording)
        return self

    def __exit__(self, *exc) -> None:
        setattr(self._module, KERNEL_NAME, self._original)

    def executed_density(self) -> float:
        """Mean over kernel calls (one per layer per forward). Reads the
        device, so call it after the timed region."""
        if not self.per_call:
            raise RuntimeError(
                "no block-sparse kernel call was recorded, so no executed "
                "density exists for this row")
        if any(bool(bad) for _, bad in self.per_call):
            raise ValueError(
                "head_mask_type was not all ones on a recorded call; what such "
                "a head executes is not modelled here")
        return float(torch.stack([d.float() for d, _ in self.per_call]).mean())

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
