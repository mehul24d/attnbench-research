"""The precomputed-mask ablation arm (T2.12; plan Section 5.5).

"Each estimator's masks computed offline and preloaded; then the same kernel.
Synchronisations: none. Separates estimator compute and synchronisation from
kernel and launch cost."

Three parts.

**`MaskRecorder`** wraps a source arm's backend and records, per layer, the
block mask the arm handed to the kernel, in the form the kernel takes:
`(1, H_q, n_blocks, n_blocks)` bool. The source arm runs once, untimed, on
the prompt; the masks it produced are the masks recorded, so each estimator's
own selection (mean-pool's head-uniform mask, the per-head selector's, the
oracle's, XAttention's own) is what gets replayed, never a re-derivation. For
XAttention the mask is captured at the estimator's return and sliced exactly
as the arm slices it for the kernel (`[:, :, :nb, :nb]`).

**`MaskSet`** is those masks with what is needed to refuse a wrong replay: the
estimator, the code path, the digest of the prompt that produced them, the
sequence length and head count. A mask is only valid for the prompt it was
computed from (layer L's mask depends on hidden states that depend on every
earlier layer's mask), so the digest travels with it and the harness checks it
against the prompt it is about to time. A set saves and loads with a sha256
sidecar.

**`PrecomputedMaskBackend`** is the arm. It makes every layer's mask
device-resident and contiguous when it is built, outside any timed region,
and its forward does the same things the XAttention arm does after its
estimator returns: repeat the KV heads, then one call to `kernels.bsa_prefill`,
the one kernel call site (so executed density is recorded for it like any
other arm's). Nothing else: no scoring, no mask build, no host read, no
device op to compute a density (those are taken at construction and appended
as they are). It takes no external mask, batch 1 only (`bsa_prefill`'s limit,
as XAttention's), and is not registered with the backend registry: it is
bound to one prompt's masks, not a sweepable kernel.

**What it separates, and what it does not.** Against the source arm on the
same prompt, the difference in prefill time is the estimator's compute, its
synchronisations and the mask build's host work. It is not a deployable
method (the masks come from running the estimator first), so the harness
labels it an ablation and refuses it any scoring call or mask build, in warm-up
as well as in a timed block (`ground_truth.PRECOMPUTED`).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from . import kernels
from .backends.base import UnsupportedConfig
from .backends.impls import _expand_kv
from .densities import block_density
from .masks import BlockSparseMask, PerHeadBlockMask

BLOCK = kernels.BLOCK_SIZE
SCHEMA = "attnbench.mask_set/1"


class MaskSetError(ValueError):
    """A mask set that is malformed, or does not belong to what it is used with."""


def prompt_digest(input_ids: torch.Tensor) -> str:
    """sha256 over the token ids (as int64) and their shape."""
    ids = input_ids.detach().to("cpu", torch.int64).contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(ids.shape)).encode())
    h.update(ids.numpy().tobytes())
    return h.hexdigest()


@dataclass
class MaskSet:
    estimator: str          # the source arm's name
    code_path: str          # its code path, as its rows record it
    mask_selector: str      # masks.MASK_SELECTORS
    prompt_digest: str
    seq_len: int
    n_heads_q: int
    masks: dict             # layer index -> bool (1, H_q, nb, nb)

    def __post_init__(self) -> None:
        nb = -(-self.seq_len // BLOCK)
        if not self.masks:
            raise MaskSetError("a mask set holds at least one layer")
        if sorted(self.masks) != list(range(len(self.masks))):
            raise MaskSetError(f"layers must be 0..{len(self.masks) - 1}, got {sorted(self.masks)}")
        for layer, m in self.masks.items():
            if m.dtype != torch.bool or tuple(m.shape) != (1, self.n_heads_q, nb, nb):
                raise MaskSetError(
                    f"layer {layer}: mask is {m.dtype} {tuple(m.shape)}, expected bool "
                    f"{(1, self.n_heads_q, nb, nb)} for seq_len {self.seq_len}")

    @property
    def n_layers(self) -> int:
        return len(self.masks)

    def mask_density(self) -> float:
        """Mean over layers of kept blocks over causally valid blocks, over heads."""
        return float(torch.stack([block_density(m.cpu()) for m in self.masks.values()]).mean())

    def save(self, directory) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"mask_set_{self.estimator}_{self.prompt_digest[:16]}.pt"
        side = path.with_name(path.name + ".sha256")
        if path.exists() or side.exists():
            raise MaskSetError(f"{path} already exists; a recorded mask set is not overwritten")
        torch.save({"schema": SCHEMA, "estimator": self.estimator, "code_path": self.code_path,
                    "mask_selector": self.mask_selector, "prompt_digest": self.prompt_digest,
                    "seq_len": self.seq_len, "n_heads_q": self.n_heads_q,
                    "masks": {int(k): v.cpu().contiguous() for k, v in self.masks.items()}}, path)
        side.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        return path

    @classmethod
    def load(cls, path) -> "MaskSet":
        path = Path(path)
        side = path.with_name(path.name + ".sha256")
        if not side.exists():
            raise MaskSetError(f"{path.name} has no sha256 sidecar")
        stated = side.read_text().split()
        if len(stated) != 2 or stated[1] != path.name:
            raise MaskSetError(f"{side.name} is not in sha256sum format for {path.name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != stated[0]:
            raise MaskSetError(f"{path.name} does not match its sidecar's sha256")
        d = torch.load(path, weights_only=True)
        if d.get("schema") != SCHEMA:
            raise MaskSetError(f"schema {d.get('schema')!r}, expected {SCHEMA!r}")
        d.pop("schema")
        return cls(**d)


class MaskRecorder:
    """A source arm's backend, recording the mask each layer hands the kernel.

    Used once, untimed, offline. `mask` arguments (a `BlockSparseMask` or a
    `PerHeadBlockMask`, which is how the model wrapper hands the oracle's and
    the inline estimator's masks to the kernel) are recorded as they are
    given; a backend that selects its own blocks and exposes `estimate`
    (XAttention) has its estimator's return captured instead.
    """

    def __init__(self, inner, *, estimator: str, code_path: str, mask_selector: str):
        self.inner, self.estimator = inner, estimator
        self.code_path, self.mask_selector = code_path, mask_selector
        self.layer_idx: Optional[int] = None
        self.recorded: dict = {}

    def __getattr__(self, name):           # reset_density, last_layer_density, ...
        return getattr(self.inner, name)

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        if hasattr(self.inner, "set_layer"):
            self.inner.set_layer(layer_idx)

    def forward(self, q, k, v, cfg, mask=None):
        if self.layer_idx is None:
            raise RuntimeError("the recorder needs set_layer() before each forward")
        captured = []
        original = None
        if mask is None and hasattr(self.inner, "estimate"):
            original = self.inner.estimate

            def capture(*a, **kw):
                out = original(*a, **kw)
                captured.append(out)
                return out
            self.inner.estimate = capture
        try:
            out = self.inner.forward(q, k, v, cfg, mask=mask)
        finally:
            if original is not None:
                del self.inner.estimate          # back to the class's method
        s, h = q.shape[2], q.shape[1]
        nb = -(-s // BLOCK)
        if isinstance(mask, PerHeadBlockMask):
            rec = mask.to_bsa()
        elif isinstance(mask, BlockSparseMask):
            if mask.block_size != BLOCK:
                raise MaskSetError(f"the kernel's block is {BLOCK}; the mask's is {mask.block_size}")
            rec = mask.to_block_sparse_attn_mask(1, h, device=q.device)
        elif captured:
            rec = captured[-1][:, :, :nb, :nb]
        else:
            raise MaskSetError(
                f"layer {self.layer_idx}: the arm handed the kernel no mask and exposes no "
                f"estimator to capture, so there is nothing to record")
        self.recorded[self.layer_idx] = rec.detach().clone().contiguous()
        return out

    def mask_set(self, input_ids: torch.Tensor, *, n_heads_q: int) -> MaskSet:
        return MaskSet(estimator=self.estimator, code_path=self.code_path,
                       mask_selector=self.mask_selector, prompt_digest=prompt_digest(input_ids),
                       seq_len=int(input_ids.shape[-1]), n_heads_q=n_heads_q,
                       masks=dict(self.recorded))


class PrecomputedMaskBackend:
    """The ablation arm. See the module docstring.

    `kernel` replaces the import of `block_sparse_attn_func`, for tests on a
    machine without the library (`kernels.bsa_prefill`'s own parameter).
    """

    name = "precomputed_mask"

    def __init__(self, mask_set: MaskSet, input_ids: torch.Tensor, *, device: str = "cuda",
                 kernel: Optional[Callable] = None):
        if prompt_digest(input_ids) != mask_set.prompt_digest:
            raise MaskSetError(
                f"the masks of {mask_set.estimator!r} were computed on another prompt; "
                f"a mask is valid only for the prompt it was computed from")
        self.mask_set, self._kernel = mask_set, kernel
        # Resident and contiguous before any clock starts.
        self._masks = {i: m.to(device).contiguous() for i, m in mask_set.masks.items()}
        self._density = {i: block_density(m) for i, m in self._masks.items()}
        self.layer_idx: Optional[int] = None
        self.last_layer_density: list = []

    @property
    def estimator(self) -> str:
        return self.mask_set.estimator

    def reset_density(self) -> None:
        self.last_layer_density = []

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx

    def claims_support(self, cfg):
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        if mask is not None:
            raise UnsupportedConfig("the precomputed arm takes its masks from its mask set; "
                                    "got an external mask")
        if cfg.mask != "causal":
            raise UnsupportedConfig(f"precomputed arm: mask kind {cfg.mask!r} (expects 'causal')")
        if q.shape[0] != 1:
            raise UnsupportedConfig("precomputed arm: batch 1 only (bsa_prefill's limit)")
        if self.layer_idx is None:
            raise RuntimeError("the precomputed arm needs set_layer() before each forward")
        ms = self.mask_set
        if q.shape[2] != ms.seq_len or q.shape[1] != ms.n_heads_q:
            raise UnsupportedConfig(
                f"masks were recorded at seq_len {ms.seq_len} with {ms.n_heads_q} query heads; "
                f"this forward has {q.shape[2]} and {q.shape[1]}")
        if self.layer_idx not in self._masks:
            raise UnsupportedConfig(f"no recorded mask for layer {self.layer_idx}")
        k_rep, v_rep = _expand_kv(k, v, cfg)
        self.last_layer_density.append(self._density[self.layer_idx])
        return kernels.bsa_prefill(q, k_rep, v_rep, self._masks[self.layer_idx],
                                   kernel=self._kernel)
