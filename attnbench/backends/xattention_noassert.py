"""XAttention with its Python asserts removed (T2.17; plan Section 5.5).

The same estimator, kernel and arguments as `XAttentionBackend`, with one
difference: `xattn_estimate` comes from the authors' package loaded through
`attnbench.assert_strip`, which compiles only that package with
`optimize=1`. Its Python `assert` statements and `if __debug__` blocks are
gone; Triton device asserts, `TORCH_CHECK` and other C-level checks stay. The
arm is accepted only if `AssertGate` finds its masks and outputs bitwise
identical to the shipped arm's on every recorded prompt, and only in a
session whose `assert_strip.session_check()` passed.

What removing them removes, at upstream commit `e379887`: in the prefill
path, two asserts on GPU tensors per query chunk in `find_blocks_chunked`
(each a `bool()` of a tensor, so a synchronisation), and asserts on Python
integers and shapes in `xattn_estimate` and the Triton wrappers, which do not
wait. `Xattention_prefill` itself is not called by either arm.

`estimate` repeats `XAttentionBackend.estimate` line for line apart from its
import; a test compares the two syntax trees, so the arms cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch

from .. import assert_strip
from ..config import AttnConfig
from .base import register
from .impls import _expand_kv
from .xattention import (XATTN_BLOCK_SIZE, XAttentionBackend, official_chunk_size,
                         triton_for_device)


@register
class XAttentionNoAssertBackend(XAttentionBackend):
    capability = replace(
        XAttentionBackend.capability, name="xattention_noassert",
        notes=XAttentionBackend.capability.notes + " Python asserts removed: "
              "the xattn package alone compiled with optimize=1 by "
              "attnbench.assert_strip (T2.17).")

    @staticmethod
    def _import_check():
        import block_sparse_attn  # noqa: F401
        assert_strip.install()
        import attnbench_xattn_noassert.src.Xattention  # noqa: F401

    def estimate(self, q: torch.Tensor, k_rep: torch.Tensor) -> torch.Tensor:
        """`XAttentionBackend.estimate`, from the stripped package."""
        assert_strip.install()
        from attnbench_xattn_noassert.src.Xattention import xattn_estimate
        use_triton = self.use_triton
        if q.device.type == "cuda":
            if self.triton_effective is None:
                self.triton_effective = triton_for_device(
                    torch.cuda.get_device_properties(q.device).name, self.use_triton)
            use_triton = self.triton_effective
        _, simple_mask = xattn_estimate(
            q, k_rep, block_size=XATTN_BLOCK_SIZE, stride=self.stride, norm=1,
            threshold=self._threshold_for(q, q.shape[1]), select_mode="inverse",
            use_triton=use_triton, causal=True,
            chunk_size=official_chunk_size(k_rep.shape[2]), kdb=1,
            keep_sink=self.keep_sink, keep_recent=self.keep_recent)
        return simple_mask


@dataclass(frozen=True)
class GateRecord:
    layer: Optional[int]
    mask_equal: bool
    output_equal: bool


class AssertGate:
    """The bitwise gate: both arms on the same inputs, layer by layer.

    Use it as the model's prefill backend over the recorded prompts. In each
    layer it runs the shipped and the stripped estimator on the same q and
    keys and compares their masks, then both forwards and compares their
    outputs, and returns the shipped output so the next layer sees exactly
    what the shipped arm would. Untimed: `torch.equal` reads the device.
    """

    def __init__(self, shipped: XAttentionBackend, stripped: XAttentionNoAssertBackend):
        if type(shipped) is not XAttentionBackend or \
                not isinstance(stripped, XAttentionNoAssertBackend):
            raise TypeError("the gate pairs the shipped arm with the asserts-removed arm")
        if (shipped.threshold, shipped.stride, shipped.keep_sink, shipped.keep_recent,
                shipped.use_triton) != (stripped.threshold, stripped.stride,
                                        stripped.keep_sink, stripped.keep_recent,
                                        stripped.use_triton):
            raise ValueError("the two arms must have identical settings")
        self.shipped, self.stripped = shipped, stripped
        self.layer_idx: Optional[int] = None
        self.records: list[GateRecord] = []

    def set_layer(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.shipped.set_layer(layer_idx)
        self.stripped.set_layer(layer_idx)

    def reset_density(self) -> None:
        self.shipped.reset_density()
        self.stripped.reset_density()

    @property
    def last_layer_density(self):
        return self.shipped.last_layer_density

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        k_rep, _ = _expand_kv(k, v, cfg)
        mask_equal = torch.equal(self.shipped.estimate(q, k_rep),
                                 self.stripped.estimate(q, k_rep))
        out = self.shipped.forward(q, k, v, cfg, mask=mask)
        other = self.stripped.forward(q, k, v, cfg, mask=mask)
        self.records.append(GateRecord(self.layer_idx, mask_equal, torch.equal(out, other)))
        return out

    @property
    def passed(self) -> bool:
        return bool(self.records) and all(r.mask_equal and r.output_equal
                                          for r in self.records)

    def failures(self) -> list[GateRecord]:
        return [r for r in self.records if not (r.mask_equal and r.output_equal)]
