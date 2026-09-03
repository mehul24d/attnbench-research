"""Configuration objects describing a single point in the sweep grid."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Literal, Optional

Pass = Literal["fwd", "fwd_bwd"]
MaskKind = Literal["full", "causal", "sliding", "block_sparse"]
Regime = Literal["prefill", "decode"]


@dataclass(frozen=True)
class AttnConfig:
    """One cell of the benchmark grid.

    Every field here is a dimension we sweep or a constant we must pin. If a
    value influences a measurement it belongs in this object, because the hash
    of this object is the primary key for every recorded result.
    """

    seq_len: int
    batch: int
    n_heads_q: int
    n_heads_kv: int          # equal to n_heads_q for MHA, smaller for GQA
    head_dim: int = 128
    dtype: str = "bfloat16"
    mask: MaskKind = "causal"
    pass_kind: Pass = "fwd"

    # Sparse-only. None for dense backends.
    sparsity: Optional[float] = None      # fraction of blocks skipped
    block_size: Optional[int] = None
    # "random" for Stage 2 kernel timing, "importance" for Stage 3 accuracy.
    # Conflating these is the failure mode this whole study exists to correct.
    mask_source: Optional[Literal["random", "importance"]] = None

    # Sliding-window only.
    window: Optional[int] = None

    # Declared but not swept: every config in this study is a padded
    # rectangular (batch, seq_len) tensor. Real varlen/packed kernels have a
    # materially different cost profile (no padding waste), but implementing
    # that path is out of scope for the compute budget here. Hashed so a
    # future varlen result can never collide with a padded one.
    varlen: bool = False

    # Decode (single/few-query attention against a KV cache or recurrent
    # state). "prefill" configs are unaffected by q_len -- seq_len alone
    # covers both Q and KV there, exactly as before this field existed.
    regime: Regime = "prefill"
    q_len: Optional[int] = None    # decode only: new query tokens this step.
                                     # seq_len continues to mean KV/context
                                     # length in both regimes.

    # Quantized kernels (e.g. SageAttention). Separate from `dtype`: a
    # kernel's internal compute precision and per-block scale factors aren't
    # expressible as a storage dtype string, and the correctness gate raises
    # on any dtype/scheme it has no tolerance registered for rather than
    # silently reusing bfloat16's -- this field is what that raise guards.
    quant_scheme: Optional[str] = None

    # Chunked linear-attention kernels (GLA, Gated DeltaNet). The linear-
    # attention analogue of block_size: None for every backend that isn't one.
    #
    # Decided, not left ambiguous: GLA's public API (fla.ops.gla.chunk_gla)
    # does not expose a chunk_size parameter -- it's fixed internally (64).
    # backends/linear.py does not forward this field to that call, and never
    # sweeps it. The field stays on AttnConfig only because a future
    # backend's public API might expose the knob (a hand-written kernel, or
    # a different chunked-attention library); if none ever does, remove this
    # field rather than let it sit unused in the hash indefinitely.
    chunk_size: Optional[int] = None

    @property
    def is_gqa(self) -> bool:
        return self.n_heads_kv != self.n_heads_q

    @property
    def group_size(self) -> int:
        if self.n_heads_q % self.n_heads_kv:
            raise ValueError(
                f"n_heads_q={self.n_heads_q} not divisible by "
                f"n_heads_kv={self.n_heads_kv}"
            )
        return self.n_heads_q // self.n_heads_kv

    @property
    def head_layout(self) -> str:
        return f"{self.n_heads_q}:{self.n_heads_kv}"

    def issued_flops(self) -> int:
        """FLOPs of the dense-equivalent problem at this shape.

        Respects causal masking (structural, not "skipped work") but ignores
        any sparsity discount. This is the wall-clock yardstick: comparable
        across sparsity levels and directly against a dense backend's number
        at the same shape. Deliberately excludes softmax and bookkeeping,
        matching the convention in the FlashAttention papers so our numbers
        stay comparable to published ones.

        `block_sparse` gets the same causal halving as `causal`: every
        block_sparse config in this study is causal too (masks.mask_for
        hardcodes it, since AttnConfig.mask has no separate causal flag for
        block_sparse -- see that function's docstring). Without this, a
        block_sparse config's `useful_flops()` (== issued_flops() * (1 -
        sparsity)) would be discounted off the *full* non-causal quadratic
        form instead of the causal one, overstating the actual work a real
        block-sparse kernel issues by 2x -- caught via timing_probe.py's
        grid-hour extrapolation, which uses useful_flops() as its FLOPs
        figure and would otherwise inherit the same 2x error.
        """
        b, s, h, d = self.batch, self.seq_len, self.n_heads_q, self.head_dim
        fwd = 4 * b * h * s * s * d          # QK^T and PV
        if self.mask in ("causal", "block_sparse"):
            fwd //= 2
        return fwd if self.pass_kind == "fwd" else int(fwd * 3.5)

    def useful_flops(self) -> int:
        """`issued_flops()` discounted by the *declared* sparsity budget.

        This is a bookkeeping split derived entirely from config -- it does
        not know whether a kernel actually skipped that work, and must not
        be read as a measurement of sparsity conversion. Whether a kernel
        converts sparsity into real speed shows up by comparing measured
        latency across sparsity levels (or profiler skip/occupancy counters),
        not from this ratio alone. Not comparable across sparsity levels for
        that reason -- it shrinks by construction as sparsity rises.
        Equal to `issued_flops()` when sparsity is None.
        """
        flops = self.issued_flops()
        if self.sparsity is not None:
            flops = int(flops * (1.0 - self.sparsity))
        return flops

    def key(self) -> str:
        """Stable short hash. Primary key for result rows."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def shape_family_key(self) -> str:
        """Stable hash of everything EXCEPT seq_len.

        Stage 1's correctness oracle cannot exist at Stage 2's longest
        lengths: a float64 naive reference needs 64 GiB at seq_len=16384 and
        256 GiB at 32768, against a 23 GiB card. So a cell at 32768 can never
        have a pass recorded at its own exact config, and matching the pass
        table on `key()` would reject every long cell -- which is precisely
        what it did on 2026-09-03 (0 of 504 cells matched).

        Correctness is therefore established per shape family -- backend,
        mask, block_size, sparsity, dtype, head geometry, pass_kind -- and
        seq_len is handled as its own axis: exact float64 agreement at the
        lengths where the oracle fits, cross-backend agreement above. Which
        one certified a given row is recorded in `check_kind`, and the
        measured error is recorded per length so fidelity-versus-length is
        readable as a result rather than assumed away.

        See docs/limitations.md.
        """
        d = asdict(self)
        d.pop("seq_len", None)
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config_key"] = self.key()
        d["shape_family_key"] = self.shape_family_key()
        d["is_gqa"] = self.is_gqa
        d["head_layout"] = self.head_layout
        return d


@dataclass
class SweepGrid:
    """The Stage 2 grid. Kept as data so it lives in YAML, not in for-loops."""

    seq_lens: tuple = (1024, 2048, 4096, 8192, 16384, 32768)
    batches: tuple = (1, 4, 16)
    head_layouts: tuple = ((32, 32), (32, 8))   # MHA, GQA
    passes: tuple = ("fwd", "fwd_bwd")
    masks: tuple = ("causal",)
    sparsities: tuple = (0.5, 0.75, 0.9)
    block_sizes: tuple = (64, 128)
    head_dim: int = 128
    dtype: str = "bfloat16"

    def dense_configs(self):
        for s in self.seq_lens:
            for b in self.batches:
                for hq, hkv in self.head_layouts:
                    for p in self.passes:
                        for m in self.masks:
                            yield AttnConfig(
                                seq_len=s, batch=b, n_heads_q=hq,
                                n_heads_kv=hkv, head_dim=self.head_dim,
                                dtype=self.dtype, mask=m, pass_kind=p,
                            )

    def sparse_configs(self, mask_source: str = "random"):
        for s in self.seq_lens:
            for b in self.batches:
                for hq, hkv in self.head_layouts:
                    for p in self.passes:
                        for sp in self.sparsities:
                            for bs in self.block_sizes:
                                yield AttnConfig(
                                    seq_len=s, batch=b, n_heads_q=hq,
                                    n_heads_kv=hkv, head_dim=self.head_dim,
                                    dtype=self.dtype, mask="block_sparse",
                                    pass_kind=p, sparsity=sp, block_size=bs,
                                    mask_source=mask_source,
                                )
