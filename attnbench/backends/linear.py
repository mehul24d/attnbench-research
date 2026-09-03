"""Gated Linear Attention (GLA), via flash-linear-attention (`fla`).

Scoped to GLA only for now. Gated DeltaNet was part of the original Phase B
plan for this module but needs its own verification pass against fla's
public API before it's written -- not silently dropped, just not yet done.

Everything below is confirmed from fla-org/flash-linear-attention's source
(fla/ops/gla/{chunk,fused_recurrent}.py), not assumed:

- `chunk_gla` (prefill/training path) and `fused_recurrent_gla` (the O(1)-
  state recurrent path) both take q, k, v, g[k] of shape (B, T, H, K) --
  our canonical layout is (B, H, S, D), so every call transposes in and out,
  same pattern as FlashAttention2/xFormers.
- `assert q.shape == k.shape == g.shape` inside chunk_gla: no native GQA
  path. KV (and the gate, which is generated against k's post-expansion
  shape) go through the same `_expand_kv` repeat_interleave every other
  no-native-GQA backend here uses, cost counted inside forward() as usual.
- `chunk_gla`'s public signature has no chunk_size parameter -- the
  underlying ops default it internally (64, per fla/ops/gla/chunk.py).
  cfg.chunk_size is therefore *not* forwarded to this backend's calls; the
  field stays on AttnConfig for whichever linear-attention backend's public
  API does expose the knob.
- `initial_state`/`final_state` (shape (B, H, K, V), required float32
  regardless of q/k/v dtype) is exactly the O(1) recurrent state this study
  cares about measuring -- make_decode_state/decode_step pass it through
  KVCacheState.payload untouched, never reshaping it into anything
  cache-shaped.
- Backward kernels (chunk_gla_bwd and the _bwd_kernel_* family) exist
  alongside the forward ones in the source -- a real backward path, not
  assumed from the package merely being popular.
- The recurrence is inherently causal; forward() rejects any cfg.mask other
  than "causal" rather than silently computing a causal result for a
  "full"-attention request.

The `g` (forget-gate) tensor has no analogue in the (q, k, v) triple
AttentionBackend.forward's signature carries, and a real GLA layer would
have it come from a learned projection. Synthesized here instead, seeded
from cfg.key() so it's deterministic per config -- and cached on the
instance, not regenerated on every timed_call rep, for the same reason
Phase A's run_once/timed_call split exists: reallocating it every rep would
pollute every latency sample with allocation cost unrelated to the kernel.

Unverified end-to-end: not yet run against an installed package (`pip
install flash-linear-attention`). Triton-based, so plausibly runs on a
wider compute-capability range than FA2/Block-Sparse-Attention's CUDA
kernels -- but that is not confirmed, and this backend was not part of the
already-validated free-tier (T4) backend set.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..config import AttnConfig
from .base import AttentionBackend, Capability, KVCacheState, UnsupportedConfig, register
from .impls import _expand_kv


@register
class GatedLinearAttention(AttentionBackend):
    capability = Capability(
        name="gla",
        family="linear",
        min_compute_capability=(7, 0),
        supports_backward=True,
        supports_gqa=True,           # via explicit KV expansion -- see module docstring
        supports_causal=True,
        supports_sliding=False,
        supports_block_sparse=False,
        supports_decode=True,
        head_dims=(64, 128),
        dtypes=("bfloat16", "float16"),
        notes=("pip install flash-linear-attention (import name `fla`). "
               "chunk_gla's public API has no chunk_size parameter -- "
               "cfg.chunk_size is not forwarded. mask must be 'causal'."),
    )

    @staticmethod
    def _import_check():
        import fla.ops.gla  # noqa: F401

    def _gate_for(self, cfg: AttnConfig, k_btwd: torch.Tensor) -> torch.Tensor:
        """Deterministic forget-gate tensor for this config, cached on the
        instance across calls -- see module docstring."""
        key = cfg.key()
        if getattr(self, "_gate_key", None) != key:
            seed = int(cfg.key(), 16) % (2**31)
            g_gen = torch.Generator(device=k_btwd.device).manual_seed(seed)
            self._gate = F.logsigmoid(
                torch.randn(k_btwd.shape, generator=g_gen,
                            device=k_btwd.device, dtype=torch.float32)
            )
            self._gate_key = key
        return self._gate

    def forward(self, q, k, v, cfg: AttnConfig, mask=None):
        from fla.ops.gla import chunk_gla

        if cfg.mask != "causal":
            raise UnsupportedConfig(
                f"gla: mask={cfg.mask!r} not applicable -- causal is "
                f"structural to the recurrence, not a configurable choice"
            )
        if cfg.regime != "prefill":
            raise UnsupportedConfig("gla: forward() is prefill-only, see decode_step")

        k_, v_ = _expand_kv(k, v, cfg)
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k_, v_))   # (B,H,S,D) -> (B,S,H,D)
        g_ = self._gate_for(cfg, k_)

        try:
            out, _ = chunk_gla(q_, k_, v_, g_, initial_state=None, output_final_state=False)
        except RuntimeError as e:
            raise UnsupportedConfig(f"gla: {e}") from e
        return out.transpose(1, 2)   # back to (B, H, S, D)

    def make_decode_state(self, cfg: AttnConfig, device: str = "cuda",
                           seed: int = 0) -> KVCacheState:
        from fla.ops.gla import chunk_gla

        q, k, v = self.make_inputs(cfg, device=device, seed=seed)   # cfg.seq_len context tokens
        k_, v_ = _expand_kv(k, v, cfg)
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k_, v_))
        g_ = self._gate_for(cfg, k_)

        try:
            _, final_state = chunk_gla(q_, k_, v_, g_, initial_state=None,
                                        output_final_state=True)
        except RuntimeError as e:
            raise UnsupportedConfig(f"gla: {e}") from e
        return KVCacheState(backend=self.name, payload=final_state)

    def decode_step(self, q_new, k_new, v_new, state: KVCacheState, cfg: AttnConfig):
        from fla.ops.gla import fused_recurrent_gla

        if state.backend != self.name:
            raise ValueError(
                f"state from backend {state.backend!r} passed to {self.name}.decode_step"
            )

        k_, v_ = _expand_kv(k_new, v_new, cfg)
        q_, k_, v_ = (t.transpose(1, 2) for t in (q_new, k_, v_))   # (B,H,q_len,D) -> (B,q_len,H,D)
        g_ = self._gate_for(cfg, k_)

        try:
            out, new_state = fused_recurrent_gla(
                q_, k_, v_, gk=g_, initial_state=state.payload, output_final_state=True
            )
        except RuntimeError as e:
            raise UnsupportedConfig(f"gla: {e}") from e
        return out.transpose(1, 2), KVCacheState(backend=self.name, payload=new_state)

    def issued_flops(self, cfg: AttnConfig) -> int:
        """O(seq_len) linear-attention FLOP count -- no S^2 term, unlike the
        inherited dense default this backend deliberately does not use.
        The state-update (K^T V accumulation) and query-readout GEMMs are
        both O(batch * heads * seq_len * head_dim^2); the intra-chunk
        quadratic term chunk_gla computes internally (chunk_size=64, not
        caller-controlled -- see module docstring) is a lower-order
        contribution at that chunk size and is not modeled here separately.
        """
        b, s, h, d = cfg.batch, cfg.seq_len, cfg.n_heads_q, cfg.head_dim
        flops = 4 * b * h * s * d * d
        return flops if cfg.pass_kind == "fwd" else int(flops * 3.5)

    def useful_flops(self, cfg: AttnConfig) -> int:
        """No sparsity concept for linear attention -- equal to issued_flops."""
        return self.issued_flops(cfg)
