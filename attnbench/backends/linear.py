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

THE FORGET GATE IS SYNTHETIC, AND THAT MAKES THIS BACKEND TIMING-ONLY.

The `g` (forget-gate) tensor has no analogue in the (q, k, v) triple
AttentionBackend.forward's signature carries, and a real GLA layer would
have it come from a learned projection. It is synthesized here instead,
seeded from cfg.key() so it's deterministic per config -- and cached on the
instance, not regenerated on every timed_call rep, for the same reason
Phase A's run_once/timed_call split exists: reallocating it every rep would
pollute every latency sample with allocation cost unrelated to the kernel.

For Stage 2 that is correct and sufficient: a kernel's throughput does not
depend on the VALUES in its gate tensor, only its shape and dtype.

For Stage 3 it is fatal, and it was allowed to run there on 2026-09-06.
`F.logsigmoid(randn)` averages -0.81 per step, so the recurrent state is
multiplied by 0.45 EVERY token: an effective memory horizon of **1.24
tokens**. At seq_len 2048 the prompt survives at exp(-1654), which is zero.
The model therefore answered from the last token or two of a prompt whose
final ~20 tokens are the same template in every example, and produced 15
distinct predictions across 300 distinct 2048-token contexts (the dense arm
produced 300 of 300). Fluent-looking garbage, a real latency, a real
stop_reason, and 900 result rows that meant nothing.

Qwen2.5 has no gate projection to borrow -- its weights were never trained
with one -- so there is no correct gate to supply, and this is not a bug
with a fix so much as a scope boundary that was not enforced. It is
enforced now: `gate_source` must be named explicitly, and the synthetic
path has to be asked for by a caller that knows it is only measuring time.

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


# The three gates this backend can run, in one place so the result schema's
# `GateSource` literal can be checked against it rather than kept in sync by
# hand -- see tests/test_gate_source_on_rows.py. A fourth value would be a
# fourth thing a row could mean, and the row schema has to know about it.
GATE_SOURCES = ("learned", "synthetic", "ungated")


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

    def __init__(self, gate_source: str = "learned"):
        """`gate_source` has no safe default, so the default is the one that
        REFUSES.

        "synthetic" must be asked for by name. It produces a kernel-accurate
        latency and a meaningless output, which is exactly the shape of thing
        that should never be reachable by omission -- see the module
        docstring for the 900 rows it produced before this existed.
        """
        if gate_source not in GATE_SOURCES:
            raise ValueError(
                f"gate_source must be one of {GATE_SOURCES}, "
                f"got {gate_source!r}")
        self.gate_source = gate_source

    def _require_usable_gate(self) -> None:
        if self.gate_source in ("synthetic", "ungated"):
            return
        raise UnsupportedConfig(
            "gla: no learned forget gate is available. This backend "
            "synthesizes g from cfg.key(), which gives a memory horizon of "
            "~1.24 tokens -- kernel-accurate for TIMING and meaningless for "
            "anything that reads the output. Qwen2.5 has no gate projection "
            "to borrow. Pass gate_source='synthetic' if you are measuring "
            "time only; there is no correct value for accuracy work.")

    def _gate_for(self, cfg: AttnConfig, k_btwd: torch.Tensor) -> torch.Tensor:
        """Deterministic forget-gate tensor for this config, cached on the
        instance across calls -- see module docstring."""
        if self.gate_source == "ungated":
            # g is a LOG decay applied as exp(cumsum(g)), so zeros mean a
            # decay factor of exactly 1: nothing is ever forgotten. This is a
            # real mechanism (ungated linear attention), not a repair of the
            # synthetic gate -- it is simply not the mechanism Qwen's weights
            # were trained for. It exists to answer one question: with the
            # state retained in full, does the output depend on the context
            # at all? See docs/gla_arm_decision.md.
            return torch.zeros(k_btwd.shape, device=k_btwd.device,
                               dtype=torch.float32)
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
        # Before the import, deliberately: a backend that cannot produce a
        # meaningful answer should say so on any machine, not only on one
        # where the optional CUDA dependency happens to be installed.
        self._require_usable_gate()
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
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as e:
            raise UnsupportedConfig(f"gla: {e}") from e
        return out.transpose(1, 2)   # back to (B, H, S, D)

    def state_from_prefill(self, k, v, cfg) -> KVCacheState:
        """Fixed-size recurrent state from a real prompt's K/V.

        Gated by the same check as forward(): a state folded through a
        synthetic gate has forgotten the prompt before it is even handed
        over, so producing one for an accuracy caller would move the failure
        one step downstream rather than prevent it.

        The bounded arm of the comparison, and the reason `KVCacheState.payload`
        is deliberately unconstrained. SDPA's payload grows with every token
        decoded; this one is `(B, H, D, D)` and does not change size no matter
        how long the context or the generation. That difference is the result
        Stage 3 can show and a single-forward kernel benchmark cannot.

        Q is required by `chunk_gla`'s signature but contributes nothing to
        `final_state` -- the recurrence folds only K and V into it -- so zeros
        of the right shape are passed rather than inventing query content that
        would look meaningful in a debugger.
        """
        self._require_usable_gate()
        from fla.ops.gla import chunk_gla

        k_, v_ = _expand_kv(k, v, cfg)
        k_, v_ = k_.transpose(1, 2), v_.transpose(1, 2)     # (B,S,H,D)
        q_ = torch.zeros_like(k_)
        g_ = self._gate_for(cfg, k_)
        try:
            _, final_state = chunk_gla(q_, k_, v_, g_, initial_state=None,
                                        output_final_state=True)
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as e:
            raise UnsupportedConfig(f"gla: {e}") from e
        return KVCacheState(backend=self.name, payload=final_state)

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
        except torch.cuda.OutOfMemoryError:
            raise
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
        except torch.cuda.OutOfMemoryError:
            raise
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
