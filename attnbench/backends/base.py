"""Backend interface.

Every attention implementation is wrapped so the harness sees one shape. The
adapter work lives here and nowhere else; if a kernel wants a different tensor
layout, mask format or head arrangement, the wrapper converts and the rest of
the codebase never learns about it.

Two ideas matter:

1. `Capability` is what the backend *claims*. It is a starting filter only.
   Documentation for these kernels is wrong often enough that claimed support
   must never be trusted as measured support.

2. `probe()` is what the backend actually *does*. Stage 0 runs it over the grid
   and records supported / unsupported / oom / incorrect per cell. Divergence
   between claim and probe is itself a finding worth reporting.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..config import AttnConfig


@dataclass
class Capability:
    """Declared support surface. Verified, never trusted."""

    name: str
    family: str                      # dense_exact | sparse | linear | reference
    min_compute_capability: tuple = (8, 0)
    supports_backward: bool = True
    supports_gqa: bool = True
    supports_causal: bool = True
    supports_sliding: bool = False
    supports_block_sparse: bool = False
    supports_decode: bool = False
    block_sizes: tuple = ()          # empty means not applicable
    head_dims: tuple = (64, 128)
    dtypes: tuple = ("bfloat16", "float16")
    notes: str = ""

    # A length above which this kernel is known to FAULT THE DEVICE, not merely
    # fail. Distinct from every other field here: the others describe what a
    # backend declines to do, this one describes what it does destructively.
    #
    # An illegal memory access corrupts the CUDA context for the whole process.
    # It cannot be caught and recovered from -- `try/except` around the call is
    # useless, because the damage is to the context, not the Python frame -- so
    # the only safe handling is not to launch it. That is why this is a
    # capability declaration rather than an exception handler.
    #
    # Set only from a REPRODUCED observation with the Xid recorded, never
    # speculatively: the cost of an unnecessary entry is silently deleted
    # coverage.
    faults_above_seq_len: Optional[int] = None
    fault_detail: str = ""


# Prefix marking a claims_support() refusal that exists because the kernel
# FAULTS THE DEVICE, not because it declines the config. `gates.probe` keys on
# it to record status="illegal_memory_access" instead of "unsupported" -- the
# difference between "this kernel does not do that" and "this kernel does that
# destructively", which a results table must not conflate.
FAULT_REASON_PREFIX = "KNOWN DEVICE FAULT: "


class UnsupportedConfig(Exception):
    """Raised by a backend when it cannot run a config. Not an error."""


@dataclass
class KVCacheState:
    """Opaque per-backend decode-state handle. The harness never inspects
    `payload` -- deliberately unconstrained, since decode backends need
    fundamentally different shapes here: a paged KV buffer + page table for
    FlashInfer, a growing tensor for a plain KV-cache backend, a fixed-size
    hidden state for a recurrent backend (GLA, Gated DeltaNet). Forcing a
    common shape onto `payload` would hide exactly the bounded-vs-unbounded
    memory result this study exists to show.
    """

    backend: str      # guards against state from backend A reaching backend B's decode_step
    payload: object


class AttentionBackend(abc.ABC):
    """One attention implementation, adapted to a common calling convention."""

    capability: Capability

    # ---- availability -----------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Whether the package imports and the current GPU is new enough.

        Import failures are expected and normal; a missing backend is recorded,
        not raised.
        """
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        if (major, minor) < cls.capability.min_compute_capability:
            return False
        try:
            cls._import_check()
        except Exception:
            return False
        return True

    @staticmethod
    def _import_check() -> None:
        """Override to import the underlying package. Raise on failure."""
        return None

    # ---- declared filter --------------------------------------------------

    @classmethod
    def claims_support(cls, cfg: AttnConfig) -> tuple[bool, str]:
        """Cheap pre-filter from the declared capability. Returns (ok, reason)."""
        c = cls.capability
        # Checked FIRST, before anything else can decline the config for a
        # milder reason. The reason string is load-bearing: `gates.probe`
        # matches on this prefix to record the cell as a FAULT rather than as
        # an ordinary "unsupported", so a destructive kernel appears in the
        # capability matrix as a finding instead of a gap.
        if (c.faults_above_seq_len is not None
                and cfg.seq_len > c.faults_above_seq_len):
            return False, f"{FAULT_REASON_PREFIX}{c.fault_detail}"
        if cfg.pass_kind == "fwd_bwd" and not c.supports_backward:
            return False, "no backward pass"
        if cfg.is_gqa and not c.supports_gqa:
            return False, "no GQA support"
        if cfg.mask == "causal" and not c.supports_causal:
            return False, "no causal mask"
        if cfg.mask == "sliding" and not c.supports_sliding:
            return False, "no sliding window"
        if cfg.mask == "block_sparse" and not c.supports_block_sparse:
            return False, "no block sparse"
        if cfg.regime == "decode" and not c.supports_decode:
            return False, "no decode support"
        if cfg.block_size and c.block_sizes and cfg.block_size not in c.block_sizes:
            return False, f"block size {cfg.block_size} unsupported"
        if cfg.head_dim not in c.head_dims:
            return False, f"head dim {cfg.head_dim} unsupported"
        if cfg.dtype not in c.dtypes:
            return False, f"dtype {cfg.dtype} unsupported"
        return True, ""

    # ---- the actual work --------------------------------------------------

    @abc.abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                cfg: AttnConfig, mask=None) -> torch.Tensor:
        """Run attention.

        Inputs are always (batch, n_heads, seq_len, head_dim), contiguous, on
        CUDA. Output must match that layout. Any transposition the kernel needs
        happens inside this method, and its cost counts as part of the method's
        cost, which is deliberate.

        Raise UnsupportedConfig for configs this backend genuinely cannot run.
        """

    def make_inputs(self, cfg: AttnConfig, device="cuda", seed: int = 0):
        """Allocate Q, K, V for a config. Shared so every backend sees identical
        tensors, which matters for the correctness gate."""
        g = torch.Generator(device=device).manual_seed(seed)
        dt = getattr(torch, cfg.dtype)
        req = cfg.pass_kind == "fwd_bwd"

        def mk(h):
            t = torch.randn(cfg.batch, h, cfg.seq_len, cfg.head_dim,
                            generator=g, device=device, dtype=dt)
            return t.requires_grad_(req)

        return mk(cfg.n_heads_q), mk(cfg.n_heads_kv), mk(cfg.n_heads_kv)

    def run_once(self, cfg: AttnConfig, mask=None) -> None:
        """Single fwd (or fwd+bwd) invocation, allocating fresh inputs.

        For Stage 0/1 probing only, where allocation cost is irrelevant.
        Stage 2 timing must not use this: see `timed_call`.
        """
        q, k, v = self.make_inputs(cfg)
        self.timed_call(q, k, v, cfg, mask=mask)

    def timed_call(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   cfg: AttnConfig, mask=None) -> None:
        """Run fwd (or fwd+bwd) against pre-allocated inputs.

        This is what Stage 2 times. Input allocation must happen once, outside
        the timed closure -- reallocating Q/K/V on every rep (as a naive
        `run_once`-in-a-loop would) pollutes every latency sample with
        `torch.randn` cost, which dominates exactly the small-batch,
        small-seq_len cells the batch=1 sweep exists to measure accurately.
        """
        if cfg.pass_kind == "fwd_bwd":
            q.grad = None
            k.grad = None
            v.grad = None
        out = self.forward(q, k, v, cfg, mask=mask)
        if cfg.pass_kind == "fwd_bwd":
            out.sum().backward()

    # ---- decode -------------------------------------------------------

    def make_decode_state(self, cfg: AttnConfig, device: str = "cuda",
                           seed: int = 0) -> "KVCacheState":
        """Allocate and pre-fill (to cfg.seq_len context tokens) the state a
        decode_step call needs. Done once, outside the timed region --
        mirrors make_inputs for prefill.

        Default raises UnsupportedConfig, so every backend without an
        override reports "no decode support" through the same claim/probe
        machinery as everything else, with no changes needed to it.
        """
        raise UnsupportedConfig(f"{self.name}: no decode support")

    def state_from_prefill(self, k: torch.Tensor, v: torch.Tensor,
                            cfg: AttnConfig) -> "KVCacheState":
        """Build decode state from a REAL prefill's K/V, not synthetic inputs.

        The distinction from `make_decode_state` is the whole reason this
        exists, and it is easy to miss because the return type is identical.
        `make_decode_state` allocates its own `make_inputs` tensors: correct
        for a Stage 2 decode sweep, which times decode against a state of the
        right *shape* and does not care what is in it. Stage 3 generates text
        from a real prompt, so its state has to carry that prompt's actual
        keys and values -- synthetic ones would decode fluent nonsense with no
        error anywhere.

        Takes k/v in the house layout `(B, n_heads_kv, S, D)`, un-expanded for
        GQA, matching what `forward` receives. Any expansion a kernel needs
        happens inside the backend, per the same rule.

        Default raises, so a backend without decode support says so through
        the existing claim/probe machinery rather than silently degrading.
        """
        raise UnsupportedConfig(f"{self.name}: no decode support")

    def decode_step(self, q_new: torch.Tensor, k_new: torch.Tensor,
                     v_new: torch.Tensor, state: "KVCacheState",
                     cfg: AttnConfig) -> tuple[torch.Tensor, "KVCacheState"]:
        """Incorporate cfg.q_len new (Q, K, V) tokens into/against `state`,
        return (output, updated_state). This -- not `forward` -- is what a
        Stage 2 decode sweep times.

        Takes k_new/v_new, not just q_new: a real decode step folds the new
        token's K/V into the state (a KV-cache backend appends them; a
        recurrent backend updates its fixed-size state with them), not only
        attends Q against a state that's already complete. An earlier
        version of this signature took only q_new, which cannot express
        either case correctly.

        Default raises UnsupportedConfig; see make_decode_state.
        """
        raise UnsupportedConfig(f"{self.name}: no decode support")

    # ---- FLOP accounting ---------------------------------------------

    def issued_flops(self, cfg: AttnConfig) -> int:
        """FLOPs of the algorithm this backend actually runs at this shape.

        Default delegates to cfg.issued_flops() (the dense causal-respecting
        S^2 formula) -- correct as-is for dense_exact and sparse families,
        so every backend that doesn't override this inherits a correct
        number for free. A linear-attention backend overrides this with its
        own O(seq_len) chunked-scan count: the FLOP shape genuinely differs
        by algorithm, not just by a sparsity discount on a shared formula.
        """
        return cfg.issued_flops()

    def useful_flops(self, cfg: AttnConfig) -> int:
        """As issued_flops, delegates to cfg.useful_flops() by default. See
        that method's docstring: this is a bookkeeping split against the
        declared sparsity budget, not a measurement of what the kernel
        actually skipped.
        """
        return cfg.useful_flops()

    # ---- identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        return self.capability.name

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}>"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[AttentionBackend]] = {}


def register(cls: type[AttentionBackend]) -> type[AttentionBackend]:
    key = cls.capability.name
    if key in _REGISTRY:
        raise KeyError(f"backend '{key}' already registered")
    _REGISTRY[key] = cls
    return cls


def get(name: str) -> type[AttentionBackend]:
    return _REGISTRY[name]


def all_backends(available_only: bool = False):
    items = sorted(_REGISTRY.items())
    if available_only:
        items = [(n, c) for n, c in items if c.is_available()]
    return dict(items)
