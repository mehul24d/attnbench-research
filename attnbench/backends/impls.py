"""Reference backend implementations.

These four are deliberately the first ones built. They are cheap to implement
and between them they exercise every awkward part of the interface: a float64
reference path, a backend with multiple internal kernels, one that compiles from
source, and one that requires torch.compile. If the interface survives these,
the sparse and linear backends will fit without redesign.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..config import AttnConfig
from .base import AttentionBackend, Capability, UnsupportedConfig, register


def _expand_kv(k: torch.Tensor, v: torch.Tensor, cfg: AttnConfig):
    """Repeat KV heads for backends with no native GQA path.

    Note this is not free. Backends that need it pay for it inside their own
    timing, which is correct: a kernel that cannot do GQA natively really does
    cost more on a GQA workload.
    """
    if not cfg.is_gqa:
        return k, v
    g = cfg.group_size
    return k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)


# ---------------------------------------------------------------------------


@register
class NaiveAttention(AttentionBackend):
    """Textbook attention. Materialises the full S x S matrix.

    Two jobs: it is the correctness oracle (in float64 via `reference()`), and
    it is the memory-wall baseline. It will OOM before anything else does, and
    that ceiling is a reportable result rather than a failure.
    """

    capability = Capability(
        name="naive",
        family="reference",
        min_compute_capability=(0, 0),
        supports_gqa=True,
        supports_sliding=True,
        # It IS the block-sparse oracle -- forward() has an explicit
        # block_sparse branch that applies the mask. Declaring False meant
        # claims_support() rejected the very configs this backend exists to
        # provide ground truth for.
        supports_block_sparse=True,
        block_sizes=(64, 128),
        head_dims=(32, 64, 128, 256),
        dtypes=("bfloat16", "float16", "float32", "float64"),
        notes="O(S^2) memory. Oracle for the correctness gate, including "
              "block_sparse (given an explicit mask).",
    )

    def forward(self, q, k, v, cfg, mask=None):
        k, v = _expand_kv(k, v, cfg)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        if cfg.mask == "causal":
            s = cfg.seq_len
            causal = torch.ones(s, s, dtype=torch.bool, device=q.device).triu(1)
            scores = scores.masked_fill(causal, float("-inf"))
        elif cfg.mask == "block_sparse":
            if mask is None:
                raise UnsupportedConfig("block_sparse requires an explicit mask")
            # mask is a masks.BlockSparseMask (masks.mask_for's return type),
            # not a dense tensor -- to_dense_bool() is the conversion its own
            # docstring already documents this call as using.
            dense = mask.to_dense_bool(device=q.device)
            scores = scores.masked_fill(~dense, float("-inf"))
        return (torch.softmax(scores, dim=-1).to(v.dtype)) @ v

    @torch.no_grad()
    def reference(self, q, k, v, cfg, mask=None) -> torch.Tensor:
        """float64 ground truth for Stage 1. Slow by design.

        dtype lives on the tensors, not on cfg, so cfg passes through unchanged;
        only the mask logic is read from it.

        `mask` must be forwarded for block_sparse configs: the oracle has to
        see the SAME mask as the backend under test, or the two are computing
        different functions and the comparison is meaningless. Omitting it
        made the sparse correctness check raise "block_sparse requires an
        explicit mask" from the oracle -- which is why block_sparse ended up
        with zero correctness rows and would have been dropped from Stage 2.
        """
        return self.forward(q.double(), k.double(), v.double(), cfg, mask=mask)


# ---------------------------------------------------------------------------


@register
class SDPABackend(AttentionBackend):
    """torch scaled_dot_product_attention.

    Dispatches internally to flash / mem-efficient / math depending on shape and
    dtype, which is a confound: two rows labelled 'sdpa' may not be the same
    kernel. We pin the backend explicitly and record which one ran.
    """

    capability = Capability(
        name="sdpa",
        family="dense_exact",
        min_compute_capability=(0, 0),
        supports_sliding=False,
        notes="Backend forced per instance; recorded in results.",
    )

    def __init__(self, kernel: str = "efficient"):
        if kernel not in ("flash", "efficient", "math", "cudnn"):
            raise ValueError(kernel)
        self.kernel = kernel

    @property
    def name(self) -> str:
        return f"sdpa_{self.kernel}"

    def _ctx(self):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel({
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }[self.kernel])

    def forward(self, q, k, v, cfg, mask=None):
        enable_gqa = cfg.is_gqa
        try:
            with self._ctx():
                return F.scaled_dot_product_attention(
                    q, k, v,
                    is_causal=(cfg.mask == "causal"),
                    enable_gqa=enable_gqa,
                )
        except RuntimeError as e:
            # SDPA raises when the requested backend rejects the shape. That is
            # an unsupported config, not a crash.
            raise UnsupportedConfig(f"sdpa/{self.kernel}: {e}") from e


# ---------------------------------------------------------------------------


@register
class FlashAttention2(AttentionBackend):
    """flash-attn v2. Expects (B, S, H, D), so we transpose in and out."""

    capability = Capability(
        name="fa2",
        family="dense_exact",
        min_compute_capability=(8, 0),
        supports_gqa=True,
        supports_sliding=True,
        head_dims=(64, 128, 256),
        notes="Compiles from source; needs ninja + matching gcc.",
    )

    @staticmethod
    def _import_check():
        import flash_attn  # noqa: F401

    def forward(self, q, k, v, cfg, mask=None):
        from flash_attn import flash_attn_func

        # FA2 wants (B, S, H, D); ours is (B, H, S, D).
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))
        window = (-1, -1)
        if cfg.mask == "sliding":
            if cfg.window is None:
                raise UnsupportedConfig("sliding mask needs cfg.window")
            window = (cfg.window, 0)
        out = flash_attn_func(
            q_, k_, v_,
            causal=(cfg.mask == "causal"),
            window_size=window,
        )
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------


@register
class FlexAttentionBackend(AttentionBackend):
    """FlexAttention via torch.compile.

    Compilation is cached per (mask_mod, shape). We warm up outside the timed
    region so we measure the kernel, not the compiler. Known to blow up memory
    on block-sparse masks; that is expected and gets recorded by the probe.
    """

    capability = Capability(
        name="flex",
        family="dense_exact",
        min_compute_capability=(8, 0),
        supports_gqa=True,
        supports_sliding=True,
        supports_block_sparse=True,
        block_sizes=(64, 128),
        notes=("O(S^2) block-mask representation can OOM at long S. "
               "block_sparse consumes masks.BlockSparseMask via "
               "to_flex_block_mask(), which pins flex's BLOCK_SIZE to the "
               "mask's block_size so the sparsity exploited is the sparsity "
               "configured. Covers block_size=64, which BSA cannot: BSA's "
               "block size is hardcoded to 128 inside block_sparse_attn_func, "
               "so without this backend every 64-wide cell in Stage 2 would "
               "have no sparse arm at all. Compilation must be warmed outside "
               "the timed region or the first measurement times the compiler."),
    )

    _compiled = None

    @staticmethod
    def _import_check():
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

    @classmethod
    def _fn(cls):
        if cls._compiled is None:
            from torch.nn.attention.flex_attention import flex_attention
            cls._compiled = torch.compile(flex_attention, dynamic=False)
        return cls._compiled

    def forward(self, q, k, v, cfg, mask=None):
        from torch.nn.attention.flex_attention import create_block_mask

        block_mask = None
        mask_mod = None

        if cfg.mask == "causal":
            def mask_mod(b, h, qi, ki):
                return qi >= ki
        elif cfg.mask == "sliding":
            w = cfg.window
            if w is None:
                raise UnsupportedConfig("sliding mask needs cfg.window")

            def mask_mod(b, h, qi, ki):
                return (qi >= ki) & (qi - ki <= w)
        elif cfg.mask == "full":
            mask_mod = None
        elif cfg.mask == "block_sparse":
            # The BlockMask comes from the shared masks.BlockSparseMask, not
            # from a pattern rebuilt here: the whole point of that type is
            # that two backends asked for "the same" mask cannot diverge.
            # Causality and the BLOCK_SIZE/block_size correspondence are
            # handled inside to_flex_block_mask -- see its docstring.
            if mask is None:
                raise UnsupportedConfig("block_sparse requires an explicit mask")
            if mask.seq_len != cfg.seq_len or mask.block_size != cfg.block_size:
                raise UnsupportedConfig(
                    f"mask is ({mask.seq_len}, block {mask.block_size}) but cfg "
                    f"is ({cfg.seq_len}, block {cfg.block_size}) -- a mismatch "
                    f"here would measure a different sparsity than configured")
            block_mask = mask.to_flex_block_mask(device=q.device)
        else:
            raise UnsupportedConfig(f"mask {cfg.mask} not wired for flex yet")

        if block_mask is None and mask_mod is not None:
            block_mask = create_block_mask(
                mask_mod, B=None, H=None,
                Q_LEN=cfg.seq_len, KV_LEN=cfg.seq_len, device=q.device,
            )

        k_, v_ = _expand_kv(k, v, cfg)
        return self._fn()(q, k_, v_, block_mask=block_mask)
