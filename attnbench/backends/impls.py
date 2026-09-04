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

    def _blocked_for(self, cfg, mask, device):
        """Positions to fill with -inf, built ONCE per (config, device).

        True where a query may NOT attend. Both branches produce the same kind
        of thing, so they share one cache entry and one `masked_fill` below.

        Built once for the same reason flex's BlockMask is: it depends only on
        cfg, so rebuilding it per call is a tax Stage 2 times and no
        deployment pays. `tests/test_timed_region_setup.py` found this after
        the flex case made the general question worth asking -- causal was
        allocating `torch.ones(S, S).triu(1)` on every call, and block_sparse
        was re-running `to_dense_bool()` plus a `~` on every call.

        The effect here is far smaller than flex's: naive is O(S^2) in its own
        right, so an O(S^2) mask build is a bounded *factor* rather than the
        unbounded *addend* that pinned flex to a 2 ms floor at every shape.
        Fixed anyway -- "the baseline is allowed to be sloppy" is not a
        principle worth defending, and naive is the memory-wall ceiling that
        every other backend is read against.
        """
        # SIZE ONE, deliberately. An unbounded dict here OOMed the Stage 1
        # probe on 2026-09-04 after 24 minutes: this tensor is (S, S) bool --
        # 268 MB at seq_len 16384 and 1.07 GB at 32768 -- and the probe reuses
        # one backend instance across all 504 configs, so every distinct cfg
        # left a copy behind. It reached 21.69 GiB and the next make_inputs
        # could not allocate 256 MB.
        #
        # Keeping only the current config loses nothing: the point of hoisting
        # is to serve the repeated calls WITHIN one cell (1 warmup + 40 reps of
        # a single cfg), not to carry masks between cells. Cross-cell reuse was
        # never a benefit, only an accumulation.
        ck = (cfg.key(), str(device))
        if self.__dict__.get("_blocked_key") == ck:
            return self.__dict__["_blocked_value"]

        if cfg.mask == "causal":
            s = cfg.seq_len
            blocked = torch.ones(s, s, dtype=torch.bool, device=device).triu(1)
        elif cfg.mask == "block_sparse":
            if mask is None:
                raise UnsupportedConfig("block_sparse requires an explicit mask")
            if mask.seq_len != cfg.seq_len or mask.block_size != cfg.block_size:
                # The oracle comparing against a mask built for another shape
                # would certify agreement on a different function than the one
                # under test -- the sub-block causality bug's failure mode.
                raise UnsupportedConfig(
                    f"mask is ({mask.seq_len}, block {mask.block_size}) but cfg "
                    f"is ({cfg.seq_len}, block {cfg.block_size})")
            # mask is a masks.BlockSparseMask (masks.mask_for's return type),
            # not a dense tensor -- to_dense_bool() is the conversion its own
            # docstring already documents this call as using.
            blocked = ~mask.to_dense_bool(device=device)
        else:
            blocked = None

        # Assign before dropping the old one so a failure leaves no stale key.
        self.__dict__["_blocked_key"] = ck
        self.__dict__["_blocked_value"] = blocked
        return blocked

    def forward(self, q, k, v, cfg, mask=None):
        blocked = self._blocked_for(cfg, mask, q.device)
        k, v = _expand_kv(k, v, cfg)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        if blocked is not None:
            scores = scores.masked_fill(blocked, float("-inf"))
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
               "the timed region or the first measurement times the compiler. "
               "block_sparse additionally pins kernel_options BLOCK_M=BLOCK_N=64: "
               "inductor's single default config (128x128 at head_dim=128) "
               "cannot lower block_size=64 and exceeds sm_89 shared memory at "
               "block_size=128 -- see _BLOCK_SPARSE_KERNEL_OPTIONS."),
    )

    _compiled = None

    # Inductor picks exactly ONE candidate config when max_autotune is off, and
    # on sm_89 at head_dim=128 that config is BLOCK_M=BLOCK_N=128. Two
    # consequences, both measured on an L4 on 2026-09-03, and between them they
    # killed every block-sparse cell in Stage 2 segment 1 (72/72):
    #
    #   block_size=64  -> ValueError: "Q and KV block size must be divisible by
    #                     BLOCK_M and BLOCK_N. We got Q_BLOCK_SIZE=64" --
    #                     64 % 128 != 0. It RAISES rather than skipping the
    #                     config precisely because there is only one candidate
    #                     (torch/_inductor/kernel/flex/flex_attention.py, the
    #                     `if len(configs) == 1: raise` branch).
    #   block_size=128 -> divisibility passes, then the template asks for
    #                     114688 B of shared memory against sm_89's 101376 B:
    #                     "No valid triton configs. OutOfMemoryError: out of
    #                     resource ... Reducing block sizes or num_stages may
    #                     help."
    #
    # 64x64 tiles satisfy both, since 64 % 64 == 0 and 128 % 64 == 0, and the
    # halved K/V tiles bring shared memory well under the cap. The lowering
    # reads these through `setdefault`, so a caller-supplied value wins over
    # inductor's default -- that is the documented override path, not a hack.
    #
    # Applied to block_sparse ONLY. Dense flex lowers fine at the default tile
    # size and already has measured rows on this card; forcing 64x64 there
    # would move dense numbers for no reason and break comparability with them.
    # The resulting asymmetry (flex-dense and flex-sparse are not tiled alike)
    # is a real confound and is recorded in docs/limitations.md.
    _BLOCK_SPARSE_KERNEL_OPTIONS = {"BLOCK_M": 64, "BLOCK_N": 64}

    @staticmethod
    def _import_check():
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401

    @classmethod
    def _fn(cls):
        if cls._compiled is None:
            from torch.nn.attention.flex_attention import flex_attention
            cls._compiled = torch.compile(flex_attention, dynamic=False)
        return cls._compiled

    def _block_mask_for(self, cfg, mask, device):
        """Build the BlockMask ONCE per (config, device), not once per call.

        `create_block_mask` / `to_flex_block_mask` are not part of the
        attention kernel. A real deployment builds a BlockMask once and reuses
        it across every call and every layer; building it inside `forward` put
        it inside Stage 2's timed region, where it is a fixed per-call tax that
        dominates exactly the short-sequence cells. Segment 1 measured flex at
        4.22 useful TFLOPS at seq_len=1024/batch=1 against FA2's 47.9 on the
        same card, with a latency floor pinned near 2.0 ms at every shape
        measured -- the signature of a constant addend, not of a slow kernel.
        It inflated `peak_memory_mb` too, since the builder materialises a
        dense Q_LEN x KV_LEN bool before reducing it to the block grid.

        Keyed on `cfg.key()`, which hashes every config field, and the mask is
        derived deterministically from cfg by `masks.mask_for` -- so one key
        cannot correspond to two different masks. `forward` additionally
        validates the supplied mask's geometry against cfg before this is
        reached, which would catch it if that ever stopped being true.
        """
        from torch.nn.attention.flex_attention import create_block_mask

        # Size one, for the same reason NaiveAttention's is -- see there. A
        # BlockMask is far smaller than a dense (S, S) bool, but it still holds
        # device tensors, and "small leak across 504 configs" is the same bug
        # with a longer fuse.
        ck = (cfg.key(), str(device))
        if self.__dict__.get("_bm_key") == ck:
            return self.__dict__["_bm_value"]

        if cfg.mask == "block_sparse":
            built = mask.to_flex_block_mask(device=device)
        elif cfg.mask == "causal":
            def mask_mod(b, h, qi, ki):
                return qi >= ki
            built = create_block_mask(mask_mod, B=None, H=None,
                                      Q_LEN=cfg.seq_len, KV_LEN=cfg.seq_len,
                                      device=device)
        elif cfg.mask == "sliding":
            w = cfg.window

            def mask_mod(b, h, qi, ki):
                return (qi >= ki) & (qi - ki <= w)
            built = create_block_mask(mask_mod, B=None, H=None,
                                      Q_LEN=cfg.seq_len, KV_LEN=cfg.seq_len,
                                      device=device)
        else:
            built = None

        self.__dict__["_bm_key"] = ck
        self.__dict__["_bm_value"] = built
        return built

    def forward(self, q, k, v, cfg, mask=None):
        kernel_options = None

        if cfg.mask in ("causal", "full"):
            pass
        elif cfg.mask == "sliding":
            if cfg.window is None:
                raise UnsupportedConfig("sliding mask needs cfg.window")
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
            kernel_options = self._BLOCK_SPARSE_KERNEL_OPTIONS
        else:
            raise UnsupportedConfig(f"mask {cfg.mask} not wired for flex yet")

        block_mask = self._block_mask_for(cfg, mask, q.device)

        k_, v_ = _expand_kv(k, v, cfg)
        return self._fn()(q, k_, v_, block_mask=block_mask,
                          kernel_options=kernel_options)
