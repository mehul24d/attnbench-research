"""FlashAttention-3 (the Hopper build of Dao-AILab/flash-attention).

Written like `FlashAttention2` in impls.py. Everything declared here was read
from the pinned source, not from memory: release v2.8.3.post2, commit
49774b31c8d1172a5b28f72e9aed0f322bf4ccba (`hopper/`, package version
3.0.0.b1), the pin in `scripts/_fa3_pin.sh`.

- The wheel installs the package `flash_attn_3` (its `_C` extension registers
  `torch.ops.flash_attn_3`) and the top-level module `flash_attn_interface`
  (`hopper/setup.py`: PACKAGE_NAME, py_modules). `flash_attn_interface`
  imports `flash_attn_3._C` itself, so importing it checks both.
- `flash_attn_func` at the pin returns `out` alone (`FlashAttnFunc.apply`).
  Other FA3 versions returned `(out, softmax_lse)`; `_unpack` takes either,
  and the tests feed both, because a wrong unpack passes a shape check on one
  version and fails on the other.
- Layout is (B, S, H, D), as in FA2. GQA is native (Q heads a multiple of KV
  heads).

**The restricted build** (researcher's decision, 28 Oct 2026, after the first
full build was OOM-killed): head dimension 128, forward, bf16 and fp16, sm_90a.
`BUILD_FLAGS` below is the one place the flags are written:
`scripts/build_fa3_wheel.sh` reads it, the wheel's manifest records it, and
`analysis/code_identity.py` fingerprints it with the timed region, so rows
from wheels built differently are never joined as the same code.

Why the backend refuses a head dim other than 128 itself, rather than letting
the kernel do it: at the pin, `run_mha_fwd_constexpr` (`hopper/flash_api.cpp`)
dispatches `d <= 64` to the hdim-64 kernel only `#ifndef
FLASHATTENTION_DISABLE_HDIM64`; with it disabled, a head dim of 64 falls
through to `d <= 128` and runs the hdim-128 kernel, padded. That is a
different kernel from the one an unrestricted wheel runs, timed as if it were
the same. Sliding windows are refused for the same reason: DISABLE_LOCAL
makes the kernel raise, and the error should name the build, not the kernel.

Not run on an H100 yet. The float64 correctness gate and the capability rows
wait for the first H100 session (docs/t64_fa3_plan.md).
"""

from __future__ import annotations

from .base import AttentionBackend, Capability, UnsupportedConfig, register

# Each flag exists in the pinned hopper/setup.py (an os.getenv of that name);
# the build script refuses one that does not, so a typo cannot be ignored.
BUILD_FLAGS = (
    "FLASH_ATTENTION_FORCE_BUILD=TRUE",          # never fetch a prebuilt wheel
    "FLASH_ATTENTION_DISABLE_BACKWARD=TRUE",
    "FLASH_ATTENTION_DISABLE_FP8=TRUE",
    "FLASH_ATTENTION_DISABLE_SM80=TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM64=TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM96=TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM192=TRUE",
    "FLASH_ATTENTION_DISABLE_HDIM256=TRUE",
    "FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE",
    "FLASH_ATTENTION_DISABLE_APPENDKV=TRUE",
    "FLASH_ATTENTION_DISABLE_SPLIT=TRUE",
    "FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE",
    "FLASH_ATTENTION_DISABLE_LOCAL=TRUE",
)
BUILT_HEAD_DIMS = (128,)


def _unpack(result):
    """The output tensor, whether the kernel returned it alone or first in a
    tuple (`(out, softmax_lse)` in other FA3 versions)."""
    if isinstance(result, (tuple, list)):
        return result[0]
    return result


@register
class FlashAttention3(AttentionBackend):
    """flash-attn v3 (Hopper). Expects (B, S, H, D), so we transpose in and out."""

    capability = Capability(
        name="fa3",
        family="dense_exact",
        min_compute_capability=(9, 0),
        supports_backward=False,
        supports_gqa=True,
        supports_sliding=False,
        head_dims=BUILT_HEAD_DIMS,
        dtypes=("bfloat16", "float16"),
        notes=("Restricted wheel built from source at a pinned commit by "
               "scripts/build_fa3_wheel.sh: head dim 128, forward, bf16/fp16, "
               "sm_90a; flags in fa3.BUILD_FLAGS."),
    )

    @staticmethod
    def _import_check():
        import flash_attn_interface  # noqa: F401  (imports flash_attn_3._C)

    def forward(self, q, k, v, cfg, mask=None):
        if cfg.head_dim not in BUILT_HEAD_DIMS:
            raise UnsupportedConfig(
                f"fa3: head dim {cfg.head_dim} is not in this wheel, which is built for "
                f"head dim {', '.join(map(str, BUILT_HEAD_DIMS))} only "
                f"(fa3.BUILD_FLAGS); the kernel would silently run a padded "
                f"hdim-128 instance instead")
        if cfg.mask not in ("causal", "full"):
            raise UnsupportedConfig(
                f"fa3: mask {cfg.mask} is not supported by this wheel "
                f"(built with FLASH_ATTENTION_DISABLE_LOCAL; causal and full only)")
        from flash_attn_interface import flash_attn_func

        # FA3 wants (B, S, H, D); ours is (B, H, S, D).
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))
        out = _unpack(flash_attn_func(q_, k_, v_, causal=(cfg.mask == "causal")))
        return out.transpose(1, 2)
