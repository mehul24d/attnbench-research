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
  heads). `window_size=(left, right)` has FA2's meaning when Q and K have the
  same length, which they do here.
- Head dimensions: the kernels are instantiated for 64, 96, 128, 192 and 256
  (`setup.py` HEAD_DIMENSIONS with no DISABLE_HDIM flag set by the build).
- The build sets FLASH_ATTENTION_DISABLE_BACKWARD, _DISABLE_FP8 and
  _DISABLE_SM80, so the wheel is forward-only, bf16/fp16 and sm_90a only.
  The capability says so: a backward cell is declined, not attempted.

Not run on an H100 yet. The float64 correctness gate and the capability rows
wait for the first H100 session (docs/t64_fa3_plan.md).
"""

from __future__ import annotations

from .base import AttentionBackend, Capability, UnsupportedConfig, register


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
        supports_sliding=True,
        head_dims=(64, 96, 128, 192, 256),
        dtypes=("bfloat16", "float16"),
        notes=("Wheel built from source at a pinned commit by "
               "scripts/build_fa3_wheel.sh; sm_90a only, forward only, no FP8."),
    )

    @staticmethod
    def _import_check():
        import flash_attn_interface  # noqa: F401  (imports flash_attn_3._C)

    def forward(self, q, k, v, cfg, mask=None):
        from flash_attn_interface import flash_attn_func

        if cfg.mask not in ("causal", "full", "sliding"):
            raise UnsupportedConfig(f"fa3: mask {cfg.mask} not wired")
        # FA3 wants (B, S, H, D); ours is (B, H, S, D).
        q_, k_, v_ = (t.transpose(1, 2) for t in (q, k, v))
        window = (-1, -1)
        if cfg.mask == "sliding":
            if cfg.window is None:
                raise UnsupportedConfig("sliding mask needs cfg.window")
            window = (cfg.window, 0)
        out = _unpack(flash_attn_func(
            q_, k_, v_,
            causal=(cfg.mask == "causal"),
            window_size=window,
        ))
        return out.transpose(1, 2)
