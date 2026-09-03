"""SageAttention: INT8-quantized (block-scaled) attention.

Named sage_attention.py, not sageattention.py, for the same reason as
xformers_backend.py -- avoids shadowing the real `sageattention` package
name in the module list.

Uses cfg.quant_scheme, not cfg.dtype, to record what's quantized: the
storage dtype of Q/K/V going in (bf16/fp16) is not what the kernel actually
computes internally (int8 Q/K, per-block scale factors). gates.py's
correctness gate now keys its tolerance lookup on quant_scheme when set, and
raises rather than silently reusing bfloat16's tolerance -- there is
deliberately no TOL entry for any sage scheme yet, since one should come
from calibration against real measured error, not a guess made while
writing this adapter.

Published SageAttention is an inference-only kernel: there is no quantized
backward path, so supports_backward=False here, unlike every other backend
in this repo so far -- a real, not incidental, capability difference.

Unverified against an installed package (`pip install sageattention`) --
not yet run, per this project's standing rule.
"""

from __future__ import annotations

import torch

from ..config import AttnConfig
from .base import AttentionBackend, Capability, UnsupportedConfig, register
from .impls import _expand_kv


@register
class SageAttention(AttentionBackend):
    """sageattention.sageattn, INT8 QK with per-block scaling."""

    capability = Capability(
        name="sage",
        family="dense_exact",   # exact math, approximate (quantized) arithmetic
        min_compute_capability=(8, 0),
        supports_backward=False,
        supports_gqa=True,
        supports_sliding=False,
        head_dims=(64, 128),
        dtypes=("bfloat16", "float16"),
        notes=("pip install sageattention. Inference-only (no backward). "
               "Requires cfg.quant_scheme set; Stage 1 raises without a "
               "matching TOL entry rather than reusing bfloat16's."),
    )

    @staticmethod
    def _import_check():
        import sageattention  # noqa: F401

    def forward(self, q, k, v, cfg, mask=None):
        from sageattention import sageattn

        if cfg.quant_scheme is None:
            raise UnsupportedConfig("sage: cfg.quant_scheme must be set")
        if cfg.mask not in ("causal", "full"):
            raise UnsupportedConfig(f"sage: mask {cfg.mask} not wired")

        k_, v_ = _expand_kv(k, v, cfg)
        try:
            return sageattn(q, k_, v_, tensor_layout="HND",
                             is_causal=(cfg.mask == "causal"))
        except RuntimeError as e:
            raise UnsupportedConfig(f"sage: {e}") from e
