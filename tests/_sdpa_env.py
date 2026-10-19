"""Why five tests xfail in a CPU-only container, and only there.

PyTorch's CPU flash SDPA kernel (2.8) runs ordinary multi-head attention but
rejects grouped-query attention: with fewer KV heads than query heads it
raises "No available kernel", which `SDPABackend` reports as
`UnsupportedConfig`. Five tests drive a GQA model through `sdpa_flash` on
CPU, so in a container without a GPU they fail for that reason alone:

- test_generation_wiring.py::test_a_backend_with_no_decode_path_gets_the_dense_one_and_says_so
- test_generation_wiring.py::test_a_pinned_fallback_decodes_through_it_and_the_row_says_pinned
- test_inline_estimator.py::test_layer_zero_matches_the_two_pass_cheap_arm
- test_inline_estimator.py::test_later_layers_see_sparse_inputs_not_the_cached_scores
- test_swappable_attention_model.py::test_cheap_scoring_pass_propagates_faithful_hidden_states

They have failed this way since at least e9e56cd (4 Oct 2026).

The mark is conditional on a probe of that exact call, not on "no GPU": if
the CPU flash kernel accepts GQA, the tests run normally and must pass. It is
strict, so an unexpected pass is reported, and it accepts only
`UnsupportedConfig`, so a test that starts failing any other way (a wrong
assertion, a different exception) is still a failure.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn.functional as F

from attnbench.backends.base import UnsupportedConfig


def cpu_flash_sdpa_supports_gqa() -> bool:
    """Whether the CPU flash SDPA kernel accepts a GQA shape."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    q = torch.zeros(1, 4, 2, 8)
    kv = torch.zeros(1, 2, 2, 8)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except RuntimeError:
        return False
    return True


xfail_without_cpu_flash_gqa = pytest.mark.xfail(
    condition=not cpu_flash_sdpa_supports_gqa(),
    reason="CPU flash SDPA rejects GQA shapes in this PyTorch build "
           "(tests/_sdpa_env.py)",
    raises=UnsupportedConfig,
    strict=True,
)
