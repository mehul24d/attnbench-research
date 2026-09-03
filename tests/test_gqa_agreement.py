"""Stage 1: verify the GQA head-grouping convention, don't assume it.

NaiveAttention expands KV heads explicitly via `repeat_interleave` in
`_expand_kv` (attnbench/backends/impls.py) -- query heads [0..g-1] share KV
head 0, [g..2g-1] share KV head 1, and so on. SDPA and FlashAttention-2 each
implement GQA internally and never call `_expand_kv`. If either kernel's
internal grouping convention (e.g. an interleaved rather than contiguous
head-to-group mapping) ever diverged from the naive expansion, every GQA
result in this study would be comparing kernels against silently mismatched
reference semantics -- a correctness-gate blind spot, since Stage 1 normally
runs backend vs. naive per-backend and would only catch a *systematic*
mismatch if this specific config were checked.

Requires CUDA; skipped otherwise. FA2 is only exercised when installed and the
device is sm80+ (not on Colab T4 / Kaggle P100, per the hardware constraints
in the project brief). SDPA's "efficient" and "flash" backends are also not a
given: on sm75 with some torch builds neither can run GQA at all and raises
UnsupportedConfig for every shape (confirmed on T4 + torch 2.11+cu128 -- Stage
0 catches it as a claim_mismatch, since SDPABackend declares supports_gqa=True
by default). A backend that raises UnsupportedConfig here is skipped, same as
Stage 0/2 treat it -- that is a capability finding, not a head-grouping
disagreement. The test only has signal if at least one candidate actually ran.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.config import AttnConfig
from attnbench.backends.base import UnsupportedConfig
from attnbench.backends.impls import NaiveAttention, FlashAttention2, SDPABackend
from attnbench.gates import TOL

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _gqa_config() -> AttnConfig:
    # n_heads_q=8, n_heads_kv=2 -> group_size=4, unambiguously GQA (not MHA/MQA).
    return AttnConfig(seq_len=256, batch=2, n_heads_q=8, n_heads_kv=2,
                       head_dim=64, dtype="bfloat16", mask="causal",
                       pass_kind="fwd")


def test_gqa_head_grouping_agreement():
    cfg = _gqa_config()
    assert cfg.is_gqa and cfg.group_size == 4

    naive = NaiveAttention()
    q, k, v = naive.make_inputs(cfg, seed=0)
    expected = naive.forward(q, k, v, cfg).double()

    candidates = [SDPABackend("efficient"), SDPABackend("math"), SDPABackend("flash")]
    if FlashAttention2.is_available():
        candidates.append(FlashAttention2())

    tol = TOL[cfg.dtype]
    checked = []
    failures = []
    for backend in candidates:
        q2, k2, v2 = backend.make_inputs(cfg, seed=0)  # identical seed -> identical tensors
        try:
            got = backend.forward(q2, k2, v2, cfg).double()
        except UnsupportedConfig:
            # This kernel/hardware/torch-build combination can't run GQA at
            # all -- Stage 0 capability-probe territory, not a head-grouping
            # disagreement. Nothing to verify against, so skip rather than fail.
            continue
        checked.append(backend.name)
        abs_err = (got - expected).abs()
        ok = bool((abs_err <= tol["atol"] + tol["rtol"] * expected.abs()).all())
        if not ok:
            failures.append(f"{backend.name}: max_abs_err={abs_err.max():.3e}")

    assert checked, (
        "every SDPA/FA2 candidate raised UnsupportedConfig for this GQA "
        "config on this hardware -- nothing was actually verified here; check "
        "the Stage 0 probe output to see which backends support GQA on this "
        "machine"
    )
    assert not failures, (
        "backend disagrees with naive-with-explicit-KV-expansion on an "
        "unambiguous GQA config -- check head-grouping convention "
        f"(group_size={cfg.group_size}), verified against {checked}: "
        + "; ".join(failures)
    )
