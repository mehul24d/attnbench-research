"""`masks.importance_block_mask_device` is the reference builder, bit for bit.

The inline-estimator arm (audit C1) and the fine-block arms (block-size axis)
both build masks on the device. They are only comparable with every banked
oracle row if the device builder makes the SAME mask the reference does --
not an equal-scoring one. The vectorised builder in
`scripts/_vec_mask_for_measurement.py` drops the tie-break jitter and was
shown on 2026-09-17 to pick different members of tied groups, which is fine
for latency and wrong for accuracy. So equality here is demanded on the
inputs that broke that one: fp16, tie-dense, and all-zero scores where the
jitter alone decides.

The CUDA case skips on the workstation and runs in the Phase A correctness
gate on the instance, where the stream property is also re-checked on the
image's torch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from attnbench import masks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

SPARSITIES = (0.5, 0.75, 0.9)
SIZES = (1, 2, 3, 5, 17, 48, 65, 129)


def _scores(n: int, kind: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    probs = torch.softmax((torch.randn(n, n, generator=g) * 6)
                          .masked_fill(upper, float("-inf")), dim=1)
    if kind == "fp16_tied":
        return probs.half()                      # what score_cache stores
    if kind == "fp32":
        return probs
    if kind == "fp64":
        return probs.double()
    if kind == "zeros":
        return torch.zeros(n, n, dtype=torch.float16)   # jitter decides everything
    raise ValueError(kind)


def _both(n, s, scores, causal=True, seed="cfg-identity"):
    ref = masks.importance_block_mask(n * 16, 16, s, scores, causal=causal,
                                      identity_seed=seed)
    dev = masks.importance_block_mask_device(n * 16, 16, s, scores, causal=causal,
                                             identity_seed=seed)
    return ref, dev


def test_one_draw_equals_the_reference_row_by_row_stream():
    """The property the builder rests on: torch's CPU generator yields the same
    values for rand(a), rand(b), ... as for rand(a + b + ...)."""
    for seed in (0, 7, 2**40 + 3):
        lens = [0, 1, 2, 5, 40, 255, 0, 1000, 3]
        g = torch.Generator().manual_seed(seed)
        rows = torch.cat([torch.rand(k, generator=g) for k in lens if k])
        g2 = torch.Generator().manual_seed(seed)
        assert torch.equal(rows, torch.rand(sum(lens), generator=g2))


@pytest.mark.parametrize("kind", ["fp16_tied", "fp32", "fp64", "zeros"])
@pytest.mark.parametrize("causal", [True, False])
def test_bitwise_equal_to_the_reference(kind, causal):
    for n in SIZES:
        for s in SPARSITIES:
            ref, dev = _both(n, s, _scores(n, kind, seed=n), causal=causal)
            assert torch.equal(ref.active, dev.active.cpu()), (kind, causal, n, s)
            assert (ref.seed, ref.source, ref.causal) == (dev.seed, dev.source, dev.causal)


def test_the_fixture_exercises_ties_the_jitterless_builder_gets_wrong():
    """Negative control: on the same tie-dense input, the latency-only
    vectorised builder (no jitter) disagrees with the reference somewhere. If
    it did not, the equality above would prove nothing about ties."""
    from _vec_mask_for_measurement import importance_block_mask_vectorised
    disagreements = 0
    for n in (48, 65, 129):
        for s in SPARSITIES:
            sc = _scores(n, "zeros")
            ref = masks.importance_block_mask(n * 16, 16, s, sc, causal=True,
                                              identity_seed="cfg-identity")
            vec = importance_block_mask_vectorised(n, s, sc, causal=True)
            disagreements += int((ref.active ^ vec).sum())
    assert disagreements > 0


def test_nesting_across_sparsity_holds():
    n = 129
    sc = _scores(n, "fp16_tied")
    masks_by_s = [masks.importance_block_mask_device(n * 16, 16, s, sc, causal=True,
                                                     identity_seed="x").active
                  for s in SPARSITIES]
    for coarse, fine in zip(masks_by_s, masks_by_s[1:]):
        assert torch.equal(fine & coarse, fine)          # 0.9 subset of 0.75 subset of 0.5


def test_a_different_identity_breaks_ties_differently():
    """The seed is load-bearing: on all-tied scores two identities must give
    different masks, or the jitter is not reaching the ranking."""
    n = 65
    sc = _scores(n, "zeros")
    a = masks.importance_block_mask_device(n * 16, 16, 0.75, sc, causal=True, identity_seed="a")
    b = masks.importance_block_mask_device(n * 16, 16, 0.75, sc, causal=True, identity_seed="b")
    assert not torch.equal(a.active, b.active)


def test_wrong_shape_refuses():
    with pytest.raises(ValueError):
        masks.importance_block_mask_device(64 * 16, 16, 0.5, torch.zeros(10, 10),
                                           causal=True, identity_seed="x")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (runs in the Phase A gate)")
def test_bitwise_equal_on_cuda():
    for kind in ("fp16_tied", "fp32", "zeros"):
        for n in (65, 129, 1024):
            for s in SPARSITIES:
                sc = _scores(n, kind, seed=n)
                ref = masks.importance_block_mask(n * 16, 16, s, sc, causal=True,
                                                  identity_seed="cfg")
                dev = masks.importance_block_mask_device(n * 16, 16, s, sc.cuda(),
                                                         causal=True, identity_seed="cfg")
                assert dev.active.is_cuda
                assert torch.equal(ref.active, dev.active.cpu()), (kind, n, s)
