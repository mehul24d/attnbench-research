"""What flex hands to the compiled kernel: tile options, and when the mask is built.

Companion to test_flex_block_sparse.py, which covers mask *semantics*. This
file covers the two changes made after Stage 2 segment 1 measured flex
block-sparse 0/72 on an L4. Both are checked at the call boundary, so they run
on CPU without the triton kernel that fails there.

Segment 1's two failures split exactly by block_size:

  block_size=64   ValueError: Q and KV block size must be divisible by BLOCK_M
                  and BLOCK_N. We got Q_BLOCK_SIZE=64 and KV_BLOCK_SIZE=64.
  block_size=128  No valid triton configs. OutOfMemoryError: out of resource:
                  Required: 114688  Hardware limit: 101376

Neither is perturbable by cache state or memory fragmentation -- 64 % 128 does
not become 0, and 114688 does not fall below 101376. Both follow from inductor
picking exactly one default config (BLOCK_M=BLOCK_N=128 at head_dim=128) when
max_autotune is off, and `kernel_options` overrides it: the lowering reads
those through `setdefault`.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.backends.impls import FlexAttentionBackend
from attnbench.config import AttnConfig
from attnbench.masks import mask_for

# head_dim=128 is the only value in Stage 2's block-sparse grid, and it is what
# makes inductor's default tile 128 wide -- i.e. it is load-bearing here, not
# incidental.
HEAD_DIM = 128


def _cfg(**kw):
    base = dict(seq_len=256, batch=1, n_heads_q=4, n_heads_kv=4,
                head_dim=HEAD_DIM, dtype="float32", mask="block_sparse",
                block_size=64, sparsity=0.5, mask_source="random")
    base.update(kw)
    return AttnConfig(**base)


class _Recorder:
    """Stands in for the compiled flex_attention, capturing what it was handed."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, k, v, block_mask=None, kernel_options=None):
        self.calls.append({"block_mask": block_mask,
                           "kernel_options": kernel_options})
        return torch.zeros_like(q)


@pytest.fixture
def flex(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(FlexAttentionBackend, "_fn", classmethod(lambda cls: rec))
    return FlexAttentionBackend(), rec


def _qkv(cfg):
    dt = getattr(torch, cfg.dtype)
    shape = (cfg.batch, cfg.n_heads_q, cfg.seq_len, cfg.head_dim)
    return tuple(torch.zeros(shape, dtype=dt) for _ in range(3))


@pytest.mark.parametrize("block_size", (64, 128))
def test_block_sparse_pins_64x64_tiles(flex, block_size):
    """Both failing block sizes must reach the kernel with tiles that divide
    them and fit sm_89 shared memory: 64 % 64 == 0 and 128 % 64 == 0."""
    be, rec = flex
    cfg = _cfg(block_size=block_size)
    be.forward(*_qkv(cfg), cfg, mask=mask_for(cfg))

    opts = rec.calls[0]["kernel_options"]
    assert opts == {"BLOCK_M": 64, "BLOCK_N": 64}
    assert block_size % opts["BLOCK_M"] == 0
    assert block_size % opts["BLOCK_N"] == 0


def test_dense_is_left_at_inductor_defaults(flex):
    """Dense flex lowers fine at the default tile size and already has measured
    rows on this card. Forcing 64x64 there would move those numbers for no
    reason, so the override is scoped to block_sparse. The resulting asymmetry
    is a real confound and is recorded in docs/limitations.md."""
    be, rec = flex
    cfg = _cfg(mask="causal", block_size=None, sparsity=None, mask_source=None)
    be.forward(*_qkv(cfg), cfg)
    assert rec.calls[0]["kernel_options"] is None


@pytest.mark.parametrize("mask_kind", ("causal", "block_sparse"))
def test_the_block_mask_is_built_once_not_once_per_call(flex, mask_kind):
    """Mask construction is not part of the attention kernel, and Stage 2 times
    whatever `forward` does. Building it per call put a fixed tax inside the
    timed region: segment 1 read flex at 4.22 useful TFLOPS at
    seq_len=1024/batch=1 against FA2's 47.9 on the same card, with a latency
    floor near 2.0 ms at every shape measured -- the signature of a constant
    addend, not of a slow kernel."""
    if mask_kind == "causal":
        cfg = _cfg(mask="causal", block_size=None, sparsity=None,
                   mask_source=None)
        mask = None
    else:
        cfg = _cfg()
        mask = mask_for(cfg)

    be, rec = flex
    q, k, v = _qkv(cfg)
    for _ in range(3):
        be.forward(q, k, v, cfg, mask=mask)

    built = [c["block_mask"] for c in rec.calls]
    assert len(built) == 3
    assert all(b is built[0] for b in built), (
        "every call must reuse the same BlockMask object")


def test_a_different_config_gets_a_different_mask(flex):
    """The cache must key on the config. Two shapes sharing one mask would
    measure a sparsity other than the one the cell claims."""
    be, rec = flex
    for sparsity in (0.5, 0.9):
        cfg = _cfg(sparsity=sparsity)
        be.forward(*_qkv(cfg), cfg, mask=mask_for(cfg))
    assert rec.calls[0]["block_mask"] is not rec.calls[1]["block_mask"]


# --- the override is capped at the length it was actually verified at -------

def test_the_override_applies_only_up_to_the_verified_length(flex):
    """1024 is not a cautious guess -- it is the literal extent of the
    evidence. The 2026-09-04 diagnostic ran under a 30-minute cap and bought
    two configs, both at seq_len=1024. The next session died to an Xid 31 MMU
    fault whose last logged configs were 32768."""
    be, rec = flex
    for seq_len in (256, 1024):
        cfg = _cfg(seq_len=seq_len)
        be.forward(*_qkv(cfg), cfg, mask=mask_for(cfg))
    assert all(c["kernel_options"] == {"BLOCK_M": 64, "BLOCK_N": 64}
               for c in rec.calls)


def test_above_the_verified_length_no_override_is_forced(flex):
    """Inductor's default then applies, which on sm_89 at head_dim=128 cannot
    lower block_size=64 and exceeds shared memory at 128 -- so flex-sparse is
    effectively capped at short lengths on this card. Failing to lower is a
    recorded result; a GPU page fault is not."""
    be, rec = flex
    cfg = _cfg(seq_len=2048)
    be.forward(*_qkv(cfg), cfg, mask=mask_for(cfg))
    assert rec.calls[0]["kernel_options"] is None


def test_the_cap_is_not_silently_raised_past_the_evidence():
    """A future edit that widens this must also widen the verification. The
    constant is the claim, so the claim is asserted."""
    assert FlexAttentionBackend._BLOCK_SPARSE_KERNEL_OPTIONS_MAX_SEQ == 1024, (
        "the override is verified at seq_len<=1024 only (two configs, "
        "2026-09-04). Raising this needs measurements at the new length, "
        "not just a larger number.")
