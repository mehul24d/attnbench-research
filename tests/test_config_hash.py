"""AttnConfig.key() is the primary key for every result row. Two bugs are
worth guarding against on every change to AttnConfig:

1. Adding or removing a field silently forks the entire keyspace, orphaning
   every already-collected result row. This actually happened while this
   study was being built -- adding regime/q_len/quant_scheme/chunk_size
   changed the hash of every existing config, including Stage 0/1 data
   already collected on Colab. Cheap to absorb there (free hardware, just
   re-run); the same mistake landing mid-Stage-2 would orphan rented-GPU
   hours instead. test_canonical_hash_pinned exists so that class of change
   fails loudly in CI instead of silently forking the keyspace -- pin the
   hash only right after a deliberate field change, never before.

2. Fields that must distinguish otherwise-identical configs (mask_source is
   the sharpest example: random-mask and importance-mask results must never
   collide) need an explicit test, since json.dumps(asdict(...)) silently
   including or excluding a field is easy to get wrong in either direction.
"""

from __future__ import annotations

from attnbench.config import AttnConfig


def _base_cfg(**overrides) -> AttnConfig:
    fields = dict(seq_len=1024, batch=4, n_heads_q=32, n_heads_kv=8,
                  head_dim=128, dtype="bfloat16", mask="causal", pass_kind="fwd")
    fields.update(overrides)
    return AttnConfig(**fields)


def test_mask_source_forks_hash():
    random_cfg = _base_cfg(mask="block_sparse", sparsity=0.9, block_size=64,
                            mask_source="random")
    importance_cfg = _base_cfg(mask="block_sparse", sparsity=0.9, block_size=64,
                                mask_source="importance")
    assert random_cfg.key() != importance_cfg.key()


def test_identical_fields_hash_identically():
    a = _base_cfg()
    b = _base_cfg()
    assert a.key() == b.key()


def test_canonical_hash_pinned():
    """Pinned right after regime/q_len/quant_scheme/chunk_size landed. If
    this fails, you changed AttnConfig's field set (or json/hash
    machinery) -- confirm that's deliberate, then re-pin, then treat every
    already-collected result row as orphaned and re-run Stage 0/1 (Stage 2/5
    too, if any exist by the time this fires).
    """
    cfg = _base_cfg()
    assert cfg.key() == "031453d2f2a3"
