"""The FA3 gate's long configurations and its exact-only rule. CPU-only.

On 6 Nov 2026 the first H100 gate failed on two batch-16, 4096-token configs
whose float64 oracle needs 64 GiB: with FA3 alone there was nothing to compare
against, and `check_kind` was "cross_backend" with no error measured. The gate
now runs exact-only, and checks the model's own head layout at 8192 to 32768
tokens against a float64 reference computed a block of queries at a time.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from attnbench import gates
from attnbench.backends.base import AttentionBackend, UnsupportedConfig
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig

ROOT = Path(__file__).resolve().parent.parent


def _cfg(s, hq, hkv, **kw):
    return AttnConfig(seq_len=s, batch=kw.pop("batch", 1), n_heads_q=hq, n_heads_kv=hkv,
                      head_dim=kw.pop("head_dim", 32), dtype="bfloat16",
                      mask=kw.pop("mask", "causal"), pass_kind="fwd", **kw)


def _inputs(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda h: torch.randn(cfg.batch, h, cfg.seq_len, cfg.head_dim, generator=g,
                               dtype=torch.float64)
    return mk(cfg.n_heads_q), mk(cfg.n_heads_kv), mk(cfg.n_heads_kv)


@pytest.mark.parametrize("hq, hkv", [(4, 4), (12, 2), (8, 2)])
@pytest.mark.parametrize("s, qb", [(64, 16), (70, 16), (33, 512), (64, 1)])
def test_the_blockwise_reference_is_the_naive_oracle(hq, hkv, s, qb):
    cfg = _cfg(s, hq, hkv)
    q, k, v = _inputs(cfg)
    want = NaiveAttention().reference(q, k, v, cfg)
    got = gates.blockwise_reference(q, k, v, cfg, q_block=qb)
    assert got.dtype == torch.float64 and got.shape == want.shape
    assert (got - want).abs().max().item() < 1e-12


def test_the_blockwise_reference_is_causal():
    cfg = _cfg(32, 4, 2)
    q, k, v = _inputs(cfg)
    base = gates.blockwise_reference(q, k, v, cfg, q_block=8)
    v2 = v.clone()
    v2[:, :, 20:] += 5.0              # later values must not reach earlier rows
    moved = gates.blockwise_reference(q, k, v2, cfg, q_block=8)
    assert torch.equal(base[:, :, :20], moved[:, :, :20])
    assert not torch.equal(base[:, :, 20:], moved[:, :, 20:])


def test_the_blockwise_reference_refuses_other_masks():
    cfg = _cfg(32, 4, 2, mask="sliding", window=8)
    q, k, v = _inputs(cfg)
    with pytest.raises(UnsupportedConfig, match="causal only"):
        gates.blockwise_reference(q, k, v, cfg)


class _Kernel(AttentionBackend):
    """A stand-in kernel: the reference's answer rounded to bf16, plus `bias`."""
    capability = NaiveAttention.capability
    bias = 0.0

    def make_inputs(self, cfg, device="cpu", seed=0):
        g = torch.Generator().manual_seed(seed)
        mk = lambda h: torch.randn(cfg.batch, h, cfg.seq_len, cfg.head_dim, generator=g,
                                   dtype=torch.bfloat16)
        return mk(cfg.n_heads_q), mk(cfg.n_heads_kv), mk(cfg.n_heads_kv)

    def forward(self, q, k, v, cfg, mask=None):
        out = gates.blockwise_reference(q, k, v, cfg, q_block=8)
        return (out + self.bias).to(torch.bfloat16)


def test_a_kernel_inside_the_tolerance_passes_the_blockwise_check_as_exact():
    cfg = _cfg(48, 4, 2)
    r = gates.check_correctness_blockwise(_Kernel(), cfg, device="cpu", q_block=8)
    assert r.passed and r.check_kind == "exact" and r.max_abs_err < 2e-2
    assert r.detail.startswith("blockwise float64 reference")


def test_a_kernel_outside_the_tolerance_fails_the_blockwise_check():
    k = _Kernel()
    k.bias = 0.2
    r = gates.check_correctness_blockwise(k, _cfg(48, 4, 2), device="cpu", q_block=8)
    assert not r.passed and r.check_kind == "exact" and r.max_abs_err > 0.1


def test_the_naive_path_is_unchanged_without_a_reference():
    r = gates.check_correctness(_Kernel(), _cfg(48, 4, 2), device="cpu")
    assert r.passed and not r.detail.startswith("blockwise")


def test_the_gates_long_configs_are_the_models_layout_at_the_oracles_blind_lengths():
    cs = gates.gate_long_configs()
    assert sorted(c.seq_len for c in cs) == [8192, 16384, 32768]
    assert {(c.n_heads_q, c.n_heads_kv, c.batch, c.mask, c.pass_kind) for c in cs} == {
        (12, 2, 1, "causal", "fwd")}
    # the naive oracle fits an 80 GB card to 16384 for this layout (25.8 GB) and not at 32768 (103 GB):
    # the blockwise reference is what makes the longest length an exact check
    budget = int(85.02 * 2**30 * gates.ORACLE_BUDGET_FRACTION)
    fits = {c.seq_len: gates.exact_oracle_fits(c, budget_bytes=budget) for c in cs}
    assert fits == {8192: True, 16384: True, 32768: False}


def _run_probe():
    spec = importlib.util.spec_from_file_location("run_probe_gate_mod",
                                                  ROOT / "scripts" / "run_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_probe_gate_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_probe_offers_exact_only_and_the_long_configs(monkeypatch, capsys):
    rp = _run_probe()
    monkeypatch.setattr(sys, "argv", ["run_probe.py", "--help"])
    with pytest.raises(SystemExit):
        rp.main()
    out = capsys.readouterr().out
    assert "--exact-only" in out and "--long-model-configs" in out


def test_the_two_failed_6_nov_configs_have_no_float64_oracle_on_an_80_gb_card():
    """The configs that failed the first H100 gate: batch 16, 32 query heads, 4096."""
    budget = int(85.02 * 2**30 * gates.ORACLE_BUDGET_FRACTION)
    for hkv in (32, 8):
        assert not gates.exact_oracle_fits(_cfg(4096, 32, hkv, batch=16, head_dim=128),
                                           budget_bytes=budget)
    assert gates.exact_oracle_fits(_cfg(4096, 12, 2, batch=16, head_dim=128), budget_bytes=budget)


def test_stage_1_routes_each_config_to_the_right_check(monkeypatch):
    rp = _run_probe()
    calls = []
    monkeypatch.setattr(rp, "check_correctness_blockwise", lambda b, c: calls.append("blockwise") or "B")
    monkeypatch.setattr(rp, "check_for_family", lambda b, c, references: calls.append("family") or "F")
    fits = {4096: True}
    monkeypatch.setattr(rp, "exact_oracle_fits", lambda c: fits.get(c.seq_len, False))
    b = type("B", (), {"name": "fa3"})()
    small, big, long_ = _cfg(4096, 12, 2), _cfg(4096, 32, 8, batch=16), _cfg(32768, 12, 2)
    assert rp.stage1_check(b, small, [], exact_only=True) == ("F", None)
    assert rp.stage1_check(b, long_, [], long_keys={long_.key()}, exact_only=True) == ("B", None)
    r, left_out = rp.stage1_check(b, _cfg(8192, 32, 8, batch=16), [], exact_only=True)
    assert r is None and left_out["backend"] == "fa3" and left_out["batch"] == 16
    assert calls == ["family", "blockwise"]
    # without exact-only an oracle-less config still takes the old cross-backend path
    assert rp.stage1_check(b, big, [], exact_only=False) == ("F", None)
