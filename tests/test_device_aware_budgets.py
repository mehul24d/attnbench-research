"""Memory budgets are a fraction of the device, not a number of bytes. CPU-only.

`EXACT_ORACLE_BUDGET_BYTES` (8 GiB) and `CROSS_BACKEND_BUDGET_BYTES` (12 GiB)
were sized by hand for a 22.03 GiB L4. Carrying them unchanged to an 80 GB
A100 would make one commit mean two different things depending on which card
ran it -- verification strength conditioned on hardware, which is precisely
what the commit-pinning machinery exists to stop. Raising them to A100 values
instead would break the L4. So the fraction is the constant.

Two properties, pulling opposite ways, and both are required:

  1. On an L4, every verdict is EXACTLY what it was before this change. A
     re-derivation that silently re-grades existing data is a baseline reset
     wearing the costume of a refactor.
  2. On an A100, the three config classes the plan promises are actually
     restored -- otherwise the A100 session is built on arithmetic nobody
     checked.

And the budget that produced a verdict travels ON the row. "declined: memory
infeasible" no longer names one number, so a reader who infers it from the GPU
model is reconstructing the standard from context and will get it wrong the
first time a third card appears. Same principle as `check_kind`.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import gates
from attnbench.config import AttnConfig, SweepGrid

GIB = 2 ** 30
L4_BYTES = gates.FALLBACK_DEVICE_BYTES        # 23.66e9, as torch reports it
A100_BYTES = 85_899_345_920                   # 80 GiB


@pytest.fixture
def as_a100(monkeypatch):
    """Pretend the visible CUDA device is an 80 GiB A100."""
    class _Props:
        total_memory = A100_BYTES
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: _Props())


def _cfg(seq_len, batch, heads=32):
    return AttnConfig(seq_len=seq_len, batch=batch, n_heads_q=heads,
                      n_heads_kv=8, head_dim=128, dtype="bfloat16",
                      mask="causal")


# ---------------------------------------------------------------------------
# 1. The L4 must be graded exactly as before.
# ---------------------------------------------------------------------------

def test_the_fractions_reproduce_the_hand_picked_l4_budgets():
    assert gates.exact_oracle_budget_bytes("cpu") / GIB == pytest.approx(8.15, abs=0.1)
    assert gates.cross_backend_budget_bytes("cpu") / GIB == pytest.approx(12.1, abs=0.1)


def test_no_l4_verdict_changed():
    """The load-bearing one. Every config in the grid must be graded now
    exactly as the old absolute constants graded it -- 8 GiB and 12 GiB."""
    g = SweepGrid()
    changed = []
    for s in g.seq_lens:
        for b in g.batches:
            for hq, hkv in g.head_layouts:
                c = AttnConfig(seq_len=s, batch=b, n_heads_q=hq, n_heads_kv=hkv,
                               head_dim=128, dtype="bfloat16", mask="causal")
                if gates.exact_oracle_fits(c, device="cpu") != (
                        gates.oracle_bytes(c) <= 8 * GIB):
                    changed.append(("oracle", s, b, hq))
                if gates.cross_backend_fits(c, device="cpu") != (
                        gates.cross_backend_bytes(c) <= 12 * GIB):
                    changed.append(("cross", s, b, hq))
    assert not changed, f"the refactor re-graded L4 cells: {changed}"


def test_the_boundary_is_not_close_to_any_real_config():
    """Why a 2% shift in the threshold is safe: oracle costs across this grid
    are widely spaced powers of two. Asserted rather than left as arithmetic
    in a comment, because the next grid edit could make it false."""
    g = SweepGrid()
    costs = sorted({gates.oracle_bytes(
        AttnConfig(seq_len=s, batch=b, n_heads_q=hq, n_heads_kv=hkv,
                   head_dim=128, dtype="bfloat16", mask="causal")) / GIB
        for s in g.seq_lens for b in g.batches for hq, hkv in g.head_layouts})
    near = [c for c in costs if 7.0 < c < 9.5]
    assert not near, f"configs sit near the 8 GiB boundary: {near} GiB"


# ---------------------------------------------------------------------------
# 2. The A100 must actually restore what the plan promises.
# ---------------------------------------------------------------------------

def test_the_budgets_scale_with_the_device(as_a100):
    assert gates.device_memory_bytes("cuda") == A100_BYTES
    assert gates.exact_oracle_budget_bytes("cuda") / GIB == pytest.approx(29.6, abs=0.5)
    assert gates.cross_backend_budget_bytes("cuda") / GIB == pytest.approx(44.0, abs=0.5)


@pytest.mark.parametrize("seq_len,batch", [(2048, 16), (4096, 4), (8192, 1)])
def test_the_three_promised_oracle_classes_are_restored(as_a100, seq_len, batch):
    """docs/a100_session_plan.md names exactly these three. If the arithmetic
    is wrong, the A100 session's headline gain does not happen and nobody
    finds out until the results come back unchanged."""
    c = _cfg(seq_len, batch)
    assert gates.oracle_bytes(c) / GIB == pytest.approx(16.0, abs=0.1)
    assert not gates.exact_oracle_fits(c, device="cpu"), "should not fit on an L4"
    assert gates.exact_oracle_fits(c, device="cuda"), "should fit on an A100"


def test_cross_backend_at_32768_batch16_stops_being_declined(as_a100):
    c = _cfg(32768, 16)
    assert gates.cross_backend_bytes(c) / GIB == pytest.approx(21.5, abs=0.2)
    assert not gates.cross_backend_fits(c, device="cpu")
    assert gates.cross_backend_fits(c, device="cuda")


def test_16384_batch1_still_does_not_fit_the_oracle(as_a100):
    """The prediction that must NOT come true.

    64 GiB against a 29.6 GiB budget on an 80 GiB card. (Reported as "68.7
    GiB" in an earlier session plan -- that was 68.7 GB, decimal, mislabelled
    as binary. The conclusion is unaffected: it did not fit either way.)
    """
    c = _cfg(16384, 1)
    assert gates.oracle_bytes(c) / GIB == pytest.approx(64.0, abs=0.5)
    assert gates.oracle_bytes(c) / 1e9 == pytest.approx(68.7, abs=0.5)
    assert not gates.exact_oracle_fits(c, device="cuda")


# ---------------------------------------------------------------------------
# 3. The budget travels on the row.
# ---------------------------------------------------------------------------

def test_a_declined_row_records_the_budget_that_declined_it(as_a100):
    from attnbench.backends.base import AttentionBackend, Capability

    class _B(AttentionBackend):
        def __init__(self, n):
            self.capability = Capability(name=n, family="dense_exact",
                                         min_compute_capability=(0, 0),
                                         dtypes=("bfloat16", "float32"))
        @property
        def name(self):
            return self.capability.name
        def claims_support(self, cfg):
            return True, ""
        def forward(self, q, k, v, cfg, mask=None):
            return v

    # Big enough to be declined even at the A100 budget.
    c = _cfg(32768, 16, heads=128)
    r = gates.check_cross_backend(_B("x"), c, [_B("a"), _B("b")], device="cuda")
    assert not r.passed and "INFEASIBLE" in r.detail
    assert r.memory_budget_bytes == gates.cross_backend_budget_bytes("cuda"), (
        "the row must name the budget that produced the verdict")
    assert "44." in r.detail or "GiB budget" in r.detail


def test_the_detail_names_the_fraction_and_the_device(as_a100):
    """A bare byte count is not enough to reproduce the decision on a card
    you do not have in front of you."""
    from attnbench.backends.base import AttentionBackend, Capability

    class _B(AttentionBackend):
        def __init__(self, n):
            self.capability = Capability(name=n, family="dense_exact",
                                         min_compute_capability=(0, 0),
                                         dtypes=("bfloat16", "float32"))
        @property
        def name(self):
            return self.capability.name
        def claims_support(self, cfg):
            return True, ""
        def forward(self, q, k, v, cfg, mask=None):
            return v

    r = gates.check_cross_backend(_B("x"), _cfg(32768, 16, heads=128),
                                  [_B("a"), _B("b")], device="cuda")
    assert "%" in r.detail, "the fraction should be visible"
    assert "device" in r.detail


def test_the_budget_column_exists_on_every_correctness_row():
    r = gates.CorrectnessResult("b", "k", True, "exact")
    assert "memory_budget_bytes" in r.to_dict()


def test_a_cpu_run_falls_back_to_the_card_the_data_came_from():
    """Every existing result was measured on an L4. A CPU test run that graded
    cells differently would stop describing the hardware the dataset came
    from."""
    assert gates.device_memory_bytes("cpu") == L4_BYTES
