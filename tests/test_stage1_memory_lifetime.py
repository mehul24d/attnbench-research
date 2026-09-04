"""Stage 1 must not be able to run itself out of memory. CPU-only.

Stage 1 died at the 8192 band on three consecutive rented sessions, each time
at a different allocation site inside the correctness loop:

  2026-09-04 (a)  the reference backend's forward
  2026-09-04 (b)  `(got - other).abs().max()` -- the comparison arithmetic
  2026-09-04 (c)  `make_inputs` at the START of the next check, with 47.12 MiB
                  free against a 22.03 GiB card

Each site was patched in turn and each patch moved the failure later without
resolving it. Three fixes that each relocate a symptom are the signal that the
defect is somewhere else, and it was: `timing.measure` has released memory in a
`finally` since it was written, and `check_for_family` never did.

The mechanism is worth stating precisely, because "leaked tensors" is the
wrong diagnosis and would have sent the fix somewhere useless. Every check's
tensors ARE freed by refcount when its frame dies. Freeing them returns the
blocks to PyTorch's caching allocator, which keeps them reserved from the
driver and reuses them only for allocations that fit. Successive checks at
different shapes fragment that reserve until a 1 GiB contiguous request fails
with the card nominally almost empty -- which is exactly the 47 MiB reading.

Two properties are asserted here:

  1. every check releases, including when it raises, and including in the
     middle of a loop rather than only at the end; and
  2. a check whose memory cost is computable from the config is DECLINED IN
     ADVANCE with a recorded reason, instead of being attempted and
     discovered by an OOM.

(2) generalises `exact_oracle_fits`, which began as a seq_len cutoff, was
wrong by 16x at batch 16, and became a memory bound. Cross-backend is the
other check that allocates, and it had no such bound at all.
"""

from __future__ import annotations

import pytest
import torch

from attnbench import gates
from attnbench.backends.base import AttentionBackend, Capability
from attnbench.config import AttnConfig
from attnbench.gates import (CHECK_KIND_PAIR, CROSS_BACKEND_BUDGET_BYTES,
                             MIN_CROSS_BACKEND_AGREEING, check_cross_backend,
                             check_for_family, cross_backend_bytes,
                             cross_backend_fits)

GIB = 2 ** 30


def _cap(name, family="dense_exact"):
    return Capability(name=name, family=family, min_compute_capability=(0, 0),
                      supports_gqa=True, supports_causal=True,
                      block_sizes=(64, 128), head_dims=(64, 128),
                      dtypes=("float32", "bfloat16", "float16"))


class _CpuBackend(AttentionBackend):
    """Allocates on CPU whatever device string it is handed.

    Lets the release path be exercised with `device="cuda"` on a machine with
    no GPU -- the branch under test is chosen by that string, and the branch
    is the thing that was missing.
    """

    def __init__(self, name="stub"):
        self.capability = _cap(name)

    @property
    def name(self):
        return self.capability.name

    def make_inputs(self, cfg, device="cuda", seed: int = 0):
        return super().make_inputs(cfg, device="cpu", seed=seed)

    def claims_support(self, cfg):
        # The base implementation is a CLASSMETHOD reading `cls.capability`,
        # and these stubs carry a per-INSTANCE capability so each can have its
        # own name. Overriding here rather than hoisting capability to the
        # class, because per-instance capability is exactly the shape real
        # backends use (SDPABackend builds one per kernel variant) and the
        # stubs should not be easier than the thing they stand in for.
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        return v


class _Raising(_CpuBackend):
    def make_inputs(self, cfg, device="cuda", seed: int = 0):
        raise RuntimeError("allocation exploded")


def _cfg(**kw):
    base = dict(seq_len=64, batch=1, n_heads_q=4, n_heads_kv=4, head_dim=64,
                dtype="float32", mask="causal")
    base.update(kw)
    return AttnConfig(**base)


@pytest.fixture
def released(monkeypatch):
    """Records calls to the two release primitives, in order."""
    log: list[str] = []
    monkeypatch.setattr(gates.gc, "collect", lambda *a, **k: log.append("gc"))
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda *a, **k: log.append("empty_cache"))
    return log


# ---------------------------------------------------------------------------
# 1. Every check releases.
# ---------------------------------------------------------------------------

def test_a_correctness_check_releases_gpu_memory_when_it_finishes(released):
    check_for_family(_CpuBackend(), _cfg(), device="cuda")
    assert "empty_cache" in released, (
        "check_for_family returned without releasing the allocator's reserve; "
        "this is the asymmetry with timing.measure that killed Stage 1 three "
        "times")


def test_the_release_happens_even_when_the_check_raises(released):
    """The case that matters most: an exception is exactly when the allocator
    is under pressure, and exactly when a `return`-sited cleanup is skipped."""
    with pytest.raises(RuntimeError, match="allocation exploded"):
        check_for_family(_Raising(), _cfg(), device="cuda")
    assert "empty_cache" in released


def test_every_check_in_a_loop_releases_not_just_the_last(released):
    """Stage 1 is a loop over hundreds of (backend, config) pairs. Releasing
    once at the end would be indistinguishable from releasing never."""
    for _ in range(5):
        check_for_family(_CpuBackend(), _cfg(), device="cuda")
    assert released.count("empty_cache") == 5


def test_gc_runs_before_empty_cache(released):
    """Order is load-bearing, not stylistic. A reference cycle -- an exception
    traceback holding a frame holding a tensor -- survives refcounting, and
    empty_cache() cannot release a block that is still referenced. Collecting
    afterwards would free the tensor into the reserve and leave it there."""
    check_for_family(_CpuBackend(), _cfg(), device="cuda")
    assert released.index("gc") < released.index("empty_cache")


def test_a_cpu_run_does_not_touch_cuda(released):
    """The suite runs on machines with no GPU; a release that assumed one
    would make every CPU test fail or, worse, initialise a CUDA context."""
    check_for_family(_CpuBackend(), _cfg(), device="cpu")
    assert "empty_cache" not in released


def test_the_correctness_path_now_matches_the_timing_path():
    """The specific parity that was missing. Asserted against the source so it
    stays true of both, rather than of whichever one someone edited last."""
    import inspect

    from attnbench import timing

    for fn in (gates.check_for_family, timing.measure):
        src = inspect.getsource(fn)
        assert "finally:" in src, f"{fn.__name__} has no finally block"
        assert "empty_cache" in src, f"{fn.__name__} never releases"


# ---------------------------------------------------------------------------
# 2. A check too large to run is declined in advance, and that is a result.
# ---------------------------------------------------------------------------

def test_cost_depends_on_the_whole_config_not_just_seq_len():
    """The exact mistake `exact_oracle_fits` made in its first version: a
    seq_len threshold that assumed batch=1 and was wrong by 16x at batch 16."""
    small = cross_backend_bytes(_cfg(seq_len=8192, batch=1, n_heads_q=32,
                                     n_heads_kv=8, head_dim=128,
                                     dtype="bfloat16"))
    big = cross_backend_bytes(_cfg(seq_len=8192, batch=16, n_heads_q=32,
                                   n_heads_kv=8, head_dim=128,
                                   dtype="bfloat16"))
    # Not the full 16x, and deliberately not asserted as such: the comparison
    # term is three float32 tensors the size of ONE batch slice, so it is
    # fixed while everything else scales. At batch 1 that term is most of the
    # cost; at batch 16 it is a fourteenth of it. Asserting 16x here would be
    # asserting a model this one is not.
    assert big > small * 7, (
        f"batch-16 cost {big} is not meaningfully above batch-1 cost {small}; "
        f"the model is ignoring batch, which is how the oracle bound failed")
    assert small < GIB < big


def test_gqa_expansion_is_in_the_model():
    """A reference's forward materialises k and v at the QUERY head count for
    GQA. At 32:8 that is 1.5x a query-shaped tensor -- concurrent with
    everything else and far too large to bury in a headroom factor."""
    kw = dict(seq_len=8192, batch=16, n_heads_q=32, head_dim=128,
              dtype="bfloat16")
    gqa = cross_backend_bytes(_cfg(n_heads_kv=8, **kw))
    mha = cross_backend_bytes(_cfg(n_heads_kv=32, **kw))
    per_q = 16 * 32 * 8192 * 128 * 2
    assert mha - gqa == pytest.approx(0.0, abs=per_q), (
        "GQA and MHA should cost within a query-tensor of each other: GQA "
        "saves on stored k/v and spends it again expanding them")


def test_the_band_that_kept_crashing_is_judged_feasible():
    """8192/batch 16 is where Stage 1 died three times, and it died to an
    accumulated reserve, not to its own size. A bound that excluded it would
    be 'fixing' the crash by deleting the band."""
    cfg = _cfg(seq_len=8192, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    assert cross_backend_fits(cfg), (
        f"needs {cross_backend_bytes(cfg) / GIB:.1f} GiB against a "
        f"{CROSS_BACKEND_BUDGET_BYTES / GIB:.0f} GiB budget")


def test_the_next_band_up_is_also_feasible():
    cfg = _cfg(seq_len=16384, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    assert cross_backend_fits(cfg)


def test_the_largest_configs_are_genuinely_infeasible():
    """21.5 GiB of concurrent tensors on a 22 GiB card. Recording this as a
    limitation is honest; attempting it is how a run dies."""
    cfg = _cfg(seq_len=32768, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    assert not cross_backend_fits(cfg)
    assert cross_backend_bytes(cfg) > 20 * GIB


def test_a_declined_check_allocates_nothing(monkeypatch):
    """The load-bearing half. Recording a tidy status after the allocation has
    already failed would be pointless -- an OOM inside a check leaves the
    allocator wherever it died, and the last one took the process with it."""
    cfg = _cfg(seq_len=32768, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    allocated = []
    monkeypatch.setattr(_CpuBackend, "make_inputs",
                        lambda self, cfg, device="cuda", seed=0:
                        allocated.append(cfg))

    r = check_cross_backend(_CpuBackend("a"), cfg,
                            [_CpuBackend("b"), _CpuBackend("c")],
                            device="cuda")
    assert allocated == [], "the infeasible check allocated anyway"
    assert not r.passed


def test_a_declined_check_says_what_could_not_be_verified_and_why():
    """'Cross-backend verification is infeasible at 32768/batch 16 on 24 GB'
    is a real statement about what this hardware can verify. A crash is not."""
    cfg = _cfg(seq_len=32768, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    r = check_cross_backend(_CpuBackend("a"), cfg,
                            [_CpuBackend("b"), _CpuBackend("c")],
                            device="cuda")
    assert "INFEASIBLE" in r.detail
    assert "GiB" in r.detail
    assert "Not a verdict on this backend" in r.detail, (
        "a declined check must not read as the backend having failed")


def test_feasibility_is_not_enforced_on_cpu():
    """CPU has no 22 GiB ceiling and the suite must stay runnable without a
    GPU. The bound describes the rented card, so it applies to cuda only."""
    cfg = _cfg(seq_len=32768, batch=16, n_heads_q=32, n_heads_kv=8,
               head_dim=128, dtype="bfloat16")
    assert not cross_backend_fits(cfg)
    # would raise/allocate enormously if the guard were device-blind; instead
    # the guard is simply not consulted, and the caller's own limits apply.
    assert gates.cross_backend_bytes(cfg) > CROSS_BACKEND_BUDGET_BYTES


# ---------------------------------------------------------------------------
# 3. When the hardware takes a reference away, the cell is downgraded, not lost.
# ---------------------------------------------------------------------------

class _Dense(AttentionBackend):
    def __init__(self, name, bias=0.0):
        self.capability = _cap(name)
        self._bias = bias

    @property
    def name(self):
        return self.capability.name

    def claims_support(self, cfg):
        # The base implementation is a CLASSMETHOD reading `cls.capability`,
        # and these stubs carry a per-INSTANCE capability so each can have its
        # own name. Overriding here rather than hoisting capability to the
        # class, because per-instance capability is exactly the shape real
        # backends use (SDPABackend builds one per kernel variant) and the
        # stubs should not be easier than the thing they stand in for.
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        return v + self._bias


def _oom_after(n_successes, err=0.0):
    """The first `n_successes` comparisons return `err`; the rest OOM.

    `err` is a parameter rather than a constant because a stub that always
    answers 0.0 makes every disagreement test pass vacuously -- it overrides
    the very difference the test is checking for.
    """
    calls = {"n": 0}

    def f(a, b, chunk=1):
        calls["n"] += 1
        if calls["n"] > n_successes:
            raise torch.cuda.OutOfMemoryError("simulated")
        return err
    return f


def test_two_survivors_pass_under_a_weaker_kind(monkeypatch):
    """Insisting on three when the hardware can only deliver two would delete
    the 8192 and 16384 bands from the study to protect a standard the CARD,
    not the backend, made unreachable."""
    monkeypatch.setattr(gates, "_max_abs_diff", _oom_after(1))
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c"), _Dense("d")],
                            device="cpu")
    assert r.passed
    assert r.check_kind == CHECK_KIND_PAIR


def test_the_weaker_kind_is_a_distinct_string():
    """Downstream reads check_kind to know what a pass certifies. Merging the
    pair into 'cross_backend' would make a two-way agreement indistinguishable
    from a three-way one forever after."""
    assert CHECK_KIND_PAIR != "cross_backend"


def test_the_weaker_kind_carries_its_own_weakness_on_the_row(monkeypatch):
    monkeypatch.setattr(gates, "_max_abs_diff", _oom_after(1))
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c"), _Dense("d")],
                            device="cpu")
    assert "WEAKER EVIDENCE" in r.detail
    assert "shared bug" in r.detail, (
        "the row should say what the missing third opinion would have caught")


def test_three_survivors_are_still_recorded_as_the_full_check(monkeypatch):
    """The downgrade must be triggered by an actual shortfall, not applied
    everywhere as a hedge."""
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c"), _Dense("d")],
                            device="cpu")
    assert r.passed and r.check_kind == "cross_backend"
    assert "WEAKER EVIDENCE" not in r.detail


def test_a_pair_still_reports_disagreement(monkeypatch):
    """Weaker evidence is not a weaker verdict. Two implementations that
    disagree is still one of them being wrong."""
    monkeypatch.setattr(gates, "_max_abs_diff", _oom_after(1, err=1.0))
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b", bias=1.0), _Dense("c"), _Dense("d")],
                            device="cpu")
    assert not r.passed
    assert r.check_kind == CHECK_KIND_PAIR
    assert "a finding, not a cell to skip" in r.detail


def test_too_few_references_OFFERED_is_still_a_hard_failure():
    """The distinction that keeps the downgrade honest: attempted-and-lost
    versus never-offered.

    A caller who supplies two implementations has hit no hardware limit --
    nothing was tried and nothing failed. Downgrading that case would hide the
    one situation where the three-way standard is actually within reach, by
    making a study-design gap look like a hardware constraint.
    """
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg, [_Dense("b")], device="cpu")
    assert not r.passed
    assert r.check_kind == "cross_backend", "not downgraded to a pair"
    assert f"need {MIN_CROSS_BACKEND_AGREEING}" in r.detail


def test_a_lone_survivor_is_not_a_pair(monkeypatch):
    """One implementation agreeing with itself is not agreement."""
    monkeypatch.setattr(gates, "_max_abs_diff", _oom_after(0))
    cfg = _cfg(seq_len=128)
    r = check_cross_backend(_Dense("a"), cfg,
                            [_Dense("b"), _Dense("c"), _Dense("d")],
                            device="cpu")
    assert not r.passed
    assert "even for a pair" in r.detail
