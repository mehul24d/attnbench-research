"""A ratio is only as precise as its denominator.

Three places in this codebase divide by a quantity that can be arbitrarily
small, and until 2026-09-06 only one of them had a floor. The two that did not
were found by accident -- `max_rel_err` reaching 3.3e+04 on real A100 rows
while `max_abs_err` stayed at 1.3e-02, and the canary's own
`CANARY_MIN_LATENCY_MS` note pointing at the same shape. This file exists so
the third, fourth and fifth are found by a test instead.

The distinction the fixes turn on: a **divide-by-zero guard** (`clamp_min(1e-8)`,
`max(..., 1e-12)`) stops the arithmetic from blowing up and says nothing about
whether the result means anything. A **resolution floor** excludes the cases
the instrument cannot resolve. They look identical in the source and do
opposite things to the reported number.
"""

from __future__ import annotations

import pytest
import torch

from attnbench.analysis.diagnostic_agreement import AGREEMENT_FLOOR, _rel
from attnbench.config import AttnConfig
from attnbench.gates import TOL, check_correctness, resolvable_rel_err


# --- max_rel_err on a masked tensor: the real defect ------------------------

def _errs(expected_values, abs_errs):
    return (torch.tensor(abs_errs, dtype=torch.float64),
            torch.tensor(expected_values, dtype=torch.float64))


BF16_ATOL = TOL["bfloat16"]["atol"]      # 2e-2


def test_rel_err_is_not_computed_against_a_denominator_of_zero():
    """The 2026-09-05 defect, in miniature. A block-sparse config has most of
    `expected` at zero by construction, and a real bf16 absolute error divided
    by the old 1e-8 clamp produced 3.3e+04 on live A100 rows while
    max_abs_err stayed at 1.3e-02."""
    abs_err, expected = _errs([1.0, 0.0, 0.0], [1e-3, 1e-3, 1e-3])
    max_rel, n_res, n_tot = resolvable_rel_err(abs_err, expected, BF16_ATOL)
    assert (n_res, n_tot) == (1, 3)
    assert max_rel == pytest.approx(1e-3)

    clamped = float((abs_err / expected.abs().clamp_min(1e-8)).max())
    assert clamped == pytest.approx(1e5), "the old behaviour, for contrast"


def test_sub_floor_elements_are_excluded_not_clamped():
    """Clamping reports a ratio against a denominator that was never
    measured. Excluding reports how many elements the answer rests on."""
    abs_err, expected = _errs([1.0, BF16_ATOL / 2], [0.5, 0.5])
    max_rel, n_res, n_tot = resolvable_rel_err(abs_err, expected, BF16_ATOL)
    assert (n_res, n_tot) == (1, 2)
    assert max_rel == pytest.approx(0.5)


def test_nothing_above_the_floor_gives_none_not_zero():
    """None means 'no element could resolve this'. 0.0 would read as perfect
    agreement -- the strongest possible claim, from an absence of data."""
    abs_err, expected = _errs([1e-9, 1e-9], [1e-12, 1e-12])
    max_rel, n_res, n_tot = resolvable_rel_err(abs_err, expected, BF16_ATOL)
    assert max_rel is None
    assert (n_res, n_tot) == (0, 2)


def test_the_floor_is_the_tolerance_not_a_magic_epsilon():
    """atol is not an arbitrary choice: below it the pass/fail bound
    `atol + rtol*|expected|` is dominated by atol, so relative error decides
    nothing there. 1e-8 was a divide-by-zero guard wearing a floor's clothes."""
    for dtype, tol in TOL.items():
        assert tol["atol"] > 1e-8, dtype


# --- end to end, through a real oracle --------------------------------------

class _Wrong:
    """A backend whose output is the oracle's plus a constant."""

    def __init__(self, offset: float):
        self.offset = offset
        self.name = "wrong"

    @staticmethod
    def claims_support(cfg):
        return True, ""

    def make_inputs(self, cfg, device="cpu", seed=0):
        from attnbench.backends.impls import NaiveAttention
        return NaiveAttention().make_inputs(cfg, device=device, seed=seed)

    def forward(self, q, k, v, cfg, mask=None):
        from attnbench.backends.impls import NaiveAttention
        return NaiveAttention().forward(q, k, v, cfg, mask=mask) + self.offset


def _cfg(**kw):
    # head_dim=64: the oracle declines anything smaller, and a declined
    # reference returns a verdict about the reference, not the backend.
    base = dict(seq_len=8, batch=1, n_heads_q=2, n_heads_kv=2, head_dim=64,
                dtype="float32")
    base.update(kw)
    return AttnConfig(**base)


def test_a_failing_verdict_explains_itself():
    """It used to carry a number and no account of it: no tolerance, no count
    of offending elements, no location. Same shape as a flag nobody reads --
    the field exists, so an auditor concludes the reason is recorded."""
    r = check_correctness(_Wrong(offset=1.0), _cfg(), device="cpu")
    assert not r.passed
    assert r.detail, "a failing verdict with an empty detail"
    for token in ("exceed", "atol=", "rtol=", "worst at flat index"):
        assert token in r.detail, f"detail does not mention {token!r}"


def test_a_passing_verdict_records_how_much_was_resolvable():
    r = check_correctness(_Wrong(offset=0.0), _cfg(), device="cpu")
    assert r.passed
    assert "above the rel_err floor" in r.detail
    assert r.rel_err_floor == TOL["float32"]["atol"]


def test_the_floor_travels_on_the_row():
    r = check_correctness(_Wrong(offset=0.0), _cfg(), device="cpu")
    assert r.rel_err_floor is not None
    assert r.rel_err_floor > 1e-8, "1e-8 is a divide-by-zero guard, not a floor"


# --- structural rows must not claim zero error ------------------------------

def test_a_structural_pass_reports_null_error_not_zero():
    """GLA contributes 30 structural rows to the A100 dataset. Under
    max_abs_err=0.0 a plot of error by backend would have shown GLA as the
    most accurate kernel in the study, having never been compared to
    anything."""
    from attnbench.gates import check_structural

    class _Linear:
        name = "gla"

        @staticmethod
        def claims_support(cfg):
            return True, ""

        def make_inputs(self, cfg, device="cpu", seed=0):
            g = torch.Generator().manual_seed(seed)
            shape = (cfg.batch, cfg.n_heads_q, cfg.seq_len, cfg.head_dim)
            q = torch.randn(shape, generator=g)
            return q, q.clone(), q.clone()

        def forward(self, q, k, v, cfg, mask=None):
            return torch.cumsum(q, dim=-2) * 0.01

    r = check_structural(_Linear(), _cfg(), device="cpu")
    assert r.check_kind == "structural"
    assert r.max_abs_err is None, "0.0 reads as perfect agreement"
    assert r.max_rel_err is None
    assert "no comparison" in r.detail


# --- the diagnostic-agreement ratio ------------------------------------------

def test_two_effectively_zero_errors_agree():
    """1e-10 against 3e-10 scored as a 67% disagreement under the old 1e-12
    divide-by-zero guard. Both numbers are zero to every tolerance in TOL."""
    assert _rel(1e-10, 3e-10) == 0.0
    assert _rel(0.0, 0.0) == 0.0


def test_a_real_disagreement_still_registers():
    """The floor must not be able to hide one. It sits three orders below the
    smallest tolerance in gates.TOL."""
    assert _rel(0.008, 0.016) == pytest.approx(0.5)
    assert AGREEMENT_FLOOR * 100 <= min(t["atol"] for t in TOL.values())


def test_the_floor_is_below_every_tolerance_that_could_matter():
    assert AGREEMENT_FLOOR < TOL["float32"]["atol"]


# --- the general sweep -------------------------------------------------------

def test_no_division_in_the_analysis_layer_uses_a_bare_epsilon_guard():
    """Static backstop for the pattern itself, not for the three known sites.

    `clamp_min(1e-8)` and `max(..., 1e-12)` are how this defect looks in
    source: a magic epsilon whose only job is to keep the division finite. Any
    new one has to justify itself as a resolution floor by name.
    """
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    offenders = []
    pattern = re.compile(r"(clamp_min\(\s*1e-(?:8|9|1\d)|max\([^)]*1e-(?:8|9|1\d)\s*\))")
    for f in sorted((repo / "attnbench").rglob("*.py")):
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{f.relative_to(repo)}:{i}: {line.strip()}")
    assert not offenders, (
        "bare epsilon guard(s) on a division -- these keep the arithmetic "
        "finite and say nothing about whether the result resolves:\n  "
        + "\n  ".join(offenders))
