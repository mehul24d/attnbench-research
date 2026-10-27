"""T2.16: dense-baseline selection (control K2; plan Section 5.5).

The real correctness gate (`gates.check_for_family`) runs here on CPU against
fake kernels whose cost is scripted on a fake clock, so the protocol is
exercised end to end: a kernel that is faster and wrong must lose, a kernel
known to fault the device must never be launched, and a record must fail to
load when it does not follow from its own samples.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from attnbench import dense_selection as ds
from attnbench import gates
from attnbench.backends.base import FAULT_REASON_PREFIX, AttentionBackend, Capability
from attnbench.backends.impls import NaiveAttention
from attnbench.config import AttnConfig

from _gt_fakes import FakeClock

CARD, SEQ, BATCH, FLOOR = "A100", 32, 1, 0.5
PLAN = Path(__file__).resolve().parent.parent / "docs" / "RESEARCH_PLAN.md"
CELL = ds.Cell(CARD, SEQ, BATCH)


def _cfg(seq=SEQ, batch=BATCH):
    return AttnConfig(seq_len=seq, batch=batch, n_heads_q=4, n_heads_kv=2, head_dim=32,
                      dtype="float32", mask="causal")


class Fake(AttentionBackend):
    """A kernel that computes `scale` x the right answer in `ms` milliseconds
    of the fake clock. `ms` may be a function of the call number."""

    def __init__(self, name, ms, clock, *, scale=1.0, fault_above=None, raises_on=None,
                 log=None):
        self.capability = Capability(name=name, family="dense_exact",
                                     min_compute_capability=(0, 0),
                                     head_dims=(32,), dtypes=("float32",))
        self.ms, self.clock, self.scale = ms, clock, scale
        self.fault_above, self.raises_on = fault_above, raises_on
        self.calls, self.log = 0, log if log is not None else []

    def claims_support(self, cfg):
        if self.fault_above is not None and cfg.seq_len > self.fault_above:
            return False, f"{FAULT_REASON_PREFIX}fake fault"
        if cfg.head_dim not in self.capability.head_dims:
            return False, f"head dim {cfg.head_dim} unsupported"
        return True, ""

    def forward(self, q, k, v, cfg, mask=None):
        assert self.fault_above is None or cfg.seq_len <= self.fault_above, \
            f"{self.name} was launched above its known fault length"
        self.calls += 1
        if self.raises_on is not None and self.calls == self.raises_on:
            raise RuntimeError("device-side assert (simulated)")
        self.log.append(self.name)
        cost = self.ms(self.calls) if callable(self.ms) else self.ms
        self.clock.advance_ms(cost)
        return NaiveAttention().forward(q, k, v, cfg, mask=mask) * self.scale


def _candidates(clock, **overrides):
    """The planned set for the card, every kernel right, with scripted costs."""
    base = {ds.SDPA_FLASH: 10.0, ds.SDPA_CUDNN: 8.0, ds.SDPA_EFFICIENT: 12.0, ds.FA2: 9.0}
    out = {}
    for name in ds.planned_candidates(CARD):
        kw = overrides.get(name, {})
        out[name] = Fake(name, kw.pop("ms", base[name]), clock, **kw)
    return out


def _select(candidates=None, clock=None, cfg=None, rounds=3, **kw):
    clock = clock or FakeClock()
    candidates = candidates if candidates is not None else _candidates(clock)
    kw.setdefault("device", "cpu")
    return ds.select_dense_baseline(CELL, cfg or _cfg(), candidates, rounds=rounds,
                                    resolution_floor_ms=FLOOR, clock=clock,
                                    wall_clock=lambda: 1_000.0 + clock.t, **kw)


def _status(choice, name):
    return next(c for c in choice.candidates if c.name == name)


# --- the protocol --------------------------------------------------------------------

def test_the_fastest_correct_candidate_is_chosen_with_its_evidence():
    ch = _select()
    assert ch.chosen == ds.SDPA_CUDNN and ch.chosen_median_ms == pytest.approx(8.0)
    assert ch.runner_up == ds.FA2 and ch.margin_ms == pytest.approx(1.0)
    assert ch.complete and ch.missing == []
    for c in ch.candidates:
        assert c.status == ds.CORRECT and len(c.warmup_ms) == 3
        assert c.check_kind == "exact"          # which check, named per candidate
        assert c.tolerance == gates.TOL["float32"]


def test_a_faster_kernel_that_is_wrong_is_never_chosen():
    clock = FakeClock()
    cands = _candidates(clock, **{ds.SDPA_CUDNN: dict(scale=1.05)})
    ch = _select(cands, clock)
    wrong = _status(ch, ds.SDPA_CUDNN)
    assert wrong.status == ds.INCORRECT and wrong.median_ms is None
    assert wrong.max_abs_err and wrong.max_abs_err > gates.TOL["float32"]["atol"]
    assert ch.chosen == ds.FA2                  # the fastest *correct* one
    assert wrong.warmup_ms == []                # a wrong kernel is never timed


def test_a_kernel_known_to_fault_the_device_is_never_launched():
    clock = FakeClock()
    cands = _candidates(clock, **{ds.SDPA_CUDNN: dict(fault_above=16)})
    ch = _select(cands, clock)
    s = _status(ch, ds.SDPA_CUDNN)
    assert s.status == ds.DECLINED and s.known_device_fault
    assert cands[ds.SDPA_CUDNN].calls == 0
    assert ch.chosen == ds.FA2 and ch.complete   # a declined fault is a result, not a gap


def test_a_candidate_that_raises_in_the_repeats_is_excluded_and_named():
    clock = FakeClock()
    cands = _candidates(clock, **{ds.SDPA_CUDNN: dict(raises_on=4)})   # 1st timed round
    ch = _select(cands, clock)
    s = _status(ch, ds.SDPA_CUDNN)
    assert s.status == ds.FAILED_REPEATS and "device-side assert" in s.reason
    assert ch.chosen == ds.FA2


def test_an_unavailable_planned_candidate_makes_the_record_incomplete_and_unusable():
    clock = FakeClock()
    cands = _candidates(clock)
    del cands[ds.FA2]
    ch = _select(cands, clock)
    assert not ch.complete and ch.missing == [ds.FA2]
    assert _status(ch, ds.FA2).status == ds.UNAVAILABLE
    with pytest.raises(ds.SelectionError, match="incomplete"):
        ds.check_control(ch, ch.chosen, ch.recorded_at + 1)


def test_the_candidate_set_is_the_protocols():
    assert ds.planned_candidates("NVIDIA L4") == ds.BASE_CANDIDATES
    assert ds.planned_candidates("NVIDIA A100-SXM4-40GB") == ds.BASE_CANDIDATES
    assert ds.planned_candidates("NVIDIA H100 80GB HBM3") == ds.BASE_CANDIDATES + (ds.FA3,)
    clock = FakeClock()
    cands = _candidates(clock)
    cands["fa9"] = Fake("fa9", 1.0, clock)
    with pytest.raises(ds.SelectionError, match="not planned candidates"):
        _select(cands, clock)


def test_the_h100_needs_fa3_and_says_so_when_it_is_absent():
    clock = FakeClock()
    cands = {n: Fake(n, 10.0, clock) for n in ds.BASE_CANDIDATES}
    cell = ds.Cell("H100", SEQ, BATCH)
    ch = ds.select_dense_baseline(cell, _cfg(), cands, rounds=3, resolution_floor_ms=FLOOR,
                                  clock=clock, device="cpu")
    assert ch.missing == [ds.FA3] and not ch.complete


def test_the_plan_names_the_candidates_the_code_plans():
    text = " ".join(PLAN.read_text().split())
    line = text[text.index("Candidates: SDPA flash"):][:200]
    for phrase in ("SDPA flash", "SDPA cuDNN", "SDPA memory-efficient", "FA2",
                   "FA3 on the H100"):
        assert phrase in line
    assert len(ds.planned_candidates("H100")) == 5 and len(ds.planned_candidates("L4")) == 4


def test_cross_backend_fallback_is_named_per_cell(monkeypatch):
    monkeypatch.setattr(gates, "exact_oracle_fits", lambda *a, **k: False)
    ch = _select()
    kinds = {c.check_kind for c in ch.candidates}
    assert kinds <= {"cross_backend", "cross_backend_pair"} and kinds
    assert all(c.status == ds.CORRECT for c in ch.candidates)


def test_the_cross_backend_fallback_is_what_a_wrong_peer_cannot_pass(monkeypatch):
    monkeypatch.setattr(gates, "exact_oracle_fits", lambda *a, **k: False)
    clock = FakeClock()
    cands = _candidates(clock, **{ds.SDPA_CUDNN: dict(scale=1.05)})
    ch = _select(cands, clock)
    assert _status(ch, ds.SDPA_CUDNN).status == ds.INCORRECT
    assert ch.chosen != ds.SDPA_CUDNN


def test_the_choice_is_independent_of_the_order_candidates_are_checked(monkeypatch):
    seen = []
    real = gates.check_for_family

    def spy(backend, cfg, **kw):
        seen.append((backend.name, sorted(r.name for r in kw["references"])))
        return real(backend, cfg, **kw)
    clock = FakeClock()
    _select(_candidates(clock, **{ds.SDPA_FLASH: dict(scale=1.05)}), clock, check=spy)
    names = ds.planned_candidates(CARD)
    for n, refs in seen:                     # every candidate sees all the others, wrong or not
        assert refs == sorted(set(names) - {n})


def test_a_tie_is_recorded_not_resolved_by_a_new_rule():
    clock = FakeClock()
    cands = _candidates(clock, **{ds.SDPA_CUDNN: dict(ms=8.0), ds.FA2: dict(ms=8.3)})
    ch = _select(cands, clock)
    assert ch.chosen == ds.SDPA_CUDNN
    assert ch.within_floor_of_chosen == [ds.FA2]       # 0.3 ms <= the 0.5 ms floor
    equal = _candidates(clock, **{ds.SDPA_FLASH: dict(ms=8.0), ds.SDPA_CUDNN: dict(ms=8.0)})
    assert _select(equal, clock).chosen == ds.SDPA_FLASH     # planned order, not a new rule


def test_the_warmup_rounds_are_interleaved_and_rotate():
    clock = FakeClock()
    log = []
    cands = {n: Fake(n, 10.0, clock, log=log) for n in ds.planned_candidates(CARD)}
    _select(cands, clock, rounds=4)
    timed = log[len(log) - 4 * 4:]            # the last 4 rounds of 4 kernels each
    rounds = [timed[i:i + 4] for i in range(0, 16, 4)]
    assert len({r[0] for r in rounds}) == 4          # a different kernel leads each round
    assert all(sorted(r) == sorted(ds.planned_candidates(CARD)) for r in rounds)


def test_the_selection_refuses_a_cell_that_is_not_its_config_and_bad_parameters():
    clock = FakeClock()
    with pytest.raises(ds.SelectionError, match="is not the cell"):
        _select(clock=clock, cfg=_cfg(seq=64))
    with pytest.raises(ValueError, match="rounds"):
        _select(clock=clock, rounds=1)
    with pytest.raises(ValueError, match="floor"):
        ds.select_dense_baseline(CELL, _cfg(), _candidates(clock), rounds=3,
                                 resolution_floor_ms=0.0, device="cpu")


# --- the repository's own backends -----------------------------------------------------

def test_the_registry_candidates_name_every_planned_kernel_and_the_cudnn_fault():
    a100 = ds.registry_candidates("A100")
    assert list(a100) == list(ds.BASE_CANDIDATES)
    ok, why = a100[ds.SDPA_CUDNN].claims_support(_cfg(seq=16384).__class__(
        seq_len=16384, batch=1, n_heads_q=12, n_heads_kv=2, head_dim=128, dtype="bfloat16"))
    assert not ok and why.startswith(FAULT_REASON_PREFIX)
    assert list(ds.registry_candidates("H100")) == list(ds.planned_candidates("H100"))
    assert ds.registry_candidates("H100")[ds.FA3] is None     # no FA3 backend exists yet


# --- the record ------------------------------------------------------------------------

def _written(tmp_path, **kw):
    ch = _select(**kw)
    return ch, ds.write_choice(ch, tmp_path)


def _rehash(path):
    data = path.read_bytes()
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(data).hexdigest()}  {path.name}\n")


def _edit(path, fn):
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d, sort_keys=True, indent=1) + "\n")


def test_a_record_round_trips_and_is_never_overwritten(tmp_path):
    ch, path = _written(tmp_path)
    back = ds.load_choice(path)
    assert back == ch and back.chosen == ds.SDPA_CUDNN
    with pytest.raises(ds.SelectionError, match="already exists"):
        ds.write_choice(ch, tmp_path)


def test_edited_bytes_are_refused(tmp_path):
    _, path = _written(tmp_path)
    _edit(path, lambda d: d.update(chosen=ds.FA2))
    with pytest.raises(ds.SelectionRecordError, match="sidecar's sha256"):
        ds.load_choice(path)


def test_a_rehashed_edit_of_the_choice_is_refused_because_it_does_not_follow(tmp_path):
    _, path = _written(tmp_path)
    _edit(path, lambda d: d.update(chosen=ds.FA2, chosen_median_ms=9.0))
    _rehash(path)
    with pytest.raises(ds.SelectionRecordError, match="do not follow from the samples"):
        ds.load_choice(path)


def test_a_rehashed_edit_of_a_median_is_refused_because_the_samples_disagree(tmp_path):
    _, path = _written(tmp_path)

    def faster(d):
        c = next(c for c in d["candidates"] if c["name"] == ds.FA2)
        c["median_ms"] = 1.0
    _edit(path, faster)
    _rehash(path)
    with pytest.raises(ds.SelectionRecordError, match="not the median of its samples"):
        ds.load_choice(path)


def test_a_rehashed_record_with_fewer_samples_than_rounds_is_refused(tmp_path):
    """The median is recomputed consistently from the shortened samples, so only
    the count against `rounds` can catch a candidate timed fewer times than
    the record says."""
    _, path = _written(tmp_path)

    def truncate(d):
        c = next(c for c in d["candidates"] if c["name"] == ds.FA2)
        c["warmup_ms"] = c["warmup_ms"][:2]
        c["median_ms"] = sum(c["warmup_ms"]) / 2
    _edit(path, truncate)
    _rehash(path)
    with pytest.raises(ds.SelectionRecordError, match="warm-up samples for 3 rounds"):
        ds.load_choice(path)


def test_a_rehashed_edit_that_hides_an_unavailable_candidate_is_refused(tmp_path):
    clock = FakeClock()
    cands = _candidates(clock)
    del cands[ds.FA2]
    _, path = _written(tmp_path, candidates=cands, clock=clock)
    _edit(path, lambda d: d.update(complete=True, missing=[]))
    _rehash(path)
    with pytest.raises(ds.SelectionRecordError, match="do not follow"):
        ds.load_choice(path)


def test_a_missing_or_malformed_sidecar_is_refused(tmp_path):
    _, path = _written(tmp_path)
    side = path.with_name(path.name + ".sha256")
    side.write_text("nonsense\n")
    with pytest.raises(ds.SelectionRecordError, match="sha256sum format"):
        ds.load_choice(path)
    side.unlink()
    with pytest.raises(ds.SelectionRecordError, match="no sha256 sidecar"):
        ds.load_choice(path)


# --- binding the ground truth to the choice -------------------------------------------

def test_the_control_must_be_the_choice_recorded_before_the_first_block():
    ch = _select()
    ds.check_control(ch, ds.SDPA_CUDNN, ch.recorded_at + 1.0)          # passes
    with pytest.raises(ds.SelectionError, match="recorded choice"):
        ds.check_control(ch, ds.SDPA_FLASH, ch.recorded_at + 1.0)      # the banked baseline
    with pytest.raises(ds.SelectionError, match="not before the first timed block"):
        ds.check_control(ch, ds.SDPA_CUDNN, ch.recorded_at)
    with pytest.raises(ds.SelectionError, match="not before"):
        ds.check_control(ch, ds.SDPA_CUDNN, ch.recorded_at - 1.0)


def test_nothing_correct_means_nothing_to_control_against():
    clock = FakeClock()
    cands = _candidates(clock)
    for c in cands.values():
        c.scale = 1.05
    ch = _select(cands, clock)
    assert ch.chosen is None and ch.complete
    with pytest.raises(ds.SelectionError, match="no candidate was correct"):
        ds.check_control(ch, ds.SDPA_FLASH, ch.recorded_at + 1)


# --- restating banked figures ---------------------------------------------------------

def test_a_measured_restatement_needs_the_chosen_kernel_in_one_session():
    ch = _select()
    r = ds.measured_restatement(ch, session="s1", dense_arm=ds.SDPA_CUDNN, dense_ms=100.0,
                                dense_session="s1", sparse_ms=80.0, sparse_session="s1")
    assert r.kind == ds.MEASURED and r.ratio == pytest.approx(1.25)
    with pytest.raises(ds.SelectionError, match="not the chosen"):
        ds.measured_restatement(ch, session="s1", dense_arm=ds.SDPA_FLASH, dense_ms=100.0,
                                dense_session="s1", sparse_ms=80.0, sparse_session="s1")
    with pytest.raises(ds.SelectionError, match="same-session"):
        ds.measured_restatement(ch, session="s1", dense_arm=ds.SDPA_CUDNN, dense_ms=100.0,
                                dense_session="s0", sparse_ms=80.0, sparse_session="s1")
    clock = FakeClock()
    cands = _candidates(clock)
    del cands[ds.FA2]
    with pytest.raises(ds.SelectionError, match="complete recorded choice"):
        ds.measured_restatement(_select(cands, clock), session="s1", dense_arm=ds.SDPA_CUDNN,
                                dense_ms=1.0, dense_session="s1", sparse_ms=1.0,
                                sparse_session="s1")


def test_an_arithmetic_restatement_is_labelled_and_is_the_banked_method():
    r = ds.arithmetic_restatement(dense_e2e_ms=1000.0, sparse_e2e_ms=800.0, layers=28,
                                  baseline_kernel_ms=2.0, chosen_kernel_ms=1.0)
    assert r.kind == ds.ARITHMETIC != ds.MEASURED
    assert r.ratio == pytest.approx((1000.0 - 28 * 1.0) / 800.0)


SWEEP = Path(__file__).resolve().parent.parent / "results/s7_sweep_hl122/sweep.parquet"
E2E = Path(__file__).resolve().parent.parent / "results/s8_vec_endtoend/vec_endtoend.parquet"


@pytest.mark.skipif(not (SWEEP.exists() and E2E.exists()), reason="banked parquets not present")
def test_the_arithmetic_restatement_reproduces_the_banked_caveat():
    """16384 against fa2: the caveat's estimated 1.072x / 1.180x / 1.260x
    (docs/limitations.md, A100 baseline caveat), recomputed from the parquets
    that test_a100_baseline_caveat.py reads, by this function."""
    import pandas as pd
    sweep, e2e = pd.read_parquet(SWEEP), pd.read_parquet(E2E)
    e2e = e2e[e2e.builder == "vectorised"]

    def kernel(backend):
        x = sweep[(sweep.pass_kind == "fwd") & (sweep.backend == backend)
                  & (sweep.seq_len == 16384)]
        return float(x.latency_ms_p50.iloc[0])
    dense = float(e2e[(e2e.band == 16384) & (e2e.backend == "sdpa_flash")]
                  .prefill_ms_mean.iloc[0])
    got = []
    for sp in (0.5, 0.75, 0.9):
        arm = float(e2e[(e2e.band == 16384) & (e2e.backend == "block_sparse")
                        & (e2e.sparsity == sp)].prefill_ms_mean.iloc[0])
        got.append(round(ds.arithmetic_restatement(
            dense_e2e_ms=dense, sparse_e2e_ms=arm, layers=28,
            baseline_kernel_ms=kernel("sdpa_flash"), chosen_kernel_ms=kernel("fa2")).ratio, 3))
    assert got == [1.072, 1.180, 1.260]
