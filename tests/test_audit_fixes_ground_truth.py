"""Fixes for audit C3 (ground_truth.py, dense_selection.py) and the
ground-truth `Controls` half of C3 finding 1. Written before the fixes.

`ground_truth` and `dense_selection` import torch at module level, so in a
torch-free container these tests skip. None of them computes with torch:
they were run here against an import-only stand-in for torch (kept out of
the repository), and must be run where torch is installed (UNRUN there).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from attnbench import clock_sampling as cs  # noqa: E402
from attnbench import ground_truth as gt  # noqa: E402

from _clock_fakes import shared_bits_reader  # noqa: E402


class _Counters:
    observes_transients = True

    def read(self):
        return cs.ClockSample(2040, 6250, 0, {"power": 0, "thermal": 0})


# --- C3 finding 1: Controls take only a sampler that sees inside the block -------------

def test_controls_refuse_a_counters_only_sampler():
    s = cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS)
    with pytest.raises(ValueError, match="counters-only"):
        gt.Controls(resolution_floor_ms=1.0, max_rerun_rounds=0, sampler=s)


def test_controls_refuse_a_counters_only_sampler_even_for_bits_it_could_count():
    s = cs.BlockSampler(_Counters(), flag_bits=cs.SW_THERMAL_SLOWDOWN | cs.HW_THERMAL_SLOWDOWN)
    with pytest.raises(ValueError, match="counters-only"):
        gt.Controls(1.0, 0, s)


@pytest.mark.parametrize("missing", [cs.HW_SLOWDOWN, cs.SW_THERMAL_SLOWDOWN,
                                     cs.HW_THERMAL_SLOWDOWN, cs.HW_POWER_BRAKE_SLOWDOWN],
                         ids=["hw_slowdown", "sw_thermal", "hw_thermal", "hw_power_brake"])
def test_controls_refuse_a_flag_set_without_the_researchers_bits(missing):
    s = cs.ProcessSampler(shared_bits_reader, (None, None),
                          flag_bits=cs.RESEARCHER_FLAG_BITS & ~missing)
    with pytest.raises(ValueError, match=cs.reason_names(missing)[0]):
        gt.Controls(1.0, 0, s)


def test_controls_take_the_process_sampler_and_an_in_block_thread():
    gt.Controls(1.0, 0, cs.ProcessSampler(shared_bits_reader, (None, None),
                                          flag_bits=cs.RESEARCHER_FLAG_BITS))
    gt.Controls(1.0, 0, cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS,
                                        interval_s=0.01))


# --- C3 finding 2: every row carries its length, batch, densities and builder kind -----

from attnbench import gt_schema as G  # noqa: E402


class _Ids:
    """Stands in for a (batch, length) token tensor: enough for the digest."""

    def __init__(self, batch=1, length=64):
        self.shape = (batch, length)

    def detach(self):
        return self

    def to(self, *a, **k):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        class _B:
            def tobytes(_):
                return b"ids"
        return _B()


class _Wrapped:
    def __init__(self):
        self.forwards = 0

    def run_measured(self, input_ids, backend, **kw):
        self.forwards += 1


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.010
        return self.t


def _controls():
    return gt.Controls(1e9, 0, cs.BlockSampler(_Counters(), flag_bits=cs.RESEARCHER_FLAG_BITS,
                                               interval_s=0.001))


def _arms(digest):
    return [gt.Arm("dense", gt.DENSE, object(), None),
            gt.Arm("pre", gt.PRECOMPUTED, object(), None, prompt_digest=digest,
                   builder_kind="gpu")]


def _densities(seen):
    def density_of(arm, run):
        seen.append(arm.name)
        run()                                # one untimed forward
        if arm.kind == gt.DENSE:
            return {"mask_density": None, "executed_density": 1.0}
        return {"mask_density": 0.25, "executed_density": 0.3}
    return density_of


def _time(ids, seen, **kw):
    from attnbench.precomputed_masks import prompt_digest
    w = _Wrapped()
    out = gt.time_arms(w, ids, _arms(prompt_digest(ids)), reps=2, controls=_controls(),
                       clock=_Clock(), density_of=_densities(seen), **kw)
    return out, w


def test_every_row_carries_the_five_fields():
    seen = []
    out, _ = _time(_Ids(batch=1, length=64), seen)
    rows = {r["arm"]: r for r in out["rows"]}
    for r in rows.values():
        G.check_row(r)
    assert rows["pre"]["context_length"] == 64 and rows["pre"]["batch"] == 1
    assert rows["pre"]["mask_density"] == 0.25 and rows["pre"]["executed_density"] == 0.3
    assert rows["pre"]["builder_kind"] == "gpu"
    assert rows["dense"]["builder_kind"] is None and rows["dense"]["executed_density"] == 1.0
    assert sorted(seen) == ["dense", "pre"]


def test_the_length_and_batch_are_the_prompts():
    out, _ = _time(_Ids(batch=4, length=96), [])
    assert {(r["context_length"], r["batch"]) for r in out["rows"]} == {(96, 4)}


def test_the_density_pass_is_untimed_and_once_per_arm():
    seen = []
    out, w = _time(_Ids(), seen)
    timed = sum(len(r["prefill_ms_samples"]) + len(r["flagged_ms_samples"]) for r in out["rows"])
    warm = 3 * 2
    assert w.forwards == warm + timed + 2         # one density forward per arm, no more
    assert all(r["prefill_ms_median"] is not None for r in out["rows"])


def test_a_density_the_probe_does_not_give_is_refused():
    from attnbench.precomputed_masks import prompt_digest
    ids = _Ids()

    def partial(arm, run):
        return {"executed_density": 1.0} if arm.kind == gt.DENSE else {"executed_density": 0.3}
    with pytest.raises(G.RowFieldError, match="mask_density"):
        gt.time_arms(_Wrapped(), ids, _arms(prompt_digest(ids)), reps=1, controls=_controls(),
                     clock=_Clock(), density_of=partial)


def test_time_arms_has_no_default_density_probe():
    from attnbench.precomputed_masks import prompt_digest
    ids = _Ids()
    with pytest.raises(TypeError, match="density_of"):
        gt.time_arms(_Wrapped(), ids, _arms(prompt_digest(ids)), reps=1, controls=_controls(),
                     clock=_Clock())


@pytest.mark.parametrize("kind, extra", [(gt.INLINE, {"inline_estimator": "minference_meanpool"}),
                                         (gt.IN_BACKEND, {})])
def test_a_deployable_arm_names_its_builder_kind(kind, extra):
    with pytest.raises(ValueError, match="builder_kind"):
        gt.Arm("a", kind, object(), None, **extra)
    with pytest.raises(ValueError, match="builder_kind"):
        gt.Arm("a", kind, object(), None, builder_kind="tpu", **extra)
    assert gt.Arm("a", kind, object(), None, builder_kind="cpu", **extra).builder_kind == "cpu"


def test_a_dense_arm_has_no_builder():
    with pytest.raises(ValueError, match="builder_kind"):
        gt.Arm("d", gt.DENSE, object(), None, builder_kind="gpu")


def test_the_harness_uses_c10s_builder_kinds():
    assert gt.BUILDER_KINDS is G.BUILDER_KINDS


def test_the_shakedown_arms_name_their_builders():
    from attnbench.components import shakedown_arms as SA
    assert {n: s.builder_kind for n, s in SA.ARMS.items()} == {
        "dense": None, "oracle": "cpu", "meanpool": "gpu", "xattention": "gpu",
        "xattention_noassert": "gpu"}


# --- C3 finding 3: the dense control is checked against the prediction's cell ----------

from attnbench import dense_selection as ds  # noqa: E402
from attnbench.predict.training import make_cell_key  # noqa: E402

CARD = "NVIDIA L4"
LABEL = f"{CARD}|torch|O0|0123456789ab"


def _choice(card=CARD, length=4096, batch=1):
    return ds.Choice(cell=ds.Cell(card, length, batch), cfg_key="k", planned=[ds.SDPA_FLASH],
                     candidates=[], rounds=3, resolution_floor_ms=0.5, recorded_at=100.0,
                     chosen=ds.SDPA_FLASH, complete=True)


def _key(length=4096, batch=1, label=LABEL):
    return make_cell_key("m", label, "dense", 1.0, length, batch)


def test_the_control_check_passes_for_its_own_cell():
    ds.check_control(_choice(), ds.SDPA_FLASH, 101.0, key=_key())
    ds.check_control(_choice(), ds.SDPA_FLASH, 101.0, key=ds.Cell(CARD, 4096, 1))


@pytest.mark.parametrize("key, match", [
    (_key(length=8192), "context length"),
    (_key(batch=4), "batch"),
    (_key(label="NVIDIA A100-SXM4-40GB|torch|O0|0123456789ab"), "card"),
    (ds.Cell(CARD, 2048, 1), "context length"),
])
def test_a_control_recorded_for_another_cell_is_refused(key, match):
    with pytest.raises(ds.SelectionError, match=match):
        ds.check_control(_choice(), ds.SDPA_FLASH, 101.0, key=key)


def test_the_control_check_needs_the_key():
    with pytest.raises(TypeError, match="key"):
        ds.check_control(_choice(), ds.SDPA_FLASH, 101.0)
