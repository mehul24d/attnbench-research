"""Gate A's leg 3 in a P1 session (`components.delay_session`), on CPU with
the injector tests' recorded sequence and fake arm, and a clock the passes
advance."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from attnbench.components import delay_session as dls
from attnbench.predict import recursion

from test_components_delay_injector import BASE, CALLS, SEQ, FakeArm

REPO = Path(__file__).resolve().parents[1]
COSTS = {m: recursion.MechanismCost(s_fixed=1.0, bandwidth=1e10, loaded_bandwidth=1e10)
         for m in recursion.PRICED_MECHANISMS}
DECL = {"arm": "dense", "length": 2048, "batch": 1,
        "mechanisms": ["synchronize", "item"],
        "counts": {"busy": [0, 1, 2], "idle": [0, 2, 4]},
        "reps": 3, "warmup": 1, "payload_bytes": 0}


class Clocked(FakeArm):
    """The fake arm on a clock: the prefill takes 100, an injection 3, and
    every timed read and fire is logged."""

    def __init__(self):
        super().__init__()
        self.now = 0.0
        self.events: list = []
        self._on_call = None

    def run_prefill(self):          # observed or not (the unobserved c = 0 pass)
        self.now += 100.0
        for c in self.calls:
            if self._on_call is not None:
                self._on_call(c.name, c.layer)

    def fire_of(self, m):
        def fire():
            self.now += 3.0
            self.events.append(("fire", m))
        return fire

    def clock(self):
        self.events.append("clock")
        return self.now


def _run(decl=DECL, arm=None, **kw):
    arm = arm or Clocked()
    written = {}

    def write_plan(rec):
        written["plan"] = copy.deepcopy(rec)
        written["events_before"] = list(arm.events)
    res = dls.run(arm.run_prefill, SEQ, BASE, COSTS, None, dls.declaration(decl),
                  observer=arm.observer, fire_of=arm.fire_of, synchronize=lambda: None,
                  clock=arm.clock, write_plan=write_plan, **kw)
    return res, written, arm


def test_the_plan_is_written_before_any_pass_is_timed_or_fired():
    res, written, arm = _run()
    assert written["events_before"] == []          # no clock read, no injection yet
    assert written["plan"] == res["plan"]
    assert "clock" in arm.events and any(e != "clock" for e in arm.events)


def test_every_variant_and_the_unobserved_pass_is_timed_reps_times():
    res, _, _ = _run()
    keys = {f"{m}/{p}/{c}" for m in DECL["mechanisms"]
            for p, cs in DECL["counts"].items() for c in cs}
    assert set(res["samples_s"]) == keys | {dls.UNOBSERVED}
    assert all(len(v) == DECL["reps"] for v in res["samples_s"].values())
    # the unobserved pass is the bare prefill, the observed c = 0 the same here
    assert res["observer_effect"]["effect_s"] == 0.0


def test_the_slopes_are_read_from_the_measured_passes():
    res, _, _ = _run()
    for m in DECL["mechanisms"]:
        e = res["mechanisms"][m]
        assert e["status"] == "measured"
        # each injection adds 3 on this clock, so the measured slope is 3
        for p in ("busy", "idle"):
            assert e["placements"][p]["measured_slope"] == pytest.approx(3.0)


def test_rounds_rotate_the_order():
    assert dls.rotated(["a", "b", "c"], 0) == ["a", "b", "c"]
    assert dls.rotated(["a", "b", "c"], 1) == ["b", "c", "a"]
    assert dls.rotated(["a", "b", "c"], 4) == ["b", "c", "a"]


def test_a_refused_pair_is_recorded_and_not_timed():
    # the base has 4 busy sites the injector accepts; 9 cannot be spread over them
    decl = dict(DECL, counts={"busy": [0, 9], "idle": [0, 2]})
    res, written, _ = _run(decl)
    assert set(written["plan"]["refused"]) == {"synchronize/busy", "item/busy"}
    assert not any("/busy/" in k for k in res["samples_s"])
    assert all(e["status"] == "measured" for e in res["mechanisms"].values())


def test_a_flat_measurement_is_unresolved_not_a_slope():
    class Flat(Clocked):
        def fire_of(self, m):
            return lambda: None
    res, _, _ = _run(arm=Flat())
    assert {e["status"] for e in res["mechanisms"].values()} == {"unresolved"}


def test_slopes_are_recomputed_from_the_record_and_a_changed_plan_is_refused():
    res, _, _ = _run()
    rec = json.loads(json.dumps(res))
    got = dls.slopes_from_record(rec)
    assert set(got) == set(DECL["mechanisms"])
    for m, s in got.items():
        assert s.slope == pytest.approx(res["mechanisms"][m]["slope"])
    rec["plan"]["variants"][1]["predicted_s"] += 1.0
    with pytest.raises(dls.LegThreeError, match="sha256"):
        dls.slopes_from_record(rec)


@pytest.mark.parametrize("change, why", [
    ({"arm": "oracle"}, "dense arm"),
    ({"mechanisms": ["synchronize", "synchronize"]}, "each once"),
    ({"mechanisms": ["spin"]}, "each one the recursion prices"),
    ({"counts": {"busy": [0, 1]}}, "per placement"),
    ({"counts": {"busy": [1, 2], "idle": [0, 2]}}, "include 0"),
    ({"reps": 0}, "reps is an int"),
    ({"payload_bytes": 64}, "payload 0"),
    ({"extra": 1}, "needs exactly"),
])
def test_the_declaration_refuses_anything_but_the_leg(change, why):
    with pytest.raises((dls.LegThreeError, recursion.CostModelError), match=why):
        dls.declaration(dict(DECL, **change))


def test_the_session_config_declares_every_priced_mechanism_on_the_dense_arm():
    cfg = json.loads((REPO / "configs" / "t31" / "p1_session.json").read_text())
    d = dls.declaration(cfg["delay_injection"])
    assert set(d.mechanisms) == set(recursion.PRICED_MECHANISMS)
    assert (d.arm, d.length, d.batch) == ("dense", 2048, 1)
    assert d.length in cfg["lengths"]
    assert d.counts == {"busy": (0, 16, 32, 64), "idle": (0, 1000, 2000, 4000)}


# --- the observer and alignment step 5 uses (torch, on CPU) ------------------------------

def test_the_observer_reports_calls_by_step_fives_rule_and_aligns_by_it():
    """A factory call that names the device counts, an empty output does not,
    and the isolated replay's own call (the embedding outside the layers) is
    neither reported nor a position any call prices."""
    torch = pytest.importorskip("torch")
    from attnbench.components import delay_injector as di
    from attnbench.components import host_profile as hp
    from attnbench.components.op_sequence import CodePath, OpSequence, SeqKind, SeqOp
    K = SeqKind.KERNEL
    ops = (SeqOp(K, layer=None, aten_op="aten::index_select", chain=(
               "aten::embedding", "aten::index_select"), calls=(0, 0),
               origin="aten::embedding", kernels=("g",), stream=0),
           SeqOp(K, layer=None, aten_op="aten::arange", chain=("aten::arange",), calls=(0,),
                 kernels=("a",), stream=0),
           SeqOp(K, layer=0, aten_op="aten::mm", chain=("aten::mm",), calls=(0,),
                 kernels=("m",), stream=0),
           SeqOp(K, layer=1, aten_op="aten::mm", chain=("aten::mm",), calls=(1,),
                 kernels=("m",), stream=0))
    seq = OpSequence(code_path=CodePath(arm="dense", gpu="cpu", implementation="t"),
                     context_length=4, batch=1, ops=ops, streams=(0,))
    iso = hp.isolated_positions(seq, {"aten::index_select"})
    assert iso == (0,)
    layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(2))
    w = torch.eye(4)
    emb = torch.nn.Embedding(8, 4)

    def run_prefill():
        with torch.no_grad():
            emb(torch.tensor([1, 2]))
            torch.arange(4)
            torch.mm(torch.zeros(0, 4), w)                 # empty: no call
            for m in layers:
                m(torch.mm(w, w))

    observe = dls.recorded_observer(seq, skip=iso, device_type="cpu", layers=layers)
    seen = []
    observe(run_prefill, lambda n, layer: seen.append((n, layer)))   # the mm runs outside them
    assert seen == [("aten::arange", None), ("aten::mm", None), ("aten::mm", None)]

    def run_in_layers():
        with torch.no_grad():
            emb(torch.tensor([1, 2]))
            torch.arange(4)
            torch.mm(torch.zeros(0, 4), w)
            for m in layers:
                m.forward = (lambda x: torch.mm(x, w))
                m(w)
    seen.clear()
    observe(run_in_layers, lambda n, layer: seen.append((n, layer)))
    assert seen == [("aten::arange", None), ("aten::mm", 0), ("aten::mm", 1)]
    calls = di.calibrate(run_in_layers, seq, observe, align=dls.recorded_alignment(iso))
    assert dls.recorded_alignment(iso)(seq, calls) == ((1,), (2,), (3,))
    with pytest.raises(di.InjectionError, match="aten::mm in layer None"):
        di.calibrate(run_prefill, seq, observe, align=dls.recorded_alignment(iso))
