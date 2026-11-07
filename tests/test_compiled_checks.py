"""The compiled arms' timed region without a probe inside it (researcher's
ruling 1 of 7 Nov 2026, after the exploratory session).

The ground truth's probe wraps the scoring functions in Python. In the
exploratory session's `compiled_default` group, Dynamo traced that wrapper
inside XAttention's compiled `forward_and_mask` and guarded on the length of
the probe's event list, so every call recompiled until the limit (256). A
Python probe cannot observe per-call work inside a compiled region, so the
probe stays out of it (`CompiledBackend.compiled_region_targets`), and what the
probe showed for an eager arm is shown for a compiled arm whose estimator is
inside its compiled function by two untimed checks after the timed blocks:

  - the Kineto twin check: the estimator's kernels in every layer, in a plain
    recording of the compiled path and in its twin pass, the same;
  - the two-prompt mask dependence check: the compiled masks change with the
    prompt, by the eager masks' change within +-0.01.

No GPU here: the traces are written by hand, the masks are tensors, and the
compiled backend runs under Dynamo's "eager" backend on a CPU. That the
profiler sees a CUDA graph's kernels under each layer's annotation is the GPU
session's to show.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest
import torch

from attnbench import compiled_checks as CC
from attnbench import compile_guard
from attnbench import ground_truth as gt
from attnbench.backends import block_sparse as BS
from attnbench.backends import compiled as C
from attnbench.config import AttnConfig

EAGER = dict(mode=None, dynamo_backend="eager")


@pytest.fixture(autouse=True)
def fresh_state():
    C.reset_compiled_state()
    compile_guard.reset_process_state()
    gt.reset_process_compile_setting()
    yield
    C.reset_compiled_state()
    compile_guard.reset_process_state()
    gt.reset_process_compile_setting()


@dataclass
class _Cap:
    name: str


class MaskArm:
    """An arm that builds its own mask inside its compiled function, as
    XAttention does: `estimate`, then `forward_and_mask`."""
    capability = _Cap("mask_arm")

    def __init__(self):
        self.layer_idx = 0

    def set_layer(self, i):
        self.layer_idx = i

    def estimate(self, q, k):
        return (q @ k.transpose(-1, -2)).mean(dim=-1, keepdim=True) > 0

    def forward_and_mask(self, q, k, v, cfg, mask=None):
        m = self.estimate(q, k)
        return v * m.float(), m

    def forward(self, q, k, v, cfg, mask=None):
        return self.forward_and_mask(q, k, v, cfg, mask=mask)[0]


class SubMaskArm(MaskArm):
    """A subclass that overrides the estimator (XAttention without asserts)."""
    capability = _Cap("sub_mask_arm")

    def estimate(self, q, k):
        return super().estimate(q, k)


def _cfg(s=16, h=2):
    return AttnConfig(seq_len=s, batch=1, n_heads_q=h, n_heads_kv=h, head_dim=8,
                      dtype="float32", mask="causal")


def _qkv(seed=0, s=16, h=2):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, h, s, 8, generator=g) for _ in range(3))


# --- the probe stays out of compiled regions -------------------------------------------------

def test_a_compiled_backend_names_the_estimators_its_compiled_function_calls():
    be = C.CompiledBackend(SubMaskArm(), "default", **EAGER)
    here = __name__
    assert set(be.compiled_region_targets) == {(here, "SubMaskArm.estimate"),
                                               (here, "MaskArm.estimate")}


def test_the_probe_patches_no_target_inside_an_arms_compiled_region(monkeypatch):
    targets = ((__name__, "MaskArm.estimate"), (__name__, "SubMaskArm.estimate"))
    monkeypatch.setattr(gt, "SCORE_TARGETS", targets)
    compiled = gt.Arm("mask@compile", gt.IN_BACKEND, C.CompiledBackend(SubMaskArm(), "default",
                                                                      **EAGER),
                      _cfg(), compile_setting="default", builder_kind="gpu")
    eager = gt.Arm("mask", gt.IN_BACKEND, SubMaskArm(), _cfg(), builder_kind="gpu")
    real = (MaskArm.__dict__["estimate"], SubMaskArm.__dict__["estimate"])
    with gt.probe_for(gt.EventLog(), compiled):
        assert (MaskArm.__dict__["estimate"], SubMaskArm.__dict__["estimate"]) == real
    with gt.probe_for(gt.EventLog(), eager):
        assert MaskArm.__dict__["estimate"] is not real[0]
        assert SubMaskArm.__dict__["estimate"] is not real[1]
    assert (MaskArm.__dict__["estimate"], SubMaskArm.__dict__["estimate"]) == real


def test_a_compiled_arm_with_its_estimator_outside_the_compiled_function_keeps_its_probe():
    # mean-pool: its estimator runs in the model's eager code, not in the backend
    class Plain:
        capability = _Cap("plain")

        def forward(self, q, k, v, cfg, mask=None):
            return v
    be = C.CompiledBackend(Plain(), "default", **EAGER)
    assert be.compiled_region_targets == ()
    arm = gt.Arm("mp@compile", gt.INLINE, be, _cfg(), inline_estimator="minference_meanpool",
                 compile_setting="default", builder_kind="gpu")
    assert not gt.needs_compiled_checks(arm)


def test_no_probe_wrapper_is_traced_by_dynamo(monkeypatch):
    """The exploratory session's defect, on a CPU: a probed estimator inside a
    compiled forward guards on the probe's log and recompiles per call."""
    targets = ((__name__, "MaskArm.estimate"),)
    monkeypatch.setattr(gt, "SCORE_TARGETS", targets)
    arm_impl = MaskArm()
    be = C.CompiledBackend(arm_impl, "default", **EAGER)
    arm = gt.Arm("mask@compile", gt.IN_BACKEND, be, _cfg(), compile_setting="default",
                 builder_kind="gpu")
    q, k, v = _qkv()
    log = gt.EventLog()
    for _ in range(6):
        with gt.probe_for(log, arm):
            be.forward(q, k, v, _cfg())
    assert len(be.events) == 1                         # one compile, then warm
    assert not log.events                              # nothing recorded inside the graph
    C.reset_compiled_state()
    be2 = C.CompiledBackend(MaskArm(), "default", **EAGER)
    log2 = gt.EventLog()
    for _ in range(6):                                 # the old probe, for contrast
        with gt.ScoringProbe(log2, score_targets=targets, mask_targets=()):
            be2.forward(q, k, v, _cfg())
    assert len(be2.events) > 1


# --- the timed region of a compiled arm whose estimator is compiled -------------------------------

PASSED = {"kineto_twin": {"passed": True}, "mask_dependence": {"passed": True}}


def _logged(reps):
    log = gt.EventLog()
    for r in reps:
        log.open_rep = r
        log.record("sync", "closing synchronise")
    log.open_rep = None
    return log


def _compiled_xa():
    return gt.Arm("xa@compile", gt.IN_BACKEND, C.CompiledBackend(MaskArm(), "default", **EAGER),
                  _cfg(), compile_setting="default", builder_kind="gpu")


def test_a_compiled_in_backend_arm_passes_on_its_compiled_checks_not_on_score_events():
    gt.check_timed_region(_logged([0, 1]), _compiled_xa(), [0, 1], compiled_checks=PASSED)


@pytest.mark.parametrize("bad", ["none", "twin", "dependence", "missing"])
def test_a_compiled_in_backend_arm_without_both_checks_passed_is_refused(bad):
    checks = {"none": None,
              "twin": dict(PASSED, kineto_twin={"passed": False, "problems": ["layer 1: ..."]}),
              "dependence": dict(PASSED, mask_dependence={"passed": False, "problems": ["x"]}),
              "missing": {"kineto_twin": {"passed": True}}}[bad]
    with pytest.raises(gt.TimedRegionError, match="compiled"):
        gt.check_timed_region(_logged([0, 1]), _compiled_xa(), [0, 1], compiled_checks=checks)


def test_an_eager_in_backend_arm_still_needs_score_events():
    arm = gt.Arm("xa", gt.IN_BACKEND, MaskArm(), _cfg(), builder_kind="gpu")
    with pytest.raises(gt.TimedRegionError, match="computed no scores"):
        gt.check_timed_region(_logged([0]), arm, [0], compiled_checks=PASSED)


# --- the Kineto twin check -----------------------------------------------------------------------

class _Trace:
    def __init__(self):
        self.ev, self.corr, self.t = [], 1, 0.0

    def x(self, cat, name, ts, dur, tid=1, **args):
        self.ev.append({"ph": "X", "cat": cat, "name": name, "pid": 1, "tid": tid,
                        "ts": ts, "dur": dur, "args": args})

    def launch(self, ts, *kernels, api="cudaLaunchKernel"):
        c = self.corr
        self.corr += 1
        self.x("cuda_runtime", api, ts, 1.0, correlation=c)
        for k in kernels:
            self.t += 2.0
            self.ev.append({"ph": "X", "cat": "kernel", "name": k, "pid": 0, "tid": 7,
                            "ts": 10_000 + self.t, "dur": 1.0, "args": {"correlation": c}})

    def dict(self):
        return {"traceEvents": self.ev}


def _structured_trace(layers=2, *, est=("e0", "e1"), skip_layer=None):
    t = _Trace()
    t.launch(1, "outside")                                        # before any layer: not counted
    for l in range(layers):
        base = 100 * (l + 1)
        t.x("user_annotation", f"{C.COMPILED_CALL}{l}", base, 50)
        if l != skip_layer:
            t.launch(base + 5, *est, api="cudaGraphLaunch")       # the compiled graph: estimator
        t.x("user_annotation", BS.KERNEL_INTERFACE, base + 20, 20)
        t.launch(base + 25, "bsa_fwd")                            # the kernel: not the estimator
        t.launch(base + 70, "o_proj")                             # after the call: not counted
    return t.dict()


def test_the_estimators_kernels_are_read_per_layer_from_the_compiled_calls():
    s = CC.estimator_structure(_structured_trace(3))
    assert s == {0: ("e0", "e1"), 1: ("e0", "e1"), 2: ("e0", "e1")}


def test_the_twin_check_passes_when_every_layer_runs_the_recorded_estimator():
    rec = CC.estimator_structure(_structured_trace(3))
    out = CC.kineto_twin_check(rec, dict(rec), 3)
    assert out["passed"] and out["problems"] == []
    assert out["kernels_per_layer"] == {"0": 2, "1": 2, "2": 2}


@pytest.mark.parametrize("bad,match", [
    ("twin lost a layer", "layer 1: the twin ran no estimator kernel"),
    ("no estimator anywhere", "layer 0: the recorded compiled path ran no estimator kernel"),
    ("other kernels", "layer 2: the twin ran"),
    ("layer beyond the model", "layer 3"),
])
def test_the_twin_check_refuses_a_layer_that_differs_or_runs_no_estimator(bad, match):
    rec = CC.estimator_structure(_structured_trace(3))
    if bad == "twin lost a layer":
        twin = CC.estimator_structure(_structured_trace(3, skip_layer=1))
    elif bad == "no estimator anywhere":
        rec = twin = CC.estimator_structure(_structured_trace(3, est=()))
    elif bad == "other kernels":
        twin = dict(rec)
        twin[2] = ("e0",)
    else:
        twin = CC.estimator_structure(_structured_trace(4))
        rec = dict(twin)
    out = CC.kineto_twin_check(rec, twin, 3)
    assert not out["passed"] and any(match in p for p in out["problems"])


# --- the two-prompt mask dependence check -----------------------------------------------------------

def _masks(n_layers, nb, seed):
    """Causal block masks, (1, 2 heads, nb, nb) per layer."""
    g = torch.Generator().manual_seed(seed)
    causal = torch.ones(nb, nb).tril().bool()
    return [(torch.rand(1, 2, nb, nb, generator=g) > 0.5) & causal for _ in range(n_layers)]


def _moved(b, a, n):
    """`b` with `n` more causal blocks of layer 0 differing from `a`: flipped
    where the two agree, so the dependence grows by exactly n blocks."""
    out = [m.clone() for m in b]
    nb = b[0].shape[-1]
    rows, cols = torch.tril_indices(nb, nb)
    done = 0
    for r, c in zip(rows.tolist(), cols.tolist()):
        if done == n:
            break
        if out[0][0, 0, r, c] == a[0][0, 0, r, c]:
            out[0][0, 0, r, c] = ~out[0][0, 0, r, c]
            done += 1
    assert done == n
    return out


def test_the_masks_depend_on_the_prompt_as_the_eager_masks_do():
    a, b = _masks(2, 10, 0), _masks(2, 10, 1)            # two prompts
    out = CC.mask_dependence_check(a, b, a, b)
    assert out["passed"] and out["difference"] == 0.0
    assert out["compiled_dependence"] == out["eager_dependence"] > 0
    assert out["compiled_vs_eager"] == [0.0, 0.0] and out["tolerance"] == 0.01


def test_a_compiled_mask_that_does_not_change_with_the_prompt_is_refused():
    a, b = _masks(2, 10, 0), _masks(2, 10, 1)
    out = CC.mask_dependence_check(a, a, a, b)           # the compiled path kept prompt A's mask
    assert not out["passed"] and out["compiled_dependence"] == 0.0
    assert any("do not depend on the prompt" in p for p in out["problems"])


def test_the_dependence_is_held_to_the_eagers_within_the_tolerance():
    # 2 layers x 2 heads x 55 causal blocks = 220 entries: 2 flipped blocks are 2/220
    a, b = _masks(2, 10, 0), _masks(2, 10, 1)
    near = CC.mask_dependence_check(a, _moved(b, a, 2), a, b)
    assert near["passed"] and near["difference"] == pytest.approx(2 / 220)
    far = CC.mask_dependence_check(a, _moved(b, a, 3), a, b)
    assert not far["passed"] and far["difference"] == pytest.approx(3 / 220)
    assert any("+-0.01" in p for p in far["problems"])


def test_two_prompts_whose_eager_masks_agree_test_nothing_and_are_refused():
    a = _masks(2, 10, 0)
    out = CC.mask_dependence_check(a, a, a, a)
    assert not out["passed"] and any("eager masks" in p for p in out["problems"])


def test_masks_that_cannot_be_compared_are_refused():
    with pytest.raises(CC.CompiledCheckError, match="layers"):
        CC.mask_dependence_check(_masks(2, 10, 0), _masks(3, 10, 1), _masks(2, 10, 0),
                                 _masks(2, 10, 1))
    with pytest.raises(CC.CompiledCheckError, match="shape"):
        CC.mask_dependence_check(_masks(2, 10, 0), _masks(2, 12, 1), _masks(2, 10, 0),
                                 _masks(2, 10, 1))


# --- what the checks read: the annotations, the masks ---------------------------------------------

def _user_annotations(prof):
    return [e.name for e in prof.events() if e.name.startswith("attnbench::")]


def test_the_compiled_backend_annotates_each_layers_call_only_when_asked():
    be = C.CompiledBackend(MaskArm(), "default", **EAGER)
    q, k, v = _qkv()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for l in range(2):
            be.set_layer(l)
            be.forward(q, k, v, _cfg())
    assert _user_annotations(prof) == []                # the timed path: no annotation
    with profile(activities=[ProfilerActivity.CPU]) as prof, be.annotate():
        for l in range(2):
            be.set_layer(l)
            be.forward(q, k, v, _cfg())
    assert _user_annotations(prof) == [f"{C.COMPILED_CALL}0", f"{C.COMPILED_CALL}1"]


def test_the_kernel_interface_is_annotated_only_when_asked(monkeypatch):
    monkeypatch.setattr(BS, "_call", lambda *a, unsupported, **k: "out")
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        BS.kernel_interface(1)
    assert _user_annotations(prof) == []
    with profile(activities=[ProfilerActivity.CPU]) as prof, BS.annotated():
        assert BS.kernel_interface(1) == "out"
    assert _user_annotations(prof) == [BS.KERNEL_INTERFACE]
    assert not BS._ANNOTATE


def test_the_twin_pass_keeps_each_layers_compiled_mask_when_asked():
    be = C.CompiledBackend(MaskArm(), "default", **EAGER)
    q, k, v = _qkv()
    with be.twin_pass(), be.keep_masks() as kept:
        for l in range(3):
            be.set_layer(l)
            be.forward(q, k, v, _cfg())
    assert len(kept) == 3 and all(torch.equal(m, MaskArm().estimate(q, k)) for m in kept)
    with be.twin_pass():
        be.forward(q, k, v, _cfg())
    assert len(kept) == 3                               # kept only while asked


def test_the_eager_masks_are_the_eager_arms_own():
    arm = MaskArm()
    rec = CC.EagerMasks(arm)
    q, k, v = _qkv()
    rec.set_layer(2)
    out = rec.forward(q, k, v, _cfg())
    assert arm.layer_idx == 2 and torch.equal(out, arm.forward(q, k, v, _cfg()))
    assert len(rec.masks) == 1 and torch.equal(rec.masks[0], arm.estimate(q, k))


# --- the session's checks, end to end on fakes --------------------------------------------------------

class _Run:
    """`time_arms`' untimed `run(backend=, input_ids=)`, on a fake model of
    `layers` layers that calls the backend once per layer."""

    def __init__(self, layers, prompts):
        self.layers, self.prompts, self.calls = layers, prompts, []

    def __call__(self, backend=None, input_ids=None):
        which = "b" if input_ids is not None else "a"
        self.calls.append((type(backend).__name__ if backend is not None else None, which))
        be = backend if backend is not None else self.backend
        for l in range(self.layers):
            be.set_layer(l)
            q, k, v = self.prompts[which]
            be.forward(q, k, v, _cfg())


def test_the_session_runs_both_checks_untimed_and_reports_them(monkeypatch):
    be = C.CompiledBackend(MaskArm(), "default", **EAGER)
    arm = gt.Arm("xa@compile", gt.IN_BACKEND, be, _cfg(), compile_setting="default",
                 builder_kind="gpu")
    run = _Run(2, {"a": _qkv(0), "b": _qkv(5)})
    run.backend = be
    profiled = []

    def profile(fn):                                    # the GPU's trace, written by hand
        fn()
        profiled.append(be._twin)
        return _structured_trace(2)
    out = CC.run_checks(arm, run, second_ids="ids_b", n_layers=2, profile=profile)
    assert profiled == [False, True]                    # the plain path, then its twin
    assert out["kineto_twin"]["passed"] and out["mask_dependence"]["passed"]
    assert ("EagerMasks", "a") in run.calls and ("EagerMasks", "b") in run.calls
    assert not BS._ANNOTATE and not be._annotate        # nothing left on


def test_time_arms_refuses_a_compiled_in_backend_arm_without_the_checks_and_reports_them():
    import test_ground_truth_timed_region as tr
    from _gt_fakes import fake_densities

    class _XAStandIn(tr.NaiveAttention):
        capability = _Cap("xa_stand_in")

        def estimate(self, q, k):
            return q[..., :1] > 0

        def forward_and_mask(self, q, k, v, cfg, mask=None):
            from dataclasses import replace
            return super().forward(q, k, v, replace(cfg, mask="causal")), self.estimate(q, k)
    be = C.CompiledBackend(_XAStandIn(), "default", **EAGER)
    be.check_recompiles = lambda gate=0: 0
    cfg = tr._cfg(mask="causal", sparsity=None, block_size=None)
    arm = gt.Arm("xa@compile-default", gt.IN_BACKEND, be, cfg, compile_setting="default",
                 builder_kind="gpu")
    kw = dict(reps=tr.REPS, density_of=fake_densities, controls=tr._controls())
    with pytest.raises(gt.TimedRegionError, match="compiled checks"):
        gt.time_arms(tr._wrapped(), tr._ids(), [arm, tr._dense()], **kw)
    gt.reset_process_compile_setting()
    seen = []

    def checks(a, run):
        seen.append(a.name)
        run(input_ids=tr._ids())                        # the second prompt goes through
        return PASSED
    out = gt.time_arms(tr._wrapped(), tr._ids(), [arm, tr._dense()], compiled_check_of=checks, **kw)
    row = out["rows"][0]
    assert seen == ["xa@compile-default"] and row["compiled_checks"] == PASSED
    assert row["score_calls_timed"] == 0                # no probe in its compiled region
    assert "compiled_checks" not in out["rows"][1]
    gt.reset_process_compile_setting()
    failed = dict(PASSED, mask_dependence={"passed": False, "problems": ["no dependence"]})
    with pytest.raises(gt.TimedRegionError, match="no dependence"):
        gt.time_arms(tr._wrapped(), tr._ids(), [arm, tr._dense()],
                     compiled_check_of=lambda a, run: failed, **kw)


def test_the_second_prompt_is_the_next_at_the_budget_cut_to_the_cells_tokens():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import t31_steps as st

    class H:
        def __init__(self, n):
            self.n, self.asked = n, []

        def input_ids(self, length, *, index=0):
            self.asked.append((length, index))
            return torch.arange(self.n)[None], f"p{index}"
    cell = torch.arange(10)[None].repeat(2, 1)              # batch 2, 10 tokens
    h = H(12)
    ids, pid = st.second_prompt(h, 16, cell)
    assert h.asked == [(16, 1)] and pid == "p1"
    assert tuple(ids.shape) == (2, 10) and torch.equal(ids[1], torch.arange(10))
    with pytest.raises(st.StepFailed, match="fewer than the cell prompt's 10"):
        st.second_prompt(H(9), 16, cell)
