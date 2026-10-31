"""The density rule (researcher, 31 Oct 2026), after Shakedown 2.

Shakedown 2 found that the oracle and mean-pool arms keep no per-layer mask
density (their masks are built by the model wrapper and handed to the
backend), and that under `compiled_default` the kernel-boundary recorder was
traced into the compiled graph and read a CUDA-graph output a later replay
had overwritten. The rule: densities are measured from the executed mask in
an untimed twin pass (or a validated bank manifest); the recorders copy; a
compiled arm's masks must equal its eager twin's; nothing records inside a
timed block. CPU only: a fake kernel module stands in for block-sparse-attn.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest
import torch

from attnbench import densities as D
from attnbench import ground_truth as gt


@pytest.fixture
def fake_bsa(monkeypatch):
    def block_sparse_attn_func(q, k, v, cu_q, cu_k, head_mask_type, streaming,
                               base_blockmask, sq, sk, **kw):
        return q.clone()
    mod = types.ModuleType("block_sparse_attn")
    mod.block_sparse_attn_func = block_sparse_attn_func
    monkeypatch.setitem(sys.modules, "block_sparse_attn", mod)
    return mod


@dataclass
class Mask:                      # the two fields a BlockSparseMask carries that the tap reads
    active: torch.Tensor
    causal: bool = True


def _mask(n_chosen: int, nb: int = 4) -> Mask:
    m = torch.zeros(nb, nb, dtype=torch.bool)
    m[:, 0] = True
    m[torch.arange(nb), torch.arange(nb)] = True
    for j in range(n_chosen):                     # extra blocks below the diagonal
        m[nb - 1, 1 + j] = True
    return Mask(m)


class HandedMaskBackend:
    """The oracle and mean-pool shape: the mask comes in, nothing is kept."""

    def __init__(self, scribble=False):
        self.scribble = scribble

    def forward(self, q, k, v, cfg, mask=None):
        import block_sparse_attn
        kernel_mask = mask.active.unsqueeze(0).unsqueeze(0)
        block_sparse_attn.block_sparse_attn_func(q, k, v, None, None,
                                                 torch.ones(1, dtype=torch.int32), None,
                                                 kernel_mask, 512, 512, is_causal=True)
        if self.scribble:                         # the caller's mask changes afterwards
            mask.active.zero_()
            kernel_mask.zero_()
        return q


class OwnMaskBackend:
    """The XAttention shape: builds its mask inside and keeps a per-layer record."""

    def __init__(self, extra=1):
        self.extra = extra
        self.last_layer_density = []

    def reset_density(self):
        self.last_layer_density = []

    def forward(self, q, k, v, cfg, mask=None):
        import block_sparse_attn
        m = _mask(self.extra).active
        block_sparse_attn.block_sparse_attn_func(q, k, v, None, None,
                                                 torch.ones(1, dtype=torch.int32), None,
                                                 m.unsqueeze(0).unsqueeze(0), 512, 512,
                                                 is_causal=True)
        self.last_layer_density.append(D.block_density(m))
        return q


class Compiled:
    """A compiled arm's wrapper: `.arm` is the eager backend, `.setting` the
    compile setting; `forward` may change what the inner arm is handed."""

    def __init__(self, arm, setting="default", change=None):
        self.arm, self.setting, self.change = arm, setting, change

    def reset_density(self):
        if hasattr(self.arm, "reset_density"):
            self.arm.reset_density()

    @property
    def last_layer_density(self):
        return self.arm.last_layer_density

    def forward(self, q, k, v, cfg, mask=None):
        if self.change is not None:
            mask = self.change(mask)
            if isinstance(self.arm, OwnMaskBackend):
                self.arm.extra = 2
        return self.arm.forward(q, k, v, cfg, mask=mask)


LAYERS = [_mask(1), _mask(2), _mask(0)]


def _runner():
    """`run(backend=)` as `time_arms` gives it: one forward, one call per layer."""
    q = torch.zeros(4, 1, 8)
    seen = []

    def run(backend=None):
        seen.append(backend)
        for m in LAYERS:
            backend.forward(q, q, q, None, mask=Mask(m.active.clone(), m.causal))
    return run, seen


def _arm(backend, *, compiled=False, kind=gt.IN_BACKEND):
    return gt.Arm("x", kind, backend, None, compile_setting="default" if compiled else None,
                  builder_kind=None if kind == gt.DENSE else "gpu")


def _mean(ms):
    return sum(float(D.block_density(m.active)) for m in ms) / len(ms)


# --- the twin pass gives every arm its density --------------------------------------

def test_a_handed_mask_arm_gets_its_density_from_the_twin_pass(fake_bsa):
    run, seen = _runner()
    got = gt.kernel_densities(_arm(HandedMaskBackend()), run)
    assert got["mask_density"] == pytest.approx(_mean(LAYERS))
    assert got["executed_density"] == pytest.approx(_mean(LAYERS))
    assert len(seen) == 1                             # an eager arm: one untimed pass


def test_an_arm_that_builds_its_own_mask_uses_its_record(fake_bsa):
    run, _ = _runner()
    got = gt.kernel_densities(_arm(OwnMaskBackend()), run)
    assert got["mask_density"] == pytest.approx(float(D.block_density(_mask(1).active)))


def test_an_arm_with_neither_is_refused(fake_bsa):
    class Opaque:
        def forward(self, q, k, v, cfg, mask=None):
            return q

    def run(backend=None):
        backend.forward(None, None, None, None, mask=None)
    with pytest.raises(RuntimeError, match="no per-layer mask density"):
        gt.kernel_densities(_arm(Opaque()), run)


def test_dense_is_unchanged():
    assert gt.kernel_densities(_arm(object(), kind=gt.DENSE), lambda backend=None: None) == \
        D.DENSE_FIELDS


# --- a compiled arm: its own compiled twin passes, reported against eager ----------
# (researcher's ruling, 31 Oct 2026: no "equals eager" gate)

def test_a_compiled_arm_is_measured_on_its_compiled_twin_twice_then_eager(fake_bsa):
    inner = HandedMaskBackend()
    comp = Compiled(inner)
    run, seen = _runner()
    got = gt.kernel_densities(_arm(comp, compiled=True), run)
    assert got["mask_density"] == pytest.approx(_mean(LAYERS))
    assert [getattr(b, "_inner", None) for b in seen] == [comp, comp, inner]
    assert got["density_vs_eager"] == 0 and got["density_vs_eager_flagged"] is False


def _drop_a_block(mask):
    a = mask.active.clone()
    a[-1, 1] = False
    return Mask(a, mask.causal)


def test_a_handed_mask_arm_compiled_executes_its_own_masks(fake_bsa):
    """The mask is the one the arm was handed; what the compiled path ran is
    the executed density, and it is that path's own."""
    run, _ = _runner()
    got = gt.kernel_densities(_arm(Compiled(HandedMaskBackend(), change=_drop_a_block),
                                   compiled=True), run)
    executed = _mean([_drop_a_block(Mask(m.active.clone())) for m in LAYERS])
    assert got["mask_density"] == pytest.approx(_mean(LAYERS))
    assert got["executed_density"] == pytest.approx(executed)
    assert got["density_vs_eager"] == 0


class RoundsDifferently(Compiled):
    """XAttention's case: the compiled estimator thresholds one more block."""

    def forward(self, q, k, v, cfg, mask=None):
        old = self.arm.extra
        self.arm.extra = old + 1
        try:
            return self.arm.forward(q, k, v, cfg, mask=mask)
        finally:
            self.arm.extra = old


def _own_runner():
    q = torch.zeros(4, 1, 8)

    def run(backend=None):
        for _ in LAYERS:
            backend.forward(q, q, q, None, mask=None)      # the arm builds its own mask
    return run


def test_a_compiled_own_mask_arm_reports_its_own_density_against_eager(fake_bsa):
    got = gt.kernel_densities(_arm(RoundsDifferently(OwnMaskBackend(extra=1)), compiled=True),
                              _own_runner())
    compiled, eager = (float(D.block_density(_mask(n).active)) for n in (2, 1))
    assert got["mask_density"] == pytest.approx(compiled)          # the code path's own
    assert got["executed_density"] == pytest.approx(compiled)
    assert got["eager_mask_density"] == pytest.approx(eager)
    assert got["density_vs_eager"] == pytest.approx(compiled - eager)
    assert got["density_vs_eager_flagged"] is True                 # 0.1 > 0.01: flagged, kept


def test_the_eager_difference_is_flagged_only_beyond_the_tolerance():
    assert D.DENSITY_FLAG_TOLERANCE == 0.01
    near = D.versus_eager(0.66582, 0.66688)                       # Shakedown 2's XAttention
    assert near["density_vs_eager"] == pytest.approx(-0.00106) and not near["density_vs_eager_flagged"]
    assert D.versus_eager(0.70, 0.68)["density_vs_eager_flagged"]
    assert not D.versus_eager(0.69, 0.68)["density_vs_eager_flagged"]


class Drifting(Compiled):
    """A compiled arm whose second pass on the same input executes a
    different mask: density would not be a function of the input."""

    def __init__(self, arm):
        super().__init__(arm)
        self.calls = 0

    def forward(self, q, k, v, cfg, mask=None):
        self.calls += 1
        if self.calls > len(LAYERS):
            mask = _drop_a_block(mask)
        return self.arm.forward(q, k, v, cfg, mask=mask)


def test_two_compiled_passes_that_disagree_are_refused(fake_bsa):
    run, _ = _runner()
    with pytest.raises(D.DensityMismatch, match="two compiled twin passes"):
        gt.kernel_densities(_arm(Drifting(HandedMaskBackend()), compiled=True), run)


class RecordDrifts(OwnMaskBackend):
    """Executes the same mask on every pass, but its per-layer record
    drifts on the second: the record check catches what the kernel's does not."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, q, k, v, cfg, mask=None):
        self.calls += 1
        out = super().forward(q, k, v, cfg, mask=mask)
        if len(LAYERS) < self.calls <= 2 * len(LAYERS):
            self.last_layer_density[-1] = self.last_layer_density[-1] - 0.05
        return out


def test_a_compiled_record_that_drifts_between_passes_is_refused(fake_bsa):
    with pytest.raises(D.DensityMismatch, match="two compiled twin passes"):
        gt.kernel_densities(_arm(Compiled(RecordDrifts()), compiled=True), _own_runner())


def test_a_compiled_arm_without_an_eager_arm_is_refused(fake_bsa):
    run, _ = _runner()
    with pytest.raises(RuntimeError, match="eager"):
        gt.kernel_densities(_arm(HandedMaskBackend(), compiled=True), run)


# --- the recorders copy -------------------------------------------------------------

def test_the_mask_tap_copies_what_it_is_handed(fake_bsa):
    tap = D.MaskTap(HandedMaskBackend(scribble=True))
    m = _mask(2)
    want = float(D.block_density(m.active))
    tap.forward(torch.zeros(4, 1, 8), None, None, None, mask=m)
    assert float(m.active.sum()) == 0                    # the caller's tensor was overwritten
    assert tap.per_layer_density("x") == [pytest.approx(want)]


def test_the_kernel_recorder_copies_what_the_kernel_is_handed(fake_bsa):
    m = _mask(2)
    want = float(D.block_density(m.active))
    with D.KernelInputRecorder() as rec:
        HandedMaskBackend(scribble=True).forward(torch.zeros(4, 1, 8), None, None, None, mask=m)
    assert rec.executed_density() == pytest.approx(want)


def test_the_tap_copies_an_own_mask_record_after_each_call(fake_bsa):
    inner = OwnMaskBackend()
    tap = D.MaskTap(inner)
    tap.forward(torch.zeros(4, 1, 8), None, None, None)
    inner.last_layer_density[-1].zero_()                 # overwritten in place, as a graph replay would
    assert tap.per_layer_density("x") == [pytest.approx(float(D.block_density(_mask(1).active)))]


# --- nothing records inside a timed block -------------------------------------------

def test_recording_is_refused_inside_a_timed_block(fake_bsa):
    with D.timed_region():
        with pytest.raises(D.RecordingRefused, match="timed"):
            with D.KernelInputRecorder():
                pass
        with pytest.raises(D.RecordingRefused, match="timed"):
            D.MaskTap(HandedMaskBackend()).forward(torch.zeros(4, 1, 8), None, None, None,
                                                   mask=_mask(0))
    rec = D.KernelInputRecorder()
    with rec:
        with D.timed_region():
            with pytest.raises(D.RecordingRefused, match="timed"):
                HandedMaskBackend().forward(torch.zeros(4, 1, 8), None, None, None, mask=_mask(0))


def test_recording_is_refused_when_traced_into_a_compiled_region(fake_bsa, monkeypatch):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(D.RecordingRefused, match="compiled"):
        D.MaskTap(HandedMaskBackend()).forward(torch.zeros(4, 1, 8), None, None, None,
                                               mask=_mask(0))
    with pytest.raises(D.RecordingRefused, match="compiled"):
        with D.KernelInputRecorder():
            HandedMaskBackend().forward(torch.zeros(4, 1, 8), None, None, None, mask=_mask(0))


def test_the_harness_marks_both_timed_sites():
    seen = []
    gt._clocked(lambda: seen.append(D.in_timed_region()), lambda: None, lambda: 0.0)
    log = gt.EventLog()
    with gt.timed_block(log, 0, synchronize=lambda: None, clock=lambda: 0.0, out=[]):
        seen.append(D.in_timed_region())
    assert seen == [True, True] and not D.in_timed_region()


def test_the_kernel_recorder_is_not_traced_into_a_compiled_function(fake_bsa):
    """Shakedown 2's `compiled_default` failure: Dynamo traced the recorder
    into the graph. Disabled, it runs eagerly between the graph's pieces and
    still records the call."""
    import torch._dynamo

    # The module object, not an `import` inside the compiled function: Dynamo
    # resolved that import to another test's fake (test_compiled_variants)
    # even after `torch._dynamo.reset()`.
    def layer(q, m):
        return fake_bsa.block_sparse_attn_func(
            q, q, q, None, None, torch.ones(1, dtype=torch.int32), None, m, 512, 512,
            is_causal=True) * 2

    torch._dynamo.reset()
    m = _mask(2).active.unsqueeze(0).unsqueeze(0)
    with D.KernelInputRecorder() as rec:
        # Dynamo would fall back to eager on the recorder's own refusal, so the
        # result alone cannot show it was never traced; its mark can.
        assert getattr(fake_bsa.block_sparse_attn_func, "_torchdynamo_disable", False)
        torch.compile(layer, backend="eager")(torch.ones(4, 1, 8), m)
    torch._dynamo.reset()
    assert rec.per_call_density() == [pytest.approx(float(D.block_density(m)))]
