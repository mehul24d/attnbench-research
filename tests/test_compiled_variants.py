"""T2.21: compiled variants, their graph-break report and mask agreement.

CPU only. Most tests compile with Dynamo's "eager" backend, which traces
the same graphs as inductor without generating code; one test runs inductor
in the plan's mode end to end. The arms are dense SDPA (math), a stand-in
whose estimator reads `.item()` (the pattern the capture options exist for),
one whose setting depends on the layer, and the real `XAttentionBackend`
over the stand-in `xattn` package of tests/test_xattention_noassert.py.

The XAttention report on the real package needs a CUDA device, since
upstream's estimator moves its inputs to "cuda"; that is the L4 dry run.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest
import torch

from attnbench import compile_guard
from attnbench.backends import compiled as C
from attnbench.backends.impls import SDPABackend
from attnbench.config import AttnConfig

pytest.importorskip("torch._dynamo")

EAGER = dict(mode=None, dynamo_backend="eager")


@pytest.fixture(autouse=True)
def fresh_dynamo():
    C.reset_compiled_state()
    compile_guard.reset_process_state()
    yield
    C.reset_compiled_state()
    compile_guard.reset_process_state()


def _cfg(s=64, h=2):
    return AttnConfig(seq_len=s, batch=1, n_heads_q=h, n_heads_kv=h, head_dim=16,
                      dtype="float32", mask="causal")


def _qkv(seed=0, s=64, h=2):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(1, h, s, 16, generator=g) for _ in range(3))


@dataclass
class _Cap:
    name: str


class ItemArm:
    """An estimator that reads a scalar back, then the kernel."""
    capability = _Cap("item_arm")

    def __init__(self):
        self.last_layer_density = []

    def reset_density(self):
        self.last_layer_density = []

    def estimate(self, q, k):
        scores = (q @ k.transpose(-1, -2)).mean(dim=(-1, -2))
        cut = scores.sum().item() / scores.numel()   # a scalar read back to Python
        return scores > cut

    def forward(self, q, k, v, cfg, mask=None):
        m = self.estimate(q, k)
        return v * m[..., None, None].float()


class BranchArm(ItemArm):
    """Python control flow on the value read back."""
    capability = _Cap("branch_arm")

    def estimate(self, q, k):
        scores = (q @ k.transpose(-1, -2)).mean(dim=(-1, -2))
        if scores.sum().item() > 0:
            return scores > 0
        return scores <= 0


class LayerArm(ItemArm):
    """A per-layer setting held as a Python int, like a threshold table row."""
    capability = _Cap("layer_arm")

    def __init__(self):
        super().__init__()
        self.layer_idx = 0

    def set_layer(self, i):
        self.layer_idx = i

    def forward(self, q, k, v, cfg, mask=None):
        return v * (self.layer_idx + 1)


# --- names, settings, one setting per process -------------------------------------

def test_variant_names_and_settings():
    assert C.variant_name("xattention", "capture") == "xattention@compile-capture"
    assert C.SETTINGS == {"default": {}, "capture": {
        "capture_scalar_outputs": True, "capture_dynamic_output_shape_ops": True}}
    assert C.MODE == "reduce-overhead"
    assert C.CompiledBackend(SDPABackend("math"), "default", **EAGER).name == "sdpa@compile-default"
    with pytest.raises(ValueError, match="setting 'max'"):
        C.CompiledBackend(SDPABackend("math"), "max", **EAGER)


def test_an_arm_is_compiled_once():
    inner = C.CompiledBackend(SDPABackend("math"), "default", **EAGER)
    with pytest.raises(TypeError, match="compiled once"):
        C.CompiledBackend(inner, "default", **EAGER)


def test_a_mode_needs_inductor():
    with pytest.raises(ValueError, match="needs the inductor backend"):
        C.CompiledBackend(SDPABackend("math"), "default", mode="reduce-overhead",
                          dynamo_backend="eager")


def test_a_second_setting_in_one_process_is_refused_until_reset():
    C.CompiledBackend(SDPABackend("math"), "default", **EAGER)
    C.CompiledBackend(ItemArm(), "default", **EAGER)       # same setting: fine
    with pytest.raises(C.CompiledVariantError, match="would reuse that compiled code"):
        C.CompiledBackend(ItemArm(), "capture", **EAGER)
    C.reset_compiled_state()
    C.CompiledBackend(ItemArm(), "capture", **EAGER)


def test_why_one_setting_per_process():
    """The behaviour the rule rests on (PyTorch 2.14.1, 25 Oct 2026). If a
    later PyTorch guards compiled code on Dynamo's settings, this fails and
    the rule can be relaxed."""
    from torch._dynamo.utils import counters

    def step(x):
        y = x * 2
        return y + y.sum().item()

    x = torch.rand(4, 8)
    counters.clear()
    torch.compile(step, backend="eager")(x)
    after_default = counters["stats"]["unique_graphs"]
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        torch.compile(step, backend="eager")(x)
    assert counters["stats"]["unique_graphs"] == after_default == 2   # reused, still split
    torch._dynamo.reset()
    counters.clear()
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        torch.compile(step, backend="eager")(x)
    assert counters["stats"]["unique_graphs"] == 1                    # fresh: no split


# --- compile events and timed blocks ------------------------------------------------

def test_the_first_call_compiles_and_later_calls_do_not():
    arm = C.CompiledBackend(SDPABackend("math"), "default", **EAGER)
    q, k, v = _qkv()
    out = arm.forward(q, k, v, _cfg())
    assert torch.allclose(out, SDPABackend("math").forward(q, k, v, _cfg()), atol=1e-6)
    assert len(arm.events) == 1 and arm.events[0].graphs >= 1
    arm.forward(*_qkv(1), _cfg())
    assert len(arm.events) == 1 and arm.recompiles == 0
    assert arm.compile_seconds == arm.events[0].seconds > 0


def test_a_per_layer_setting_recompiles_once_and_a_cold_timed_block_is_flagged():
    """Layer 0 compiles; layer 1 recompiles with the layer index made
    symbolic (Dynamo's automatic dynamic), and no later layer compiles. A
    new dtype cannot be made symbolic, so it compiles inside the block."""
    arm = C.CompiledBackend(LayerArm(), "default", **EAGER)
    q, k, v = _qkv()
    for layer in range(4):
        arm.set_layer(layer)
        out = arm.forward(q, k, v, _cfg())
        assert torch.equal(out, v * (layer + 1))
    assert [e.layer for e in arm.events] == [0, 1] and arm.recompiles == 1
    with arm.timed_block() as warm:
        for layer in range(4):
            arm.set_layer(layer)
            arm.forward(q, k, v, _cfg())
    assert not warm.flagged and warm.graphs_inside == 0
    with arm.timed_block() as cold:
        arm.forward(*(t.double() for t in (q, k, v)), _cfg())
    assert cold.flagged and cold.graphs_inside >= 1
    assert arm.events[-1].layer == 3


def test_each_call_traces_under_its_variants_setting():
    q, k, v = _qkv()
    default = C.CompiledBackend(ItemArm(), "default", **EAGER)
    default.forward(q, k, v, _cfg())
    assert default.events[0].graphs > 1                 # split at .item()
    C.reset_compiled_state()
    capture = C.CompiledBackend(ItemArm(), "capture", **EAGER)
    capture.forward(q, k, v, _cfg())
    assert capture.events[0].graphs == 1                # captured


def test_a_fallback_to_eager_is_refused():
    arm = C.CompiledBackend(SDPABackend("math"), "default", **EAGER)
    compile_guard._PROCESS_REASONS.append("hit config.recompile_limit (8)")
    with pytest.raises(compile_guard.CompileFallbackError, match="sdpa@compile-default"):
        arm.forward(*_qkv(), _cfg())


def test_the_recompile_limit_is_raised():
    C.CompiledBackend(SDPABackend("math"), "default", **EAGER)
    assert torch._dynamo.config.recompile_limit == compile_guard.RECOMPILE_LIMIT


def test_hooks_pass_through_to_the_eager_arm():
    inner = LayerArm()
    arm = C.CompiledBackend(inner, "default", **EAGER)
    arm.set_layer(5)
    assert inner.layer_idx == 5 and arm.layer_idx == 5
    inner.last_layer_density.append(1)
    assert arm.last_layer_density == [1]
    arm.reset_density()
    assert inner.last_layer_density == []


def test_inductor_in_the_plans_mode_runs_on_cpu():
    arm = C.CompiledBackend(SDPABackend("math"), "default")
    q, k, v = _qkv()
    out = arm.forward(q, k, v, _cfg())
    assert torch.allclose(out, SDPABackend("math").forward(q, k, v, _cfg()), atol=1e-5)
    assert arm.mode == "reduce-overhead"


# --- the graph-break report ---------------------------------------------------------

def _layers(n=2):
    return [(i, _qkv(i), _cfg()) for i in range(n)]


def test_an_item_read_splits_by_default_and_not_with_the_capture_options():
    default = C.graph_break_report(ItemArm(), _layers(), "default")
    capture = C.graph_break_report(ItemArm(), _layers(), "capture")
    # One break, at the read back. The graph count is more than two because
    # the break is inside a call Dynamo had inlined, which splits the caller
    # too; it is not pinned, since it depends on the PyTorch version.
    assert [r.layer for r in default] == [0, 1]
    assert all(r.graphs > 1 and len(r.breaks) == 1 for r in default)
    br = default[0].breaks[0]
    assert "item" in br.reason.lower()
    assert br.where[-1].startswith("test_compiled_variants.py:") and br.where[-1].endswith("in estimate")
    assert [(r.graphs, r.breaks) for r in capture] == [(1, ()), (1, ())]
    assert not C.same_report(default, capture)


def test_branching_on_the_value_splits_even_with_the_capture_options():
    for setting in ("default", "capture"):
        report = C.graph_break_report(BranchArm(), _layers(1), setting)
        assert report[0].graphs > 1 and len(report[0].breaks) >= 1, setting
    (br,) = C.graph_break_report(BranchArm(), _layers(1), "capture")[0].breaks
    assert br.where[-1].endswith("in estimate")


def test_dense_attention_is_one_graph_either_way():
    a = C.graph_break_report(SDPABackend("math"), _layers(), "default")
    b = C.graph_break_report(SDPABackend("math"), _layers(), "capture")
    assert [(r.graphs, r.breaks) for r in a] == [(1, ()), (1, ())]
    assert C.same_report(a, b)


def test_the_report_sets_each_layer_and_claims_no_setting():
    arm = LayerArm()
    report = C.graph_break_report(arm, _layers(3), "default")
    assert [r.layer for r in report] == [0, 1, 2] and arm.layer_idx == 2
    C.CompiledBackend(SDPABackend("math"), "capture", **EAGER)   # nothing claimed


def test_the_report_refuses_to_run_beside_a_compiled_variant():
    C.CompiledBackend(SDPABackend("math"), "capture", **EAGER)
    with pytest.raises(C.CompiledVariantError, match="process of its own"):
        C.graph_break_report(ItemArm(), _layers(1), "default")
    C.reset_compiled_state()
    C.graph_break_report(ItemArm(), _layers(1), "default")


def test_the_report_leaves_no_compiled_code_cached():
    from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list
    C.graph_break_report(ItemArm(), _layers(1), "default")
    for code in (ItemArm.forward.__code__, ItemArm.estimate.__code__):
        assert _debug_get_cache_entry_list(code) == []


def test_the_report_refuses_an_unknown_setting():
    with pytest.raises(ValueError, match="setting"):
        C.graph_break_report(ItemArm(), _layers(), "fast")


# --- XAttention, over the stand-in package ------------------------------------------

# The stand-in package and its fixture, shared with the T2.17 tests.
import test_xattention_noassert as X                                  # noqa: E402
from test_xattention_noassert import fake_xattn                       # noqa: E402,F401


def _xa_layers(n=2):
    return [(i, X._qkv(i), X._cfg()) for i in range(n)]


def test_the_xattention_report_runs_through_the_arms_own_forward(fake_xattn):
    from attnbench.backends.xattention import XAttentionBackend
    arm = XAttentionBackend(threshold=0.9)
    report = C.graph_break_report(arm, _xa_layers(), "default")
    # The stand-in's asserts are the plain and try/except forms, which the
    # banked check found do not split; nothing else in the path reads back.
    # The one break is the kernel interface, kept out of Dynamo on purpose
    # (rulings, 1 Nov 2026): the code before and after it compiles.
    assert [(r.layer, r.graphs, len(r.breaks)) for r in report] == [(0, 3, 1), (1, 3, 1)]
    for r in report:
        (b,) = r.breaks
        assert b.reason == "Attempted to call function marked as skipped"
        assert b.where[-1].startswith("block_sparse.py:") and \
            b.where[-1].endswith(" in kernel_interface")
    C.check_declared(report, arm, "default")


def test_compiled_masks_agree_with_eager_on_every_prompt_and_layer(fake_xattn):
    from attnbench.backends.xattention import XAttentionBackend
    gate = C.MaskAgreement(XAttentionBackend(threshold=0.9), "default", **EAGER)
    eager = XAttentionBackend(threshold=0.9)
    for prompt in range(2):
        gate.next_prompt()
        for layer in range(2):
            gate.set_layer(layer)
            q, k, v = X._qkv(10 * prompt + layer)
            out = gate.forward(q, k, v, X._cfg())
            assert torch.equal(out, eager.forward(q, k, v, X._cfg()))
    assert [(r.prompt, r.layer, r.equal) for r in gate.records] == \
        [(0, 0, True), (0, 1, True), (1, 0, True), (1, 1, True)]
    assert gate.same_arm and gate.agreement_share == 1.0 and gate.differing_prompts() == ()


def test_a_mask_that_differs_on_one_prompt_makes_a_different_arm(fake_xattn):
    from attnbench.backends.xattention import XAttentionBackend
    gate = C.MaskAgreement(XAttentionBackend(threshold=0.9), "default", **EAGER)
    real = gate._estimate
    gate._estimate = lambda q, k: ~real(q, k) if gate.prompt == 1 else real(q, k)
    for prompt in range(3):
        gate.next_prompt()
        for layer in range(2):
            gate.set_layer(layer)
            gate.forward(*X._qkv(prompt), X._cfg())
    assert not gate.same_arm and gate.differing_prompts() == (1,)
    assert gate.agreement_share == pytest.approx(4 / 6)


def test_mask_agreement_needs_an_estimator_and_prompts():
    with pytest.raises(TypeError, match="no estimator of its own"):
        C.MaskAgreement(SDPABackend("math"), "default", **EAGER)
    gate = C.MaskAgreement(ItemArm(), "default", **EAGER)
    with pytest.raises(RuntimeError, match="next_prompt"):
        gate.forward(*_qkv(), _cfg())
    with pytest.raises(RuntimeError, match="no layers recorded"):
        gate.agreement_share
    assert not gate.same_arm
