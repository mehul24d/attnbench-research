"""Every arm's patches resolve to the modules the arm actually calls (the
adviser's rulings dated 13 Nov 2026, item 2).

P2 at `lock-2026-11-13` stopped in step 7 on `xattention_noassert_dcal_L2048_b1`
with the refusal of 12 Nov 2026 that the Triton null entries were meant to
fix: the entries nulled `xattn.src.kernels`, and the asserts-removed arm
launches the kernels of its stripped copy, `attnbench_xattn_noassert.src.
kernels`, a different module object (`assert_strip`). Its unit test passed
because it stood a fake module in for both arms.

So these tests run the real arms, behind the CPU boundary on the pinned
x-attention checkout, and read what each arm calls:

  - **Triton launches.** Each XAttention arm's estimator is sent down its
    Triton path (the device named as an A100, which is the authors' own test)
    with every Triton kernel of both packages replaced by one that names its
    module and stops. The arm's launch must come from its backend's
    `kernels_module`, and `t31_steps.null_entries` must null exactly that
    module's kernels. Then a real null pass (`host_profile._entries`) over the
    same call must meet the launch as a null entry, not run it.
  - **Patch sites in one forward.** For every arm of P2's and P4's configs,
    one forward with the session's patch sites on: the mask capture and the
    kernel-input recorder (both on `block_sparse_attn.block_sparse_attn_func`,
    shared by every sparse arm) see one call per layer, and the ground-truth
    probe (`ScoringProbe`) logs a score event for each deployable arm. A
    dense forward reaches none of them.

The block-sparse extension call (`host_profile.block_sparse_entry`) is not
exercised here: the CPU stub's kernel is one custom operator, which the null
pass's dispatch mode nulls first, as it does the pinned build's; the
pre-launch CPU pass runs that null pass on every arm.
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch

from attnbench import cpu_boundary as cb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not cb.have_xattn(),
                                reason="needs the x-attention clone (ATTNBENCH_XATTN_DIR)")

XATTN_ARMS = ("xattention", "xattention_noassert")
CONFIGS = ("configs/t61/p2_session.json", "configs/t64/p4_session.json")
A100 = "NVIDIA A100-SXM4-80GB"


class Launched(Exception):
    """A Triton kernel ran for real; `args[0]` is its module, `args[1]` its name."""


class Stop(Exception):
    """The null pass's device was asked for its null launch."""


class RealKernel:
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            raise Launched(self.module, self.name)
        return launch


@pytest.fixture(scope="module")
def boundary():
    """(harness, shipped kernels module, stripped kernels module), behind the
    CPU boundary, with the device named as an A100 so the estimators take
    their Triton path."""
    with cb.installed():
        import attnbench_xattn_noassert.src.kernels as stripped
        import xattn.src.kernels as shipped
        from attnbench.components import shakedown_arms as sa
        h = sa.Harness(device="cpu", dry_run=True, score_cache_dir=tempfile.mkdtemp())
        yield h, shipped, stripped


@pytest.fixture
def real_kernels(boundary, monkeypatch):
    """Every Triton kernel of both packages replaced by one that raises
    `Launched` (no Triton runs on a CPU)."""
    from attnbench.components import host_profile as hp
    _, shipped, stripped = boundary
    for mod in (shipped, stripped):
        for n in hp.XATTN_TRITON_KERNELS:
            monkeypatch.setattr(mod, n, RealKernel(mod, n))
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda *a, **k: types.SimpleNamespace(name=A100))


def _estimate(h, arm):
    from attnbench.components import shakedown_arms as sa
    be = h._backend(sa.ARMS[arm])
    be.use_triton = True
    be.set_layer(0)
    c = h.model.config
    heads, dim = c.num_attention_heads, c.hidden_size // c.num_attention_heads
    q, k = torch.randn(1, heads, 1024, dim), torch.randn(1, heads, 1024, dim)
    return be, (lambda: be.estimate(q, k))


def _triton_path():
    import t31_steps as st
    return types.SimpleNamespace(implementation=st.XATTN_TRITON)


def test_the_two_xattention_arms_launch_from_different_kernel_modules(boundary):
    _, shipped, stripped = boundary
    assert shipped is not stripped
    assert (shipped.__name__, stripped.__name__) == (
        "xattn.src.kernels", "attnbench_xattn_noassert.src.kernels")


@pytest.mark.parametrize("arm", XATTN_ARMS)
def test_each_arm_launches_its_triton_kernels_from_its_backend_s_kernels_module(
        boundary, real_kernels, arm):
    h, _, _ = boundary
    be, run = _estimate(h, arm)
    with pytest.raises(Launched) as e:
        run()
    module, name = e.value.args
    assert module is be.kernels_module()
    assert name == "flat_group_gemm_fuse_reshape_kernel"


@pytest.mark.parametrize("arm", XATTN_ARMS)
def test_null_entries_null_the_kernels_module_the_arm_launches_from(boundary, real_kernels, arm):
    import t31_steps as st
    from attnbench.components import host_profile as hp
    h, _, _ = boundary
    be, run = _estimate(h, arm)
    with pytest.raises(Launched) as e:
        run()
    launched_from = e.value.args[0]
    triton = [x for x in st.null_entries(arm, _triton_path(), be) if x.grid]
    assert [x.attr for x in triton] == list(hp.XATTN_TRITON_KERNELS)
    assert all(x.module is launched_from for x in triton), \
        f"{arm} launches from {launched_from.__name__}; its null entries patch " \
        f"{sorted({x.module.__name__ for x in triton})}"


@pytest.mark.parametrize("arm", XATTN_ARMS)
def test_the_null_pass_meets_each_arm_s_first_triton_launch_as_a_null_entry(
        boundary, real_kernels, arm):
    import t31_steps as st
    from attnbench.components import host_profile as hp
    h, _, _ = boundary
    be, run = _estimate(h, arm)

    def null_launch():
        raise Stop
    log = hp.PassLog()
    device = hp.NullDevice(launch=null_launch, device_type="cpu")
    with hp._entries(st.null_entries(arm, _triton_path(), be), device, log,
                     hp._State(lambda: 0.0, 0.0), lambda: 0.0):
        with pytest.raises(Stop):       # `Launched` here: the kernel ran for real
            run()
    mod = be.kernels_module()
    assert all(isinstance(getattr(mod, n), RealKernel) for n in hp.XATTN_TRITON_KERNELS)


# --- every patch site, one forward per arm ----------------------------------------------

def _config_arms() -> list:
    arms = []
    for c in CONFIGS:
        for a in json.loads((ROOT / c).read_text())["arms"]:
            if a not in arms:
                arms.append(a)
    return arms


def test_p2_and_p4_run_the_same_five_arms():
    assert _config_arms() == ["dense", "oracle", "meanpool", *XATTN_ARMS]


@pytest.mark.parametrize("arm", _config_arms())
def test_every_patch_site_sees_the_arm_s_own_calls_in_one_forward(boundary, arm):
    from attnbench import densities as D
    from attnbench import ground_truth as gt
    from attnbench.components.mask_bank import MaskCapture
    h, _, _ = boundary
    names = ["dense"] if arm == "dense" else ["dense", arm]
    wrapped, ids, _, arms, score_kwargs = h.timing_arms(
        names, 1024, densities=[None] + [None if arm.startswith("xattention") else 0.25][
            :len(names) - 1], batch=1)
    a = arms[-1]
    log = gt.EventLog()
    try:
        with gt.probe_for(log, a):
            kw = gt.untimed_setup(wrapped, ids, a, score_kwargs=score_kwargs)
            log.phase = gt.WARMUP
            with MaskCapture() as cap, D.KernelInputRecorder() as rec:
                wrapped.run_measured(ids, a.backend, cfg=a.cfg, logits_to_keep=1, **kw)
    finally:
        h.unwrap()
    layers = int(h.model.config.num_hidden_layers)
    scores = [e for e in log.events if e.kind == "score" and e.phase == gt.WARMUP]
    if arm == "dense":
        assert (len(cap.calls), len(rec.per_call), len(scores)) == (0, 0, 0)
        return
    assert len(cap.calls) == layers, f"{arm}: the mask capture saw {len(cap.calls)} calls"
    assert len(rec.per_call) == layers, f"{arm}: the recorder saw {len(rec.per_call)} calls"
    if a.deployable:
        assert scores, f"{arm} is deployable and the ground-truth probe saw no score call"
