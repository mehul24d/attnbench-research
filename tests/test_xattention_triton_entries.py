"""XAttention's Triton kernels as null entries (adviser's rulings of 13 Nov 2026,
items 1 and 2).

P2's relaunch stopped in step 7 (`host_components`, 12 Nov 2026) on
`xattention_dcal_L2048_b1`: "the recording's call 35 is to an entry point in
layer 0 (position 38, 1 operation(s)), and the pass's launching call 35 was
to aten::_to_copy in layer 0". The recording (read-only from the bucket on
the user's yes of 13 Nov 2026; `sequence.json`, sha256 89c14538...cc8, the
file its `recording.json` names) holds positions 38 and 39 of each layer as
`cuLaunchKernelEx` launches with no aten operator around them:
`flat_group_gemm_fuse_reshape_kernel` and `softmax_fuse_block_sum_kernel_causal`,
the estimator's Triton path. No entry nulled them, so they ran for real
inside the null pass and made no launching call, and the pass's call 35 was
position 40's `aten::_to_copy`, the recording's call 37. P1 ran on the L4,
which has no Triton path (`xattention.triton_for_device`), so the check met
these launches for the first time on the A100.

The committed fixture is that sequence up to the first operation of layer 1
(166 of 4360 operations, unchanged;
`tests/fixtures/p2_xattention_triton_2048_layer0.json.gz`). The pass's calls
are the dispatcher's view of the recording, as in
`test_shakedown6_rulings.py`. The structure check (`align_calls`) is not
changed: the fix is that the passes null the kernels (`host_profile.
xattention_triton_entries`, a `NullEntry` launched as `kernel[grid](...)`),
and the check still refuses a pass that misses one.
"""

from __future__ import annotations

import gzip
import json
import sys
import types
from pathlib import Path

import pytest
import torch

from attnbench import cpu_boundary as cb
from attnbench.components import bare_timer as bt
from attnbench.components import host_profile as hp
from attnbench.components import op_sequence as S

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
FIXTURE = HERE / "fixtures" / "p2_xattention_triton_2048_layer0.json.gz"
WHERE_SCALAR = ("aten::where", "aten::scalar_tensor", "aten::fill_")
P2_REFUSAL = ("the recording's call 35 is to an entry point in layer 0 (position 38, "
              "1 operation(s)), and the pass's launching call 35 was to aten::_to_copy "
              "in layer 0")


def _fixture() -> S.OpSequence:
    return S.from_record(json.loads(gzip.decompress(FIXTURE.read_bytes())))


def _dispatcher_view(seq: S.OpSequence):
    """(isolated positions, the pass's launching calls as (operator, layer)),
    each Triton launch nulled: one call per recorded call, at the operator
    the pass meets it at, and an entry point's as None."""
    iso = hp.isolated_positions(seq)
    visible = lambda ch: next((n for n in ch if not cb.composite_implicit(n)), ch[0])
    names = {visible(hp.chain_of(o)) for i, o in enumerate(seq.ops)
             if hp._is_issue(o) and i not in iso and o.chain}
    met = hp.met_at(seq, names | {"aten::scalar_tensor"}, skip=iso)
    issued = []
    for name, layer, pos in hp.recorded_calls(seq, met, skip=iso):
        issued += [("aten::scalar_tensor", layer) for p in pos
                   if seq.ops[p].chain == WHERE_SCALAR]
        issued.append((name, layer))
    return iso, issued


def _log(issued) -> hp.PassLog:
    log = hp.PassLog()
    for k, (name, layer) in enumerate(issued):
        log.issues.append(hp.Issue(name=name, layer=layer, gap_s=(k + 1) * 1e-6,
                                   launch_s=3e-6))
        log.launches.append(3e-6)
    log.isolated.append(5e-6)            # the embedding's gather, kept out
    return log


# --- the recording ----------------------------------------------------------------------

def test_the_fixture_is_p2s_triton_path_with_two_triton_launches_in_layer_0():
    seq = _fixture()
    assert seq.code_path.implementation == "xattn:triton"
    assert seq.code_path.gpu == "NVIDIA A100-SXM4-80GB"
    entries = [(i, o.api, o.kernels[0]) for i, o in enumerate(seq.ops)
               if hp._is_issue(o) and o.aten_op is None]
    assert entries == [(38, "cuLaunchKernelEx", "flat_group_gemm_fuse_reshape_kernel"),
                       (39, "cuLaunchKernelEx", "softmax_fuse_block_sum_kernel_causal")]
    assert all(seq.ops[i].layer == 0 and not seq.ops[i].chain for i, _, _ in entries)
    assert seq.ops[40].chain == ("aten::to", "aten::_to_copy", "aten::copy_")


def test_the_pass_at_the_lock_tag_reproduces_p2s_refusal():
    # The Triton launches ran for real, unnulled, and made no launching call.
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    at_tag = [c for c in issued if c[0] is not None]
    assert len(issued) - len(at_tag) == 2
    with pytest.raises(hp.ProfileMismatch) as e:
        hp.align(seq, _log(at_tag), isolated=iso)
    assert str(e.value) == P2_REFUSAL


def test_with_the_triton_kernels_nulled_the_pass_meets_the_recording_once_each():
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    log = _log(issued)
    host, source, _ = hp.align(seq, log, isolated=iso)
    met = hp.met_at(seq, (e.name for e in log.issues), skip=iso)
    groups = hp.align_calls(seq, met, issued, skip=iso)
    assert [i for i, _, _ in groups] == list(range(len(issued)))
    by_call = {i: share for i, _, share in groups}
    k = issued.index((None, 0))
    assert by_call[k] == (38,) and by_call[k + 1] == (39,)
    assert issued[k + 2] == ("aten::_to_copy", 0) and by_call[k + 2] == (40,)
    assert host[38] == pytest.approx((k + 1) * 1e-6 + 3e-6)
    assert source[38] == source[39] == hp.NULL_PASS
    shares = [p for _, _, share in groups for p in share]
    assert sorted(shares) == sorted(host) and len(shares) == len(set(shares))


@pytest.mark.parametrize("damage", ["one launch missing", "an extra entry", "swapped"])
def test_the_structure_check_still_refuses_a_genuine_mismatch(damage):
    seq = _fixture()
    iso, issued = _dispatcher_view(seq)
    k = issued.index((None, 0))
    bad = list(issued)
    if damage == "one launch missing":
        del bad[k + 1]
    elif damage == "an extra entry":
        bad.insert(k + 3, (None, 0))
    else:
        bad[k + 1], bad[k + 2] = bad[k + 2], bad[k + 1]
    with pytest.raises(hp.ProfileMismatch):
        hp.align(seq, _log(bad), isolated=iso)


# --- the entries ------------------------------------------------------------------------

class FakeTritonKernel:
    """`kernel[grid](*args)`, as a Triton `JITFunction` launches."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, args))


def fake_kernels():
    return types.SimpleNamespace(**{n: FakeTritonKernel() for n in hp.XATTN_TRITON_KERNELS},
                                 __name__="xattn.src.kernels")


def wrapper(mod, x):
    """As `flat_group_gemm_fuse_reshape` does: allocate, then launch."""
    out = torch.empty_like(x)
    mod.flat_group_gemm_fuse_reshape_kernel[(2, 1, 4)](x, out, 8)
    mod.softmax_fuse_block_sum_kernel_causal[(2, 4, 1)](out, x)
    return out


def test_every_triton_kernel_the_pinned_module_defines_is_an_entry():
    mod = fake_kernels()
    entries = hp.xattention_triton_entries(mod)
    assert [e.attr for e in entries] == list(hp.XATTN_TRITON_KERNELS)
    assert hp.XATTN_TRITON_KERNELS[:2] == ("flat_group_gemm_fuse_reshape_kernel",
                                           "softmax_fuse_block_sum_kernel_causal")
    assert all(e.grid and e.module is mod for e in entries)
    assert entries[0].label == "xattn.src.kernels.flat_group_gemm_fuse_reshape_kernel"


def test_the_null_pass_nulls_each_triton_launch_and_restores_the_kernels():
    mod = fake_kernels()
    originals = {n: getattr(mod, n) for n in hp.XATTN_TRITON_KERNELS}
    x = torch.ones(4)
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]
    device = hp.NullDevice(launch=lambda: None, device_type="cpu")
    log = hp.PassLog()
    state = hp._State(clock, clock())
    with hp._entries(hp.xattention_triton_entries(mod), device, log, state, clock):
        out = wrapper(mod, x)
    assert out.shape == x.shape
    assert [(i.name, i.layer) for i in log.issues] == [(None, None), (None, None)]
    assert all(not k.launches for k in originals.values())      # nothing ran for real
    assert {n: getattr(mod, n) for n in hp.XATTN_TRITON_KERNELS} == originals
    wrapper(mod, x)
    assert len(originals["flat_group_gemm_fuse_reshape_kernel"].launches) == 1


def test_a_triton_entry_is_restored_when_the_prefill_raises():
    mod = fake_kernels()
    originals = {n: getattr(mod, n) for n in hp.XATTN_TRITON_KERNELS}
    device = hp.NullDevice(launch=lambda: None, device_type="cpu")
    with pytest.raises(RuntimeError):
        with hp._entries(hp.xattention_triton_entries(mod), device, hp.PassLog(),
                         hp._State(lambda: 0.0, 0.0), lambda: 0.0):
            wrapper(mod, torch.ones(2))
            raise RuntimeError("x")
    assert {n: getattr(mod, n) for n in hp.XATTN_TRITON_KERNELS} == originals


def test_the_bare_pass_launches_each_triton_kernel_for_real_and_prices_it_per_grid():
    mod = fake_kernels()
    kern = mod.flat_group_gemm_fuse_reshape_kernel
    pricer = bt.Pricer(reps=2, sync=lambda: None, min_reps=1)
    out = bt.BarePass()
    state = hp._State(lambda: 0.0, 0.0)
    mode = bt.BareMode(pricer, frozenset(), "cpu", state, out)
    with bt._bare_entries(hp.xattention_triton_entries(mod), mode):
        wrapper(mod, torch.ones(2))
    assert [c.name for c in out.calls] == [None, None]
    grids = [g for g, _ in kern.launches]
    assert grids[0] == (2, 1, 4) and set(grids) == {(2, 1, 4)}
    assert len(grids) == 1 + 1 + 2     # the call, then the bare timer's warm-up and two reps
    labels = {k[0] for k in pricer.prices}
    assert "xattn.src.kernels.flat_group_gemm_fuse_reshape_kernel[(2, 1, 4)]" in labels
    assert "xattn.src.kernels.softmax_fuse_block_sum_kernel_causal[(2, 4, 1)]" in labels
    assert mod.flat_group_gemm_fuse_reshape_kernel is kern


def test_the_residual_times_a_triton_launch_as_one_call():
    from attnbench.components import layer_residual as lr
    mod = fake_kernels()
    calls = []

    class Timer:
        inside = 0

        def call(self):
            import contextlib

            @contextlib.contextmanager
            def cm():
                calls.append("call")
                yield
            return cm()
    with lr._entries(hp.xattention_triton_entries(mod), Timer()):
        wrapper(mod, torch.ones(2))
    assert calls == ["call", "call"]
    assert len(mod.flat_group_gemm_fuse_reshape_kernel.launches) == 1


# --- which cells null them (t31_steps.null_entries) -------------------------------------

def test_only_the_triton_path_nulls_the_triton_kernels_in_the_backend_s_own_module(monkeypatch):
    import t31_steps as st
    bsa = object()
    monkeypatch.setattr(hp, "block_sparse_entry", lambda: bsa)
    path = lambda impl: types.SimpleNamespace(implementation=impl)
    shipped, stripped = fake_kernels(), fake_kernels()
    be = lambda mod: types.SimpleNamespace(kernels_module=lambda: mod)
    assert st.null_entries("dense", path("eager"), None) == ()
    assert st.null_entries("oracle", path("eager"), object()) == (bsa,)
    assert st.null_entries("xattention", path("xattn:torch"), be(shipped)) == (bsa,)
    for mod in (shipped, stripped):
        got = st.null_entries("xattention_noassert", path("xattn:triton"), be(mod))
        assert got[0] is bsa and len(got) == 5 and all(e.module is mod for e in got[1:])
    compiled = types.SimpleNamespace(arm=be(stripped))      # a compiled arm's eager backend
    assert all(e.module is stripped for e in st.null_entries(
        "xattention", path("xattn:triton"), compiled)[1:])
    with pytest.raises(st.StepFailed, match="names no kernels module"):
        st.null_entries("xattention", path("xattn:triton"), object())


# --- the preflight's Triton smoke check (item 3) ----------------------------------------

def test_the_triton_pairings_run_the_smoke_check_and_the_l4_rehearsal_does_not():
    import t61_rehearsal as R
    import t61_session as T61
    import t64_session as P4
    assert T61.triton_pairing(T61.T61) and T61.triton_pairing(P4.T64)
    assert not T61.triton_pairing(R.REHEARSAL)


@pytest.mark.parametrize("rc, written, problem", [
    (0, {"cell": "xattention_dcal_L2048_b1"}, None),
    (1, {"error": "ProfileMismatch: x"}, "ProfileMismatch: x"),
    (0, {"error": "StepFailed: y"}, "StepFailed: y"),
    (1, None, "no triton_smoke.json"),
    (0, None, "no triton_smoke.json"),
])
def test_the_smoke_check_runs_step_5s_cell_in_its_own_process_and_a_failure_is_a_problem(
        tmp_path, monkeypatch, rc, written, problem):
    import t31_steps as st
    import t61_session as T61
    cmds = []

    def run(cmd, **kw):
        cmds.append(cmd)
        if written is not None:
            (tmp_path / "triton_smoke" / "triton_smoke.json").write_text(json.dumps(written))
        return rc
    monkeypatch.setattr(T61, "run", run)
    args = types.SimpleNamespace(session="results/t61_p2_x")
    found, problems = T61.run_triton_smoke(T61.T61, args, tmp_path)
    assert cmds[0][2:5] == [st.TRITON_SMOKE, "--out", str(tmp_path / "triton_smoke")]
    assert cmds[0][-1].endswith("configs/t61/p2_session.json")
    assert (problems == []) if problem is None else (problem in problems[0])


def test_the_smoke_cell_is_xattention_at_2048_and_its_child_is_a_cli_step():
    import t31_steps as st
    assert st.TRITON_SMOKE_CELL == ("xattention", None, 2048, 1)
    assert st.parser().parse_args([st.TRITON_SMOKE, "--out", "o", "--session", "s"]).step \
        == st.TRITON_SMOKE


def _smoke_fakes(monkeypatch, st, impl="xattn:triton", identical=True):
    """`triton_smoke` with the GPU's parts faked: the harness, step 5's cell,
    the recording and the replay; returns what was called."""
    from attnbench.components import recorder
    from attnbench.components import replay
    seen = []
    arm = types.SimpleNamespace(code_path=types.SimpleNamespace(implementation=impl),
                                run_prefill=lambda: None)
    h = types.SimpleNamespace(ready=None, unwrap=lambda: seen.append("unwrap"))
    monkeypatch.setattr(st, "_need_gpu", lambda what: None)
    monkeypatch.setattr(st, "_harness", lambda args: h)
    monkeypatch.setattr(st, "ready_cell", lambda h_, *cell: seen.append(("ready", cell)) or arm)
    monkeypatch.setattr(st, "bare_pricer", lambda: "pricer")
    monkeypatch.setattr(st, "_graph_backend", lambda: "graph")
    monkeypatch.setattr(st, "host_cell", lambda h_, cfg, out, cell, pricer: seen.append(
        ("host_cell", cell, pricer)) or {"dir": "xattention_dcal_L2048_b1",
                                         "host_cost_sources": {hp.NULL_PASS: 3}})
    rec = types.SimpleNamespace(sequence=_fixture(), tape="tape", masks="masks")
    monkeypatch.setattr(recorder, "load_recording", lambda d: seen.append(("load", d)) or rec)
    monkeypatch.setattr(replay, "capture", lambda run, tape, **kw: seen.append(
        ("capture", tape, kw["masks"], kw["backend"])) or "cap")
    monkeypatch.setattr(replay, "mask_agreement",
                        lambda cap: {"identical": identical})
    monkeypatch.setattr(replay, "profile_replay", lambda cap, path: path)
    monkeypatch.setattr(replay, "replay_nodes", lambda trace: [])
    monkeypatch.setattr(replay, "match", lambda seq, nodes: types.SimpleNamespace(
        unmatched_positions=(1, 2)))
    monkeypatch.setattr(replay, "needs_original", lambda seq, m: True)
    return seen


def test_the_smoke_check_runs_step_5s_cell_then_step_6s_mask_check(tmp_path, monkeypatch):
    import t31_steps as st
    seen = _smoke_fakes(monkeypatch, st)
    args = types.SimpleNamespace(out=str(tmp_path), session="s")
    found = st.triton_smoke(args, {})
    assert ("host_cell", st.TRITON_SMOKE_CELL, "pricer") in seen
    assert ("capture", "tape", "masks", "graph") in seen and seen[-1] == "unwrap"
    assert found["implementation"] == "xattn:triton" and found["entry_points"] == 2
    assert found["replay_needs_original"] is True and found["replay_unmatched_positions"] == 2
    assert json.loads((tmp_path / "triton_smoke.json").read_text()) == found


def test_the_smoke_check_refuses_a_cell_off_the_triton_path_before_step_5(tmp_path, monkeypatch):
    import t31_steps as st
    seen = _smoke_fakes(monkeypatch, st, impl="xattn:torch")
    with pytest.raises(st.StepFailed, match="took xattn:torch"):
        st.triton_smoke(types.SimpleNamespace(out=str(tmp_path), session="s"), {})
    assert not any(s[0] == "host_cell" for s in seen if isinstance(s, tuple))


def test_the_smoke_check_refuses_live_masks_that_differ_from_the_bank(tmp_path, monkeypatch):
    import t31_steps as st
    _smoke_fakes(monkeypatch, st, identical=False)
    with pytest.raises(st.StepFailed, match="live masks differ"):
        st.triton_smoke(types.SimpleNamespace(out=str(tmp_path), session="s"), {})
