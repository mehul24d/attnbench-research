"""The adapter's fix for Block-Sparse-Attention's boundary read, and the
kernel-contract check T3.1's preflight runs (researcher's ruling, 2 Nov 2026).
CPU only: the kernel itself is the preflight's to check, on the card."""
from __future__ import annotations

import json
import sys
import types

import pytest
import torch

from attnbench import bsa_contract as bc


def _reference_rows(mask: torch.Tensor) -> list:
    """The library's row form, from its definition: each row's selected key
    blocks in descending order, then -1 to the row's end."""
    out = []
    for row in mask.reshape(-1, mask.shape[-1]).tolist():
        sel = [j for j, x in enumerate(row) if x][::-1]
        out.append(sel + [-1] * (len(row) - len(sel)))
    return out


def _mask(h=3, nb=5, seed=0, full_last=()):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand(1, h, nb, nb, generator=g) < 0.5
    for head in range(h):
        m[0, head, -1, :] = head in full_last
        m[0, head, -1, 0] = True
    return m


def test_the_row_form_is_the_libraries():
    for seed in range(5):
        m = _mask(seed=seed, full_last=(0, 2))
        rows = bc.row_blockmask(m)
        assert rows.dtype == torch.int32 and rows.is_contiguous()
        assert tuple(rows.shape) == tuple(m.shape)
        assert rows.reshape(-1, m.shape[-1]).tolist() == _reference_rows(m)
        shipped = bc.row_blockmask(m, tail_rows=0)
        assert torch.equal(rows, shipped)


def test_the_buffer_has_a_terminator_row_after_it():
    m = _mask(full_last=(0, 1, 2))
    rows = bc.row_blockmask(m)
    n, c = rows.numel(), rows.shape[-1]
    assert bc.storage_elements(rows) == n + c
    tail = torch.empty(0, dtype=torch.int32).set_(rows.untyped_storage())[n:n + c]
    assert tail.tolist() == [-1] * c
    assert bc.storage_elements(bc.row_blockmask(m, tail_rows=0)) == n


def test_the_kernels_walk_reads_past_the_buffer_only_on_a_full_last_row():
    full = _mask(full_last=(2,))                 # the last head's last row
    assert bc.reads_past_storage(bc.row_blockmask(full, tail_rows=0))
    assert not bc.reads_past_storage(bc.row_blockmask(full))
    inner = _mask(full_last=(0, 1))              # other heads only: inside the buffer
    assert not bc.reads_past_storage(bc.row_blockmask(inner, tail_rows=0))
    assert bc.past_row_reads(bc.row_blockmask(inner, tail_rows=0)) >= 2
    none = _mask(full_last=())
    assert bc.past_row_reads(bc.row_blockmask(none, tail_rows=0)) == sum(
        bool(r.all()) for r in none.reshape(-1, none.shape[-1]))


def _fake_library(monkeypatch):
    iface = types.ModuleType("block_sparse_attn.block_sparse_attn_interface")
    iface.convert_blockmask_row_reverse = lambda m, causal=False: bc.row_blockmask(m, tail_rows=0)
    pkg = types.ModuleType("block_sparse_attn")
    pkg.__version__ = "0.0.2"
    pkg.block_sparse_attn_interface = iface
    monkeypatch.setitem(sys.modules, "block_sparse_attn", pkg)
    monkeypatch.setitem(sys.modules, "block_sparse_attn.block_sparse_attn_interface", iface)
    return iface


def test_install_replaces_the_conversion_once_and_keeps_the_shipped_one(monkeypatch):
    iface = _fake_library(monkeypatch)
    shipped = iface.convert_blockmask_row_reverse
    bc.install()
    assert iface.convert_blockmask_row_reverse is bc.row_blockmask
    assert getattr(iface, bc.SHIPPED_ATTR) is shipped
    bc.install()
    assert getattr(iface, bc.SHIPPED_ATTR) is shipped
    assert bc.installed()


def test_every_kernel_call_installs_it(monkeypatch):
    iface = _fake_library(monkeypatch)
    seen = []
    sys.modules["block_sparse_attn"].block_sparse_attn_func = \
        lambda *a, **k: seen.append(iface.convert_blockmask_row_reverse)
    from attnbench.backends import block_sparse
    block_sparse.kernel_interface(1, 2)
    assert seen == [bc.row_blockmask]


def test_bsa_prefill_installs_it_with_the_real_kernel(monkeypatch):
    iface = _fake_library(monkeypatch)
    seen = []

    def kernel(q, *a, **k):
        seen.append(iface.convert_blockmask_row_reverse)
        return q
    sys.modules["block_sparse_attn"].block_sparse_attn_func = kernel
    from attnbench import kernels
    q = torch.zeros(1, 2, 128, 8)
    kernels.bsa_prefill(q, q, q, torch.ones(1, 2, 1, 1, dtype=torch.bool))
    assert seen == [bc.row_blockmask]


def test_the_identity_names_the_version_the_interface_and_the_fix(monkeypatch, tmp_path):
    f = tmp_path / "block_sparse_attn_interface.py"
    f.write_bytes(b"x")
    monkeypatch.setattr(bc, "_interface_path", lambda: f)
    monkeypatch.setattr(bc, "_version", lambda: "0.0.2")
    ident = bc.identity()
    assert ident["block_sparse_attn"] == "0.0.2"
    assert ident["bsa_interface_sha256"] == bc._sha256(f)
    assert ident["bsa_row_mask_fix"] == bc.ROW_MASK_FIX
    with pytest.raises(bc.ContractError, match="interface"):
        bc.require_pinned(ident)
    ok = dict(ident, bsa_interface_sha256=bc.INTERFACE_SHA256)
    bc.require_pinned(ok)
    with pytest.raises(bc.ContractError, match="version"):
        bc.require_pinned(dict(ok, block_sparse_attn="0.0.3"))


def test_provenance_records_it():
    from attnbench import provenance
    fields = {f for f in provenance.Provenance.__dataclass_fields__}
    assert {"block_sparse_attn", "bsa_interface_sha256", "bsa_row_mask_fix"} <= fields
    assert provenance.capture().bsa_row_mask_fix == bc.ROW_MASK_FIX


def test_synthetic_masks_have_the_shapes_the_check_needs():
    m = bc.synthetic_mask(12, 16, full_last=range(12), seed=0)
    assert m.dtype == torch.bool and tuple(m.shape) == (1, 12, 16, 16)
    assert m[0, :, -1, :].all()
    assert m[0, :, 8, :].all()                         # a full row inside the buffer too
    part = bc.synthetic_mask(12, 16, full_last=range(11), seed=0)
    assert part[0, :11, -1, :].all() and not part[0, 11, -1, :].all()
    assert part[0, :, torch.arange(16), torch.arange(16)].all()   # every diagonal block


def test_the_check_refuses_when_the_walk_leaves_the_buffer(monkeypatch):
    """The emulated walk is the CPU's positive control: with the shipped
    conversion it must find the read past the buffer, with the fix it must not."""
    rep = bc.walk_check(lengths=(1951,), heads=4)
    assert rep["shipped_reads_past_buffer"] is True
    assert rep["fixed_reads_past_buffer"] is False
    monkeypatch.setattr(bc, "TAIL_ROWS", 0)
    with pytest.raises(bc.ContractError, match="past"):
        bc.walk_check(lengths=(1951,), heads=4)


def test_the_check_refuses_a_vacuous_positive_control(monkeypatch):
    monkeypatch.setattr(bc, "reads_past_storage", lambda rows: False)
    with pytest.raises(bc.ContractError, match="positive control"):
        bc.walk_check(lengths=(1951,), heads=4)


def test_memcheck_summary_is_parsed():
    assert bc.memcheck_errors("x\n========= ERROR SUMMARY: 130 errors\n") == 130
    assert bc.memcheck_errors("========= ERROR SUMMARY: 0 errors\n") == 0
    assert bc.memcheck_errors("no summary") is None


def test_the_sanitizer_verdicts():
    good = {"shipped": {"errors": 4, "rc": 99}, "fixed": {"errors": 0, "rc": 0}}
    bc.judge_sanitizer(good)
    with pytest.raises(bc.ContractError, match="fixed"):
        bc.judge_sanitizer(dict(good, fixed={"errors": 2, "rc": 99}))
    with pytest.raises(bc.ContractError, match="positive control"):
        bc.judge_sanitizer(dict(good, shipped={"errors": 0, "rc": 0}))
    with pytest.raises(bc.ContractError, match="no summary"):
        bc.judge_sanitizer(dict(good, fixed={"errors": None, "rc": 1}))


def test_the_sanitizer_command_turns_the_caching_allocator_off(tmp_path):
    cmd, env = bc.sanitizer_command("/x/compute-sanitizer", "fixed", tmp_path / "m.log")
    assert cmd[:3] == ["/x/compute-sanitizer", "--tool", "memcheck"]
    assert "--error-exitcode" in cmd and cmd[-2:] == ["--variant", "fixed"]
    assert env["PYTORCH_NO_CUDA_MEMORY_CACHING"] == "1"


def test_the_output_comparisons(monkeypatch):
    """`compare` is bitwise for the padding questions and toleranced against
    the reference, and the tolerance must separate a one-block change."""
    a = torch.zeros(4)
    assert bc.max_abs_diff(a, a) == 0.0
    assert bc.max_abs_diff(a, a + 0.5) == 0.5
    bc.judge_outputs({"vs_reference": [0.01], "tolerance_separates": [0.4],
                      "tail_value_effect": [0.0], "padded_vs_unpadded": [0.0]})
    for key, bad in (("vs_reference", [0.5]), ("tolerance_separates", [0.01]),
                     ("tail_value_effect", [1e-3]), ("padded_vs_unpadded", [1e-3])):
        rep = {"vs_reference": [0.01], "tolerance_separates": [0.4],
               "tail_value_effect": [0.0], "padded_vs_unpadded": [0.0]}
        rep[key] = bad
        with pytest.raises(bc.ContractError, match=key):
            bc.judge_outputs(rep)


def test_the_cpu_boundary_has_the_interface_and_the_check_passes_behind_it(tmp_path):
    from attnbench import cpu_boundary
    with cpu_boundary.installed(xattn=False):
        iface = sys.modules["block_sparse_attn.block_sparse_attn_interface"]
        rep = bc.contract_check(tmp_path, sanitizer=None, lengths=(1951,), heads=4)
        assert iface.convert_blockmask_row_reverse is bc.row_blockmask
    saved = json.loads((tmp_path / "kernel_contract.json").read_text())
    assert saved["status"] == "passed"
    assert saved["sanitizer"]["status"].startswith("not run")
    assert saved["walk"]["shipped_reads_past_buffer"] is True
    assert rep["outputs"]["padded_vs_unpadded"] == [0.0]


# --- T3.1's preflight ------------------------------------------------------------------

def _steps():
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts"))
    import t31_steps
    return t31_steps


def test_the_preflight_runs_the_contract_and_a_refusal_fails_the_step(tmp_path, monkeypatch):
    st = _steps()
    monkeypatch.setattr(st, "_t227", lambda *a: None)
    seen = []

    def check(out, *, sanitizer):
        seen.append(out)
        return {"status": "passed", "outputs": {"vs_reference": [0.01]},
                "sanitizer": {"status": "ran", "shipped": {"errors": 3}, "fixed": {"errors": 0}}}
    monkeypatch.setattr(bc, "contract_check", check)
    args = types.SimpleNamespace(out=str(tmp_path), dry_run=False, cpu_boundary=True)
    cfg = st.load_config(None, dry_run=True)
    res = st.step_preflight(args, cfg)
    assert seen == [tmp_path / "kernel_contract"]
    assert res["kernel_contract"]["status"] == "passed"

    def refuse(out, *, sanitizer):
        out.mkdir(parents=True, exist_ok=True)
        (out / "kernel_contract.json").write_text(json.dumps({"status": "failed",
                                                              "reason": "memcheck fixed"}))
        raise bc.ContractError("memcheck fixed: 2 errors with the fix")
    monkeypatch.setattr(bc, "contract_check", refuse)
    with pytest.raises(st.StepFailed, match="kernel contract: memcheck fixed") as e:
        st.step_preflight(args, cfg)
    assert e.value.findings["status"] == "failed"


def test_a_dry_run_records_that_the_contract_did_not_run(tmp_path, monkeypatch):
    st = _steps()
    monkeypatch.setattr(st, "_t227", lambda *a: None)
    monkeypatch.setattr(bc, "contract_check", lambda *a, **k: pytest.fail("ran"))
    args = types.SimpleNamespace(out=str(tmp_path), dry_run=True, cpu_boundary=False)
    res = st.step_preflight(args, st.load_config(None, dry_run=True))
    assert res["kernel_contract"] == {"status": "not run: dry run"}


def test_a_test_stand_in_for_the_library_is_left_alone(monkeypatch, tmp_path):
    stand_in = types.ModuleType("block_sparse_attn")
    stand_in.block_sparse_attn_func = lambda *a, **k: "ran"
    monkeypatch.setitem(sys.modules, "block_sparse_attn", stand_in)
    monkeypatch.delitem(sys.modules, "block_sparse_attn.block_sparse_attn_interface",
                        raising=False)
    assert bc.install() is None and not bc.installed()
    from attnbench.backends import block_sparse
    assert block_sparse.kernel_interface() == "ran"
    with pytest.raises(bc.ContractError, match="not installed"):
        bc.contract_check(tmp_path, sanitizer=None, lengths=(1951,), heads=2)


def test_the_exposure_count_finds_the_rows_the_kernel_reads_past():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import bsa_row_exposure as ex
    a = _mask(h=3, nb=4, full_last=(2,)).numpy()          # last head: past the buffer
    b = _mask(h=3, nb=4, full_last=(0,)).numpy()          # inside it only
    rep = ex.exposure([a, b])
    assert rep["layers"] == 2 and rep["pairs"] == 6
    assert rep["out_of_buffer_layers"] == [0]
    assert rep["full_last_row"] == [(0, 2), (1, 0)]
    assert rep["full_rows_any"] == int(a.all(-1).sum() + b.all(-1).sum())


def test_the_code_path_names_the_fix(monkeypatch):
    _fake_library(monkeypatch)
    from attnbench.components import shakedown_arms
    libs = dict(shakedown_arms.libraries())
    assert libs["block_sparse_attn"] == "0.0.2"
    assert libs["bsa_row_mask_fix"] == bc.ROW_MASK_FIX
