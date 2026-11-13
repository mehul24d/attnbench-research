"""The block-sparse kernel's contract in P2's and P4's preflight (the
adviser's rulings dated 13 Nov 2026, second set, item 3; audit item E,
`docs/gpu_path_audit.md`).

P1's preflight ran the contract (`t31_steps.kernel_contract`); P2's and P4's
did not, so a read past the row buffer whose value is discarded (1 Nov 2026)
could pass on the A100 or H100 unseen. The session now runs the same check in
its own process before the smoke check, and refuses a refusal, a missing
result or a result that is not clean: the padded and the library's own
buffer bitwise equal over every layer of a real XAttention prompt.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from attnbench import bsa_contract as bc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import t61_session as T61  # noqa: E402
import t64_session as P4  # noqa: E402

PAIRINGS = {"P2": T61.T61, "P4": P4.T64}
CLEAN = {"status": "passed", "clean": True,
         "real_prompt": {"layers": 28, "unpadded": "compared", "padded_vs_unpadded": 0.0},
         "sanitizer": {"status": "ran", "shipped": {"errors": 130}, "fixed": {"errors": 0}}}


def _run(written, rc=0, cmds=None):
    def run(cmd, **kw):
        if cmds is not None:
            cmds.append(cmd)
        d = Path(cmd[cmd.index("--out") + 1])
        if written is not None:
            (d / "kernel_contract.json").write_text(json.dumps(written))
        return rc
    return run


@pytest.mark.parametrize("pairing", sorted(PAIRINGS))
def test_p2_and_p4_run_the_contract_in_its_own_process(tmp_path, monkeypatch, pairing):
    import t31_xa_diag
    cmds = []
    monkeypatch.setattr(T61, "run", _run(CLEAN, cmds=cmds))
    monkeypatch.setattr(t31_xa_diag, "find_sanitizer", lambda: "/cs")
    found, problems = T61.run_kernel_contract(PAIRINGS[pairing], tmp_path)
    assert problems == []
    assert cmds == [[sys.executable, "-m", "attnbench.bsa_contract", "check", "--out",
                     str(tmp_path / "kernel_contract"), "--sanitizer", "/cs"]]
    assert found["status"] == "passed" and found["clean"] is True
    assert found["real_prompt"]["layers"] == 28
    assert found["sanitizer"]["shipped"] == {"errors": 130}


def test_a_pairing_off_the_triton_path_runs_no_contract(tmp_path, monkeypatch):
    import t61_rehearsal as R
    monkeypatch.setattr(T61, "run", lambda cmd, **kw: pytest.fail("ran"))
    assert T61.run_kernel_contract(R.REHEARSAL, tmp_path) == (None, [])


@pytest.mark.parametrize("written, rc, problem", [
    ({"status": "failed", "reason": "real_prompt padded_vs_unpadded: the output moved by "
                                    "0.0078"}, 1, "refused (rc 1): real_prompt padded_vs"),
    ({"status": "failed", "reason": "memcheck fixed: 2 errors with the fix"}, 0,
     "refused (rc 0): memcheck fixed"),
    (None, 1, "no kernel_contract.json"),
    (None, 0, "no kernel_contract.json"),
    ({**CLEAN, "clean": False,
      "real_prompt": {"unpadded": "not compared: the library's own buffer exited -11 "
                                  "(the known read)", "padded_vs_unpadded": None}}, 0,
     "not clean: padded against unpadded is None (not compared"),
])
def test_a_violated_or_unproven_contract_is_a_problem(tmp_path, monkeypatch, written, rc, problem):
    monkeypatch.setattr(T61, "run", _run(written, rc))
    for cfg in PAIRINGS.values():
        found, problems = T61.run_kernel_contract(cfg, tmp_path)
        assert len(problems) == 1 and problem in problems[0], problems
        (tmp_path / "kernel_contract" / "kernel_contract.json").unlink(missing_ok=True)


def _fake_real(diff):
    """`tests/test_bsa_contract.py`'s runner: every variant's layer outputs
    equal, except `diff[variant]` added to one layer."""
    def runner(variants, out):
        for v in variants:
            layers = [torch.ones(1, 2, 4, 3) * i for i in range(3)]
            layers[1] = layers[1] + diff.get(v, 0.0)
            torch.save({"prompt_id": "qa_1_2048_0", "length": 1951, "layers": layers},
                       Path(out) / f"real_{v}.pt")
        return 0
    return runner


@pytest.mark.parametrize("diff, problem", [
    ({}, None),
    ({"unpadded": 2.0 ** -7}, "real_prompt padded_vs_unpadded: the output moved by 0.00781"),
    ({"tail0": 2.0 ** -7}, "real_prompt tail_value_effect"),
])
def test_the_real_check_behind_the_cpu_boundary_refuses_a_padded_unpadded_difference(
        tmp_path, monkeypatch, diff, problem):
    """The contract's own code (`bsa_contract.main`, the session's command)
    on the CPU boundary's kernel, with the real prompt's outputs moved by one
    bf16 step in one layer: the session refuses it."""
    from attnbench import cpu_boundary
    orig = bc.real_prompt_check
    monkeypatch.setattr(bc, "real_prompt_check",
                        lambda out, runner: orig(out, runner=_fake_real(diff)))
    monkeypatch.setattr(T61, "run", lambda cmd, **kw: bc.main(cmd[3:]))
    with cpu_boundary.installed(xattn=False):
        found, problems = T61.run_kernel_contract(T61.T61, tmp_path)
    if problem is None:
        assert problems == [] and found["status"] == "passed" and found["clean"] is True
    else:
        assert len(problems) == 1 and problem in problems[0], problems
        assert found["status"] == "failed"


def test_a_refused_contract_fails_the_preflight_before_the_smoke_check(tmp_path, monkeypatch):
    monkeypatch.setattr(T61, "run", lambda cmd, **kw: 0)
    monkeypatch.setattr(T61, "_host_speed_row", lambda out, tag: {})
    monkeypatch.setattr(T61, "check_provenance_row", lambda cfg, row, integrity: [])
    monkeypatch.setattr(T61, "import_check", lambda cfg: ([], {"required": [],
                                                              "kernel_problems": []}))
    monkeypatch.setattr(T61, "lightgbm_problem", lambda: None)
    monkeypatch.setattr(T61, "run_gpu_variants", lambda out: ({}, []))
    monkeypatch.setattr(T61, "run_kernel_contract",
                        lambda cfg, out: ({"status": "failed"}, ["the kernel contract refused"]))
    monkeypatch.setattr(T61, "run_preflight_smoke",
                        lambda cfg, args, out: pytest.fail("the smoke ran"))
    args = types.SimpleNamespace(out=str(tmp_path), session="s")
    for cfg in PAIRINGS.values():
        with pytest.raises(T61.StepFailed) as e:
            T61.real_preflight(cfg, args)
        assert "the kernel contract refused" in e.value.findings["problems"]
        assert e.value.findings["kernel_contract"] == {"status": "failed"}
        assert e.value.findings["smoke"] is None
