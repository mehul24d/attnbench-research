#!/usr/bin/env python3
"""The P4 session (T6.4: Sapphire Rapids + H100, FA3 in the dense set).

The runner, its order rules (prediction synced at least 60 s before the
first clocked block, the strict first-clocked-block rule, the late check)
and its dry run are `scripts/t61_session.py`'s; this file holds what differs
on P4:

  - the instance: `a3-highgpu-1g` on DWS Flex-start only, launched from the
    Mac by `scripts/gcp_launch_h100.sh` (the async submit and bounded poll
    of `scripts/_flex_request.sh`, on main from PR #22);
  - `masks`: XAttention's Triton masks recorded untimed on the target, the
    input T2.24 compares with P2's after this session;
  - `fa3_import`, before `dense_select`: `scripts/fa3_import_check.sh` with
    the wheel's sha256 recorded at its build (`--fa3-sha256`), which checks
    the hash, the Python and torch tags, installs it without dependencies,
    imports it and checks the `fa3` backend is available on sm_90. A failure
    stops the session: T6.4 depends on an FA3 backend (T2.16), so K2 does
    not run without it;
  - the null-kernel floor: preflight reports P4's floor beside the two
    published figures the plan names (Section 5.6). The multiple is fixed
    at lock and the gate is read before H3, not here;
  - training: P4 is predicted from P1 and P2, so both banking times are
    needed (`--banked-at P1=... --banked-at P2=...`).

The launch card is `docs/launch_cards/t64_p4_h100.md`.

Usage, on the H100 from the repo root:
    python3 scripts/t64_session.py run --gate-only --fa3-sha256 HEX    # the FA3 gate alone
    python3 scripts/t64_session.py run --fa3-sha256 HEX --banked-at P1=ISO --banked-at P2=ISO
    python3 scripts/t64_session.py run --from STEP --session results/t64_p4_<UTC> ...
    python3 scripts/t64_session.py run --dry-run --fa3-sha256 HEX --session /tmp/t64_dry
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t61_session as S  # noqa: E402

# Section 5.6: the published floors, measured the same way.
PUBLISHED_FLOORS_US = {"TaxBreak, Xeon 8480C with H100": 4.7,
                       "SKIP, Xeon 8468V with H100 PCIe": 2.37}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def floor_findings(row: dict) -> dict:
    us = S._us(row.get("null_launch_s_median"))
    return {"null_kernel_floor": {
        "p4_us": us,
        "ratio_to_published": None if us is None else
        {k: us / v for k, v in PUBLISHED_FLOORS_US.items()},
        "note": "the multiple is fixed at lock; the gate is read before H3 (Section 7)"}}


def _sha(args) -> str:
    sha = getattr(args, "fa3_sha256", None)
    if not sha or not _HEX64.match(sha):
        raise S.StepFailed("--fa3-sha256 must be the 64-hex sha256 recorded when the wheel "
                           f"was built, got {sha!r}")
    return sha


def real_fa3_import(cfg, args) -> dict:
    sha = _sha(args)
    script = S.ROOT / "scripts" / "fa3_import_check.sh"
    if not script.exists():
        raise S.StepFailed("scripts/fa3_import_check.sh is not in the deployed tree "
                           "(it is on main from PR #22)")
    rc = S.run(["bash", str(script), sha])
    if rc != 0:
        raise S.StepFailed(f"fa3_import_check.sh exited {rc}: no FA3 backend, so K2 would "
                           f"record dense_set_incomplete='fa3:...'; T6.4 needs it",
                           {"exit": rc})
    return {"fa3_wheel_sha256": sha, "available": True}


def fake_fa3_import(cfg, args) -> dict:
    sha = _sha(args)
    S.write_json(Path(args.out) / "fa3_import.json", {"fake": True, "sha256": sha})
    return {"fa3_wheel_sha256": sha, "available": True, "fake": True}


def _t64() -> S.SessionConfig:
    reals, fakes = S.common_bodies()
    reals["masks"] = S.not_built("T2.23", "recording XAttention's Triton masks at the grid "
                                          "lengths, untimed, for T2.24")
    fakes["masks"] = S.fake_files("masks.json", recorded=True)
    reals["fa3_import"], fakes["fa3_import"] = real_fa3_import, fake_fa3_import
    cfg = S.SessionConfig(
        task="T6.4", pairing="P4", title="The P4 session (T6.4: Sapphire Rapids + H100)",
        script="t64_session.py", phase_prefix="t64", session_prefix="t64_p4",
        gpu="H100", instance_type="a3-highgpu-1g", provisioning=("FLEX_START",),
        code_path={"gpu": "H100", "implementation": "xattention:triton",
                   "libraries": [["torch", "dry"], ["triton", "dry"], ["flash_attn_3", "dry"]],
                   "python_optimize": 0},
        banked=("P1", "P2"),
        steps=S.COMMON_STEPS_HEAD + (
            S.Step("masks", "T2.23, T2.24", "Triton masks recorded untimed at the grid "
                                            "lengths, for the comparison with P2's"),
        ) + S.COMMON_STEPS_COMPONENTS + (
            S.Step("fa3_import", "T2.16", "FA3 wheel: hash, tags, install, import, backend"),
        ) + S.COMMON_STEPS_TAIL,
        requirements=S.PHASE2 + (
            S.Requirement("T2.23", "the mask recording P4's Triton masks use"),
        ) + S.GATES + (
            S.Requirement("T6.1", "P2 banked: P4 is trained on P1 and P2"),
        ),
        files=S.BUILT_FILES + (
            ("scripts/fa3_import_check.sh", None),
            ("scripts/_fa3_pin.sh", None),
            ("attnbench/backends/fa3.py", None),
            ("attnbench/fa3_wheel.py", None),
        ),
        fakes=fakes, reals=reals, preflight_findings=floor_findings)
    S.validate(cfg)
    return cfg


T64 = _t64()


# --- the FA3 correctness gate (`--gate-only`) ---------------------------------------
#
# Before P4, the H100 is used once for the FA3 correctness gate alone (plan
# Section 8 calendar, 31 Oct 2026): bootstrap with the wheel's sha256, import
# FA3, run the float64 correctness gate, time nothing, halt. The same runner
# and flex-start path as the session, with its own four steps. Every other
# step, timed or not, is refused in this mode: the step subcommand's choices
# are these four, so `step ground_truth --gate-only` is an argparse error.

GATE_FLAG = "--gate-only"
GATE_MAX_SEQ = 4096        # the float64 oracle is allocated up to here (gates.py)
GATE_PROBE = "probe"       # run_probe.py's output, under the gate step's directory


def _gate_provenance(cfg, row: dict) -> list:
    """The session's instance checks on the probe's stamp, less the null-kernel
    burst: the gate times nothing, so it has none."""
    return [p for p in S.check_provenance_row(cfg, row, [])
            if not p.startswith("no null-kernel burst")]


def read_gate(cfg, probe_dir: Path) -> dict:
    """The verdict from `run_probe.py`'s files: FA3 was checked on at least one
    configuration, every check was the float64 one ("exact"), every check
    passed, and the stamp is this instance's. Raises StepFailed otherwise."""
    import pandas as pd
    corr = Path(probe_dir) / "correctness.parquet"
    if not corr.exists():
        raise S.StepFailed(f"run_probe.py wrote no {corr.name}")
    df = pd.read_parquet(corr)
    fa3 = df[df.backend == "fa3"]
    findings = {
        "configs_checked": int(len(fa3)),
        "check_kinds": sorted(set(fa3.check_kind)) if len(fa3) else [],
        "failed": sorted(fa3[~fa3.passed.astype(bool)].config_key) if len(fa3) else [],
        "max_abs_err": None if fa3.empty or fa3.max_abs_err.isna().all()
        else float(fa3.max_abs_err.max()),
        "max_seq": GATE_MAX_SEQ}
    stamp = S.read_json(Path(probe_dir) / "versions.json") or {}
    findings["provenance_problems"] = _gate_provenance(cfg, stamp)
    if fa3.empty:
        raise S.StepFailed("FA3 was checked on no configuration: the probe found it "
                           "unsupported everywhere up to the oracle's length", findings)
    if findings["check_kinds"] != ["exact"]:
        raise S.StepFailed(f"FA3's checks were {findings['check_kinds']}, not only the "
                           f"float64 one ('exact')", findings)
    if findings["failed"]:
        raise S.StepFailed(f"FA3 failed the float64 gate on {len(findings['failed'])} "
                           f"configuration(s)", findings)
    if findings["provenance_problems"]:
        raise S.StepFailed("the probe's stamp is not this session's instance: "
                           + "; ".join(findings["provenance_problems"]), findings)
    return findings


def real_fa3_gate(cfg, args) -> dict:
    """`run_probe.py` on FA3 alone, up to `GATE_MAX_SEQ`: its capability probe
    and its correctness gate against the float64 naive reference. Neither is a
    timing measurement (`run_probe.py`'s docstring)."""
    out = Path(args.out) / GATE_PROBE
    rc = S.run(S.py("run_probe.py", "--out", str(out), "--only-backends", "fa3",
                    "--max-seq", str(GATE_MAX_SEQ), "--no-resume"))
    if rc != 0:
        raise S.StepFailed(f"run_probe.py exited {rc}", {"exit": rc})
    return read_gate(cfg, out)


def fake_fa3_gate(cfg, args) -> dict:
    import pandas as pd
    from attnbench.provenance import PROVISIONING_MODELS
    out = Path(args.out) / GATE_PROBE
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"backend": "fa3", "config_key": f"dry{n}", "passed": True,
                   "check_kind": "exact", "max_abs_err": 1e-3} for n in (1, 2)]
                 ).to_parquet(out / "correctness.parquet")
    S.write_json(out / "versions.json", {
        "cloud_provider": "gcp", "instance_type": cfg.instance_type,
        "instance_generation": "a3", "cpu_microcode": "0xdry",
        "provisioning_model": PROVISIONING_MODELS[cfg.provisioning[0]],
        "provisioning_model_problem": None, "python_optimize": 0, "git_dirty": False,
        "gpu_name": f"fake {cfg.gpu}"})
    return {**read_gate(cfg, out), "fake": True}


def real_halt(cfg, args) -> dict:
    """Self-halt one minute on, so `run_phase.sh` syncs this step first. The
    teardown from the Mac deletes the stopped VM."""
    rc = S.run(["sudo", "shutdown", "-h", "+1"])
    if rc != 0:
        raise S.StepFailed(f"shutdown exited {rc}: the VM is still billing; halt it by hand",
                           {"exit": rc})
    return {"halt_in_minutes": 1}


def fake_halt(cfg, args) -> dict:
    return {"halt_in_minutes": 1, "fake": True}


def _t64_gate() -> S.SessionConfig:
    reals = {"prereqs": S.body_prereqs, "fa3_import": real_fa3_import,
             "fa3_gate": real_fa3_gate, "halt": real_halt}
    fakes = {"prereqs": S.body_prereqs, "fa3_import": fake_fa3_import,
             "fa3_gate": fake_fa3_gate, "halt": fake_halt}
    cfg = S.SessionConfig(
        task="T6.4-gate", pairing="P4", title="The H100 FA3 correctness gate (no timing)",
        script="t64_session.py", phase_prefix="t64gate", session_prefix="t64_fa3_gate",
        gpu="H100", instance_type=T64.instance_type, provisioning=T64.provisioning,
        code_path=T64.code_path, banked=(),
        steps=(S.Step("prereqs", "T2.16", "deployed files present"),
               T64.step("fa3_import"),
               S.Step("fa3_gate", "T2.16", "FA3 against the float64 naive reference, untimed"),
               S.Step("halt", "T6.4", "self-halt; the teardown deletes the VM")),
        requirements=(),
        files=S.BUILT_FILES + (("scripts/fa3_import_check.sh", None),
                               ("scripts/_fa3_pin.sh", None),
                               ("scripts/run_probe.py", None),
                               ("attnbench/backends/fa3.py", None),
                               ("attnbench/fa3_wheel.py", None)),
        fakes=fakes, reals=reals, step_args=(GATE_FLAG,))
    if cfg.names[:1] != ("prereqs",) or cfg.names[-1] != "halt":
        raise RuntimeError("the gate starts with prereqs and ends with halt")
    timed = set(T64.names) - {"prereqs", "fa3_import"}
    if timed & set(cfg.names):
        raise RuntimeError(f"the gate holds session steps {sorted(timed & set(cfg.names))}")
    return cfg


T64_GATE = _t64_gate()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if GATE_FLAG in argv:
        argv.remove(GATE_FLAG)
        return S.main(argv, cfg=T64_GATE)
    return S.main(argv, cfg=T64)


if __name__ == "__main__":
    raise SystemExit(main())
