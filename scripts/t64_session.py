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


def main(argv=None) -> int:
    return S.main(argv, cfg=T64)


if __name__ == "__main__":
    raise SystemExit(main())
