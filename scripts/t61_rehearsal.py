#!/usr/bin/env python3
"""The L4 rehearsal of the P2 session runner (adviser's ruling of 10 Nov
2026): `scripts/t61_session.py`'s steps, order rules, blocks and exit, run on
the L4 before an A100 session is spent. **Exploratory: no row it produces
enters training, the floors or any claim** (`EXPLORATORY_NOTE`, written into
every session's manifest and `EXPLORATORY.md`). The card is
`docs/launch_cards/t61_rehearsal_l4.md`.

What differs from P2 (`t61_session.T61`), and why each is not the code under
test:

  - the instance: the L4 (`g2-standard-8`, on-demand), so the preflight's
    provenance checks expect it;
  - the pairing is `P2-R`: its prediction goes to `predictions/P2-R/`, its
    block spec is `configs/t61/rehearsal_lock.json`'s, and it is trained on
    P1's training file as P2 is;
  - `prereqs` checks the deployed files and records the tasks P2's own check
    would refuse on (`p2_would_refuse`), without refusing: the rehearsal is
    what some of those tasks wait for (T2.17, T2.26);
  - `mask_gate`: the L4 has no Triton path (`xattention.triton_for_device`:
    the device name must contain "100"), and P1 is this card. So the step
    records XAttention's PyTorch path on P1's prompts through P2's recorder
    (`record_both_paths`) and compares it with P1's banked masks through P2's
    comparison and gate (`mask_gate.compare_banks`, `mask_gate.gate`; the
    recording labelled `@rehearsal`, since the gate refuses one code path
    against itself) at the ±0.01 density bound (`densities.DENSITY_FLAG_TOLERANCE`; adviser's ruling
    3 of the eleventh 9 Nov 2026 paste). On one card and one path the masks
    should agree: a failure is a defect in the recording or comparison code.
    Block agreement is descriptive. `p2_mask_gate`'s own two-path verdict
    cannot run here and is not rehearsed;
  - `predict`: the learned predictor's configuration is a lock value, unset,
    so the rehearsal fits the working one, configuration 4 of
    `docs/learned_cv_p1_20261109.json` (provisional), and says so;
  - the grid is 2048 and 8192 at batch 1 (`configs/t61/rehearsal_l4.json`);
    the measurement protocol is the working values', unchanged.

Two more commands, run after a block, never inside one:

  - `compiled --session DIR --out DIR`: the compiled groups' ground truth at
    2048 (scope v) through P2's harness (`t61_session.real_harness`) on
    `configs/t61/rehearsal_compiled_l4.json`, against block b1's committed
    prediction and dense choice; the recompile gate and the compiled checks
    are the harness's own;
  - `trace-sizes --session DIR`: each Kineto trace's size, by arm and cell,
    into `trace_sizes.json` (scope i), so the logs-and-manifests pull carries
    them.

Usage, on the L4 from the repo root, under `scripts/session_exit.sh`:
    python3 scripts/t61_rehearsal.py run --lock configs/t61/rehearsal_lock.json --block b1 \\
        --banked-at P1=<ISO>
    python3 scripts/t61_rehearsal.py compiled --session results/<b1> --out results/<b1>/compiled
    python3 scripts/t61_rehearsal.py trace-sizes --session results/<b1>
    python3 scripts/t61_rehearsal.py run --dry-run --lock configs/t61/rehearsal_lock.json --block b1 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import t61_session as S  # noqa: E402

PAIRING = "P2-R"
LEARNED_CV = ROOT / "docs" / "learned_cv_p1_20261109.json"
COMPILED_CONFIG = "configs/t61/rehearsal_compiled_l4.json"
LOCK = "configs/t61/rehearsal_lock.json"
EXPLORATORY_NOTE = ("Exploratory (adviser's ruling of 10 Nov 2026): the L4 rehearsal of the P2 "
                    "runner. No row of this session enters training, the floors, Gate A, the "
                    "lock or any claim, and its prediction is not P2's.")


def rehearsal_prereqs(cfg, args) -> dict:
    """The deployed files, checked as P2 checks them; P2's unmet tasks are
    recorded, not refused on."""
    text = (S.ROOT / "docs" / "RESEARCH_PLAN.md").read_text()
    would = S.unmet(S.T61.requirements, S.plan_statuses(text))
    findings = S.body_prereqs(cfg, args)
    findings["p2_would_refuse"] = would
    return findings


def density_bound() -> float:
    from attnbench.densities import DENSITY_FLAG_TOLERANCE
    return float(DENSITY_FLAG_TOLERANCE)


REHEARSAL_LABEL = "@rehearsal"


def relabelled(bank) -> list:
    """The rehearsal's recording under its own implementation name
    (`xattn:torch@rehearsal`): P1's bank is this card's same path, and
    `mask_gate.compare_records` refuses two records of one code path (it
    guards against a bank compared with itself). Only the label changes,
    never a mask."""
    return [dataclasses.replace(r, code_path=dataclasses.replace(
        r.code_path, implementation=r.code_path.implementation + REHEARSAL_LABEL))
        for r in bank]


def rehearsal_mask_gate(cfg, args, *, record=None) -> dict:
    """P1's banked XAttention masks against the same prompts recorded again
    on this card's PyTorch path, through P2's recorder, comparison and gate.
    Like P2's, a failed verdict does not stop the session; the masks and
    comparisons stay recorded."""
    from dataclasses import asdict
    from attnbench.components import mask_gate as MG
    session, out = Path(args.session), Path(args.out)
    p1 = S.banked_masks(session / S.P1_MASKS)
    cells = [(r.code_path.arm, budget, r.batch, r.prompt_id) for budget, r in p1]
    banks = (record or S.record_both_paths)(args, cells, out / "masks", paths=("torch",))
    ref = [r for _, r in p1]
    bound = density_bound()
    try:
        comps = MG.compare_banks(ref, relabelled(banks["torch"]))
        S.write_json(out / "mask_comparisons.json",
                     {"torch_vs_p1_same_card": S._comparisons(comps)})
        res = MG.gate(comps, MG.Bounds(max_density_difference=bound))
    except MG.MaskGateError as e:
        raise S.StepFailed(f"the mask gate cannot compare these banks: {e}") from e
    verdict = "pass" if res.passed else "fail"
    S.write_json(out / "mask_gate.json", {"verdict": verdict, "rehearsal": True,
                                          "candidate_label": REHEARSAL_LABEL, **asdict(res)})
    return {"verdict": verdict, "cells": len(cells), "bound": bound,
            "failures": len(res.failures), "min_agreement": res.min_agreement,
            "identical_layers": res.identical_layers, "n_layers": res.n_layers,
            "max_abs_density_difference": res.max_abs_density_difference,
            "not_rehearsed": "P2's two-path verdict (no Triton path on the L4)"}


def working_learned_config():
    """Configuration 4 of the 9 Nov re-selection: the working choice, not a
    lock value."""
    from attnbench.predict.learned import LearnedConfig
    d = json.loads(LEARNED_CV.read_text())
    won = next(c for c in d["configs"] if c["index"] == d["winner"])
    return d["winner"], LearnedConfig(target=won["target"], params=dict(won["params"]))


def rehearsal_predict(cfg, args) -> dict:
    """P2's predict, with the learned predictor on the working configuration
    for this process only."""
    from attnbench.predict import learned as LM
    index, config = working_learned_config()
    before = LM.LOCKED_CONFIG
    LM.LOCKED_CONFIG = config
    try:
        found = S.body_predict(cfg, args)
    finally:
        LM.LOCKED_CONFIG = before
    return {**found, "learned_config": f"working: configuration {index} of {LEARNED_CV.name}, "
                                      f"not a lock value"}


def _rehearsal() -> S.SessionConfig:
    reals, fakes = S.common_bodies()
    reals["prereqs"] = fakes["prereqs"] = rehearsal_prereqs
    reals["mask_gate"] = rehearsal_mask_gate
    fakes["mask_gate"] = S.fake_files("mask_gate.json", verdict="pass", rehearsal=True)
    reals["predict"] = fakes["predict"] = rehearsal_predict
    cfg = S.SessionConfig(
        task="T6.1-rehearsal", pairing=PAIRING,
        title="The L4 rehearsal of the P2 session runner (exploratory)",
        script="t61_rehearsal.py", phase_prefix="t61r", session_prefix="t61r_l4",
        gpu="L4", instance_type="g2-standard-8", provisioning=("STANDARD",),
        code_path={"gpu": "L4", "implementation": "xattention:torch",
                   "libraries": [["torch", "dry"], ["triton", "dry"]], "python_optimize": 0},
        banked=("P1",), components_config="configs/t61/rehearsal_l4.json",
        steps=S.T61.steps, requirements=(),
        files=S.BUILT_FILES + ((LOCK, None), (COMPILED_CONFIG, None),
                               ("docs/learned_cv_p1_20261109.json", None)),
        fakes=fakes, reals=reals, exploratory=EXPLORATORY_NOTE)
    S.validate(cfg)
    if cfg.names != S.T61.names:
        raise RuntimeError("the rehearsal runs P2's steps in P2's order")
    return cfg


REHEARSAL = _rehearsal()


# --- after a block: the compiled ground truth and the trace sizes ---------------------------

def compiled(args, *, harness=None) -> int:
    """The compiled groups at 2048 against block b1's prediction (scope v).
    Refused unless b1's predict and ground truth passed in `--session`."""
    session, out = Path(args.session), Path(args.out)
    m = S.load_manifest(session)
    done = S.last_status(m)
    if (m.get("block") or {}).get("id") != "b1" or done.get("predict") != S.PASSED \
            or done.get("ground_truth") != S.PASSED:
        print(f"REFUSING: {session} is not a block b1 whose predict and ground truth passed",
              file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    cfg = dataclasses.replace(REHEARSAL, components_config=COMPILED_CONFIG)
    ns = argparse.Namespace(out=str(out), session=str(session), dry_run=False)
    try:
        result = (harness or S.real_harness)(cfg, ns)
        status, reason = (S.PASSED, "") if result.get("complete") is True else \
            (S.FAILED, "the compiled ground truth is incomplete")
    except S.StepFailed as e:
        result, status, reason = {"findings": e.findings}, S.FAILED, str(e)
    S.write_json(out / "compiled_ground_truth.json", result)
    S.write_json(out / "result.json", {"step": "compiled", "status": status, "reason": reason,
                                       "exploratory": EXPLORATORY_NOTE})
    print(f"[compiled] {status.upper()}{': ' + reason if reason else ''}", flush=True)
    return S.EXIT[status]


_TRACE = re.compile(r"^trace_L(?P<length>\d+)_b(?P<batch>\d+)_(?P<label>.+)\.json$")


def trace_sizes(session: Path) -> dict:
    """Every Kineto trace under the session, by group, cell and arm label,
    with its bytes; the totals per arm label (the label's leading index
    dropped) and per cell."""
    rows = []
    for p in sorted(Path(session).rglob("traces/trace_L*_b*_*.json")):
        m = _TRACE.match(p.name)
        if m is None:
            continue
        rows.append({"path": str(p.relative_to(session)), "length": int(m["length"]),
                     "batch": int(m["batch"]), "label": m["label"],
                     "arm": m["label"].split("_", 1)[-1], "bytes": p.stat().st_size})
    per_arm: dict = {}
    for r in rows:
        k = f"{r['arm']}@L{r['length']}_b{r['batch']}"
        per_arm[k] = per_arm.get(k, 0) + r["bytes"]
    return {"traces": rows, "bytes_by_arm_and_cell": per_arm,
            "total_bytes": sum(r["bytes"] for r in rows), "exploratory": EXPLORATORY_NOTE}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["compiled"]:
        ap = argparse.ArgumentParser(description=compiled.__doc__)
        ap.add_argument("--session", required=True)
        ap.add_argument("--out", required=True)
        return compiled(ap.parse_args(argv[1:]))
    if argv[:1] == ["trace-sizes"]:
        ap = argparse.ArgumentParser(description=trace_sizes.__doc__)
        ap.add_argument("--session", required=True)
        a = ap.parse_args(argv[1:])
        res = trace_sizes(Path(a.session))
        S.write_json(Path(a.session) / "trace_sizes.json", res)
        print(json.dumps({"traces": len(res["traces"]), "total_bytes": res["total_bytes"]}))
        return 0
    return S.main(argv, cfg=REHEARSAL)


if __name__ == "__main__":
    raise SystemExit(main())
