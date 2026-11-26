#!/usr/bin/env python3
"""P6's prediction, committed and pushed before any P6 measurement (the
researcher's task of 26 Nov 2026 after P3's push, item 2a; EXPLORATORY).

P6 is the H100 on another host (AWS p5.4xlarge, AMD EPYC 7R13). It is built
as P3's was (`scripts/p3_prediction_from_p1.py`, 22 Nov 2026): the same-GPU
mirror, here of P4's committed prediction (T6.4, written 16 Nov 2026 on the
H100 at `7a24cba`), so each P6 cell's prediction is P4's committed value for
the same cell, every predictor P4's file holds. No estimator, threshold or
matching rule is changed, no P6 component exists when it is written, and the
timed-path pin and every earlier prediction are untouched.

P4's record is read only after its sha256 sidecar checks, and its cells must
be `docs/p6_cells.json`'s 35, key for key (arm, density, length budget,
batch; XAttention's density is the one its masks measure, so the cells file
leaves it open). The P6 record is canonical JSON (`attnbench.predict.commit`)
with its own `sha256sum`-format sidecar, never overwritten. Nothing here is a
T2.11 record: the runner does not read it, and it scores no hypothesis.

    python3 scripts/p6_prediction_from_p4.py P4_PREDICTION.json OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.predict import commit as C  # noqa: E402
import p3_prediction_from_p1 as G  # noqa: E402  (P3's builder: its sidecar check and budgets)

SCHEMA = "attnbench.p6_transfer_prediction/1"
LABEL = "EXPLORATORY"
CELLS = ROOT / "docs" / "p6_cells.json"
BASIS = ("same-GPU mirror of P4: P6's host and device components are both taken as P4's "
         "(T6.4, the H100), so each cell's prediction is P4's committed value for the same "
         "cell, every predictor P4's file holds. Built as P3's prediction from P1 was. "
         "Written before any P6 measurement; no P6 component, ground truth or host-effect "
         "statistic existed when it was written.")
RefusedError = G.RefusedError


def cell_keys(path: Path = CELLS) -> list:
    """docs/p6_cells.json's cells: (arm, density or None, budget, batch)."""
    return [(c["arm"], c["density"], c["context_length"], c["batch"])
            for c in json.loads(path.read_text())["cells"]]


def transfer(p4: dict, *, source_name: str, source_sha: str, written_at: datetime,
             code: dict, cells: list) -> dict:
    if p4.get("pairing") != "P4":
        raise RefusedError(f"the source is pairing {p4.get('pairing')!r}, not P4")
    out = []
    for e in p4["cells"]:
        preds = dict(e["predictions"])
        if not preds:
            raise RefusedError(f"{e['arm']} L{e['context_length']}: P4 has no prediction")
        out.append({"arm": e["arm"], "model": e["model"], "batch": e["batch"],
                    "density": e["density"], "context_length": e["context_length"],
                    "budget": G.budget(e["context_length"]),
                    "p4_code_path": e["code_path"], "predictions_s": preds})
    keys = [(c["arm"], c["density"], c["budget"], c["batch"]) for c in out]
    if len(set(keys)) != len(keys):
        raise RefusedError("two P4 cells share an arm, density, budget and batch")
    # XAttention's density is open in the cells file, so it is compared as None there
    open_arms = {a for a, d, _, _ in cells if d is None}
    want = sorted(cells, key=repr)
    got = sorted(((c["arm"], None if c["arm"] in open_arms else c["density"], c["budget"], c["batch"])
                  for c in out), key=repr)
    if got != want:
        raise RefusedError(f"P4's cells are not docs/p6_cells.json's: {len(got)} against {len(want)}, "
                           f"first difference {next((x for x in got if x not in want), None)}")
    return {"schema": SCHEMA, "label": LABEL, "pairing": "P6", "basis": BASIS,
            "statistic": p4["statistic"],
            "source": {"pairing": "P4", "file": source_name, "sha256": source_sha,
                       "written_at": p4["written_at"], "code": p4["code"]},
            "written_at": written_at.isoformat(timespec="microseconds"),
            "code": code, "cells": out}


def write(record: dict, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = datetime.fromisoformat(record["written_at"])
    name = f"prediction_P6_exploratory_from_P4_{t:%Y%m%dT%H%M%S%fZ}.json"
    data = C.canonical_bytes(record)
    digest = C.sha256_hex(data)
    path = out_dir / name
    if path.exists():
        raise RefusedError(f"{path} exists: a committed prediction is never overwritten")
    path.write_bytes(data)
    path.with_name(name + ".sha256").write_text(C.sidecar_text(digest, name))
    return path, digest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("p4_prediction", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    try:
        p4, sha, _ = G.read_checked(a.p4_prediction)
        rec = transfer(p4, source_name=a.p4_prediction.name, source_sha=sha,
                       written_at=datetime.now(timezone.utc), code=G.code_stamp(),
                       cells=cell_keys())
        path, digest = write(rec, a.out_dir)
    except (RefusedError, C.PredictionCommitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"{len(rec['cells'])} cells, {LABEL}: {path.name} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
