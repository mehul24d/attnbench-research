#!/usr/bin/env python3
"""P3's prediction, committed before any P3 measurement (adviser's task of
23 Nov 2026, item 4c; EXPLORATORY).

P3 is the L4 on another host (AWS g6.2xlarge, AMD EPYC 7R13). When this is
written, no P3 component exists, so the only prediction that can come before
P3's measurements is the full transfer from P1: host and device components
both P1's (T3.2d, the session P1's training file comes from), which makes
each P3 cell's prediction P1's committed value for the same cell. If P3's
host matters, this prediction is off by the host effect; the host-effect
rule (`scripts/p6_host_effect.py`) measures that separately, from P3's own
host components.

The P1 record is read only after its sha256 sidecar checks, and the P3
record is written as canonical JSON (`attnbench.predict.commit`) with its
own `sha256sum`-format sidecar. Nothing here is a T2.11 record: it is not
read by the runner, and it scores no hypothesis.

    python3 scripts/p3_prediction_from_p1.py P1_PREDICTION.json OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from attnbench.predict import commit as C  # noqa: E402

SCHEMA = "attnbench.p3_transfer_prediction/1"
LABEL = "EXPLORATORY"
BUDGETS = (2048, 4096, 8192, 16384, 32768)
BASIS = ("full transfer from P1: P3's host and device components are both taken as "
         "P1's (T3.2d), so each cell's prediction is P1's committed value for the same "
         "cell. Written before any P3 measurement; no P3 component, ground truth or "
         "host-effect statistic existed when it was written.")


class RefusedError(RuntimeError):
    pass


def budget(context_length: int) -> int:
    """The length budget a prompt was cut to: the smallest one it fits."""
    for b in BUDGETS:
        if context_length <= b:
            return b
    raise RefusedError(f"context length {context_length} is over every budget")


def read_checked(path: Path) -> tuple[dict, str, bytes]:
    data = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.exists():
        raise RefusedError(f"{side} is missing: P1's record is read only against its sidecar")
    want = C.parse_sidecar(side.read_text(), path.name)
    got = C.sha256_hex(data)
    if got != want:
        raise RefusedError(f"{path.name}: sha256 {got} is not its sidecar's {want}")
    return json.loads(data), got, data


def transfer(p1: dict, *, source_name: str, source_sha: str, written_at: datetime,
             code: dict) -> dict:
    if p1.get("pairing") != "P1":
        raise RefusedError(f"the source is pairing {p1.get('pairing')!r}, not P1")
    cells = []
    for e in p1["cells"]:
        preds = dict(e["predictions"])
        if not preds:
            raise RefusedError(f"{e['arm']} L{e['context_length']}: P1 has no prediction")
        cells.append({"arm": e["arm"], "model": e["model"], "batch": e["batch"],
                      "density": e["density"], "context_length": e["context_length"],
                      "budget": budget(e["context_length"]),
                      "p1_code_path": e["code_path"], "predictions_s": preds})
    keys = [(c["arm"], c["density"], c["budget"], c["batch"]) for c in cells]
    if len(set(keys)) != len(keys):
        raise RefusedError("two P1 cells share an arm, density, budget and batch")
    return {"schema": SCHEMA, "label": LABEL, "pairing": "P3", "basis": BASIS,
            "statistic": p1["statistic"],
            "source": {"pairing": "P1", "file": source_name, "sha256": source_sha,
                       "written_at": p1["written_at"], "code": p1["code"]},
            "written_at": written_at.isoformat(timespec="microseconds"),
            "code": code, "cells": cells}


def write(record: dict, out_dir: Path) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    t = datetime.fromisoformat(record["written_at"])
    name = f"prediction_P3_exploratory_from_P1_{t:%Y%m%dT%H%M%S%fZ}.json"
    data = C.canonical_bytes(record)
    digest = C.sha256_hex(data)
    path = out_dir / name
    if path.exists():
        raise RefusedError(f"{path} exists: a committed prediction is never overwritten")
    path.write_bytes(data)
    path.with_name(name + ".sha256").write_text(C.sidecar_text(digest, name))
    return path, digest


def code_stamp() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip())
    return {"git_commit": head, "git_dirty": dirty}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("p1_prediction", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args(argv)
    try:
        p1, sha, _ = read_checked(a.p1_prediction)
        rec = transfer(p1, source_name=a.p1_prediction.name, source_sha=sha,
                       written_at=datetime.now(timezone.utc), code=code_stamp())
        path, digest = write(rec, a.out_dir)
    except (RefusedError, C.PredictionCommitError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(f"{len(rec['cells'])} cells, {LABEL}: {path} sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
