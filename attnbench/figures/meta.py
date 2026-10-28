"""A figure's `meta.json`: which committed prediction, which ground truth and
which other files a figure was drawn from, each with its sha256, so a reader
can check that the figure follows from banked inputs (rule 3, rule 6).

**What it records.**

  - `prediction`: the committed file's name, sha256 and its sidecar's, the
    pairing, the statistic, the write and sync times, and the code stamp it
    was computed at. The prediction must come from `load_committed` (it
    carries a sync time, so the K9 order was checked) and the file on disk
    must be the bytes that were committed.
  - `cell`: the figure's cell key and every committed prediction for it.
  - `measured`: the ground-truth file's name and sha256, the arm row read,
    the statistic (the file's own), and the value in seconds. The result
    must be untraced and complete; its first timed block's start is
    recorded beside the sync time. Ground-truth rows carry no density,
    length or batch (audit C3, finding 2), so the row is matched to the
    cell by arm and the rest of the key is the caller's statement, recorded
    as such.
  - `inputs`: every other file the figure read (traces, sequence records,
    events tables), by role, name and sha256.
  - `generated_by`: the commit the metadata was made at, which must be
    clean.

Names are file names, not paths, so no local directory is recorded; two
inputs with one name are refused. The file is written once, never
overwritten.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence, Union

from ..predict.commit import CodeStamp, CommittedPrediction
from .results import (ResultsError, check_loaded, check_result, measured_seconds,
                      sha256_file)

META_SCHEMA = "attnbench.figure_meta/1"
ROLES = ("trace", "sequence", "events", "other")


class MetaError(ValueError):
    """Inputs a figure's metadata cannot be made from."""


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _entry(role: str, path: Union[str, Path]) -> dict:
    p = Path(path)
    if not p.is_file():
        raise MetaError(f"{role} input {p.name!r} is not a file")
    return {"role": role, "name": p.name, "sha256": sha256_file(p), "bytes": p.stat().st_size}


def figure_meta(*, figure: str, prediction: CommittedPrediction,
                prediction_path: Union[str, Path], cell: Sequence,
                result: Mapping, result_path: Union[str, Path],
                inputs: Sequence[tuple[str, Union[str, Path]]],
                code: CodeStamp) -> dict:
    """The metadata (module docstring). `cell` is (model, code path, arm,
    density, context length, batch); `inputs` is (role, path) pairs."""
    if not isinstance(figure, str) or not figure.strip():
        raise MetaError("name the figure")
    if not isinstance(code, CodeStamp) or code.git_dirty is not False \
            or len(code.git_commit or "") != 40:
        raise MetaError(f"metadata is made at a clean, known commit, got {code!r}")
    try:
        cp = check_loaded(prediction)
    except ResultsError as e:
        raise MetaError(str(e)) from e
    pred_path = Path(prediction_path)
    side_path = pred_path.with_name(pred_path.name + ".sha256")
    if not pred_path.is_file() or sha256_file(pred_path) != cp.sha256:
        raise MetaError(f"{pred_path.name} is not the committed file (sha256 {cp.sha256})")
    if not side_path.is_file() or not side_path.read_text(encoding="ascii").startswith(cp.sha256):
        raise MetaError(f"{side_path.name} is missing or names another digest")
    try:
        c = cp.cell(*cell)
    except KeyError:
        raise MetaError(f"the prediction has no cell {tuple(cell)}") from None
    res_path = Path(result_path)
    try:
        check_result(result, res_path.name)
        seconds = measured_seconds(result, c.arm, cp.statistic, res_path.name)
    except ResultsError as e:
        raise MetaError(str(e)) from e

    entries = [_entry("prediction", pred_path), _entry("prediction_sidecar", side_path),
               _entry("result", res_path)]
    for role, path in inputs:
        if role not in ROLES:
            raise MetaError(f"input role {role!r} is not one of {ROLES}")
        entries.append(_entry(role, path))
    names = [e["name"] for e in entries]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise MetaError(f"two inputs named {dup}; names identify inputs here")

    return {
        "schema": META_SCHEMA,
        "figure": figure,
        "generated_by": {"git_commit": code.git_commit, "git_dirty": code.git_dirty},
        "prediction": {
            "name": pred_path.name, "sha256": cp.sha256, "pairing": cp.pairing,
            "statistic": cp.statistic, "queue_depth": cp.queue_depth,
            "written_at": _iso(cp.written_at), "synced_at": _iso(cp.synced_at),
            "code": {"git_commit": cp.code.git_commit, "git_dirty": cp.code.git_dirty}},
        "cell": {"model": c.model, "code_path": c.code_path, "arm": c.arm,
                 "density": c.density, "context_length": c.context_length,
                 "batch": c.batch, "predictions_s": dict(sorted(c.predictions.items()))},
        "measured": {
            "name": res_path.name, "arm_row": c.arm, "statistic": cp.statistic,
            "seconds": seconds,
            "first_timed_block_started_at": result.get("first_timed_block_started_at"),
            "matched_by": "arm; density, context length and batch as stated by the "
                          "caller (ground-truth rows do not carry them)"},
        "inputs": entries,
    }


def write_meta(meta: Mapping, path: Union[str, Path]) -> Path:
    """Write once; never overwrite."""
    if meta.get("schema") != META_SCHEMA:
        raise MetaError("not a figure metadata record")
    path = Path(path)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return path
