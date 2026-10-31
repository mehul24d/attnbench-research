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
  - `predicted_layer` (when the figure is of one layer): the recursion's
    predicted end of that layer, for the figure's predicted-end marker.
    It is derived from the recursion's own per-operation completion times
    (`Timeline.device_done`) over the committed cell's priced components,
    so it is the committed prediction's, not a new one: the end is the
    latest completion among the layer's operations, and the layer starts
    where the host reaches its first operation, H of the operation before
    it (0 for the first), which is where the trace's layer annotation
    starts. `predicted_end_ms` is end minus start, in milliseconds, and the
    whole-prefill D_N is recorded beside it. The layer's operations must
    be contiguous. `predicted_end_ms` is what the figure reads, and it
    refuses a value that is the whole prefill's (researcher's decision,
    28 Oct 2026).
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
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

from ..predict.commit import CodeStamp, CommittedPrediction
from ..predict.recursion import PricedSequence, recurse
from .results import (ResultsError, check_loaded, check_result, measured_seconds,
                      seconds_to_ms, sha256_file)

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


def layer_window(priced: PricedSequence, queue_depth: Optional[int], layer: int) -> dict:
    """The recursion's predicted window for one layer, in seconds of
    prefill time (module docstring)."""
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise MetaError(f"a layer is an int, got {layer!r}")
    idx = [k for k, l in enumerate(priced.layers) if l == layer]
    if not idx:
        raise MetaError(f"the committed sequence has no operation in layer {layer}")
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise MetaError(f"layer {layer}'s operations are not contiguous in the sequence")
    tl = recurse(priced, queue_depth)
    start = tl.host_issued[idx[0] - 1] if idx[0] > 0 else 0.0
    return {"layer": layer, "first_op": idx[0], "last_op": idx[-1], "start_s": start,
            "end_s": max(tl.device_done[k] for k in idx), "whole_prefill_s": tl.end_to_end}


def figure_meta(*, figure: str, prediction: CommittedPrediction,
                prediction_path: Union[str, Path], cell: Sequence,
                result: Mapping, result_path: Union[str, Path],
                inputs: Sequence[tuple[str, Union[str, Path]]],
                code: CodeStamp, layer: Optional[int]) -> dict:
    """The metadata (module docstring). `cell` is (model, code path, arm,
    density, context length, batch); `inputs` is (role, path) pairs;
    `layer` is the figure's layer, or None for a figure of no one layer."""
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

    predicted_layer = None
    if layer is not None:
        w = layer_window(c.priced, cp.queue_depth, layer)
        committed = c.predictions["recursion"]
        if not math.isclose(w["whole_prefill_s"], committed, rel_tol=1e-12, abs_tol=0.0):
            raise MetaError(f"the recursion over the committed components gives "
                            f"{w['whole_prefill_s']}, not the committed {committed}")
        predicted_layer = {
            "layer": layer, "predictor": "recursion", "scope": "layer",
            "first_op": w["first_op"], "last_op": w["last_op"],
            "layer_start_ms": seconds_to_ms(w["start_s"]),
            "predicted_end_ms": seconds_to_ms(w["end_s"] - w["start_s"]),
            "whole_prefill_ms": seconds_to_ms(w["whole_prefill_s"]),
            "relative_to": "the host reaching the layer's first operation"}

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
        "predicted_layer": predicted_layer,
        "inputs": entries,
    }


def predicted_end_ms(meta: Mapping, layer: int) -> float:
    """The predicted end of `layer` in ms from the layer's start, as the
    figure reads it. Refuses metadata with no per-layer value, for another
    layer, or whose value is the whole prefill's (D_N), which would put the
    marker at the end of the prefill rather than the layer."""
    if meta.get("schema") != META_SCHEMA:
        raise MetaError("not a figure metadata record")
    pl = meta.get("predicted_layer")
    if not isinstance(pl, Mapping):
        raise MetaError("the metadata holds no per-layer prediction; make it with the "
                        "figure's layer")
    if pl.get("scope") != "layer" or pl.get("layer") != layer:
        raise MetaError(f"the per-layer prediction is for layer {pl.get('layer')!r} "
                        f"(scope {pl.get('scope')!r}), not layer {layer}")
    v = pl.get("predicted_end_ms")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise MetaError(f"predicted_end_ms is {v!r}")
    whole = [pl.get("whole_prefill_ms"),
             seconds_to_ms(meta.get("cell", {}).get("predictions_s", {}).get("recursion", 0.0))]
    if any(isinstance(w, (int, float)) and w > 0 and (v >= w or math.isclose(v, w))
           for w in whole):
        raise MetaError(f"predicted_end_ms {v} is the whole prefill's ({whole}), not "
                        f"layer {layer}'s")
    return float(v)


def write_meta(meta: Mapping, path: Union[str, Path]) -> Path:
    """Write once; never overwrite."""
    if meta.get("schema") != META_SCHEMA:
        raise MetaError("not a figure metadata record")
    path = Path(path)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return path
