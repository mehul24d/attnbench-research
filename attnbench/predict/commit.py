"""The prediction commit of plan Sections 5.1, 5.3 step 6 and 5.4 (T2.11, K9).

One prediction file per pairing holds, for every cell, the recursion's
prediction and its baselines' committed predictions together. A cell is keyed
by model, code path, arm, density, context length and batch (plan Section
5.3 step 6; Appendix A, 25 Oct 2026), so two cells that differ only in the
model or in the code path (XAttention's Triton and PyTorch paths on one card,
for instance) are two cells and never one:

  - computed here, from the cell's priced components, so no predictor can
    be priced from different numbers: the recursion at the measured Q, the
    Sum as written, the Sum + s_0 and the KernelSight-style step form;
  - fitted elsewhere and named with what they were trained on: the fitted
    sum (and, labelled separately, the fitted sum + s_0) and the learned
    predictor. Each record names the pairings it was trained on, and those
    must be exactly the pairings banked before this prediction (Section
    5.4, "Fitting the baselines"). With none banked (P1) neither exists, and
    the file says why.

The file also carries every cell's priced components, so the analysis
recomputes each deterministic prediction from them and refuses a file whose
numbers do not follow from its own inputs, and the queue sensitivity
(T2.15) re-runs exactly the committed inputs.

**The order of events this module enforces.** Section 5.1: "The prediction
file and its sha256 are synced before the timed workload starts; the
analysis refuses any prediction stamped later." So:

  1. `build_record` checks and computes, `write_prediction` writes the file
     and a sha256 sidecar in `sha256sum` format (checkable with
     `shasum -a 256 -c`). Neither overwrites an existing file.
  2. `sync_prediction` uploads both and reads them back; it fails unless the
     bytes that came back are the bytes written. The session runner starts
     the timed workload only after it returns.
  3. `load_committed` is what the analysis reads. It refuses a file whose
     hash does not match its sidecar, whose sync time is not before the
     timed workload started, whose own write stamp is later than its sync
     time, which was written at a dirty or unknown commit, whose committed
     baselines were trained on a pairing banked after the prediction was
     written, or whose deterministic predictions do not follow from its
     components.

**Which clock.** The sync time is the store's, not the instance's: an
object store's creation time is not something the session can set. For
GCS an overwrite makes a new object generation with a new creation time,
so re-uploading a file later can only make it later, never backdate it.
`GsutilStore.created` returns an upper bound (gsutil prints whole
seconds). The timed workload's start comes from the instance clock, so the
caller states the clock skew it allows between the two; there is no
default.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Protocol, Sequence

from .baselines import FittedSum, step_form, sum_as_written, sum_plus_s0
from .intervals import Interval
from .learned import LIGHTGBM_VERSION, LearnedModel
from . import learned as _learned
from .naive import Nearest, nearest_pairing, with_constant_host_cost
from .recursion import CostModelError, OpKind, PricedSequence, recurse
from .training import (COMMITTED, GROUND_TRUTH_STATISTIC, STATISTICS, Cell,
                       CellKey, make_cell_key)

# /2: the cell key gained its model and code path, and the file carries each
# code path's definition. A /1 file has neither and is refused.
SCHEMA = "attnbench.prediction/2"

# Computed here from every cell's components.
COMPUTED = ("recursion", "sum", "sum_plus_s0", "step_form")
# Fitted on the pairings banked before this one; both required when any is.
TRAINED_REQUIRED = ("fitted_sum", "learned")
# Fitted, labelled separately (not in the plan), never required.
TRAINED_OPTIONAL = ("fitted_sum_plus_s0",)

NOT_FITTED_REASON = ("no pairing was banked before this prediction, so no "
                     "fitted baseline exists; Gate A uses the sum and the "
                     "step form (Section 5.4)")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class PredictionCommitError(ValueError):
    """A prediction that cannot be committed, or a committed file the
    analysis refuses."""


class LatePredictionError(PredictionCommitError):
    """The prediction was not synced before the timed workload started (K9)."""


@dataclass(frozen=True)
class CodeStamp:
    """The commit the prediction was computed at. Rule 3: clean only."""

    git_commit: str
    git_dirty: bool


def current_code_stamp() -> CodeStamp:
    """This checkout's commit and dirty flag, from `provenance._git_state`.

    Imported lazily because provenance imports torch. Refuses when there is
    no usable commit rather than stamping one that looks populated.
    """
    from attnbench.provenance import _git_state
    commit, dirty = _git_state()
    if commit is None or dirty is None:
        raise PredictionCommitError(
            "no usable git commit for this checkout; a prediction is "
            "commit-stamped (rule 3)")
    return CodeStamp(git_commit=commit, git_dirty=dirty)


def _check_code(code: CodeStamp) -> None:
    if not isinstance(code, CodeStamp) or not isinstance(code.git_commit, str) \
            or not _SHA_RE.match(code.git_commit):
        raise PredictionCommitError(
            f"the code stamp needs a 40-hex git commit, got {code!r}")
    if code.git_dirty is not False:
        raise PredictionCommitError(
            f"the prediction was computed at a dirty or unknown tree "
            f"(git_dirty={code.git_dirty!r}); rule 3 stamps measurements "
            f"and predictions with git_dirty = False")


def _check_aware(t: datetime, name: str) -> datetime:
    if not isinstance(t, datetime) or t.tzinfo is None \
            or t.tzinfo.utcoffset(t) is None:
        raise PredictionCommitError(
            f"{name} must be a timezone-aware datetime, got {t!r}; a naive "
            f"time cannot be compared with a store's clock")
    return t.astimezone(timezone.utc)


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _from_iso(s: str) -> datetime:
    return _check_aware(datetime.fromisoformat(s), "a stamp in the file")


def utc_from_epoch(seconds: float) -> datetime:
    """A provenance `timestamp` (time.time() on the instance) as UTC."""
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc)


def ground_truth_start(timestamps: Iterable) -> datetime:
    """The earliest of the timed workload's stamps (epoch seconds or aware
    datetimes). The caller passes the stamp of the first timed block, not
    of a warm-up or of the structural probe."""
    ts = [utc_from_epoch(t) if isinstance(t, (int, float)) and not
          isinstance(t, bool) else _check_aware(t, "a ground-truth stamp")
          for t in timestamps]
    if not ts:
        raise PredictionCommitError("no ground-truth stamps, so no start time")
    return min(ts)


# --------------------------------------------------------------------------
# Building the record
# --------------------------------------------------------------------------

def cell_key(model: str, code_path: str, arm: str, density: float,
             context_length: int, batch: int) -> CellKey:
    """The key of a cell, in the plan's order (`training.CellKey`)."""
    try:
        return make_cell_key(model, code_path, arm, density, context_length,
                             batch)
    except ValueError as e:
        raise PredictionCommitError(str(e)) from e


# What a code path is (plan Section 5.1): the GPU architecture, the library
# versions and which implementation runs, and `sys.flags.optimize` (K8). The
# arm is not part of it; the key has the arm already.
_CODE_PATH_FIELDS = ("gpu", "implementation", "libraries", "python_optimize")


def code_path_definition(code_path) -> dict:
    """The definition of an `op_sequence.CodePath`, as the file records it."""
    return {"gpu": code_path.gpu, "implementation": code_path.implementation,
            "libraries": [list(x) for x in code_path.libraries],
            "python_optimize": code_path.python_optimize}


def _check_definition(defn: Mapping, where: str) -> dict:
    if not isinstance(defn, Mapping) or set(defn) != set(_CODE_PATH_FIELDS):
        raise PredictionCommitError(
            f"{where}: a code path is defined by {list(_CODE_PATH_FIELDS)}, "
            f"got {sorted(defn) if isinstance(defn, Mapping) else defn!r}")
    gpu, impl, libs, opt = (defn[f] for f in _CODE_PATH_FIELDS)
    if not (isinstance(gpu, str) and gpu and isinstance(impl, str) and impl):
        raise PredictionCommitError(
            f"{where}: gpu and implementation are non-empty strings")
    if isinstance(opt, bool) or not isinstance(opt, int) or opt not in (0, 1, 2):
        raise PredictionCommitError(
            f"{where}: python_optimize is 0, 1 or 2, got {opt!r}")
    try:
        libs = sorted([str(n), str(v)] for n, v in libs)
    except (TypeError, ValueError) as e:
        raise PredictionCommitError(
            f"{where}: libraries are (name, version) pairs") from e
    return {"gpu": gpu, "implementation": impl, "libraries": libs,
            "python_optimize": opt}


def code_path_label(defn: Mapping) -> str:
    """A label that names a code path and cannot be reused for another: the
    GPU, the implementation, the optimise flag, and a hash of the whole
    definition (so a library version changes the label)."""
    d = _check_definition(defn, "code path")
    digest = hashlib.sha256(json.dumps(d, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return f"{d['gpu']}|{d['implementation']}|O{d['python_optimize']}|{digest[:12]}"


def _components(p: PricedSequence) -> dict:
    return {
        "host": list(p.host), "link": list(p.link), "device": list(p.device),
        "sync_cost": list(p.sync_cost), "is_sync": list(p.is_sync),
        "kinds": [k.value for k in p.kinds], "layers": list(p.layers),
        "mechanisms": list(p.mechanisms),
        "payload_bytes": list(p.payload_bytes),
    }


def priced_from_components(c: Mapping) -> PricedSequence:
    """The `PricedSequence` a committed cell was computed from."""
    try:
        return PricedSequence(
            host=tuple(float(x) for x in c["host"]),
            link=tuple(float(x) for x in c["link"]),
            device=tuple(float(x) for x in c["device"]),
            sync_cost=tuple(float(x) for x in c["sync_cost"]),
            is_sync=tuple(bool(x) for x in c["is_sync"]),
            kinds=tuple(OpKind(k) for k in c["kinds"]),
            layers=tuple(c["layers"]),
            mechanisms=tuple(c["mechanisms"]),
            payload_bytes=tuple(int(x) for x in c["payload_bytes"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise PredictionCommitError(f"unreadable components: {e}") from e


def _check_closing_sync(p: PricedSequence, where: str) -> None:
    # Appendix A, 22 Oct 2026 (2): a prefill ends with an explicit closing
    # synchronise. Priced components carry no labels, so the check is on
    # what the closing synchronise is: a payload-free explicit synchronise,
    # last.
    if not (len(p) and p.is_sync[-1] and p.mechanisms[-1] == "synchronize"
            and p.payload_bytes[-1] == 0):
        raise PredictionCommitError(
            f"{where} does not end with the closing synchronise; build the "
            f"sequence with recursion.close_prefill")


def computed_predictions(p: PricedSequence, queue_depth: int) -> dict:
    """The predictions this module computes itself, by label."""
    return {
        "recursion": recurse(p, queue_depth).end_to_end,
        "sum": sum_as_written(p),
        "sum_plus_s0": sum_plus_s0(p),
        "step_form": step_form(p),
    }


def _finite_positive(x: float, where: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) \
            or not math.isfinite(x) or x <= 0:
        raise PredictionCommitError(
            f"{where} must be a positive finite number of seconds, got {x!r}")
    return float(x)


def _fitted_record(fs: FittedSum, label: str, pairing: str,
                   banked: tuple[str, ...], statistic: str) -> dict:
    if fs.label != label:
        raise PredictionCommitError(
            f"the {label} slot holds a {fs.label}; the two are labelled "
            f"separately")
    _check_trained(fs.kind, fs.target, fs.trained_on, fs.statistic, label,
                   pairing, banked, statistic)
    cond = fs.condition_number
    return {
        "kind": fs.kind, "trained_on": list(fs.trained_on),
        "statistic": fs.statistic, "n_rows": fs.n_rows,
        "a": fs.a, "b": fs.b, "c": fs.c, "add_s0": fs.add_s0,
        "training_loss": fs.training_loss,
        "condition_number": cond if math.isfinite(cond) else "inf",
        "unidentified": list(fs.unidentified),
    }


def _check_trained(kind, target, trained_on, statistic, label, pairing,
                   banked, want_statistic) -> None:
    if kind != COMMITTED:
        raise PredictionCommitError(
            f"the {label} was fitted on a {kind} training set; only a "
            f"committed fit (pairings banked before {pairing}) is a committed "
            f"prediction, and leave-one-pairing-out is the retrospective "
            f"secondary")
    if target != pairing:
        raise PredictionCommitError(
            f"the {label} was fitted for {target}, not {pairing}")
    if tuple(trained_on) != banked:
        raise PredictionCommitError(
            f"the {label} was trained on {list(trained_on)} but the pairings "
            f"banked before {pairing} are {list(banked)}; the training set is "
            f"whatever was banked when the prediction is committed")
    if statistic != want_statistic:
        raise PredictionCommitError(
            f"the {label} was fitted to the {statistic}, and this file "
            f"predicts the {want_statistic}")


def _learned_record(lm: LearnedModel, pairing: str, banked: tuple[str, ...],
                    statistic: str) -> dict:
    _check_trained(lm.kind, lm.target, lm.trained_on, lm.statistic,
                   "learned predictor", pairing, banked, statistic)
    locked = _learned.LOCKED_CONFIG
    if locked is None:
        raise PredictionCommitError(
            "the learned predictor's target and hyperparameters are fixed at "
            "lock (Gate B) and learned.LOCKED_CONFIG is still None, so no "
            "learned prediction can be committed yet")
    if lm.config != locked:
        raise PredictionCommitError(
            f"the learned predictor was fitted with {lm.config!r}, not the "
            f"locked configuration {locked!r}")
    return {
        "kind": lm.kind, "trained_on": list(lm.trained_on),
        "statistic": lm.statistic, "n_rows": lm.n_rows,
        "config": {"target": lm.config.target,
                   "params": dict(lm.config.params)},
        "lightgbm_version": LIGHTGBM_VERSION,
    }


def build_record(pairing: str, cells: Sequence[Cell], queue_depth: int, *,
                 code_paths: Mapping[str, Mapping],
                 banked: Iterable[str],
                 fitted_sum: Optional[FittedSum],
                 learned: Optional[LearnedModel],
                 code: CodeStamp, written_at: datetime,
                 fitted_sum_plus_s0: Optional[FittedSum] = None,
                 constant_host_cost: Optional[float] = None,
                 nearest: Optional[tuple[Nearest, Mapping]] = None,
                 intervals: Optional[Mapping[tuple, Mapping[str, Interval]]] = None,
                 statistic: str = GROUND_TRUTH_STATISTIC) -> dict:
    """Check everything and compute every prediction for one pairing.

    `code_paths` defines every code path the cells name, by label
    (`code_path_definition` and `code_path_label` build them from an
    `op_sequence.CodePath`). A label with no definition, a definition whose
    label is not the one its content gives, and a definition no cell uses
    are all refused: the label is what the key carries, and it must say the
    same thing to every reader.

    `banked` is the pairings whose ground truth was banked before this
    prediction is written (the same list `training.committed_training_set`
    was given). `queue_depth` is the measured Q; "no queue term" is a
    sensitivity (T2.15), never the committed prediction.

    Optional, from T2.19: `constant_host_cost`, the per-host constant
    (`naive.median_launch_cost`), which adds the constant-host-cost
    baseline, computed here; `nearest`, the result of
    `naive.nearest_pairing_predictions` for this pairing and `banked`;
    `intervals`, by cell key and prediction label, each an
    `intervals.Interval` whose point is that committed prediction.
    """
    if not isinstance(pairing, str) or not pairing:
        raise PredictionCommitError(f"pairing must be a name, got {pairing!r}")
    if isinstance(queue_depth, bool) or not isinstance(queue_depth, int) \
            or queue_depth < 1:
        raise PredictionCommitError(
            f"the committed prediction uses the measured Q, an int >= 1; got "
            f"{queue_depth!r} (no queue term is a sensitivity, T2.15)")
    if statistic not in STATISTICS:
        raise PredictionCommitError(
            f"statistic must be one of {STATISTICS}, got {statistic!r}")
    _check_code(code)
    written_at = _check_aware(written_at, "written_at")
    banked = tuple(sorted(set(banked)))
    if pairing in banked:
        raise PredictionCommitError(
            f"{pairing} is listed as banked before its own prediction")

    cells = list(cells)
    if not cells:
        raise PredictionCommitError("a prediction file needs at least one cell")
    seen = set()
    for c in cells:
        if c.pairing != pairing:
            raise PredictionCommitError(
                f"a cell of {c.pairing} in {pairing}'s prediction file")
        k = cell_key(*c.key)
        if k in seen:
            raise PredictionCommitError(f"cell {k} appears twice")
        seen.add(k)
        _check_closing_sync(c.priced, f"cell {k}")
    code_paths = _check_code_paths(code_paths, {k.code_path for k in seen})

    training: dict[str, Any] = {}
    not_fitted: dict[str, str] = {}
    if banked:
        if fitted_sum is None or learned is None:
            missing = [n for n, m in (("fitted_sum", fitted_sum),
                                      ("learned", learned)) if m is None]
            raise PredictionCommitError(
                f"{pairing} has pairings banked before it ({list(banked)}), so "
                f"the fitted sum and the learned predictor are committed with "
                f"the recursion (T2.11); missing: {missing}")
        training["fitted_sum"] = _fitted_record(
            fitted_sum, "fitted_sum", pairing, banked, statistic)
        training["learned"] = _learned_record(learned, pairing, banked,
                                              statistic)
        if fitted_sum_plus_s0 is not None:
            training["fitted_sum_plus_s0"] = _fitted_record(
                fitted_sum_plus_s0, "fitted_sum_plus_s0", pairing, banked,
                statistic)
    else:
        given = [n for n, m in (("fitted_sum", fitted_sum), ("learned", learned),
                                ("fitted_sum_plus_s0", fitted_sum_plus_s0))
                 if m is not None]
        if given:
            raise PredictionCommitError(
                f"no pairing is banked before {pairing}, yet {given} were "
                f"given; a fit on nothing is not a committed prediction")
        for name in TRAINED_REQUIRED + TRAINED_OPTIONAL:
            not_fitted[name] = NOT_FITTED_REASON

    if constant_host_cost is not None:
        # Validated by the baseline itself; refuse here with this module's error.
        try:
            with_constant_host_cost(cells[0].priced, constant_host_cost)
        except CostModelError as e:
            raise PredictionCommitError(str(e)) from e
    near_record, near_values = None, {}
    if nearest is not None:
        near, near_values = nearest
        want = nearest_pairing(pairing, banked)
        if near != want:
            raise PredictionCommitError(
                f"the nearest-pairing baseline names {near}, but for {pairing} "
                f"with {list(banked)} banked it is {want}")
        near_record = {"source": near.source, "rule": near.rule}
        near_values = {tuple(k): v for k, v in near_values.items()}
        extra = set(near_values) - seen
        if extra:
            raise PredictionCommitError(
                f"nearest-pairing values for cells not in this file: "
                f"{sorted(extra)}")
    intervals = {tuple(k): v for k, v in (intervals or {}).items()}
    extra = set(intervals) - seen
    if extra:
        raise PredictionCommitError(
            f"intervals for cells not in this file: {sorted(extra)}")

    learned_values = learned.predict(cells) if banked else None
    out_cells = []
    for i, c in enumerate(cells):
        k = c.key
        preds = computed_predictions(c.priced, queue_depth)
        if banked:
            preds["fitted_sum"] = fitted_sum.predict(c.priced)
            preds["learned"] = learned_values[i]
            if fitted_sum_plus_s0 is not None:
                preds["fitted_sum_plus_s0"] = fitted_sum_plus_s0.predict(c.priced)
        if constant_host_cost is not None:
            preds["constant_host_cost"] = _constant_host_prediction(
                c.priced, constant_host_cost, queue_depth)
        not_predicted = {}
        if nearest is not None:
            v = near_values.get(k)
            if v is None:
                not_predicted["nearest_pairing"] = (
                    "no pairing banked before this one shares its GPU or host "
                    "platform" if near_record["source"] is None else
                    f"{near_record['source']} has no ground truth for this cell")
            else:
                preds["nearest_pairing"] = v
        preds = {n: _finite_positive(v, f"the {n} prediction for cell {k}")
                 for n, v in preds.items()}
        cell_intervals = {}
        for name, iv in intervals.get(k, {}).items():
            cell_intervals[name] = _check_interval(iv, name, preds, k)
        out_cells.append({
            "model": k.model, "code_path": k.code_path, "arm": k.arm,
            "context_length": k.context_length, "density": k.density,
            "batch": k.batch,
            "components": _components(c.priced),
            "predictions": preds,
            "not_predicted": not_predicted,
            "intervals": cell_intervals,
        })

    return {
        "schema": SCHEMA, "pairing": pairing, "statistic": statistic,
        "queue_depth": queue_depth, "banked_before": list(banked),
        "written_at": _iso(written_at),
        "code": {"git_commit": code.git_commit, "git_dirty": code.git_dirty},
        "training": training, "not_fitted": not_fitted,
        "constant_host_cost": constant_host_cost,
        "nearest_pairing": near_record,
        "code_paths": code_paths,
        "cells": out_cells,
    }


def _check_code_paths(code_paths: Mapping[str, Mapping],
                      used: set) -> dict:
    """The definitions, checked against the labels the cells use."""
    if not isinstance(code_paths, Mapping):
        raise PredictionCommitError(
            f"code_paths maps a label to a definition, got {code_paths!r}")
    out = {}
    for label, defn in code_paths.items():
        d = _check_definition(defn, f"code path {label!r}")
        want = code_path_label(d)
        if label != want:
            raise PredictionCommitError(
                f"code path {label!r} is defined as {d}, whose label is "
                f"{want!r}; a label says one thing")
        out[label] = d
    missing = sorted(used - set(out))
    if missing:
        raise PredictionCommitError(
            f"cells name code path(s) {missing} that code_paths does not "
            f"define")
    unused = sorted(set(out) - used)
    if unused:
        raise PredictionCommitError(
            f"code_paths defines {unused}, which no cell uses")
    return out


def _constant_host_prediction(p: PricedSequence, constant: float,
                              queue_depth: int) -> float:
    return recurse(with_constant_host_cost(p, constant), queue_depth).end_to_end


# An interval's point is the committed prediction it belongs to, recomputed
# by the caller's bootstrap; allow for a different summation order only.
INTERVAL_POINT_RTOL = 1e-9


def _check_interval(iv, name: str, preds: Mapping[str, float], k) -> dict:
    if not isinstance(iv, Interval):
        raise PredictionCommitError(
            f"cell {k}: the {name} interval must be an intervals.Interval")
    d = iv.to_dict()
    _check_interval_dict(d, name, preds, f"cell {k}")
    return d


def _check_interval_dict(d: Mapping, name: str, preds: Mapping[str, float],
                         where: str) -> None:
    if name not in preds:
        raise PredictionCommitError(
            f"{where}: an interval for {name!r}, which has no prediction here")
    if not (d["lo"] <= d["point"] <= d["hi"]):
        raise PredictionCommitError(
            f"{where}: the {name} interval [{d['lo']}, {d['hi']}] does not "
            f"contain its point {d['point']}")
    if not math.isclose(d["point"], preds[name], rel_tol=INTERVAL_POINT_RTOL,
                        abs_tol=0.0):
        raise PredictionCommitError(
            f"{where}: the {name} interval is centred on {d['point']}, not on "
            f"the committed prediction {preds[name]}")


def canonical_bytes(record: Mapping) -> bytes:
    """The one serialisation that is hashed: sorted keys, no whitespace,
    ASCII, no NaN or infinity, a final newline. Floats are written with
    Python's repr, so every number reads back to the same double."""
    try:
        text = json.dumps(record, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise PredictionCommitError(
            f"the record cannot be written as canonical JSON: {e}") from e
    return (text + "\n").encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar_text(digest: str, filename: str) -> str:
    """`sha256sum` format, so `shasum -a 256 -c` checks the pair."""
    return f"{digest}  {filename}\n"


def parse_sidecar(text: str, filename: str) -> str:
    m = re.fullmatch(r"([0-9a-f]{64})  (\S+)\n", text)
    if not m:
        raise PredictionCommitError(f"unreadable sha256 sidecar: {text!r}")
    if m.group(2) != filename:
        raise PredictionCommitError(
            f"the sidecar names {m.group(2)!r}, not {filename!r}")
    return m.group(1)


# --------------------------------------------------------------------------
# Writing and syncing
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CommittedFile:
    path: Path
    sidecar: Path
    sha256: str


def prediction_filename(record: Mapping) -> str:
    t = _from_iso(record["written_at"])
    return f"prediction_{record['pairing']}_{t:%Y%m%dT%H%M%S%fZ}.json"


def write_prediction(record: Mapping, directory: Path | str) -> CommittedFile:
    """Write the file and its sidecar. Never overwrites either."""
    data = canonical_bytes(record)
    digest = sha256_hex(data)
    directory = Path(directory)
    name = prediction_filename(record)
    path = directory / name
    sidecar = directory / (name + ".sha256")
    for p in (path, sidecar):
        if p.exists():
            raise PredictionCommitError(
                f"{p} exists; a committed prediction is never overwritten")
    try:
        with open(path, "xb") as f:
            f.write(data)
        with open(sidecar, "x", encoding="ascii") as f:
            f.write(sidecar_text(digest, name))
    except FileExistsError as e:
        raise PredictionCommitError(
            f"{e.filename} exists; a committed prediction is never "
            f"overwritten") from e
    return CommittedFile(path=path, sidecar=sidecar, sha256=digest)


class Store(Protocol):
    """Where predictions are synced. `created` is an upper bound on when the
    object at `remote` was created, by the store's clock."""

    def put(self, local: Path, remote: str) -> None: ...
    def get(self, remote: str) -> bytes: ...
    def created(self, remote: str) -> datetime: ...


class GsutilStore:
    """A GCS bucket through `gsutil`. `put` never clobbers (`cp -n`): an
    object already at the path is left alone, and the read-back in
    `sync_prediction` then fails on the mismatch."""

    def __init__(self, gsutil: str = "gsutil"):
        self.gsutil = gsutil

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        try:
            p = subprocess.run([self.gsutil, *args], capture_output=True,
                               timeout=300)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PredictionCommitError(f"gsutil {args[0]} failed: {e}") from e
        if p.returncode != 0:
            raise PredictionCommitError(
                f"gsutil {' '.join(args)} exited {p.returncode}: "
                f"{p.stderr.decode(errors='replace').strip()}")
        return p

    @staticmethod
    def _check_uri(remote: str) -> None:
        if not remote.startswith("gs://"):
            raise PredictionCommitError(f"not a gs:// URI: {remote!r}")

    def put(self, local: Path, remote: str) -> None:
        self._check_uri(remote)
        self._run(["-q", "cp", "-n", str(local), remote])

    def get(self, remote: str) -> bytes:
        self._check_uri(remote)
        return self._run(["cat", remote]).stdout

    def created(self, remote: str) -> datetime:
        """`gsutil stat`'s creation time plus one second: it prints whole
        seconds, so the object was created before the returned time."""
        self._check_uri(remote)
        out = self._run(["stat", remote]).stdout.decode(errors="replace")
        m = re.search(r"^\s*Creation time:\s*(.+?)\s*$", out, re.MULTILINE)
        if not m:
            raise PredictionCommitError(
                f"gsutil stat gave no creation time for {remote}")
        try:
            t = parsedate_to_datetime(m.group(1))
        except (TypeError, ValueError) as e:
            raise PredictionCommitError(
                f"unreadable creation time {m.group(1)!r}") from e
        return _check_aware(t, "the creation time") + timedelta(seconds=1)


@dataclass(frozen=True)
class SyncReceipt:
    remote: str
    remote_sidecar: str
    sha256: str
    synced_at: datetime   # the later of the two objects' creation times

    def to_dict(self) -> dict:
        return {"remote": self.remote, "remote_sidecar": self.remote_sidecar,
                "sha256": self.sha256, "synced_at": _iso(self.synced_at)}


def _join(remote_dir: str, name: str) -> str:
    return remote_dir.rstrip("/") + "/" + name


def sync_prediction(committed: CommittedFile, store: Store,
                    remote_dir: str) -> SyncReceipt:
    """Upload the file and its sidecar, read both back, and return when the
    store holds exactly what was written. The timed workload starts only
    after this returns."""
    data = committed.path.read_bytes()
    if sha256_hex(data) != committed.sha256:
        raise PredictionCommitError(
            f"{committed.path} changed after it was written")
    side = committed.sidecar.read_text(encoding="ascii")
    remote = _join(remote_dir, committed.path.name)
    remote_side = _join(remote_dir, committed.sidecar.name)
    store.put(committed.path, remote)
    store.put(committed.sidecar, remote_side)
    if store.get(remote) != data:
        raise PredictionCommitError(
            f"{remote} does not hold the bytes written; an object already "
            f"there is never replaced")
    if store.get(remote_side).decode("ascii", errors="replace") != side:
        raise PredictionCommitError(f"{remote_side} does not hold the sidecar")
    synced_at = max(_check_aware(store.created(remote), "a creation time"),
                    _check_aware(store.created(remote_side), "a creation time"))
    return SyncReceipt(remote=remote, remote_sidecar=remote_side,
                       sha256=committed.sha256, synced_at=synced_at)


# --------------------------------------------------------------------------
# Reading, for the analysis
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CommittedCell:
    model: str
    code_path: str
    arm: str
    context_length: int
    density: float
    batch: int
    priced: PricedSequence
    predictions: Mapping[str, float]

    @property
    def key(self) -> CellKey:
        return cell_key(self.model, self.code_path, self.arm, self.density,
                        self.context_length, self.batch)


@dataclass(frozen=True)
class CommittedPrediction:
    pairing: str
    statistic: str
    queue_depth: int
    banked_before: tuple[str, ...]
    written_at: datetime
    synced_at: Optional[datetime]   # None from `verify_bytes` alone
    sha256: str
    code: CodeStamp
    training: Mapping[str, Mapping]
    not_fitted: Mapping[str, str]
    cells: tuple[CommittedCell, ...]
    constant_host_cost: Optional[float] = None
    code_paths: Mapping[str, Mapping] = dataclasses.field(default_factory=dict)

    def cell(self, model: str, code_path: str, arm: str, density: float,
             context_length: int, batch: int) -> CommittedCell:
        k = cell_key(model, code_path, arm, density, context_length, batch)
        for c in self.cells:
            if c.key == k:
                return c
        raise KeyError(k)


def _fitted_from_record(r: Mapping, pairing: str) -> FittedSum:
    cond = r["condition_number"]
    return FittedSum(
        a=r["a"], b=r["b"], c=r["c"], add_s0=r["add_s0"], target=pairing,
        kind=r["kind"], trained_on=tuple(r["trained_on"]),
        statistic=r["statistic"], n_rows=r["n_rows"],
        training_loss=r["training_loss"],
        condition_number=math.inf if cond == "inf" else cond,
        unidentified=tuple(r["unidentified"]))


def verify_bytes(data: bytes, sidecar: str, filename: str) -> CommittedPrediction:
    """Integrity only, no timing: the hash, the schema, the code stamp, the
    training records, and every deterministic prediction recomputed from the
    file's own components."""
    digest = sha256_hex(data)
    if parse_sidecar(sidecar, filename) != digest:
        raise PredictionCommitError(
            f"{filename}'s sha256 is {digest}, not the one its sidecar holds")
    try:
        rec = json.loads(data.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PredictionCommitError(f"{filename} is not canonical JSON") from e
    if canonical_bytes(rec) != data:
        raise PredictionCommitError(f"{filename} is not in canonical form")
    if rec.get("schema") != SCHEMA:
        raise PredictionCommitError(
            f"{filename} has schema {rec.get('schema')!r}, expected {SCHEMA}")
    if prediction_filename(rec) != filename:
        raise PredictionCommitError(
            f"{filename} does not match its own pairing and write stamp")
    try:
        code = CodeStamp(**rec["code"])
    except TypeError as e:
        raise PredictionCommitError(f"{filename}: unreadable code stamp") from e
    _check_code(code)
    pairing, Q = rec["pairing"], rec["queue_depth"]
    if isinstance(Q, bool) or not isinstance(Q, int) or Q < 1:
        raise PredictionCommitError(f"{filename}: queue depth {Q!r}")
    if rec["statistic"] not in STATISTICS:
        raise PredictionCommitError(f"{filename}: statistic {rec['statistic']!r}")
    banked = tuple(rec["banked_before"])
    training = rec["training"]
    unknown = sorted(set(training) - set(TRAINED_REQUIRED + TRAINED_OPTIONAL))
    if unknown:
        raise PredictionCommitError(f"{filename}: unknown baselines {unknown}")

    if banked:
        missing = [n for n in TRAINED_REQUIRED if n not in training]
        if missing:
            raise PredictionCommitError(
                f"{filename} has pairings banked before it but no {missing}")
        for name, r in training.items():
            if tuple(r["trained_on"]) != banked or r["kind"] != COMMITTED:
                raise PredictionCommitError(
                    f"{filename}: the {name} record does not name the "
                    f"committed training pairings {list(banked)}")
    elif training:
        raise PredictionCommitError(
            f"{filename} has no pairing banked before it but carries "
            f"{sorted(training)}")
    fitted = {n: _fitted_from_record(training[n], pairing)
              for n in ("fitted_sum", "fitted_sum_plus_s0") if n in training}
    constant = rec["constant_host_cost"]
    near = rec["nearest_pairing"]
    if near is not None:
        want_near = nearest_pairing(pairing, banked)
        if (near["source"], near["rule"]) != (want_near.source, want_near.rule):
            raise PredictionCommitError(
                f"{filename}: the nearest pairing is recorded as {near}, but "
                f"with {list(banked)} banked it is {want_near}")

    try:
        code_paths = _check_code_paths(
            rec["code_paths"], {c["code_path"] for c in rec["cells"]})
    except KeyError as e:
        raise PredictionCommitError(
            f"{filename}: a cell or the file lacks {e}; the key carries a "
            f"model and a code path") from e
    cells = []
    seen_keys = set()
    for c in rec["cells"]:
        p = priced_from_components(c["components"])
        try:
            k = cell_key(c["model"], c["code_path"], c["arm"], c["density"],
                         c["context_length"], c["batch"])
        except KeyError as e:
            raise PredictionCommitError(
                f"{filename}: a cell lacks {e}; a cell is keyed by model, "
                f"code path, arm, density, context length and batch") from e
        if k in seen_keys:
            raise PredictionCommitError(f"{filename}: cell {k} appears twice")
        seen_keys.add(k)
        _check_closing_sync(p, f"{filename} cell {k}")
        want = computed_predictions(p, Q)
        want.update({n: fs.predict(p) for n, fs in fitted.items()})
        if constant is not None:
            want["constant_host_cost"] = _constant_host_prediction(p, constant, Q)
        got = c["predictions"]
        expected = set(want) | ({"learned"} if "learned" in training else set())
        skipped = c["not_predicted"]
        if near is not None:
            if set(skipped) - {"nearest_pairing"}:
                raise PredictionCommitError(
                    f"{filename} cell {k}: unknown entries in not_predicted")
            if "nearest_pairing" not in skipped:
                expected.add("nearest_pairing")
        elif skipped:
            raise PredictionCommitError(
                f"{filename} cell {k}: not_predicted without a nearest pairing")
        for name, d in c["intervals"].items():
            _check_interval_dict(d, name, got, f"{filename} cell {k}")
        if set(got) != expected:
            raise PredictionCommitError(
                f"{filename} cell {k} has predictions {sorted(got)}, expected "
                f"{sorted(expected)}")
        for n, v in want.items():
            if got[n] != v:
                raise PredictionCommitError(
                    f"{filename} cell {k}: the committed {n} prediction "
                    f"{got[n]!r} does not follow from the committed "
                    f"components ({v!r})")
        cells.append(CommittedCell(model=k.model, code_path=k.code_path,
                                   arm=k.arm, context_length=k.context_length,
                                   density=k.density, batch=k.batch, priced=p,
                                   predictions=dict(got)))
    return CommittedPrediction(
        pairing=pairing, statistic=rec["statistic"], queue_depth=Q,
        banked_before=banked, written_at=_from_iso(rec["written_at"]),
        synced_at=None, sha256=digest, code=code, training=training,
        not_fitted=rec["not_fitted"], cells=tuple(cells),
        constant_host_cost=constant, code_paths=code_paths)


def load_committed(store: Store, remote: str, *,
                   ground_truth_started_at: datetime,
                   clock_skew: timedelta,
                   banked_at: Optional[Mapping[str, datetime]] = None,
                   local: Optional[Path | str] = None) -> CommittedPrediction:
    """The committed prediction the analysis may score, or a refusal.

    `ground_truth_started_at` is when the pairing's first timed block began
    (`ground_truth_start`). `clock_skew` is the largest difference allowed
    between the store's clock and the instance's; it only ever makes the
    check stricter. `banked_at` gives, for every pairing a committed
    baseline was trained on, when that pairing's ground truth was banked.
    `local`, if given, must hold the same bytes as the store.
    """
    start = _check_aware(ground_truth_started_at, "ground_truth_started_at")
    if not isinstance(clock_skew, timedelta) or clock_skew < timedelta(0):
        raise PredictionCommitError(
            f"clock_skew must be a timedelta >= 0, got {clock_skew!r}")
    filename = remote.rstrip("/").rsplit("/", 1)[-1]
    remote_side = remote + ".sha256"
    data = store.get(remote)
    side = store.get(remote_side).decode("ascii", errors="replace")
    if local is not None and Path(local).read_bytes() != data:
        raise PredictionCommitError(
            f"{local} and {remote} differ; the store's copy is the committed one")
    cp = verify_bytes(data, side, filename)
    synced_at = max(_check_aware(store.created(remote), "a creation time"),
                    _check_aware(store.created(remote_side), "a creation time"))

    if synced_at + clock_skew >= start:
        raise LatePredictionError(
            f"{filename} was synced by {_iso(synced_at)} (store clock, allowing "
            f"{clock_skew} of skew), not before the timed workload started at "
            f"{_iso(start)}; the analysis refuses it (K9)")
    if cp.written_at > synced_at + clock_skew:
        raise PredictionCommitError(
            f"{filename} says it was written at {_iso(cp.written_at)}, after "
            f"the store created it ({_iso(synced_at)}); the clocks disagree "
            f"by more than the {clock_skew} allowed")

    trained = sorted({p for r in cp.training.values() for p in r["trained_on"]})
    banked_at = dict(banked_at or {})
    for p in trained:
        if p not in banked_at:
            raise PredictionCommitError(
                f"{filename} has baselines trained on {p}, and no banking time "
                f"was given for it, so the order cannot be checked")
        b = _check_aware(banked_at[p], f"banked_at[{p!r}]")
        if b + clock_skew >= cp.written_at:
            raise LatePredictionError(
                f"{filename} was written at {_iso(cp.written_at)}, but {p}'s "
                f"ground truth, which its baselines were trained on, was "
                f"banked at {_iso(b)}: not before the prediction")
    return dataclasses.replace(cp, synced_at=synced_at)
