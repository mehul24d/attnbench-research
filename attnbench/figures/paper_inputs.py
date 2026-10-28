"""What the paper's tables, figures and numbers read, checked once (C10).

One spec file names every input. Nothing is read that the spec does not
name, and every named file carries the sha256 it was written for:

  - `prediction_sha256`: the committed prediction the spec was written for.
    The prediction itself is read by `load_committed` (the caller passes the
    `CommittedPrediction`); `results.check_loaded` refuses one with no sync
    time. A prediction whose digest is not the spec's is refused as stale.
  - `results`: each ground-truth file (`ground_truth.time_arms`) with its
    sha256 and, per arm, the committed cell its row measures. Ground-truth
    rows carry no density, length or batch, so the spec states the cell
    (`results.measured_cells`). A file whose bytes are not the stated digest
    is refused.
  - `result_classes`: every committed arm's class, one of `CLASSES`. Oracle,
    deployable-estimator, CPU-builder and GPU-builder results are never
    labelled alike (rule 2): each class has its own label, every table row
    and figure panel names it, and no figure panel holds two classes. A
    class is checked against the row's own `arm_kind` and `deployable`
    where they decide it (`_check_class`).
  - `admissible`: per sparse cell, whether it is admissible (an accuracy
    input, measured elsewhere; no default). Read for decision error only.
  - `crossover` (optional): the bootstrap protocol for the measured
    difference, and a hashed file of the predicted difference intervals
    (T2.19's joint bootstrap of the component repeats, which the committed
    file does not carry). Without it no crossover is computed, and the
    outputs say so.
  - `relative_floor` (the noise floor, a relative error), whether it is
    provisional, and `resolution_floor_s` (seconds): lock values with no
    default (Section 5.6).

Every grouping keeps the model and the code path, so nothing is pooled
across code paths (rule 3).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from ..predict import intervals as IV
from ..predict.commit import COMPUTED, CommittedPrediction, cell_key
from ..predict.training import CellKey
from . import results as RS

SPEC_SCHEMA = "attnbench.paper_spec/1"
INTERVALS_SCHEMA = "attnbench.crossover_intervals/1"

DENSE = "dense"
ORACLE = "oracle"
DEPLOYABLE = "deployable_estimator"
CPU_BUILDER = "cpu_builder"
GPU_BUILDER = "gpu_builder"
PRECOMPUTED = "precomputed_ablation"
CLASSES = {
    DENSE: "dense",
    ORACLE: "oracle (non-deployable reference)",
    DEPLOYABLE: "deployable estimator",
    CPU_BUILDER: "CPU mask builder",
    GPU_BUILDER: "GPU mask builder",
    PRECOMPUTED: "ablation (masks precomputed offline)",
}
if len(set(CLASSES.values())) != len(CLASSES):   # pragma: no cover
    raise AssertionError("two result classes share a label")

_STATISTIC = {"median": statistics.median,
              "p90": lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.90)),
              "p99": lambda x: float(np.quantile(np.asarray(x, dtype=float), 0.99))}


class PaperInputError(ValueError):
    """Inputs the paper artifacts refuse."""


def class_label(cls: str) -> str:
    try:
        return CLASSES[cls]
    except KeyError:
        raise PaperInputError(f"result class {cls!r} is not one of {sorted(CLASSES)}") from None


def _check_class(arm: str, cls: str, row: Mapping, name: str) -> None:
    """The row's own kind, where it decides the class."""
    kind, dep = row.get("arm_kind"), row.get("deployable")
    if not isinstance(kind, str) or not isinstance(dep, bool):
        raise PaperInputError(f"{name}: arm {arm!r}'s row has no arm_kind and deployable, "
                              f"so its class cannot be checked")
    want = {"dense": DENSE, "oracle": ORACLE, "precomputed": PRECOMPUTED}.get(kind)
    if want is not None and cls != want:
        raise PaperInputError(f"{name}: arm {arm!r} is a {kind!r} arm, stated as {cls!r}")
    if want is None and cls in (DENSE, ORACLE, PRECOMPUTED):
        raise PaperInputError(f"{name}: arm {arm!r} is a {kind!r} arm, stated as {cls!r}")
    if cls == DEPLOYABLE and not dep:
        raise PaperInputError(f"{name}: arm {arm!r} is stated deployable; its row says not")


def _finite(x, what: str, *, lo: float = 0.0, strict: bool = False) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) \
            or x < lo or (strict and x <= lo):
        raise PaperInputError(f"{what} is {x!r}")
    return float(x)


def _sha_file(base: Path, entry: Mapping, what: str) -> tuple[Path, str]:
    if not isinstance(entry, Mapping) or "file" not in entry:
        raise PaperInputError(f"{what}: name its file")
    sha = entry.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        raise PaperInputError(f"{what} {entry['file']!r} states no sha256; unhashed inputs "
                              f"are refused")
    p = base / entry["file"]
    if not p.is_file():
        raise PaperInputError(f"{what} {p.name!r} is not a file")
    got = RS.sha256_file(p)
    if got != sha:
        raise PaperInputError(f"{what} {p.name!r} is not the file the spec was written for "
                              f"(sha256 {got}, spec {sha}); refusing stale input")
    return p, sha


@dataclass(frozen=True)
class Inputs:
    cp: CommittedPrediction
    measured: Mapping[CellKey, float]            # seconds, the file's statistic
    samples_s: Mapping[CellKey, tuple]           # each measured cell's clean repeats
    classes: Mapping[str, str]                   # arm -> class
    admissible: Mapping[CellKey, bool]
    dense_arm: str
    predictors: tuple[str, ...]
    relative_floor: float
    floor_provisional: bool
    resolution_floor_s: float
    crossover: Optional[Mapping]                 # protocol and predicted intervals
    files: tuple[dict, ...]                      # every input by role, name, sha256

    def cls(self, key) -> str:
        return self.classes[key.arm]


def read_spec(path: Union[str, Path]) -> dict:
    p = Path(path)
    spec = json.loads(p.read_text(encoding="utf-8"))
    if spec.get("schema") != SPEC_SCHEMA:
        raise PaperInputError(f"{p.name} is not a {SPEC_SCHEMA} spec")
    return spec


def ground_truth(spec: Mapping, base: Union[str, Path]) -> list[tuple[str, str, Mapping, Mapping]]:
    """(name, sha256, result, {arm: cell}) per ground-truth file, hash-checked."""
    base = Path(base)
    out, names = [], set()
    for r in spec.get("results") or ():
        p, sha = _sha_file(base, r, "ground-truth result")
        if p.name in names:
            raise PaperInputError(f"two ground-truth files named {p.name!r}")
        names.add(p.name)
        out.append((p.name, sha, json.loads(p.read_text(encoding="utf-8")), r.get("arms") or {}))
    if not out:
        raise PaperInputError("the spec names no ground truth")
    return out


def _samples(result: Mapping, arm: str, name: str, statistic: str, seconds: float) -> tuple:
    row = [r for r in result["rows"] if r.get("arm") == arm][0]
    xs = row.get("prefill_ms_samples")
    if xs is None:
        return ()
    if not isinstance(xs, list) or not xs or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0
            for v in xs):
        raise PaperInputError(f"{name}: arm {arm!r} has unusable samples")
    s = tuple(RS.ms_to_seconds(float(v)) for v in xs)
    if not math.isclose(_STATISTIC[statistic](s), seconds, rel_tol=1e-9):
        raise PaperInputError(f"{name}: arm {arm!r}'s {statistic} is not its samples' "
                              f"{statistic}")
    return s


def build_inputs(spec: Mapping, base: Union[str, Path], cp: CommittedPrediction) -> Inputs:
    """Every check of the module docstring. `cp` comes from `load_committed`."""
    RS.check_loaded(cp)
    base = Path(base)
    if spec.get("prediction_sha256") != cp.sha256:
        raise PaperInputError(f"the spec was written for prediction "
                              f"{spec.get('prediction_sha256')!r}, not {cp.sha256}; "
                              f"refusing stale input")
    rel_floor = _finite(spec.get("relative_floor"), "relative_floor (a lock value, no default)")
    res_floor = _finite(spec.get("resolution_floor_s"),
                        "resolution_floor_s (a lock value, no default)")
    prov = spec.get("floor_provisional")
    if not isinstance(prov, bool):
        raise PaperInputError("say whether the floors are provisional (floor_provisional)")
    dense_arm = spec.get("dense_arm")
    predictors = tuple(spec.get("predictors") or COMPUTED)
    if len(set(predictors)) != len(predictors):
        raise PaperInputError(f"name each predictor once, got {predictors}")

    classes = dict(spec.get("result_classes") or {})
    for arm, c in classes.items():
        class_label(c)
    committed_arms = {c.arm for c in cp.cells}
    missing = sorted(committed_arms - set(classes))
    if missing:
        raise PaperInputError(f"no result class for committed arm(s) {missing}")
    if not isinstance(dense_arm, str) or classes.get(dense_arm) != DENSE:
        raise PaperInputError(f"dense_arm {dense_arm!r} must be a committed arm of class dense")
    if [a for a, c in classes.items() if c == DENSE and a in committed_arms] != [dense_arm]:
        raise PaperInputError("exactly one committed arm is the dense arm")

    gt = ground_truth(spec, base)
    try:
        measured = RS.measured_cells([(n, r, a) for n, _, r, a in gt], statistic=cp.statistic)
    except RS.ResultsError as e:
        raise PaperInputError(str(e)) from e
    samples = {}
    for name, _, result, arms in gt:
        for arm, cell in arms.items():
            if arm not in classes:
                raise PaperInputError(f"{name}: arm {arm!r} has no result class")
            row = [r for r in result["rows"] if r.get("arm") == arm][0]
            _check_class(arm, classes[arm], row, name)
            k = cell_key(*cell)
            samples[k] = _samples(result, arm, name, cp.statistic, measured[k])

    admissible = {}
    for e in spec.get("admissible") or ():
        k = cell_key(*e["cell"])
        if not isinstance(e.get("admissible"), bool):
            raise PaperInputError(f"admissible for {k} is not a bool")
        if k in admissible:
            raise PaperInputError(f"admissibility stated twice for {k}")
        admissible[k] = e["admissible"]

    files = [{"role": "prediction", "name": None, "sha256": cp.sha256}]
    files += [{"role": "ground_truth", "name": n, "sha256": s} for n, s, _, _ in gt]
    crossover = None
    if spec.get("crossover") is not None:
        crossover = _crossover_inputs(spec["crossover"], base, cp, files)
    return Inputs(cp=cp, measured=measured, samples_s=samples, classes=classes,
                  admissible=admissible, dense_arm=dense_arm, predictors=predictors,
                  relative_floor=rel_floor, floor_provisional=prov,
                  resolution_floor_s=res_floor, crossover=crossover, files=tuple(files))


def _crossover_inputs(x: Mapping, base: Path, cp: CommittedPrediction, files: list) -> dict:
    level = x.get("level")
    if isinstance(level, bool) or not isinstance(level, float) or not 0 < level < 1:
        raise PaperInputError(f"crossover level is {level!r}")
    n_boot, seed = x.get("n_boot"), x.get("seed")
    for v, n in ((n_boot, "n_boot"), (seed, "seed")):
        if isinstance(v, bool) or not isinstance(v, int) or v < (1 if n == "n_boot" else 0):
            raise PaperInputError(f"crossover {n} is {v!r}")
    p, sha = _sha_file(base, x.get("predicted_intervals") or {}, "predicted crossover intervals")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("schema") != INTERVALS_SCHEMA:
        raise PaperInputError(f"{p.name} is not a {INTERVALS_SCHEMA} file")
    if doc.get("prediction_sha256") != cp.sha256:
        raise PaperInputError(f"{p.name} was made for another prediction; refusing stale input")
    entries = {}
    for e in doc.get("entries") or ():
        sk, dk = cell_key(*e["sparse_cell"]), cell_key(*e["dense_cell"])
        pred = e["predictor"]
        try:
            want = cp.cell(*sk).predictions[pred] - cp.cell(*dk).predictions[pred]
        except KeyError:
            raise PaperInputError(f"{p.name}: no committed {pred!r} for {sk} or {dk}") from None
        iv = IV.Interval(point=float(e["point"]), lo=float(e["lo"]), hi=float(e["hi"]),
                         level=float(e["level"]), n_boot=int(e["n_boot"]), seed=int(e["seed"]))
        if not math.isclose(iv.point, want, rel_tol=1e-9, abs_tol=1e-15):
            raise PaperInputError(f"{p.name}: the {pred} interval at {sk} is centred on "
                                  f"{iv.point}, not the committed difference {want}")
        if iv.level != level:
            raise PaperInputError(f"{p.name}: level {iv.level}, the measured side's is {level}")
        if not iv.lo <= iv.point <= iv.hi:
            raise PaperInputError(f"{p.name}: point outside its interval at {sk}")
        if (pred, sk) in entries:
            raise PaperInputError(f"{p.name}: {pred} at {sk} twice")
        entries[(pred, sk)] = (dk, iv)
    files.append({"role": "crossover_intervals", "name": p.name, "sha256": sha})
    return {"level": level, "n_boot": n_boot, "seed": seed, "predicted": entries}


def measured_difference(inp: Inputs, sparse: CellKey, dense: CellKey) -> IV.Interval:
    """Sparse minus dense, measured, with its bootstrap interval over the
    clean repeats (`intervals.bootstrap`, the file's statistic)."""
    x = inp.crossover
    s, d = inp.samples_s.get(sparse), inp.samples_s.get(dense)
    if not s or not d:
        raise PaperInputError(f"no repeats for {sparse if not s else dense}; the measured "
                              f"crossover needs them")
    b = IV.bootstrap({"s": s, "d": d}, lambda v: {"s": v["s"], "d": v["d"]},
                     n_boot=x["n_boot"], seed=x["seed"],
                     statistic=_STATISTIC[inp.cp.statistic])
    return b.difference("s", "d", x["level"])
