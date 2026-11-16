"""Score H0 to H3 on P2 and P4 at the lock (the adviser's rulings dated
16 Nov 2026, scoring wave, items 1 and 6; `attnbench/analysis/hypotheses.py`).

CPU only. The lock is the tag's (`git show <tag>:docs/lock_values.json`), and
the run refuses if it differs from `main`'s anywhere but `plan_sha256`, which
`main` re-syncs after every plan edit. The verdicts come from the analysis
commit's code: every module of this repo the run imports, apart from this
driver, must be byte-identical to its blob at the lock's `analysis_commit`
(`HY.check_analysis_commit` is then called with that commit; any other file
refuses, naming it). Each committed prediction is read through
`predict.commit.load_committed`, the frozen path `late_check` ran: the store's
sync times against the pairing's first ground-truth start, the training order
from the pairing ledger, and the learned predictor refitted on the banked
training files.

What is scored is the lock's `scored_pairings`: H0 and H2 on P2 (host side,
P1 the reference and the source), H1 on P2 and P4, H3 on P4; H4, H5 and H6
are reported "not scored" with the lock's reasons. Inputs (`--inputs`, JSON):

    {"bucket": "gs://...", "dense_arm": "dense",
     "training_files": {"P1": <path>, "P2": <path>},
     "P1": {"remote": "predictions/P1/...json", "local": <path>,
            "ground_truth_started_at": <ISO>, "ground_truth_started_at_from": <doc>},
     "P2": {"remote": ..., "local": ..., "ground_truth_dir": "12_ground_truth",
            "sessions": [<session dir>]},
     "P4": {"remote": ..., "local": ..., "ground_truth_dir": "13_ground_truth",
            "sessions": [<block session dir>, ...]}}

P1's ground truth is not re-read (it is the reference file only): its first
ground-truth start is the one its own record states, and the ledger's, if any,
is taken when earlier. A pairing run in blocks (P4) keeps its passed blocks
(`block_drift.collect`); a cell timed in more than one block (the anchor)
pools its clean samples across them, as the training files do (the adviser's
rulings dated 12 Nov 2026, item 3 (1)), and its within-session spread is
taken about each block's own median.

    python3 scripts/score_hypotheses.py --inputs <inputs.json> --mirror <dir> --live --out <dir>
    python3 scripts/score_hypotheses.py --inputs <inputs.json> --mirror <dir> --out <dir>

`--live` reads the store (ask before running it) and records every object it
reads, with its creation time, under `--mirror`; without it the run reads
only the mirror, so a second run reproduces the first offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from attnbench.analysis import block_drift as BD  # noqa: E402
from attnbench.analysis import gate_a as G  # noqa: E402
from attnbench.analysis import hypotheses as HY  # noqa: E402
from attnbench.predict import commit as C  # noqa: E402
from attnbench.predict import crossover as X  # noqa: E402
from attnbench.predict import p1_training as PT  # noqa: E402
from attnbench.predict import p2_training as P2T  # noqa: E402
from attnbench.predict.intensity import shape_for  # noqa: E402
from attnbench.predict.ledger import Ledger, record_cells  # noqa: E402
import gate_a as GA  # noqa: E402  (scripts/gate_a.py: step 9's group files)
import hypotheses as DRY  # noqa: E402  (scripts/hypotheses.py: the estimators' grids)

TAG = "lock-2026-11-14-r3"
LOCK_FILE = "docs/lock_values.json"
MAY_DIFFER = ("plan_sha256",)
SCORED_HERE = ("H0", "H1", "H2", "H3")
THIS = Path(__file__).resolve()


class ScoreError(ValueError):
    pass


# --- the lock: the tag's, checked against main's -------------------------------------

def lock_at(tag: str, root: Path = ROOT) -> dict:
    p = subprocess.run(["git", "-C", str(root), "show", f"{tag}:{LOCK_FILE}"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise ScoreError(f"no {LOCK_FILE} at {tag}: {p.stderr.strip()}")
    return json.loads(p.stdout)


def differing_paths(a, b, path: str = "") -> list:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}")
            else:
                out += differing_paths(a[k], b[k], f"{path}/{k}")
        return out
    return [] if a == b else [path or "/"]


def check_lock_source(tagged: Mapping, main: Mapping) -> list:
    """The paths where `main`'s lock differs from the tag's; refused unless
    they are only `MAY_DIFFER`."""
    diff = differing_paths(tagged, main)
    bad = [p for p in diff if p.lstrip("/") not in MAY_DIFFER]
    if bad:
        raise ScoreError(f"main's {LOCK_FILE} differs from the tag's at {bad}: the analysis "
                         f"reads the tag's values only")
    return diff


def locked_from(lock: Mapping) -> HY.Locked:
    """`HY.Locked` from the lock, as `Locked.from_lock` reads it for H0 to H3.
    `from_lock` (at the analysis commit) also requires H4's power target and
    n, which the lock types not applicable (H4 and H5 not scored, Appendix A,
    12 Nov 2026), so it refuses this lock; H4's fields are left unset here and
    H4 is never scored (a deviation, flagged in Appendix A, 16 Nov 2026)."""
    sp = lock.get("scored_pairings") or {}
    if not (isinstance(sp.get("H4"), dict) and sp["H4"].get("not_scored")):
        return HY.Locked.from_lock(lock)
    vals = lock.get("values") or {}

    def value(k):
        v = (vals.get(k) or {}).get("value")
        if v is None:
            raise HY.HypothesisError(f"the lock's {k} is not fixed (T4.1)")
        return v
    tol = value("h0_to_h3_tolerances")
    try:
        floors = HY.FloorTable.from_lock(lock)
    except HY.FloorsError as e:
        raise HY.HypothesisError(str(e)) from e
    try:
        boot = HY.IV.settings_from_lock(lock)
    except ValueError as e:
        raise HY.HypothesisError(str(e)) from e
    scored = {h: tuple(p) for h, p in sp.items() if not isinstance(p, dict)}
    for h in ("H1", "H2", "H3"):
        if HY.DEVELOPMENT_PAIRING in scored.get(h, ()):
            raise HY.HypothesisError(f"{h} names P1, the development pairing")
    return HY.Locked(
        tolerances={h: HY._fraction(tol[h], f"{h}'s tolerance") for h in ("H0", "H1", "H2")},
        floors=floors, bootstrap=boot, h4_bound_pts=float("nan"), h4_tests=(),
        h4_alpha=float("nan"), h4_n=0, scored=scored,
        sensitivity_s={HY._cell(k): float(v) / 1000.0
                       for k, v in value("marginal_label_s_sensitivity_floors").items()},
        analysis_commit=str(value("analysis_commit")))


# --- the analysis commit ---------------------------------------------------------------

def imported_files(root: Path = ROOT, modules=None) -> list:
    """This repo's files among the loaded modules, relative paths, this driver left out."""
    out = set()
    for m in list((modules if modules is not None else sys.modules).values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p == THIS or root not in p.parents:
            continue
        out.add(p.relative_to(root).as_posix())
    return sorted(out)


def not_at_commit(files, commit: str, root: Path = ROOT) -> list:
    """The files whose bytes differ from their blob at `commit` (or that it lacks)."""
    bad = []
    for rel in files:
        p = subprocess.run(["git", "-C", str(root), "show", f"{commit}:{rel}"],
                           capture_output=True)
        if p.returncode != 0 or p.stdout != (root / rel).read_bytes():
            bad.append(rel)
    return bad


def at_analysis_commit(locked: HY.Locked, root: Path = ROOT, modules=None) -> dict:
    files = imported_files(root, modules)
    bad = not_at_commit(files, locked.analysis_commit, root)
    if bad:
        raise ScoreError(f"these imported files differ from the analysis commit "
                         f"{locked.analysis_commit[:12]}: {bad}")
    HY.check_analysis_commit(locked, locked.analysis_commit)
    return {"analysis_commit": locked.analysis_commit, "files_checked": len(files)}


# --- the store: live and recorded, or the recording -------------------------------------

class RecordingStore:
    """A live store whose every read is kept, with its creation time, under `mirror`."""

    def __init__(self, store, mirror: Path):
        self.store, self.mirror = store, Path(mirror)
        self.index = _read_index(self.mirror)

    def _obj(self, remote):
        return self.mirror / "objects" / remote.split("://", 1)[-1]

    def _save(self):
        (self.mirror / "index.json").write_text(json.dumps(self.index, indent=1, sort_keys=True))

    def get(self, remote):
        data = self.store.get(remote)
        p = self._obj(remote)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        e = self.index.setdefault(remote, {})
        e.update(exists=True, sha256=hashlib.sha256(data).hexdigest())
        self._save()
        return data

    def exists(self, remote):
        ok = bool(self.store.exists(remote))
        self.index.setdefault(remote, {})["exists"] = ok
        self._save()
        return ok

    def created(self, remote):
        t = self.store.created(remote)
        self.index.setdefault(remote, {}).update(exists=True, created=t.isoformat())
        self._save()
        return t


class MirrorStore:
    """What a RecordingStore kept; anything it did not keep is refused, never
    read as absent."""

    def __init__(self, mirror: Path):
        self.mirror = Path(mirror)
        self.index = _read_index(self.mirror)

    def _entry(self, remote, key):
        e = self.index.get(remote)
        if e is None or key not in e:
            raise ScoreError(f"the mirror did not record {key} of {remote}")
        return e

    def get(self, remote):
        if not self._entry(remote, "exists")["exists"]:
            raise FileNotFoundError(remote)
        data = (self.mirror / "objects" / remote.split("://", 1)[-1]).read_bytes()
        if hashlib.sha256(data).hexdigest() != self._entry(remote, "sha256")["sha256"]:
            raise ScoreError(f"the mirror's copy of {remote} is not what was read")
        return data

    def exists(self, remote):
        return bool(self._entry(remote, "exists")["exists"])

    def created(self, remote):
        return datetime.fromisoformat(self._entry(remote, "created")["created"])


def _read_index(mirror: Path) -> dict:
    p = Path(mirror) / "index.json"
    return json.loads(p.read_text()) if p.exists() else {}


# --- ground truth ----------------------------------------------------------------------

def passed_sessions(dirs, gt_dir: str) -> tuple:
    """([(block id, [(name, result)])], [start stamps], {"discarded", "unpassed"}).
    Blocks go through `block_drift.collect`; an unblocked session (P2 on
    Flex-start) is one block, its id the directory's name, refused unless the
    last attempt of every step passed, ground truth and the late check among them."""
    dirs = [Path(d) for d in dirs]
    blocked = [bool((json.loads((d / "manifest.json").read_text())).get("block")) for d in dirs]
    if all(blocked):
        got = BD.collect(dirs, ground_truth_dir=gt_dir)
        out, starts = [], []
        for d in dirs:
            m = json.loads((d / "manifest.json").read_text())
            if m["block"]["id"] not in got["blocks"] or not (d / gt_dir).exists():
                continue
            if "ground_truth" not in m["block"]["steps"]:
                continue
            results, st = GA.ground_truth(d / gt_dir)
            out.append((m["block"]["id"], results))
            starts += st
        return out, starts, {"discarded": got["discarded"], "unpassed": got["unpassed"]}
    if any(blocked) or len(dirs) != 1:
        raise ScoreError(f"one unblocked session, or blocks only: {[str(d) for d in dirs]}")
    d = dirs[0]
    m = json.loads((d / "manifest.json").read_text())
    if m.get("dry_run") or (d / BD.DISCARD).exists():
        raise ScoreError(f"{d} is a dry run's or carries {BD.DISCARD}")
    last = {}
    for a in m.get("attempts") or ():
        last[a["step"]] = a.get("status")
    bad = [s for s, v in last.items() if v != "passed"]
    bad += [s for s in ("ground_truth", "late_check") if s not in last]
    if bad:
        raise ScoreError(f"{d}: steps not passed: {bad}")
    results, starts = GA.ground_truth(d / gt_dir)
    return [(d.name, results)], starts, {"discarded": [], "unpassed": []}


@dataclass(frozen=True)
class BlockCell:
    """A cell as one block timed it: the within-session spread is taken about
    each block's own median."""
    block: str
    key: tuple

    @property
    def context_length(self):
        return self.key.context_length

    @property
    def batch(self):
        return self.key.batch

    def __lt__(self, other):
        return (self.block, self.key) < (other.block, other.key)


def measure(cp, blocks, *, min_clean_rounds: int) -> dict:
    """{"seconds", "samples" (pooled, seconds), "per_block", "nominal",
    "excluded", "unmeasured", "pooled_cells"} over the passed blocks."""
    by: dict = {}
    pooled, per, nominal, excluded = {}, {}, {}, []
    for bid, results in blocks:
        m = G.measured(cp, results, min_clean_rounds=min_clean_rounds)
        excluded += [{**e, "block": bid} for e in m["excluded"]]
        for name, result in results:
            if G.short_of_rounds(result, min_clean_rounds):
                continue
            for row in result["rows"]:
                k = G.place_row(cp, row, name).key
                xs = [x / 1000.0 for x in row["prefill_ms_samples"]]
                per[BlockCell(bid, k)] = xs
                pooled.setdefault(k, []).extend(xs)
                nominal[k] = row.get("nominal_density") or "calibrated"
                by.setdefault(k, []).append(m["seconds"][k])
    many = sorted(k for k, v in by.items() if len(v) > 1)
    if many and cp.statistic != "median":
        raise ScoreError(f"pooling across blocks takes the median; the file's statistic "
                         f"is {cp.statistic}")
    seconds = {k: (v[0] if len(v) == 1 else float(statistics.median(pooled[k])))
               for k, v in by.items()}
    return {"seconds": seconds, "samples": pooled, "per_block": per, "nominal": nominal,
            "excluded": excluded, "pooled_cells": [list(k) for k in many],
            "unmeasured": sorted(c.key for c in cp.cells if c.key not in seconds)}


# --- the committed files ---------------------------------------------------------------

def sha256_of(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_inputs(spec: Mapping, lock: Mapping, banked: Mapping) -> tuple:
    """(rows, {pairing: sha256}) of the banked training files, each checked
    against its ledger entry (and P1's against the lock)."""
    files, rows = {}, []
    for p, path in sorted((spec.get("training_files") or {}).items()):
        sha = sha256_of(path)
        if p not in banked or banked[p][1]["training_file_sha256"] != sha:
            raise ScoreError(f"{path} has sha256 {sha}; the ledger's {p} entry is "
                             f"{banked.get(p, (None, {}))[1].get('training_file_sha256')}")
        if p == "P1" and (lock.get("p1_training") or {}).get("file_sha256") != sha:
            raise ScoreError(f"{path} is not the lock's P1 training file")
        files[p] = sha
        mod = {"P1": PT, "P2": P2T}[p]
        rows += mod.rows_from_record(mod.load(Path(path)), model=shape_for("primary"))
    return rows, files


def load(store, prefix: str, pairing: str, ps: Mapping, starts, *, banked: Mapping,
         rows, files: Mapping) -> tuple:
    """(the committed prediction through `load_committed`, how it was read).
    `local` is a copy kept on the Mac (P2's banked session), checked byte for
    byte against the store; P1's and P4's exist only in the store (null)."""
    local = Path(ps["local"]) if ps.get("local") else None
    record = json.loads(store.get(prefix + ps["remote"]))
    named = list(record.get("banked_before") or ())
    stated = []
    if ps.get("ground_truth_started_at"):
        stated = [datetime.fromisoformat(ps["ground_truth_started_at"].replace("Z", "+00:00"))]
    start = min([C.ground_truth_start(starts)] if starts else stated or [None])
    if start is None:
        raise ScoreError(f"{pairing}: no ground-truth start")
    first = Ledger(store, prefix, pairing).earliest_start(record_cells(record))
    if first is not None:
        start = min(start, first)
    cp = C.load_committed(store, prefix + ps["remote"], ground_truth_started_at=start,
                          clock_skew=timedelta(seconds=C.CLOCK_SKEW_ALLOWED_S),
                          banked_at={p: t for p, (t, _) in banked.items()}, local=local,
                          training_rows=rows if named else None,
                          training_files={p: files[p] for p in named} if named else None)
    if cp.pairing != pairing:
        raise ScoreError(f"{ps['remote']} is {cp.pairing}'s, not {pairing}'s")
    return cp, {"remote": ps["remote"], "sha256": cp.sha256, "trained_on": named,
                "ground_truth_started_at": start.isoformat(),
                "ledger_earliest_start": first.isoformat() if first else None}


# --- the scoring -----------------------------------------------------------------------

def h3_inputs(cp, m: Mapping, *, dense_arm: str, settings) -> tuple:
    """({estimator: predicted GridDifferences}, {estimator: measured}) at batch 1,
    as the P1 dry run builds them (`scripts/hypotheses.py`)."""
    pred = {c.key: c.predictions["recursion"] for c in cp.cells}
    predicted, measured = {}, {}
    for e, (s, d) in DRY.grids(cp, m["seconds"], m["nominal"], dense_arm=dense_arm).items():
        measured[e] = HY.measured_differences(m["samples"], s, d, statistic=statistics.median,
                                              settings=settings)
        predicted[e] = [X.GridDifference(int(a.context_length), pred[a] - pred[b],
                                         pred[a] - pred[b], pred[a] - pred[b])
                        for a, b in zip(s, d)]
    return predicted, measured


def spread(m: Mapping, *, quantile: float, budget_of) -> dict:
    return HY.within_session_spread(m["per_block"], quantile=quantile, budget_of=budget_of)


def score(lock: Mapping, locked: HY.Locked, cps: Mapping, ms: Mapping, *, dense_arm: str) -> dict:
    """Every hypothesis the lock scores here, on the loaded files and measurements."""
    sp = lock["scored_pairings"]
    want = {"H0": ["P2"], "H1": ["P2", "P4"], "H2": ["P2"], "H3": ["P4"]}
    for h, ps in want.items():
        if list(sp.get(h) or ()) != ps:
            raise ScoreError(f"the lock scores {h} on {sp.get(h)}; this driver scores it on {ps}")
    q = float(lock["values"]["floors_quantile"]["value"])
    spreads = {p: spread(ms[p], quantile=q, budget_of=locked.floors.budget_of) for p in ms}
    out = {"H1": {p: HY.h1(cps[p], ms[p]["seconds"], locked=locked, dense_arm=dense_arm,
                           spread_s=spreads[p]) for p in sp["H1"]}}
    out["H0"] = {"P2": HY.h0(cps["P1"], cps["P2"], side="host", floors=locked.floors,
                             tolerance=locked.tolerances["H0"])}
    out["H2"] = {"P2": HY.h2(cps["P2"], cps["P1"], ms["P2"]["seconds"], locked=locked,
                             dense_arm=dense_arm, h0_verdict=out["H0"]["P2"]["verdict"],
                             side="host", spread_s=spreads["P2"])}
    pred, meas = h3_inputs(cps["P4"], ms["P4"], dense_arm=dense_arm, settings=locked.bootstrap)
    out["H3"] = {"P4": HY.h3(pred, meas, floors=locked.floors, pairing="P4")}
    for h in ("H4", "H5", "H6"):
        ns = (sp.get(h) or {}).get("not_scored") if isinstance(sp.get(h), dict) else None
        if not ns:
            raise ScoreError(f"the lock scores {h}; this driver scores only {SCORED_HERE}")
        out[h] = {"verdict": HY.NOT_SCORED, **ns}
    out["within_session_spread_ms"] = {p: {f"{a}x{b}": v * 1000 for (a, b), v in s.items()}
                                       for p, s in spreads.items()}
    return out


# --- the report ------------------------------------------------------------------------

def _f(x, d=4):
    return "—" if x is None else f"{x:.{d}f}"


def to_markdown(rep: Mapping) -> str:
    s = rep["scores"]
    L = [f"# H0 to H3 scored at `{rep['lock']['tag']}` (CPU only)", "",
         f"Analysis commit `{rep['analysis']['analysis_commit'][:12]}` "
         f"({rep['analysis']['files_checked']} imported files byte-identical); "
         f"lock read from the tag, `main` differing only at {rep['lock']['main_differs_at']}.", ""]
    L += ["| Hypothesis | Pairing | Statistic | Threshold | Verdict |", "|---|---|---|---|---|"]
    h0 = s["H0"]["P2"]
    L.append(f"| H0 (host, P1→P2) | P2 | median abs. relative difference "
             f"{_f(h0.get('median_abs_relative_difference'))} over {h0['operations_scored']} "
             f"operations | ≤ {h0['tolerance']} | **{h0['verdict']}** |")
    for p, h in s["H1"].items():
        riv = ", ".join(f"{k} {_f(v)}" for k, v in h.get("rivals_median_abs_relative", {}).items())
        L.append(f"| H1 | {p} | recursion median abs. relative error "
                 f"{_f(h.get('recursion_median_abs_relative'))} over {h.get('n_cells')} cells; "
                 f"rivals {riv}; sign {h['sign']['agree']} agree, {len(h['sign']['disagree'])} "
                 f"disagree; XAttention pair {h['xattention_pair']['agree']} agree, "
                 f"{h['xattention_pair']['disagree']} disagree | ≤ {h['tolerance']}, below every "
                 f"rival, every sign, the pair within {h['tolerance']} | **{h['verdict']}** |")
    h2 = s["H2"]["P2"]
    L.append(f"| H2 (host from P1) | P2 | median abs. relative error "
             f"{_f(h2.get('median_abs_relative'))} over {h2.get('n_cells')} cells | "
             f"≤ {h2.get('tolerance')} and every sign | **{h2['verdict']}** |")
    h3 = s["H3"]["P4"]
    cnt = {}
    for r in h3["estimators"].values():
        cnt[r["verdict"]] = cnt.get(r["verdict"], 0) + 1
    L.append(f"| H3 | P4 | estimators: {', '.join(f'{v} {k}' for k, v in sorted(cnt.items()))} | "
             f"measured bracket inside the predicted | **{h3['verdict']}** |")
    for h in ("H4", "H5", "H6"):
        L.append(f"| {h} | — | — | — | **{s[h]['verdict']}** ({s[h]['appendix_a']}) |")
    L += ["", "## H1 parts", ""]
    for p, h in s["H1"].items():
        L.append(f"- {p}: {h['parts']}; recursion below: {h['recursion_below']}; "
                 f"marginal signs: {len(h['sign']['marginal'])}; inside the floor: "
                 f"{h['sign']['inside_floor']}; unmeasured: {len(h['unmeasured'])}.")
    L += ["", "## H3 per estimator (P4)", "", "| Estimator | Status | Verdict |", "|---|---|---|"]
    for e, r in h3["estimators"].items():
        L.append(f"| {e} | {r['status']} | {r['verdict']} |")
    L += ["", "## Inputs", ""]
    for p, r in rep["files"].items():
        L.append(f"- {p}: `{r['remote']}` sha256 `{r['sha256'][:12]}…`, trained on "
                 f"{r['trained_on'] or 'nothing'}, first ground truth {r['ground_truth_started_at']}.")
    for p, m in rep["measured"].items():
        L.append(f"- {p} measured: {m['cells']} cells, blocks {m['blocks']}, pooled across blocks "
                 f"{m['pooled_cells']}, groups left out {len(m['excluded'])}, unmeasured "
                 f"{len(m['unmeasured'])}.")
    return "\n".join(L) + "\n"


def run(args) -> dict:
    spec = json.loads(Path(args.inputs).read_text())
    tagged = lock_at(args.tag)
    main_differs = check_lock_source(tagged, json.loads((ROOT / LOCK_FILE).read_text()))
    locked = locked_from(tagged)
    if args.live:
        Path(args.mirror).mkdir(parents=True, exist_ok=True)
        store = RecordingStore(C.GsutilStore(), Path(args.mirror))
    else:
        store = MirrorStore(Path(args.mirror))
    prefix = spec["bucket"].rstrip("/") + "/"
    banked = {}
    for p in ("P1", "P2", "P4"):
        b = Ledger(store, prefix, p).banked()
        if b is not None:
            banked[p] = b
    rows, files = training_inputs(spec, tagged, banked)
    mcr = int(tagged["values"]["rerun_budget"]["value"]["min_clean_rounds"])
    cps, read, ms, meta = {}, {}, {}, {}
    for p in ("P1", "P2", "P4"):
        ps = spec[p]
        blocks, starts, left = ([], [], {})
        if p != "P1":
            blocks, starts, left = passed_sessions(ps["sessions"], ps["ground_truth_dir"])
        cps[p], read[p] = load(store, prefix, p, ps, starts, banked=banked, rows=rows,
                               files=files)
        if p != "P1":
            ms[p] = measure(cps[p], blocks, min_clean_rounds=mcr)
            meta[p] = {"blocks": [b for b, _ in blocks], "cells": len(ms[p]["seconds"]),
                       "pooled_cells": ms[p]["pooled_cells"], "excluded": ms[p]["excluded"],
                       "unmeasured": [list(k) for k in ms[p]["unmeasured"]], **left}
    analysis = at_analysis_commit(locked)
    scores = score(tagged, locked, cps, ms, dense_arm=spec.get("dense_arm", "dense"))
    return {"lock": {"tag": args.tag, "main_differs_at": main_differs,
                     "plan_sha256_at_tag": tagged["plan_sha256"]},
            "analysis": analysis, "files": read, "measured": meta,
            "banked": {p: t.isoformat() for p, (t, _) in banked.items()}, "scores": scores}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--mirror", required=True)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        rep = run(args)
    except (ScoreError, HY.HypothesisError, G.GateAError, BD.BlockDriftError,
            C.PredictionCommitError, PT.TrainingRowsError, OSError, KeyError, ValueError) as e:
        print(f"score_hypotheses: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hypotheses_scored.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    (out / "hypotheses_scored.md").write_text(to_markdown(rep))
    print(f"wrote {out}/hypotheses_scored.json and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
