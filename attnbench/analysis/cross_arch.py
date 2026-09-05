"""Combining Stage 2 timing results measured on different GPUs.

The whole hardware-conditional claim of this study -- "backend X wins on an
L4 but loses on an H100" -- rests on this path, and it is the one part of
the pipeline that must NEVER silently do the obvious thing.

The rule, in one line: **a speedup ratio is only ever computed between two
measurements taken on the same physical machine.** Cross-architecture
comparison happens between *ratios*, never between raw latencies.

Why that matters concretely. If backend A runs at 10ms on an L4 and the
dense baseline runs at 12ms on an H100, "A is 1.2x faster" is not a claim
about A -- it is a claim about the two GPUs, wearing A's name. The two
machines differ in clocks, memory bandwidth, driver, and thermal state.
Nothing downstream of a results file can recover which machine a row came
from once a ratio has been taken across them, so the guard has to live at
the point the ratio is formed.

The workflow this supports:

1. Each machine writes its OWN results file. `sweep.check_host_continuity`
   already refuses to resume into a file written on a different host, so
   this is enforced at collection time, not merely conventional.
2. Every row carries `host` and `gpu_name` (from `provenance.capture()`).
3. Files are joined ONLY here, at analysis time, by
   `load_segments`.
4. Speedups are computed per-host against a baseline measured on that same
   host (`speedup_within_host`), which requires the baseline backend to
   have been re-measured on every machine -- enforced, not assumed.
5. Only then are per-host speedups compared across architectures
   (`compare_across_architectures`).

**Preemption is the same case.** A Spot/preemptible instance that dies
mid-sweep and resumes elsewhere produces exactly this shape: two partial
segments on two machines. The recovery path is a new results file per
segment, joined here -- NOT resuming into the original file, which
`check_host_continuity` blocks. Nothing in this module assumes one file per
architecture, so N segments per GPU are fine, and two GPUs of the same model
(same `gpu_name`, different `host`) stay separate because grouping is by
host.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Columns every segment must carry for a cross-architecture join to be
# meaningful. `host` identifies the physical machine (the unit a ratio may
# be taken within); `gpu_name` identifies the architecture (the unit results
# are compared across). They are different things: two rented L4s are one
# gpu_name and two hosts.
REQUIRED_COLUMNS = ("host", "gpu_name", "backend", "config_key")


class CrossArchError(RuntimeError):
    """A cross-architecture combination that would produce an invalid number."""


@dataclass(frozen=True)
class Speedup:
    """One backend's speedup against the baseline, on ONE machine.

    `host` and `gpu_name` are both retained: the ratio is only valid within
    `host`, and `gpu_name` is what it may later be compared across.

    `clocks_locked` travels with the ratio because it describes how noisy
    that ratio is, and the harm from unlocked clocks lands here -- in a
    comparison -- rather than at the moment of measurement. Gating the
    measurement instead would block the second architecture entirely:
    `nvidia-smi -lgc` needs root and fails on most rental hosts, and losing
    the hardware-conditional result costs far more than the extra variance.

    True only when BOTH sides were locked. A ratio is as noisy as its noisier
    half, so "locked" cannot mean "locked somewhere in the pair". None means
    unknown -- rows predating the field, which must not be reported as
    unlocked: absence of evidence is not evidence of absence, and a flag that
    fires on missing data gets ignored within a day.
    """

    host: str
    gpu_name: str
    backend: str
    config_key: str
    latency_ms: float
    baseline_latency_ms: float
    clocks_locked: Optional[bool] = None

    @property
    def speedup(self) -> float:
        """Baseline over backend: >1 means the backend is faster."""
        return self.baseline_latency_ms / self.latency_ms


def segment_digest(df: pd.DataFrame) -> str:
    """Order-independent content hash of a segment's measurements.

    Hashes the sorted (config_key, backend, host, latency) tuples rather than
    the file bytes: parquet encoding, column order and row order are all
    incidental, and a digest that changes when they do would cry wolf. Two
    segments with the same digest recorded the same measurements.
    """
    cols = [c for c in ("config_key", "backend", "host", "latency_ms_p50")
            if c in df.columns]
    payload = df[cols].astype(str).agg("|".join, axis=1).sort_values().str.cat(sep="\n")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _check_commit_consistency(frames: list[tuple[Path, pd.DataFrame]],
                              *, allow_unverified: bool,
                              allow_mixed_commits: bool) -> None:
    """A segment measured at a different commit is a different experiment.

    This is not hypothetical. On 2026-09-03 a bug was found in
    `masks.to_dense_bool`: causal block-sparse masks permitted attention to
    up to `block_size - 1` future tokens inside diagonal blocks. Any segment
    measured before that fix and any segment measured after it are answers to
    different questions, and averaging them would produce a number belonging
    to neither.

    Silently merging is the failure this refuses. Mixing is still possible,
    but only by asking for it in the call, which puts the decision in the
    caller's code where a reader can see it.
    """
    commits: dict[str, list[str]] = {}
    unverified = []
    for path, df in frames:
        if "git_commit" not in df.columns:
            unverified.append(f"{path} (no git_commit column)")
            continue
        values = sorted({str(c) for c in df["git_commit"].dropna().unique()})
        # "HEAD" is what git echoes to stdout when the repo has no commits;
        # provenance now records None instead, but files written before that
        # fix carry the literal string and must not be read as a real commit.
        real = [c for c in values if c not in ("None", "nan", "HEAD", "")]
        if not real:
            unverified.append(f"{path} (commit is {values or ['missing']})")
            continue
        for c in real:
            commits.setdefault(c, []).append(str(path))

    if unverified and not allow_unverified:
        raise CrossArchError(
            "segment(s) carry no usable git commit, so it cannot be shown they "
            "were measured from the same code:\n  " + "\n  ".join(unverified) +
            "\n\nThis is what a repository with no commits produces. Commit the "
            "code and re-measure, or pass allow_unverified=True to state "
            "explicitly that you are joining results whose provenance cannot "
            "be checked."
        )

    if len(commits) > 1 and not allow_mixed_commits:
        detail = "\n  ".join(f"{c[:12]}: {', '.join(paths)}"
                             for c, paths in sorted(commits.items()))
        raise CrossArchError(
            f"segments were measured at {len(commits)} different commits:\n  "
            f"{detail}\n\nA code change between segments makes them different "
            f"experiments -- the 2026-09-03 to_dense_bool causality fix is a "
            f"concrete example of a change that alters results. Re-measure the "
            f"older segments, or pass allow_mixed_commits=True if you have "
            f"established the difference cannot affect these rows."
        )


def _check_no_duplicate_cells(df: pd.DataFrame) -> None:
    """Two measurements of the same (config_key, backend, host) mean something
    went wrong, and silently keeping one of them hides which.

    Deliberately NOT resolved by last-wins. A duplicate is either a re-measure
    that should have gone into a new segment with a recorded reason, or a
    checkpoint appended twice -- and those want opposite treatment. Neither is
    served by picking one quietly.
    """
    keys = [c for c in ("config_key", "backend", "host") if c in df.columns]
    if len(keys) < 3:
        return
    dupes = df[df.duplicated(subset=keys, keep=False)]
    if dupes.empty:
        return
    listing = (dupes.groupby(keys, dropna=False)
                    .agg(n=("backend", "size"),
                         segments=("segment", lambda s: sorted(set(s))))
                    .reset_index().head(10))
    raise CrossArchError(
        f"{len(dupes)} row(s) duplicate a (config_key, backend, host) cell that "
        f"was already measured. Showing up to 10 groups:\n{listing.to_string(index=False)}"
        f"\n\nA cell measured twice is a question, not a detail: it is either a "
        f"re-measurement that belongs in its own segment with a recorded reason, "
        f"or the same checkpoint joined twice. Resolve it deliberately rather "
        f"than letting the join pick one."
    )


def load_segments(paths: Iterable[Path | str], *,
                  allow_unverified: bool = False,
                  allow_mixed_commits: bool = False) -> pd.DataFrame:
    """Load and concatenate per-machine results files.

    Deliberately takes many paths rather than one directory: a machine may
    contribute several files (a preemption splits a sweep into segments),
    and an architecture may be represented by several machines. Nothing here
    assumes one file per architecture or per host.

    Raises if a segment is missing the columns needed to keep hosts
    distinguishable -- a file without `host`/`gpu_name` cannot be safely
    joined with anything, and silently concatenating it would produce
    exactly the cross-host ratio this module exists to prevent.
    """
    frames = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise CrossArchError(f"segment not found: {p}")
        df = pd.read_parquet(p)
        if df.empty:
            continue
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise CrossArchError(
                f"{p} is missing {missing}. Every results row must carry its "
                f"provenance -- without host/gpu_name a row cannot be kept on "
                f"the right side of a same-machine comparison."
            )
        df = df.copy()
        df["segment"] = str(p)
        df["segment_digest"] = segment_digest(df)
        frames.append((p, df))

    if not frames:
        raise CrossArchError("no non-empty segments to load")

    _check_commit_consistency(frames, allow_unverified=allow_unverified,
                              allow_mixed_commits=allow_mixed_commits)
    joined = pd.concat([df for _, df in frames], ignore_index=True)
    _check_no_duplicate_cells(joined)
    return joined


def hosts_by_architecture(df: pd.DataFrame) -> dict[str, list[str]]:
    """Which physical machines contributed each gpu_name.

    More than one host per architecture is normal and fine (a re-rental, a
    preemption). It is surfaced because a reader comparing architectures
    should know whether "the L4 result" came from one machine or three.
    """
    out: dict[str, list[str]] = {}
    for gpu_name, group in df.groupby("gpu_name"):
        out[str(gpu_name)] = sorted(str(h) for h in group["host"].unique())
    return out


def speedup_within_host(df: pd.DataFrame, *, baseline_backend: str,
                         latency_column: str = "latency_ms_p50",
                         ) -> list[Speedup]:
    """Speedup of every backend against `baseline_backend`, computed
    strictly within each (host, config_key).

    The baseline must have been re-measured on every host present. If a
    machine contributed backend rows but no baseline row, its speedups are
    not computable, and this raises rather than falling back to another
    machine's baseline -- which is exactly the invalid comparison the module
    exists to prevent, and would look completely ordinary in the output.
    """
    if latency_column not in df.columns:
        raise CrossArchError(f"no {latency_column!r} column in the joined results")

    usable = df[df[latency_column].notna()]
    if "ok" in usable.columns:
        usable = usable[usable["ok"].astype(bool)]

    hosts = sorted(str(h) for h in usable["host"].unique())
    hosts_without_baseline = [
        h for h in hosts
        if usable[(usable["host"] == h)
                  & (usable["backend"] == baseline_backend)].empty
    ]
    if hosts_without_baseline:
        raise CrossArchError(
            f"baseline backend {baseline_backend!r} was not measured on "
            f"host(s) {hosts_without_baseline}. A speedup for those rows "
            f"could only be formed against another machine's baseline, which "
            f"would be a comparison between two GPUs rather than between two "
            f"backends. Re-measure the baseline on every machine."
        )

    results: list[Speedup] = []
    for (host, config_key), cell in usable.groupby(["host", "config_key"]):
        base_rows = cell[cell["backend"] == baseline_backend]
        if base_rows.empty:
            # This (host, config) has no baseline -- e.g. the baseline OOM'd
            # at this shape. Skipped rather than raised: it is a legitimate
            # per-cell gap, unlike a whole machine missing the baseline.
            continue
        baseline_latency = float(base_rows[latency_column].iloc[0])
        gpu_name = str(cell["gpu_name"].iloc[0])
        base_locked = _locked(base_rows.iloc[0])
        for _, row in cell.iterrows():
            if row["backend"] == baseline_backend:
                continue
            results.append(Speedup(
                host=str(host), gpu_name=gpu_name, backend=str(row["backend"]),
                config_key=str(config_key),
                latency_ms=float(row[latency_column]),
                baseline_latency_ms=baseline_latency,
                clocks_locked=_both_locked(base_locked, _locked(row))))
    return results


def _locked(row) -> Optional[bool]:
    """`clocks_locked` for one row: True, False, or None for unknown.

    None for a missing column and for NaN, deliberately. `bool(nan)` is True,
    so a naive read would report every row predating the field as LOCKED --
    the silent direction, and the one that would quietly certify a noisy
    comparison as clean.
    """
    try:
        v = row["clocks_locked"]
    except (KeyError, IndexError, TypeError):
        return None
    if v is None or (isinstance(v, float) and v != v):
        return None
    return bool(v)


def _both_locked(a: Optional[bool], b: Optional[bool]) -> Optional[bool]:
    """A ratio is as noisy as its noisier half."""
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


@dataclass(frozen=True)
class ArchitectureComparison:
    """One backend/config's speedup on each architecture.

    This is the hardware-conditional claim's unit: two RATIOS, each formed
    within its own machine, compared with each other. No raw latency from
    one machine is ever divided by a raw latency from another.
    """

    backend: str
    config_key: str
    speedup_by_architecture: dict[str, float]
    # gpu_name -> whether every contributing ratio had both sides locked.
    clocks_locked_by_architecture: dict[str, Optional[bool]] = field(
        default_factory=dict)

    @property
    def architectures(self) -> list[str]:
        return sorted(self.speedup_by_architecture)

    @property
    def unlocked_architectures(self) -> list[str]:
        """Architectures whose ratio came from unlocked clocks.

        Non-empty means this comparison is noisier than the numbers suggest.
        It is NOT a reason to discard the comparison: `nvidia-smi -lgc` needs
        root and fails on most rental hosts, so refusing here would delete the
        second architecture and with it the hardware-conditional finding --
        which costs far more than the variance does.
        """
        return sorted(a for a, locked in self.clocks_locked_by_architecture.items()
                      if locked is False)

    @property
    def locked_architectures(self) -> list[str]:
        """Architectures whose ratio came from locked clocks.

        Exists so `asymmetrically_controlled` can tell the two failure shapes
        apart. `is True` rather than `not locked is False`, because unknown
        (a row predating the field, or NaN) is not evidence of locking.
        """
        return sorted(a for a, locked in self.clocks_locked_by_architecture.items()
                      if locked is True)

    @property
    def asymmetrically_controlled(self) -> bool:
        """One side locked, the other not.

        Distinct from "unlocked", and the distinction is the point. If both
        sides are unlocked the comparison is uniformly noisy, which a reader
        discounts uniformly. If one side is locked, the halves are measured to
        DIFFERENT precision and the noisier one is not identifiable from the
        ratio -- so a difference between architectures can be read as hardware
        behaviour when part of it is measurement quality.

        This is the shape the A100 session produces if its clock lock
        succeeds: every L4 row in the dataset was measured unlocked.
        """
        return bool(self.locked_architectures) and bool(self.unlocked_architectures)

    def flips(self, *, threshold: float = 1.0) -> bool:
        """True when the backend beats the baseline on one architecture and
        loses on another -- the hardware-conditional result this study is
        looking for."""
        values = list(self.speedup_by_architecture.values())
        return any(v > threshold for v in values) and any(v <= threshold for v in values)

    def caveat(self) -> str:
        """The sentence that must accompany this number, or "".

        A flip is the study's headline claim, and a flip between 0.98 and 1.02
        from unlocked clocks is not a flip at all -- DRIFT_TOLERANCE is 5%,
        which is the scale of the effect being claimed. So the caveat is
        harshest exactly where the result is most interesting.
        """
        unlocked = self.unlocked_architectures
        if not unlocked:
            return ""
        base = (f"clocks were NOT locked on {', '.join(unlocked)}, so this "
                f"ratio carries run-to-run variance of roughly the same "
                f"magnitude as the canary tolerance (5%)")
        if self.asymmetrically_controlled:
            # Naming only the unlocked side reads as "everything here is
            # unlocked", which invites discounting both halves equally. The
            # asymmetry is the thing a reader cannot recover from the number.
            base += (f"; clocks WERE locked on "
                     f"{', '.join(self.locked_architectures)}, so the two "
                     f"sides of this comparison are NOT controlled to the "
                     f"same precision and the difference between them is "
                     f"partly a difference in measurement quality")
        if self.flips():
            return (base + ". This comparison FLIPS, and a flip within that "
                    "margin is not evidence of hardware-conditional behaviour "
                    "-- re-measure with locked clocks before claiming it")
        return base


def compare_across_architectures(speedups: list[Speedup],
                                  ) -> list[ArchitectureComparison]:
    """Group per-host speedups by architecture, for backend/config pairs
    measured on more than one architecture.

    Where an architecture was measured on several hosts, their speedups are
    averaged -- averaging RATIOS across machines is valid in a way averaging
    latencies never is, because each ratio is already normalised by its own
    machine's baseline.

    Pairs present on only one architecture are excluded: there is no
    cross-architecture claim to make about them, and including them with a
    single entry invites reading a one-sided result as a comparison.
    """
    by_key: dict[tuple[str, str], dict[str, list[float]]] = {}
    locks: dict[tuple[str, str], dict[str, list[Optional[bool]]]] = {}
    for s in speedups:
        arches = by_key.setdefault((s.backend, s.config_key), {})
        arches.setdefault(s.gpu_name, []).append(s.speedup)
        locks.setdefault((s.backend, s.config_key), {}) \
             .setdefault(s.gpu_name, []).append(s.clocks_locked)

    out = []
    for (backend, config_key), arches in sorted(by_key.items()):
        if len(arches) < 2:
            continue
        # Averaging ratios across hosts of one architecture is valid; the lock
        # state is not averaged but REDUCED -- one unlocked host makes the
        # architecture's mean ratio an unlocked number, because the variance
        # it contributed is still in there.
        by_arch_lock: dict[str, Optional[bool]] = {}
        for gpu, vals in locks[(backend, config_key)].items():
            reduced: Optional[bool] = True
            for v in vals:
                reduced = _both_locked(reduced, v)
            by_arch_lock[gpu] = reduced
        out.append(ArchitectureComparison(
            backend=backend, config_key=config_key,
            speedup_by_architecture={
                gpu: sum(vals) / len(vals) for gpu, vals in arches.items()},
            clocks_locked_by_architecture=by_arch_lock))
    return out
