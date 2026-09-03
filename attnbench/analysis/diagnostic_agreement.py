"""Does the pipeline reproduce what the diagnostic measured?

Segment 2 re-runs Stage 1 under `compile_guard`, and its flex block-sparse
rows must agree with the 2026-09-04 diagnostic before 504 cells are measured
through that path. If they do not, something differs between the diagnostic
path (a small script calling `check_for_family` directly) and the pipeline
path (`run_probe.py` over the full grid), and that is worth knowing while it
costs one comparison rather than a segment.

**Why this can name its failure instead of merely detecting one.** Eager and
compiled flex are different implementations, and they leave different
numerical fingerprints at the same config, seed, machine and commit:

    config (seq_len 1024, sparsity 0.9)   eager      compiled
    block_size=64                         0.015746   0.008404
    block_size=128                        0.016893   0.008983

Consistently ~1.9x apart -- eager materialises the score matrix and takes one
softmax in the compute dtype, while the compiled kernel accumulates online in
fp32, so it is the more accurate of the two. That gap is itself independent
evidence that the two paths are different code, arrived at without reading a
single log line.

It also means a re-run that reports ~0.0157 has not merely "disagreed": it has
fallen back to eager again, and this module says so by name
(`EAGER_SIGNATURE`) rather than emitting a generic mismatch that a reader
would have to diagnose from scratch a second time.

**Coverage, stated plainly.** The 2026-09-04 diagnostic ran two sparse configs
in six minutes, so this compares 2 of the 72 flex block-sparse cells. That is
adequate for the failure it targets and inadequate for others, and the
difference is worth being explicit about: the recompile-limit fallback is
**process-scoped**, so once it fires every later flex call in that process is
eager and any two cells will show it. A per-config defect affecting only cells
this does not cover would pass here. This is a spot-check on a global
property, not coverage of the grid.

Validated against segment 1's own correctness parquet, which it correctly
reports as EAGER_SIGNATURE on both cells at 0.0% distance from the eager
fingerprint -- the known-bad run is the best available test case.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_COMPILED_REF = Path(
    "results/diagnostics/20260904_flex_recheck/flex_recheck.json")
DEFAULT_EAGER_REF = Path(
    "results/stage2/segment_20260903_seg1/probe_final/correctness.parquet")

# Fractional agreement required. Deliberately loose: the claim is "this is the
# same implementation", and the two candidates are ~1.9x apart, so anything
# under ~40% separates them cleanly while tolerating ordinary run-to-run
# variation in a reduction over random inputs. A tight tolerance here would
# manufacture failures without distinguishing anything the loose one cannot.
REL_TOL = 0.15

_ERR_RE = re.compile(r"max_abs_err=([0-9.eE+-]+)")


class DiagnosticAgreementError(RuntimeError):
    """The pipeline did not reproduce the diagnostic."""


@dataclass
class AgreementRow:
    config_key: str
    observed: Optional[float]
    compiled_ref: Optional[float]
    eager_ref: Optional[float]
    verdict: str          # AGREES | EAGER_SIGNATURE | DISAGREES | NOT_MEASURED
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _rel(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def load_compiled_reference(path: Path = DEFAULT_COMPILED_REF) -> dict[str, float]:
    """{config_key: max_abs_err} for cells the diagnostic ran compiled.

    Reads `max_abs_err` if the row carries it as a field, and otherwise parses
    it out of `detail` -- the 2026-09-04 run recorded it only in the free-text
    detail string, and re-deriving the number by hand would put a transcription
    error between the reference and the thing it references.
    """
    payload = json.loads(Path(path).read_text())
    out: dict[str, float] = {}
    for row in payload.get("rows", []):
        if row.get("verdict") != "OK":
            continue
        val = row.get("max_abs_err")
        if val is None:
            m = _ERR_RE.search(str(row.get("detail", "")))
            if not m:
                continue
            val = float(m.group(1))
        out[row["config_key"]] = float(val)
    return out


def load_eager_reference(path: Path = DEFAULT_EAGER_REF,
                         backend: str = "flex") -> dict[str, float]:
    """{config_key: max_abs_err} from the run that was silently eager.

    Not a baseline to match -- a fingerprint to recognise. Matching it is the
    failure this module exists to name.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    df = df[(df.backend == backend) & (df["mask"] == "block_sparse")]
    df = df[df.max_abs_err.notna()]
    return {r.config_key: float(r.max_abs_err) for r in df.itertuples()}


def compare(observed: dict[str, float],
            compiled_ref: dict[str, float],
            eager_ref: Optional[dict[str, float]] = None,
            *, rel_tol: float = REL_TOL) -> list[AgreementRow]:
    """One row per reference cell. Cells absent from `observed` are reported,
    not skipped: a pipeline that stopped producing a cell has changed too."""
    eager_ref = eager_ref or {}
    rows: list[AgreementRow] = []

    for key, ref in sorted(compiled_ref.items()):
        got = observed.get(key)
        eager = eager_ref.get(key)
        if got is None:
            rows.append(AgreementRow(key, None, ref, eager, "NOT_MEASURED",
                                     "cell present in the diagnostic but "
                                     "absent from the pipeline run"))
            continue

        d_compiled = _rel(got, ref)
        if d_compiled <= rel_tol:
            rows.append(AgreementRow(key, got, ref, eager, "AGREES",
                                     f"within {d_compiled:.1%} of the "
                                     f"diagnostic"))
            continue

        if eager is not None and _rel(got, eager) <= rel_tol:
            rows.append(AgreementRow(
                key, got, ref, eager, "EAGER_SIGNATURE",
                f"matches the EAGER fingerprint ({eager:.6f}) to "
                f"{_rel(got, eager):.1%}, not the compiled one ({ref:.6f}). "
                f"torch.compile fell back again -- check compile_guard and "
                f"the recompile limit before trusting any flex row"))
            continue

        rows.append(AgreementRow(
            key, got, ref, eager, "DISAGREES",
            f"{d_compiled:.1%} from the diagnostic ({ref:.6f}) and does not "
            f"match the eager fingerprint either -- a third behaviour"))

    return rows


def assert_agreement(rows: Iterable[AgreementRow]) -> None:
    bad = [r for r in rows if r.verdict != "AGREES"]
    if not bad:
        return
    lines = [f"  {r.config_key} [{r.verdict}] observed={r.observed} "
             f"{r.detail}" for r in bad]
    raise DiagnosticAgreementError(
        f"{len(bad)} of {len(list(rows))} flex cells did not reproduce the "
        f"2026-09-04 diagnostic:\n" + "\n".join(lines))


def observed_from_correctness(path: Path, backend: str = "flex",
                              mask: str = "block_sparse") -> dict[str, float]:
    """{config_key: max_abs_err} from a Stage 1 correctness parquet."""
    import pandas as pd

    df = pd.read_parquet(path)
    df = df[(df.backend == backend) & (df["mask"] == mask)]
    df = df[df.max_abs_err.notna()]
    return {r.config_key: float(r.max_abs_err) for r in df.itertuples()}
