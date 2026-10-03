"""Recall rows of the estimator-frontier study: the `split` column, budget
selection on the selection split only, and the firewall in front of the
evaluation analysis (pre-registration sec. 5, "Firewall order", and 7.2).

The recall pass itself (the GPU-side exact-mass computation) is not here. This
module is what its rows must look like and who may read them.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

import pandas as pd

from ..masks import MASK_SELECTORS
from .frontier_prereg import SPLIT_INDEX, SplitLeak

SPLITS = tuple(SPLIT_INDEX)
D_NOMS = (0.50, 0.25, 0.10, 0.05)
SELECTION_THRESHOLD = 0.95          # sec. 7.2: mean normalised recall
SELECTION_FALLBACK = 0.50
SELECTION_BLOCK = 128
RECALL_COLUMNS = ("model", "task", "band", "example_id", "split", "arm", "mask_selector",
                  "block_size", "d_nom", "recall_raw", "recall_norm", "realised_density",
                  "git_commit", "git_dirty")


def example_index(example_id: str) -> int:
    m = re.search(r"_(\d+)$", example_id)
    if not m:
        raise SplitLeak(f"example id {example_id!r} carries no index")
    return int(m.group(1))


def split_of(example_id: str) -> str:
    """The split an example index belongs to (sec. 5). An index in no split
    is refused: it is neither held out nor selectable."""
    i = example_index(example_id)
    for split, (first, n) in SPLIT_INDEX.items():
        if first <= i < first + n:
            return split
    raise SplitLeak(f"{example_id!r} (index {i}) is in no split of sec. 5")


@dataclass(frozen=True)
class RecallRow:
    """One (example, arm, block size, budget) recall measurement. `split` is
    required and must be the split the example's index belongs to, so a row
    cannot be relabelled into the selection split."""
    model: str
    task: str
    band: int
    example_id: str
    split: str
    arm: str
    mask_selector: str
    block_size: int
    d_nom: Optional[float]
    recall_raw: float
    recall_norm: float
    realised_density: float
    git_commit: str
    git_dirty: bool

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split {self.split!r} is not one of {SPLITS}")
        if split_of(self.example_id) != self.split:
            raise SplitLeak(f"{self.example_id!r} is in the {split_of(self.example_id)} split, "
                            f"not {self.split!r}")
        if self.mask_selector not in MASK_SELECTORS:
            raise ValueError(f"mask_selector {self.mask_selector!r} is not one of {MASK_SELECTORS}")
        if self.mask_selector == "per_head" and self.d_nom not in D_NOMS:
            raise ValueError(f"an era-4 row needs d_nom in {D_NOMS}, got {self.d_nom!r}")
        if self.git_dirty:
            raise ValueError("a recall row from an uncommitted tree is refused")


def to_frame(rows) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows], columns=list(RECALL_COLUMNS))


def validate_recall_frame(frame: pd.DataFrame) -> None:
    """Every row carries its split, and the split is its example's."""
    missing = [c for c in RECALL_COLUMNS if c not in frame.columns]
    if missing:
        raise SplitLeak(f"recall rows lack {missing}; a row with no split is refused, "
                        "not assumed to be selection")
    if frame["split"].isna().any():
        raise SplitLeak("recall rows with an empty split are refused")
    for eid, split in frame[["example_id", "split"]].drop_duplicates().itertuples(index=False):
        if split not in SPLITS or split_of(eid) != split:
            raise SplitLeak(f"{eid!r} is labelled {split!r}")


def select_budgets(frame: pd.DataFrame) -> dict:
    """Sec. 7.2. Per (arm, band): b* is the smallest d_nom in {0.50, 0.25,
    0.10, 0.05} whose mean normalised recall is at least 0.95, or 0.50 if
    none is. Era-4 rows at b = 128 only.

    It refuses a frame holding any row that is not from the selection split:
    a budget chosen with evaluation recall in view is the leak the splits
    exist to prevent."""
    validate_recall_frame(frame)
    other = frame.loc[frame["split"] != "selection", "split"].value_counts().to_dict()
    if other:
        raise SplitLeak(f"budget selection reads the selection split only; the frame also "
                        f"holds {other}")
    use = frame[(frame["mask_selector"] == "per_head") & (frame["block_size"] == SELECTION_BLOCK)]
    if use.empty:
        raise SplitLeak("no era-4 selection rows at b = 128 to select a budget from")
    out = {}
    for (arm, band), g in use.groupby(["arm", "band"]):
        means = g.groupby("d_nom")["recall_norm"].mean()
        absent = [d for d in D_NOMS if d not in means.index]
        if absent:
            raise SplitLeak(f"{arm} at {band} has no rows at d_nom {absent}")
        ok = [d for d in D_NOMS if means[d] >= SELECTION_THRESHOLD]
        out[f"{arm}@{int(band)}"] = {
            "b_star": min(ok) if ok else SELECTION_FALLBACK, "qualified": bool(ok),
            "mean_recall_norm": {f"{d:.2f}": round(float(means[d]), 6) for d in D_NOMS},
            "n_examples": int(g["example_id"].nunique())}
    return out


def _digest(selection: Mapping) -> str:
    return hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()


def write_budget_selection(path: Path, selection: Mapping, *, git_commit: str) -> str:
    doc = {"selection": selection, "sha256": _digest(selection), "git_commit": git_commit,
           "threshold": SELECTION_THRESHOLD, "block_size": SELECTION_BLOCK}
    Path(path).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc["sha256"]


def require_budget_selection(path: Path, *, repo: Path) -> dict:
    """The firewall's third step. The evaluation analysis may run only if
    `budget_selection.json` is committed at HEAD, unchanged in the working
    tree, and its digest matches its contents."""
    path, repo = Path(path).resolve(), Path(repo).resolve()
    if not path.exists():
        raise SplitLeak(f"{path.name} does not exist: budgets are selected and committed "
                        "before evaluation recall is analysed")
    rel = path.relative_to(repo).as_posix()
    head = subprocess.run(["git", "-C", str(repo), "show", f"HEAD:{rel}"],
                          capture_output=True)
    if head.returncode != 0:
        raise SplitLeak(f"{rel} is not committed at HEAD")
    if head.stdout != path.read_bytes():
        raise SplitLeak(f"{rel} differs from the copy at HEAD")
    doc = json.loads(path.read_text())
    if doc.get("sha256") != _digest(doc.get("selection", {})):
        raise SplitLeak(f"{rel}: the digest does not match the selection")
    return doc


def evaluation_frame(frame: pd.DataFrame, budget_selection: Path, *, repo: Path) -> pd.DataFrame:
    """The evaluation rows, handed over only behind the firewall. Returns
    them with the committed selection attached as `frame.attrs`."""
    doc = require_budget_selection(budget_selection, repo=repo)
    validate_recall_frame(frame)
    out = frame[frame["split"] == "evaluation"].copy()
    out.attrs["budget_selection"] = doc
    return out
