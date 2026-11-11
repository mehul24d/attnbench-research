"""The pairing-level prediction ledger (adviser's ruling 1c of 11 Nov 2026;
T4.3 read B, finding 3; plan Section 6, rule 6, and Section 5.4).

Before this, a second prediction was refused only inside one session
directory, so a later session of the same pairing could predict after an
earlier session's ground truth had begun, and pass its own late check. The
ledger lives in the bucket beside the predictions, one object per cell and
event, written with the store's no-clobber `put`, so it is append-only and
its creation times are the store's clock:

  - `<cell>/prediction.json`: the first committed prediction that covers the
    cell (its session, block, file, sha256, sync receipt, and the digest of
    its predictions);
  - `<cell>/ground_truth_started.json`: written before the first ground-truth
    process of any session of the pairing that clocks the cell. Its store
    creation time is an upper bound on nothing earlier than the clocking, so
    it is the conservative start.

**The rules.**

  - One prediction per cell: a prediction covering a cell that already has
    one is refused, unless the session is a redone block (the same block id
    as the entry's) and its predictions are bit-for-bit those of the entry
    (`predictions_digest`: the whole record but its write time). A redone
    block whose re-prediction differs stops. An equal re-prediction commits
    nothing new: the session uses the original file, which was synced before
    any ground truth.
  - A prediction covering a cell whose ground truth has started anywhere in
    the pairing is refused (unless it is that equal re-prediction).
  - The late check and the analysis read the pairing's earliest ground-truth
    start over the prediction's cells (`earliest_start`), not only their own
    session's.

**Banking** (ruling 1e): `banked.json` at the pairing's ledger root is
written once, when the pairing's ground truth is banked
(`scripts/bank_pairing.py`), naming its training file's sha256 and its
banking time, the latest teardown of the sessions that banked it, read from
their records (adviser's ruling 2 dated 12 Nov 2026: never typed). It is the
time every later prediction's `--banked-at` was typed from before. A time
after the entry's own store creation time is refused: nothing is banked
after it was recorded as banked.

**The cell** is (arm, token budget, batch): the budget is the smallest power
of two at or above the realised length, as the grid doubles (lock table row
27: the realised count is the key, the budget a label). Density is not in
it, so a ground-truth start at one density of an arm blocks a new prediction
at every density of that arm at the cell: stricter than per density, and
P2's and P4's predictions are one file per pairing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional

LEDGER_DIR = "ledger"
PREDICTION = "prediction.json"
GT_STARTED = "ground_truth_started.json"
BANKED = "banked.json"


class LedgerError(ValueError):
    """A prediction or a ground-truth start the ledger refuses."""


def budget_of(length) -> int:
    L = int(length)
    if L < 1:
        raise LedgerError(f"a context length is positive, got {length!r}")
    return 1 << (L - 1).bit_length()


def cell_name(arm: str, length, batch) -> str:
    return f"{arm}_L{budget_of(length)}_b{int(batch)}"


def record_cells(record: Mapping) -> list[str]:
    """The ledger cells a prediction record covers."""
    return sorted({cell_name(c["arm"], c["context_length"], c["batch"])
                   for c in record["cells"]})


def predictions_digest(record: Mapping) -> str:
    """sha256 of the record without its write time: two records with the same
    digest predict the same thing, bit for bit."""
    body = {k: v for k, v in record.items() if k != "written_at"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"))
                          .encode()).hexdigest()


def _HEX64(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def _banked_at(entry: Mapping) -> datetime:
    """The entry's banking time: an ISO time with its offset."""
    v = entry.get("banked_at")
    try:
        t = datetime.fromisoformat(v)
    except (TypeError, ValueError):
        t = None
    if t is None or t.tzinfo is None:
        raise LedgerError(f"a banked entry states banked_at, an ISO time with its offset "
                          f"read from the sessions' records, got {v!r}")
    return t


def ledger_root(prefix: str, pairing: str) -> str:
    return f"{prefix}predictions/{pairing}/{LEDGER_DIR}/"


def _exists(store, remote: str) -> bool:
    ex = getattr(store, "exists", None)
    if ex is not None:
        return bool(ex(remote))
    try:
        store.get(remote)
        return True
    except (FileNotFoundError, KeyError):
        return False


class Ledger:
    """The ledger of one pairing in one store."""

    def __init__(self, store, prefix: str, pairing: str):
        self.store, self.root, self.pairing = store, ledger_root(prefix, pairing), pairing

    def _uri(self, cell: str, what: str) -> str:
        return f"{self.root}{cell}/{what}"

    def get(self, cell: str, what: str) -> Optional[dict]:
        uri = self._uri(cell, what)
        if not _exists(self.store, uri):
            return None
        return json.loads(self.store.get(uri))

    def created(self, cell: str, what: str) -> datetime:
        return self.store.created(self._uri(cell, what))

    def _put(self, cell: str, what: str, obj: Mapping) -> bool:
        """Write `obj` unless an entry is there; True if this one is it."""
        return self._put_uri(self._uri(cell, what), obj)

    def _put_uri(self, uri: str, obj: Mapping) -> bool:
        data = json.dumps(obj, sort_keys=True, indent=1).encode()
        what = uri.rsplit("/", 1)[-1]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / what
            p.write_bytes(data)
            self.store.put(p, uri)          # never clobbers
        return self.store.get(uri) == data

    def check_prediction(self, cells: Iterable[str], digest: str, *,
                         block: Optional[str]) -> Optional[dict]:
        """None for a new prediction of `cells`; the original entry for an
        equal re-prediction by the same block; LedgerError otherwise."""
        cells = sorted(set(cells))
        if not cells:
            raise LedgerError("a prediction covers at least one cell")
        entries = {c: self.get(c, PREDICTION) for c in cells}
        have = {c: e for c, e in entries.items() if e is not None}
        if have:
            originals = {e["sha256"] for e in have.values()}
            e = next(iter(have.values()))
            if block is None or len(originals) != 1 or e.get("block") != block \
                    or set(have) != set(cells):
                raise LedgerError(
                    f"{self.pairing} already has a prediction for {sorted(have)} "
                    f"(session {e.get('session')}, block {e.get('block')}): one prediction "
                    f"per cell; only a redone block may re-predict, and only bit for bit")
            if e["predictions_digest"] != digest:
                raise LedgerError(
                    f"block {block} re-predicted {self.pairing} differently from its "
                    f"discarded attempt (digest {digest[:12]} against "
                    f"{e['predictions_digest'][:12]}): stop (ruling 1c, 11 Nov 2026)")
            return e
        started = [c for c in cells if self.get(c, GT_STARTED) is not None]
        if started:
            raise LedgerError(
                f"ground truth for {started} has started in {self.pairing}, so a new "
                f"prediction of them is refused (Section 6, rule 6)")
        return None

    def record_prediction(self, cells: Iterable[str], entry: Mapping) -> None:
        for c in sorted(set(cells)):
            if not self._put(c, PREDICTION, entry):
                raise LedgerError(f"another prediction of {c} reached the ledger first")

    def record_ground_truth_start(self, cells: Iterable[str], entry: Mapping) -> None:
        """Before any of `cells` is clocked; the first entry per cell stays."""
        for c in sorted(set(cells)):
            self._put(c, GT_STARTED, entry)

    def record_banked(self, entry: Mapping) -> None:
        """Once, when the pairing's ground truth is banked."""
        if not _HEX64(entry.get("training_file_sha256")):
            raise LedgerError("a banked entry names its training file's sha256 (64 hex)")
        _banked_at(entry)
        if not self._put_uri(self.root + BANKED, {**entry, "pairing": self.pairing}):
            raise LedgerError(f"{self.pairing} is already banked: {self.banked()[1]}")

    def banked(self) -> Optional[tuple[datetime, dict]]:
        """(the entry's banking time, the entry), or None if not banked."""
        uri = self.root + BANKED
        if not _exists(self.store, uri):
            return None
        entry = json.loads(self.store.get(uri))
        at, made = _banked_at(entry), self.store.created(uri)
        if at > made:
            raise LedgerError(f"{self.pairing}'s banked entry says {at.isoformat()}, after "
                              f"the entry itself was written ({made.isoformat()})")
        return at, entry

    def earliest_start(self, cells: Iterable[str]) -> Optional[datetime]:
        """The store's creation time of the first ground-truth start of any
        of `cells` in the pairing, or None."""
        ts = [self.created(c, GT_STARTED) for c in sorted(set(cells))
              if self.get(c, GT_STARTED) is not None]
        return min(ts) if ts else None
