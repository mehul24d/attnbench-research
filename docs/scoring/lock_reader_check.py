"""The lock-reader check for the scoring report of 16 Nov 2026 (the adviser's rulings
of 16 Nov 2026, closing the scoring wave, item 1).

Runs `scripts/score_hypotheses.py` offline with the tag's lock wrapped so that every
key it reads is recorded with the function that reads it, and asserts the lock it is
handed is the tag's `docs/lock_values.json` as parsed. Then compares `locked_from(tag)`
with the analysis commit's `Locked.from_lock` on a copy whose only changes are a
placeholder `power_target_and_n` and H4's `scored_pairings` entry as a list, H4's fields
aside. Not part of the suite (it needs the scoring mirror, kept outside the repo):

    PATH=<fake gsutil/gcloud>:$PATH python3 docs/scoring/lock_reader_check.py <worktree> \
        <result.json> --inputs <inputs.json> --mirror <mirror> --out <dir>
"""
import copy, json, math, subprocess, sys
from pathlib import Path
WT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(WT)); sys.path.insert(0, str(WT / "scripts"))
import score_hypotheses as SH
from attnbench.analysis import hypotheses as HY

READ = set()
WHO = {}
import traceback
class Rec(dict):
    def __init__(self, d, path):
        super().__init__({k: wrap(v, f"{path}/{k}") for k, v in d.items()}); self._p = path
    def _r(self, k):
        path = f"{self._p}/{k}"; READ.add(path)
        fr = [f for f in traceback.extract_stack()[:-2] if str(WT) in f.filename]
        WHO.setdefault(path, set()).add(f"{Path(fr[-1].filename).name}:{fr[-1].name}" if fr else "?")
    def __getitem__(self, k): self._r(k); return super().__getitem__(k)
    def get(self, k, default=None):
        self._r(k); return super().get(k, default)
    def items(self):
        for k in super().keys(): self._r(k)
        return super().items()
    def values(self):
        for k in super().keys(): self._r(k)
        return super().values()
def wrap(v, p):
    return Rec(v, p) if isinstance(v, dict) else v

raw_text = subprocess.run(["git", "-C", str(WT), "show", f"{SH.TAG}:{SH.LOCK_FILE}"],
                          capture_output=True, text=True, check=True).stdout
raw = json.loads(raw_text)
orig_lock_at = SH.lock_at
def rec_lock_at(tag, root=SH.ROOT):
    got = orig_lock_at(tag, root)
    assert got == raw, "lock_at is not the tag's file"
    return Rec(got, "")
SH.lock_at = rec_lock_at
rc = SH.main(sys.argv[3:])
def at(d, path):
    for k in path.strip("/").split("/"):
        d = d[k]
    return d
# leaves read: drop a path when a longer read path below it exists
leaves = sorted(p for p in READ if not any(q.startswith(p + "/") for q in READ))
rows = [{"path": p, "equals_tag": True, "value_is_null": at(raw, p) is None} for p in leaves]
# value-level equality: the values the run saw were built from raw; check against a fresh parse
fresh = json.loads(raw_text)
for r in rows:
    r["equals_tag"] = at(fresh, r["path"]) == at(raw, r["path"])
# reader equivalence
dummy = copy.deepcopy(raw)
dummy["values"]["power_target_and_n"]["value"] = {"alpha": 0.05, "n": 1}
dummy["scored_pairings"] = {h: (p if not isinstance(p, dict) else []) for h, p in raw["scored_pairings"].items()}
a, b = SH.locked_from(raw), HY.Locked.from_lock(dummy)
fields = [f for f in a.__dataclass_fields__] if hasattr(a, "__dataclass_fields__") else list(vars(a))
cmp = {}
for f in fields:
    if f.startswith("h4_"):
        continue
    x, y = getattr(a, f), getattr(b, f)
    if f == "scored":
        x = {h: v for h, v in x.items() if h in ("H0", "H1", "H2", "H3")}
        y = {h: v for h, v in y.items() if h in ("H0", "H1", "H2", "H3")}
    cmp[f] = (x == y)
null_read = [r["path"] for r in rows if "power_target_and_n" in r["path"]]
for r in rows: r["read_by"] = sorted(WHO.get(r["path"], ()))
OUT.write_text(json.dumps({"rc": rc, "fields_read": rows, "power_target_and_n_read": null_read,
                           "null_fields_in_tag": sorted(k for k, v in raw["values"].items() if v.get("value") is None),
                           "locked_from_equals_from_lock_on_placeholder": cmp}, indent=1))
print("rc", rc, "leaves", len(rows), "null read", null_read, "cmp", cmp)
print("null leaves read:", [r["path"] for r in rows if r["value_is_null"]])

from collections import Counter
print(Counter(w for r in rows for w in r["read_by"]))
print("power_target_and_n readers:", sorted({w for r in rows if "power_target_and_n" in r["path"] for w in r["read_by"]}))
print("null-value readers:", {r["path"]: r["read_by"] for r in rows if r["value_is_null"]})
