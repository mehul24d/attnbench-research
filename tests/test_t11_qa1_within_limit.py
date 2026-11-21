"""scripts/t11_qa1_within_limit.py (EXPLORATORY, adviser's task of 22 Nov
2026, item 5): the union exclusion rule on a toy pair, and, with the banked
T1.1 Session B files, the numbers docs/paper_claims_audit.md quotes."""
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
pd = pytest.importorskip("pandas")
import t11_qa1_within_limit as w  # noqa: E402

DOC = ROOT / "docs" / "paper_claims_audit.md"
needs_banked = pytest.mark.skipif(not all(p.exists() for p in w.files(w.DEFAULT)),
                                  reason="T1.1 Session B files not in this checkout")


def test_the_union_of_both_files_is_excluded():
    dense = pd.DataFrame({"example_id": list("abcd"), "positions_over_limit": [0, 3, 0, 0]})
    arm = pd.DataFrame({"example_id": list("abcd"), "positions_over_limit": [0, 0, 7, 0]})
    assert w.over_limit_union(dense, arm) == {"b", "c"}


@needs_banked
def test_the_banked_recompute_is_what_the_audit_quotes():
    df = w.r.load([str(p) for p in w.files(w.DEFAULT)], allow_mixed_commits=False)
    res = w.recompute(df).set_index(["arm", "rows"])
    a = res.loc[("xattention_inline@cal=authors", "all rows (certified path)")]
    assert (a.n, a.dense_correct, a.sparse_correct, round(a.lower_pts, 1)) == (100, 46, 50, -6.0)
    i = res.loc[("xattention_inline@cal=authors", "inside the limit")]
    got = (int(i.excluded), int(i.n), int(i.dense_correct), int(i.sparse_correct),
           int(i.b_sparse_only), int(i.c_dense_only), round(float(i.lower_pts), 1))
    m = re.search(r"EXPLORATORY sensitivity.*?(\d+) rows excluded, n = (\d+): dense (\d+), "
                  r"`authors` (\d+), (\d+) pairs to XAttention and (\d+) to dense, bound "
                  r"(−[0-9.]+)", DOC.read_text(), re.S)
    assert m, "the audit's sensitivity line"
    want = tuple(int(x) for x in m.groups()[:6]) + (-float(m.group(7)[1:]),)
    assert got == want, (got, want)
    assert bool(i.non_inferior)
