"""The estimator-frontier pre-registration, as code (lock gate L1, 2026-10-03).

`docs/estimator_frontier_preregistration.md` states each hypothesis, its
input ratio, tolerance, indeterminate rule and pass rule, plus the split
firewall, the resolution floor, the cost bracket and the hard stop. Prose is
an unimplemented test, so every one of those is a function here, and
`tests/test_frontier_prereg_plan.py` pins each function to the numbers in the
document and watches it fail and go indeterminate.

Nothing here reads a result file by path. The analysis scripts load data and
hand it to these functions, so the scoring rules can be tested before a
single row exists.

Verdicts are strings, one of `VERDICTS`. "not run (cost stop)" is a verdict,
never folded into pass or fail (sec. 7.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from .exact_noninferiority import paired_lower_bound

# --------------------------------------------------------------------------
# Pre-registered constants. Each is pinned to the document by the plan test.
# --------------------------------------------------------------------------

TAU_K = 0.05            # kernel-level input-ratio tolerance (sec. 3)
TAU_E = 0.07            # end-to-end tolerance, and the near-parity band (3, 7.3)
F_MIN = 0.02            # resolution floor's fixed term (7.4)
MARGIN_PTS = 10.0       # non-inferiority margin, points (2.4)
ALPHA = 0.025           # one-sided (2.4)
BOOT_B = 10_000         # recall bootstrap resamples (2.3)
BOOT_SEED = 20261003    # recall bootstrap seed (2.3)

H1_BAND = (0.75, 1.33)
H1_MIN_IN_BAND = 15
H1_MAX_INDETERMINATE = 3
H2A_MIN_MATCH = 0.90
H2A_WRONG_SIGN_GUARD = 0.15
H2B_MIN_WITHIN = 0.80
H2B_REL = 0.20
H2D_RANGE = (1.25, 1.80)
H3_MIN_DIFF = 0.05
H3_MIN_CELLS = 16
H3_DENSITY_TOL = 0.01
H3B_LIMIT = 0.95
H4_RATIO = 2.0
H4_P = 0.025
H4_BOUNDARY = 0.005
H4_MIN_PER_TERCILE = 30
H6A_PASS = 0.85
H6A_STOP = 0.70
H6B_MIN_RHO = 0.80
H6B_N_ENTRIES = 996
H7A_FACTOR = 0.75
H7_MIN_CELLS = 16
H7_DENSITY_TOL = 0.05
H7_MAX_INDETERMINATE = 4
R1_REL = 0.05
R1_MIN_FRAC = 0.90
R2_MIN_FRAC = 0.90
R3_MIN_AGREE = 0.85
R3_MAX_DELTA_PTS = 5.0
R_N_IDS = 400

CAP_INR = 12_000
STORAGE_PER_MONTH_INR = 212     # two images + the bucket (8.2)
DISK_PER_HOUR_INR = 2.4

PASS, FAIL = "pass", "fail"
UNRESOLVED, UNDERPOWERED, UNSCORABLE = "unresolved", "underpowered", "unscorable"
NOT_RUN = "not run (cost stop)"
VERDICTS = (PASS, FAIL, UNRESOLVED, UNDERPOWERED, UNSCORABLE, NOT_RUN)


@dataclass(frozen=True)
class Verdict:
    hypothesis: str
    verdict: str
    detail: str = ""
    labels: tuple = ()
    counts: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")


# --------------------------------------------------------------------------
# Profitability and the resolution floor (2.2, 7.4)
# --------------------------------------------------------------------------

def resolution_floor(f_aa: float, sigma_s: Optional[float] = None,
                     k: Optional[int] = None) -> float:
    """f = max(0.02, f_AA, t_{0.975,k-1} * sigma_s / sqrt(k)). The third term
    exists only with k >= 2 replicate sessions."""
    terms = [F_MIN, f_aa]
    if sigma_s is not None and k is not None and k >= 2:
        from scipy import stats
        terms.append(stats.t.ppf(0.975, k - 1) * sigma_s / math.sqrt(k))
    return max(terms)


def f_aa(dense_a: Sequence[float], dense_b: Sequence[float]) -> float:
    """|mean(a)/mean(b) - 1| + 2 SE over reps (7.4), SE by the delta method."""
    a, b = np.asarray(dense_a, float), np.asarray(dense_b, float)
    r = a.mean() / b.mean()
    se = r * math.sqrt(a.var(ddof=1) / len(a) / a.mean() ** 2
                       + b.var(ddof=1) / len(b) / b.mean() ** 2)
    return abs(r - 1) + 2 * se


def classify_profit(s_lower: float, s_upper: float, f: float) -> str:
    """profitable / unprofitable / unresolved, from the 97.5% bounds on s."""
    if s_lower > 1 + f:
        return "profitable"
    if s_upper < 1 - f:
        return "unprofitable"
    return "unresolved"


def near_parity(m: Optional[float] = None, s: Optional[float] = None) -> bool:
    """|1 - m| <= 0.07 or |1 - s| <= 0.07 (7.3)."""
    return any(x is not None and abs(1 - x) <= TAU_E for x in (m, s))


# --------------------------------------------------------------------------
# Bootstrap (2.3)
# --------------------------------------------------------------------------

def bootstrap_means(values: Sequence[float], *, b: int = BOOT_B,
                    seed: int = BOOT_SEED) -> np.ndarray:
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(b, len(v)))
    return v[idx].mean(axis=1)


def one_sided_lower(values: Sequence[float], **kw) -> float:
    """The one-sided 97.5% percentile-bootstrap lower bound of the mean."""
    return float(np.percentile(bootstrap_means(values, **kw), 2.5))


# --------------------------------------------------------------------------
# H1: cost across cards
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class H1Cell:
    c1: float
    c2: float
    D: float          # T_dense^card1 / T_dense^card2
    W: float          # bandwidth card2 / card1
    xattn_path: Optional[str] = None


def score_h1(cells: Sequence[H1Cell]) -> Verdict:
    """Q = (c2/c1)/(D/W) in [0.75, 1.33] widened by x/÷ 1.05^2. A pair with
    |ln(D/W)| < 2 ln 1.05 is indeterminate. More than 3 indeterminate:
    unresolved; else >= 15 in band passes and fewer fails."""
    refuse_torch_fallback(c.xattn_path for c in cells)
    lo, hi = H1_BAND[0] / (1 + TAU_K) ** 2, H1_BAND[1] * (1 + TAU_K) ** 2
    ind = in_band = 0
    for c in cells:
        if abs(math.log(c.D / c.W)) < 2 * math.log(1 + TAU_K):
            ind += 1
            continue
        q = (c.c2 / c.c1) / (c.D / c.W)
        in_band += lo <= q <= hi
    counts = {"cells": len(cells), "indeterminate": ind, "in_band": in_band}
    if ind > H1_MAX_INDETERMINATE:
        return Verdict("H1", UNRESOLVED, f"{ind} indeterminate cells", counts=counts)
    return Verdict("H1", PASS if in_band >= H1_MIN_IN_BAND else FAIL, counts=counts)


# --------------------------------------------------------------------------
# H2: the component model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class H2Cell:
    m: float
    s_lower: float
    s_upper: float
    f: float
    s_meas: float          # measured saving, ms (S_meas)
    s_pred: float          # L * (T_dense - T_est - T_sel - T_bsa) - n_sync t_sync
    t_dense_e2e: float


def s_pred(L: int, t_dense: float, t_est: float, t_sel: float, t_bsa: float,
           n_sync: int = 0, t_sync: float = 0.0) -> float:
    return L * (t_dense - t_est - t_sel - t_bsa) - n_sync * t_sync


def score_h2ab(cells: Sequence[H2Cell]) -> tuple[Verdict, Verdict]:
    det = [c for c in cells if abs(1 - c.m) > TAU_E]
    if not det:
        v = Verdict("H2a", UNRESOLVED, "no determinate cell")
        return v, Verdict("H2b", UNRESOLVED, "no determinate cell")
    match = wrong_far = within = 0
    for c in det:
        cls = classify_profit(c.s_lower, c.s_upper, c.f)
        predicted = "profitable" if c.m < 1 else "unprofitable"
        match += cls == predicted
        wrong = cls not in (predicted, "unresolved")
        wrong_far += wrong and abs(1 - c.m) > H2A_WRONG_SIGN_GUARD
        within += abs(c.s_meas - c.s_pred) <= max(H2B_REL * abs(c.s_pred),
                                                  c.f * c.t_dense_e2e)
    n = len(det)
    a_ok = match / n >= H2A_MIN_MATCH and wrong_far == 0
    counts = {"determinate": n, "match": match, "wrong_far": wrong_far, "within": within}
    return (Verdict("H2a", PASS if a_ok else FAIL, counts=counts),
            Verdict("H2b", PASS if within / n >= H2B_MIN_WITHIN else FAIL, counts=counts))


def score_h2c(points: Sequence[Mapping]) -> Verdict:
    """No deployable point profitable at 8192 on the A100 or the H100.
    `points`: dicts with band, card, deployable, s_lower, s_upper, f."""
    pts = [p for p in points if p["band"] == 8192 and p["card"] in ("A100", "H100")
           and p["deployable"]]
    if not pts:
        return Verdict("H2c", UNSCORABLE, "no deployable point at 8192")
    prof = [p for p in pts if classify_profit(p["s_lower"], p["s_upper"], p["f"]) == "profitable"]
    return Verdict("H2c", FAIL if prof else PASS, labels=("prior anchored on visible data",),
                   counts={"points": len(pts), "profitable": len(prof)})


def score_h2d(points: Sequence[Mapping]) -> Verdict:
    """A100, 32768, per-head MP at d_nom 0.25 and 0.10: profitable, point s in
    [1.25, 1.80]. Unresolved profitability: unresolved."""
    want = {0.25, 0.10}
    pts = {p["d_nom"]: p for p in points if p["card"] == "A100" and p["band"] == 32768
           and p["arm"] == "MP" and p["d_nom"] in want}
    if set(pts) != want:
        return Verdict("H2d", UNSCORABLE, f"cells present: {sorted(pts)}")
    labels = ("prior anchored on visible data",)
    cls = {d: classify_profit(p["s_lower"], p["s_upper"], p["f"]) for d, p in pts.items()}
    if any(c == "unresolved" for c in cls.values()):
        return Verdict("H2d", UNRESOLVED, str(cls), labels=labels)
    ok = all(cls[d] == "profitable" and H2D_RANGE[0] <= pts[d]["s"] <= H2D_RANGE[1]
             for d in want)
    return Verdict("H2d", PASS if ok else FAIL, str(cls), labels=labels)


# --------------------------------------------------------------------------
# H3, H7: recall comparisons
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class H3Cell:
    diffs: Sequence[float]         # per-example R~_XA8 - R~_MP
    density_ratio: float           # realised density XA8 / MP


def score_h3(cells: Sequence[H3Cell]) -> Verdict:
    """>= 16 of 24 cells with mean diff >= 0.05 and one-sided 97.5% lower
    bound > 0. A cell outside the G7 density ratio 1 +/- 0.01 is
    indeterminate: if that leaves 16 unreachable either way, unresolved."""
    ok = ind = 0
    for c in cells:
        if abs(c.density_ratio - 1) > H3_DENSITY_TOL:
            ind += 1
            continue
        d = np.asarray(c.diffs, float)
        ok += d.mean() >= H3_MIN_DIFF and one_sided_lower(d) > 0
    counts = {"cells": len(cells), "pass_cells": ok, "indeterminate": ind}
    if ok >= H3_MIN_CELLS:
        return Verdict("H3", PASS, counts=counts)
    if ok + ind >= H3_MIN_CELLS:
        return Verdict("H3", UNRESOLVED, "indeterminate cells decide it", counts=counts)
    return Verdict("H3", FAIL, counts=counts)


def score_h3b(best_r_at_010: Sequence[float]) -> Verdict:
    if not best_r_at_010:
        return Verdict("H3b", UNSCORABLE)
    k = sum(r < H3B_LIMIT for r in best_r_at_010)
    return Verdict("H3b", PASS if k * 3 >= 2 * len(best_r_at_010) else FAIL,
                   counts={"cells": len(best_r_at_010), "below": k})


@dataclass(frozen=True)
class H7Cell:
    r_mp_16: float
    r_mp_128: float
    gap_16: float          # R~_XA8 - R~_MP at b = 16
    gap_128: float
    density_ratio: float   # realised density b=16 / b=128


def score_h7(cells: Sequence[H7Cell]) -> tuple[Verdict, Verdict]:
    det = [c for c in cells if abs(c.density_ratio - 1) <= H7_DENSITY_TOL]
    ind = len(cells) - len(det)
    if ind > H7_MAX_INDETERMINATE:
        v = (UNRESOLVED, f"{ind} indeterminate cells")
        return Verdict("H7a", *v), Verdict("H7b", *v)
    a = sum((1 - c.r_mp_16) <= H7A_FACTOR * (1 - c.r_mp_128) for c in det)
    b = sum(c.gap_16 < c.gap_128 for c in det)
    counts = {"determinate": len(det), "indeterminate": ind, "a": a, "b": b}
    return (Verdict("H7a", PASS if a >= H7_MIN_CELLS else FAIL, counts=counts),
            Verdict("H7b", PASS if b >= H7_MIN_CELLS else FAIL, counts=counts))


# --------------------------------------------------------------------------
# H4: recall predicts accuracy (Mantel-extension trend test)
# --------------------------------------------------------------------------

def terciles(r: Sequence[float]) -> np.ndarray:
    """1 / 2 / 3 within a stratum. An example within +/-0.005 of a tercile
    boundary goes to the lower tercile (sec. 3, H4)."""
    r = np.asarray(r, float)
    q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
    t = np.where(r <= q1 + H4_BOUNDARY, 1, np.where(r <= q2 + H4_BOUNDARY, 2, 3))
    return t


def score_h4(strata: Mapping[str, Mapping[str, Sequence]]) -> Verdict:
    """`strata`: name -> {r_tilde, dense_correct, sparse_correct}. Among
    dense-correct examples, loss = sparse wrong. Pass: loss rate in tercile 1
    >= 2x tercile 3 and one-sided Mantel-extension p < 0.025 for a loss
    decreasing with tercile. Fewer than 30 dense-correct per tercile:
    underpowered."""
    from scipy import stats
    pooled_n = {1: 0, 2: 0, 3: 0}
    pooled_loss = {1: 0, 2: 0, 3: 0}
    T = V = 0.0
    for s in strata.values():
        t = terciles(s["r_tilde"])
        dc = np.asarray(s["dense_correct"], bool)
        loss = dc & ~np.asarray(s["sparse_correct"], bool)
        x, y = t[dc], loss[dc].astype(float)
        if len(x) < 2:
            continue
        for k in (1, 2, 3):
            pooled_n[k] += int((x == k).sum())
            pooled_loss[k] += int(y[x == k].sum())
        n, ysum = len(x), y.sum()
        T += (x * y).sum() - x.sum() * ysum / n
        V += ysum * (n - ysum) / (n * n * (n - 1)) * (n * (x ** 2).sum() - x.sum() ** 2) if n > 1 else 0
    counts = {"n": pooled_n, "loss": pooled_loss}
    if min(pooled_n.values()) < H4_MIN_PER_TERCILE:
        return Verdict("H4", UNDERPOWERED, counts=counts)
    rate = {k: pooled_loss[k] / pooled_n[k] for k in pooled_n}
    ratio = math.inf if rate[3] == 0 and rate[1] > 0 else (rate[1] / rate[3] if rate[3] else 0.0)
    p = float(stats.norm.cdf(T / math.sqrt(V))) if V > 0 else 1.0
    counts.update(ratio=ratio, p=p)
    return Verdict("H4", PASS if ratio >= H4_RATIO and p < H4_P else FAIL, counts=counts)


# --------------------------------------------------------------------------
# H5, H8, P-T4, R1-R4: accuracy
# --------------------------------------------------------------------------

def ni_lower_pts(dense: Sequence[bool], sparse: Sequence[bool]) -> float:
    d, s = np.asarray(dense, bool), np.asarray(sparse, bool)
    return 100 * paired_lower_bound(int((s & ~d).sum()), int((d & ~s).sum()), len(d), ALPHA)


def certified(dense: Sequence[bool], sparse: Sequence[bool]) -> bool:
    return ni_lower_pts(dense, sparse) > -MARGIN_PTS


def score_h5(points: Sequence[Mapping]) -> tuple[Verdict, Verdict]:
    """points: band, arm, is_xa, profit (classify_profit), certified. The
    count is of points both profitable and certified; unresolved is not
    profitable."""
    out = []
    for band, name in ((16384, "H5a"), (32768, "H5b")):
        pts = [p for p in points if p["band"] == band]
        if not pts:
            out.append(Verdict(name, UNSCORABLE, f"no point at {band}"))
            continue
        hits = [p for p in pts if p["profit"] == "profitable" and p["certified"]]
        if band == 16384:
            ok = len(hits) == 0
        else:
            ok = len(hits) <= 1 and all(p["is_xa"] for p in hits)
        out.append(Verdict(name, PASS if ok else FAIL,
                           labels=("prior anchored on visible data",) if band == 16384 else (),
                           counts={"points": len(pts), "profitable_certified": len(hits)}))
    return out[0], out[1]


def score_h6a(mean_raw_r: Optional[float]) -> Verdict:
    if mean_raw_r is None:
        return Verdict("H6a", UNSCORABLE, "not measured")
    labels = []
    if mean_raw_r < H6A_PASS:
        labels.append("triggers I2c-B (double-BOS arm)")
    if mean_raw_r < H6A_STOP:
        labels.append("G6 STOP")
    return Verdict("H6a", PASS if mean_raw_r >= H6A_PASS else FAIL, labels=tuple(labels))


def i2cb_readout(mean_raw_r_double_bos: float) -> str:
    """sec. 4.9: >= 0.85 on the authors' encoding attributes the miss to BOS."""
    return ("passes on the authors' encoding, fails on ours"
            if mean_raw_r_double_bos >= H6A_PASS else "BOS does not explain the miss")


def score_h6b(rho: Optional[float], n_entries: int) -> Verdict:
    if rho is None or n_entries != H6B_N_ENTRIES:
        return Verdict("H6b", UNSCORABLE, f"rho={rho}, entries={n_entries}")
    return Verdict("H6b", PASS if rho >= H6B_MIN_RHO else FAIL)


def score_h8(cells: Sequence[Mapping], *, gates_pass: bool) -> Verdict:
    """cells: primary Llama cells {dense, sparse} boolean arrays."""
    if not gates_pass:
        return Verdict("H8", UNSCORABLE, "G1, G2 or G9 did not pass on Llama")
    if not cells:
        return Verdict("H8", UNSCORABLE, "no primary cell selected")
    ok = all(certified(c["dense"], c["sparse"]) for c in cells)
    return Verdict("H8", PASS if ok else FAIL, counts={"cells": len(cells)})


def score_p_t4(primary: Mapping[int, Mapping]) -> Verdict:
    """0 of 2 primary cells certified (qa_1 at 16384 and 32768, n = 100)."""
    if set(primary) != {16384, 32768}:
        return Verdict("P-T4", UNSCORABLE, f"cells: {sorted(primary)}")
    k = sum(certified(c["dense"], c["sparse"]) for c in primary.values())
    return Verdict("P-T4", PASS if k == 0 else FAIL, counts={"certified": k})


def _aligned(a: Mapping, b: Mapping) -> Optional[list]:
    ids = sorted(a)
    return ids if sorted(b) == ids and len(ids) == R_N_IDS else None


def score_r1(a100_density: Mapping[str, float], l4_density: Mapping[str, float]) -> Verdict:
    ids = _aligned(a100_density, l4_density)
    if ids is None:
        return Verdict("R1", UNSCORABLE, "id sets differ or n != 400")
    k = sum(abs(a100_density[i] / l4_density[i] - 1) <= R1_REL for i in ids)
    return Verdict("R1", PASS if k / len(ids) >= R1_MIN_FRAC else FAIL, counts={"within": k})


def score_r2(a100_pred: Mapping[str, str], l4_pred: Mapping[str, str]) -> Verdict:
    ids = _aligned(a100_pred, l4_pred)
    if ids is None:
        return Verdict("R2", UNSCORABLE, "id sets differ or n != 400")
    k = sum(a100_pred[i] == l4_pred[i] for i in ids)
    return Verdict("R2", PASS if k / len(ids) >= R2_MIN_FRAC else FAIL, counts={"identical": k})


def _labels(r1: Verdict, r2: Verdict) -> tuple:
    return tuple(l for v, l in ((r1, "path-divergent"), (r2, "card-confounded"))
                 if v.verdict != PASS)


def score_r3(cells: Mapping[str, Mapping], *, primary: Iterable[str],
             r1: Verdict, r2: Verdict) -> Verdict:
    """cells: name -> {a100: bools, l4: bools} aligned. >= 85% agreement in
    every cell; |delta accuracy| <= 5 points in each primary cell."""
    if not cells:
        return Verdict("R3", UNSCORABLE)
    ok = True
    for name, c in cells.items():
        a, l = np.asarray(c["a100"], bool), np.asarray(c["l4"], bool)
        ok &= (a == l).mean() >= R3_MIN_AGREE
        if name in set(primary):
            ok &= abs(100 * (a.mean() - l.mean())) <= R3_MAX_DELTA_PTS
    return Verdict("R3", PASS if ok else FAIL, labels=_labels(r1, r2))


def score_r4(claims: Mapping[int, tuple], *, r1: Verdict, r2: Verdict) -> Verdict:
    """claims: band -> (a100_certified, t4_certified) on the shared qa_1 ids."""
    if set(claims) != {16384, 32768}:
        return Verdict("R4", UNSCORABLE, f"bands: {sorted(claims)}")
    ok = all(a == t for a, t in claims.values())
    return Verdict("R4", PASS if ok else FAIL, labels=_labels(r1, r2))


# --------------------------------------------------------------------------
# Cross-card refusal (4.9)
# --------------------------------------------------------------------------

class TorchFallbackRefusal(ValueError):
    """An L4 XA torch-fallback row offered to a cross-card computation."""


def refuse_torch_fallback(paths: Iterable[Optional[str]], *, allow_fallback: bool = False) -> None:
    if allow_fallback:
        return
    if any(p == "torch_fallback" for p in paths):
        raise TorchFallbackRefusal(
            "torch_fallback rows are never on the generational curve (sec. 4.9); "
            "pass allow_fallback=True only for the separately-labelled descriptive table")


def cross_card_ratio(a_rows, b_rows, column: str, *, allow_fallback: bool = False) -> float:
    for rows in (a_rows, b_rows):
        if "xattn_path" in rows:
            refuse_torch_fallback(rows["xattn_path"], allow_fallback=allow_fallback)
    return float(np.median(b_rows[column]) / np.median(a_rows[column]))


# --------------------------------------------------------------------------
# Splits (sec. 5)
# --------------------------------------------------------------------------

class SplitLeak(ValueError):
    """Two splits share an example by any of the four identities."""


SPLIT_KEYS = ("example_id", "context_sha256", "qa_question", "niah_needles")


def check_splits_disjoint(splits: Mapping[str, Sequence[Mapping]]) -> None:
    """Every pair of splits disjoint by example_id, context sha256, QA
    question index AND gold-document set (as `qa_question`, a hashable
    (index, docs) or just the index), and NIAH (key, value) pairs. A shared
    `qa_1` question under another seed and context is a leak."""
    names = sorted(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in SPLIT_KEYS:
                sa = _identities(splits[a], key)
                sb = _identities(splits[b], key)
                both = sa & sb
                if both:
                    raise SplitLeak(f"{a} and {b} share {key}: {sorted(map(str, both))[:3]}")


def _identities(rows, key):
    out = set()
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        if key == "qa_question":
            # The index alone decides: the same SQuAD question is a leak
            # whatever its distractors (ruler.py: `index=i`).
            out.add(v[0] if isinstance(v, tuple) else v)
        elif key == "niah_needles":
            out.update(tuple(p) for p in v)
        else:
            out.add(v)
    return out


# --------------------------------------------------------------------------
# Cost: the bracket, the reservation check and rebracket (sec. 8)
# --------------------------------------------------------------------------

RATE = {"A100": 284, "H100": 425, "L4": 80, "CPU": 130}


@dataclass(frozen=True)
class BracketInputs:
    """Every input of the sec. 8.2 bracket, as (low, high) where it is a
    range. Defaults are the sixth draft's, with sources in sec. 8.1."""
    pre_a100: Mapping[int, float] = field(default_factory=lambda: {8192: .18987, 16384: .43592, 32768: 1.10181})
    pre_h100: Mapping[int, float] = field(default_factory=lambda: {8192: .07942, 16384: .18921, 32768: .50614})
    l4_row_32768: float = 5.398
    a7_row_16384: float = 2.726
    mu: tuple = (6.19, 9.08)
    wall: tuple = (1.25, 1.69)
    overhead_min: tuple = (8, 15)
    ex: tuple = (1.0, 1.6)
    decode_s: tuple = (0.020, 0.035)
    block_mult: tuple = (1.1, 1.5)
    n_gen: Mapping[int, float] = field(default_factory=lambda: {16384: 28, 32768: 36.5, 65536: 40})
    p7_lo: Mapping[int, float] = field(default_factory=lambda: {16384: 1.85, 32768: 4.19})
    decode_llama_s: tuple = (0.025, 0.045)
    # The C line's text factor: sum over the 121 Qwen-counted text.json
    # texts <= 32768 of (n/16384)^1.6. Per-text counts are not committed.
    c_text_factor: float = 117.71112258763596
    # The Llama oracle lines, minutes, from scripts/derive_llama_oracle_cost.py.
    llama_oracle_min: Mapping[str, tuple] = field(default_factory=lambda: {
        "I2c": (24.752, 55.381), "I2d": (47.427, 103.799), "I2e": (69.179, 149.18)})


@dataclass(frozen=True)
class BracketLine:
    name: str
    card: str
    minutes: tuple
    star: bool
    group: str

    @property
    def inr(self) -> tuple:
        return tuple(m / 60 * RATE[self.card] for m in self.minutes)


def bracket(inp: BracketInputs = BracketInputs()) -> list[BracketLine]:
    """The sec. 8.2 table. Formulas as in the sixth draft."""
    K = (0, 1)
    pa, ph = inp.pre_a100, inp.pre_h100
    p7 = {16384: (inp.p7_lo[16384], inp.a7_row_16384),
          32768: (inp.p7_lo[32768], inp.a7_row_16384 * 1102 / 436)}
    ext = lambda k: (3, 10)[k]  # noqa: E731
    lines: list[BracketLine] = []

    def add(name, card, f, star, group):
        lines.append(BracketLine(name, card, tuple(f(k) for k in K), star, group))

    add("FA3", "CPU", lambda k: (60, 150)[k], True, "intrinsic")
    add("C", "A100", lambda k: p7[16384][k] * inp.c_text_factor * inp.mu[k] / 60
        + inp.overhead_min[k] + ext(k), True, "intrinsic")

    def i1(k):
        e2e = sum(15 * 23 * 2 * pa[x] for x in pa) * inp.ex[k] / 60
        rec_sel = 96 * (pa[16384] + pa[32768]) * inp.mu[k] / 60
        rec_eval = 400 * (pa[16384] + pa[32768]) * inp.mu[k] * inp.block_mult[k] / 60
        return ext(k) + e2e + rec_sel + rec_eval + inp.overhead_min[k] + 1
    add("I1", "A100", i1, True, "intrinsic")
    add("I2a", "A100", lambda k: 60 * (p7[16384][k] + p7[32768][k]) * inp.mu[k]
        * inp.block_mult[k] / 60 + inp.overhead_min[k] + ext(k), True, "intrinsic")
    add("I2b", "A100", lambda k: 15 * 13 * p7[32768][k] * inp.ex[k] / 60, False, "intrinsic")
    add("I2c", "A100", lambda k: inp.llama_oracle_min["I2c"][k], True, "intrinsic")
    add("I2c-B", "A100", lambda k: inp.llama_oracle_min["I2c"][k], False, "intrinsic")
    add("I2d", "A100", lambda k: inp.llama_oracle_min["I2d"][k], False, "intrinsic")
    add("I2e", "A100", lambda k: inp.llama_oracle_min["I2e"][k], False, "intrinsic")
    add("I3", "H100", lambda k: (5, 15)[k] + sum(15 * 23 * 2 * ph[x] for x in ph)
        * inp.ex[k] / 60 + inp.overhead_min[k], True, "intrinsic")
    add("I4", "L4", lambda k: (3, 10)[k] + inp.overhead_min[k], False, "intrinsic")

    def rep(p):
        return lambda k: (0, 3 * (inp.overhead_min[1] + sum(15 * 23 * 2 * p[x] for x in p)
                                  * inp.ex[1] / 60))[k]
    add("R A100x3", "A100", rep(pa), True, "intrinsic")
    add("R H100x2", "H100", lambda k: rep(ph)(k) * 2 / 3, True, "intrinsic")
    add("R H100#3", "H100", lambda k: rep(ph)(k) / 3, False, "intrinsic")

    def row15(bd, k):
        return (pa[bd] + inp.n_gen[bd] * inp.decode_s[0],
                {16384: inp.a7_row_16384, 32768: inp.l4_row_32768}[bd])[k]

    def ext15(n, arms):
        return lambda k: n * (1 + arms[k]) * (row15(16384, k) + row15(32768, k)) * inp.wall[k] / 60
    add("X0", "A100", lambda k: 12 * (1 + (5, 8)[k]) * (row15(16384, k) + row15(32768, k)) / 2
        * inp.wall[k] / 60 + inp.overhead_min[k] + ext(k), True, "extrinsic")
    add("X primary", "A100", lambda k: ext15(300, (5, 8))(k) + 300 * (pa[16384] + pa[32768])
        * inp.mu[k] / 60 + 2 * inp.overhead_min[k], True, "extrinsic")
    add("X secondary", "A100", lambda k: ext15(100, (5, 8))(k) + 100 * (pa[16384] + pa[32768])
        * inp.mu[k] / 60, True, "extrinsic")
    add("X-rep", "A100", lambda k: (15, 61)[k], True, "extrinsic")

    pl_lo = {16384: 0.124 * 3.05 + 0.312 * 5.4, 32768: 0.496 * 3.05 + 0.606 * 5.4,
             65536: 1.984 * 3.05 + 1.212 * 5.4}

    def rowL(bd, k):
        return (pl_lo[bd], 1.6 * pl_lo[bd])[k] + inp.n_gen[bd] * inp.decode_llama_s[k]

    def xl(bands, n, arms=4):
        return lambda k: n * (1 + arms) * sum(rowL(x, k) for x in bands) * inp.wall[k] / 60
    add("XL0", "A100", lambda k: 25 * sum(rowL(x, k) for x in (16384, 32768, 65536))
        * inp.wall[k] / 60 + inp.overhead_min[k] + ext(k), False, "extrinsic")
    add("XL primary", "A100", lambda k: xl((16384, 32768), 300)(k) + inp.overhead_min[k],
        False, "extrinsic")
    add("XL secondary", "A100", xl((16384, 32768), 100), False, "extrinsic")
    add("XL 65536", "A100", lambda k: xl((65536,), 400)(k) + inp.overhead_min[k],
        False, "extrinsic")
    return lines


def totals(lines: Sequence[BracketLine]) -> dict:
    def tot(sel):
        ls = [l for l in lines if sel(l)]
        return (sum(l.inr[0] for l in ls), sum(l.inr[1] for l in ls),
                sum(l.minutes[0] for l in ls) / 60, sum(l.minutes[1] for l in ls) / 60)
    return {"core": tot(lambda l: l.star), "all": tot(lambda l: True),
            "intrinsic": tot(lambda l: l.group == "intrinsic" and l.name != "X-rep"),
            "extrinsic": tot(lambda l: l.group == "extrinsic" and l.name != "X-rep")}


def s_res(months_remaining: int, hours_remaining_upper: float) -> float:
    """S_res = Rs 212 x months (rounded up) + Rs 2.4 x upper-bound hours."""
    return STORAGE_PER_MONTH_INR * math.ceil(months_remaining) + DISK_PER_HOUR_INR * hours_remaining_upper


def reservation_check(*, spent: float, next_u: float, remaining_star_u: Sequence[float],
                      s_res_inr: float, r_b: float = 0.0, cap: float = CAP_INR) -> str:
    """'PROCEED' or 'STOP' (sec. 8.3)."""
    need = spent + next_u + sum(remaining_star_u) + s_res_inr + r_b
    return "PROCEED" if need <= cap else "STOP"


def worst_case(lines: Sequence[BracketLine], *, months: int = 2) -> dict:
    t = totals(lines)
    core_hi, core_h_hi = t["core"][1], t["core"][3]
    x_primary = next(l for l in lines if l.name == "X primary").inr[1]
    others = max(l.inr[1] for l in lines if l.star and l.name != "X primary")
    rerun = max(x_primary / 2, others)
    i2cb = next(l for l in lines if l.name == "I2c-B").inr[1]
    sres = s_res(months, core_h_hi)
    total = core_hi + sres + rerun + i2cb
    return {"core": core_hi, "s_res": sres, "rerun": rerun, "i2cb": i2cb,
            "total": total, "spare": CAP_INR - total}


# The leak guard (sec. 8.3): no canary result may enter a funding decision.
LEAK_COLUMNS = frozenset({"correct", "predicted", "score", "R", "R_tilde",
                          "recall", "recall_raw", "recall_norm"})


class LeakRefusal(ValueError):
    """A result column offered to rebracket()."""


def check_no_result_columns(columns: Iterable[str]) -> None:
    bad = sorted(c for c in columns
                 if c in LEAK_COLUMNS or "density" in c.lower() or "recall" in c.lower())
    if bad:
        raise LeakRefusal(f"rebracket() reads only timing, mu and cost columns; refusing {bad}")


def rebracket(frames: Iterable, *, measured: Mapping) -> dict:
    """DP1 / DP2. `frames` are the timing parquets the decision reads (checked
    by the leak guard and nothing else); `measured` overrides BracketInputs
    fields with measured values (mu, block_mult, overhead_min, ...). Returns
    the lines, totals and worst case."""
    for f in frames:
        check_no_result_columns(f.columns)
    inp = BracketInputs(**dict(measured))
    lines = bracket(inp)
    return {"lines": lines, "totals": totals(lines), "worst_case": worst_case(lines)}
