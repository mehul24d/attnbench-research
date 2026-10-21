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

from ..masks import era_budget
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
H1_GUARD_MARGIN = 1.10          # the null must lie this far outside the widened band
H1_MIN_FRACTION = 5 / 6         # of the determinate cells, in band
H1_MIN_DETERMINATE = 6
H1_INDEPENDENT_PAIRS = (("L4", "A100"), ("A100", "H100"))   # L4-H100 is their product
H1_SIGMA_Q = 0.035              # assumed log-SD of a measured Q (sec. 3, H1)
H2A_MIN_MATCH = 0.90
H2A_WRONG_SIGN_GUARD = 0.15
H2B_MIN_WITHIN = 0.80
H2B_REL = 0.20
H2D_RANGE = (1.25, 1.80)
H3_MIN_DIFF = 0.05
H3_MIN_CELLS = 16
H3_MIN_7B_CELLS = 6
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
H7_MIN_FRACTION = 2 / 3         # 16 of 24, or 8 of 12 under cut 4
H7_DENSITY_TOL = 0.01
H7_MAX_DESCRIPTIVE_FRACTION = 1 / 6   # 4 of 24, or 2 of 12
H7_MIN_PARENT_ROW = 7           # b = 128 rows scored by H7: p >= 7 (tokens >= 896)
H7_CELL_COUNTS = (24, 12)       # full, or 1.5B only (cut 4)
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


def near_parity(m: float) -> bool:
    """|1 - m| <= 0.07 (7.3). m is an input ratio from components. The
    measured speedup s never triggers a replicate: that would be a stopping
    rule on the outcome."""
    return abs(1 - m) <= TAU_E


def marginal(lower: float, upper: float, threshold: float) -> bool:
    """A point-estimate rule whose 95% interval contains its threshold is
    labelled "marginal" beside its verdict (7.5). The verdict is unchanged."""
    return lower <= threshold <= upper


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
    pair: tuple = ("L4", "A100")      # (card1, card2)


def h1_band() -> tuple:
    """[0.75, 1.33] widened x/÷ 1.05^2 for the tolerance on D and W."""
    return H1_BAND[0] / (1 + TAU_K) ** 2, H1_BAND[1] * (1 + TAU_K) ** 2


def h1_determinate(D: float, W: float) -> bool:
    """A pair discriminates only if the null's value, Q = 1/(D/W) (c the same
    on both cards), lies outside the widened band by a further factor of
    1.10. Otherwise a cell passes whether or not H1 is true."""
    lo, hi = h1_band()
    q_null = W / D
    return q_null < lo / H1_GUARD_MARGIN or q_null > hi * H1_GUARD_MARGIN


def h1_false_pass(p: float, n: int = 6, k: int = 5) -> tuple:
    """(independent cells, fully correlated cells): the chance that at least
    k of n null cells land in band when each does so with probability p."""
    ind = sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))
    return ind, p


def h1_null_cell_pass_probability(sigma_q: float = H1_SIGMA_Q) -> float:
    """At the guard's edge a null cell is in band only if its measured Q is
    off by the guard margin: P(Z > ln 1.10 / sigma_Q)."""
    from scipy import stats
    return float(stats.norm.sf(math.log(H1_GUARD_MARGIN) / sigma_q))


def score_h1(cells: Sequence[H1Cell]) -> Verdict:
    """Q = (c2/c1)/(D/W) in [0.75, 1.33] widened by x/÷ 1.05^2, on the
    independent card pairs only. A cell whose pair does not discriminate
    (`h1_determinate`) is indeterminate. Fewer than 6 determinate cells:
    unresolved. Else at least 5/6 of them in band passes, and fewer fails."""
    refuse_torch_fallback(c.xattn_path for c in cells)
    lo, hi = h1_band()
    ind = in_band = det = descriptive = 0
    for c in cells:
        if tuple(c.pair) not in H1_INDEPENDENT_PAIRS:
            descriptive += 1
            continue
        if not h1_determinate(c.D, c.W):
            ind += 1
            continue
        det += 1
        q = (c.c2 / c.c1) / (c.D / c.W)
        in_band += lo <= q <= hi
    counts = {"cells": len(cells), "determinate": det, "indeterminate": ind,
              "in_band": in_band, "descriptive": descriptive}
    if det < H1_MIN_DETERMINATE:
        return Verdict("H1", UNRESOLVED, f"{det} determinate cells", counts=counts)
    need = math.ceil(H1_MIN_FRACTION * det - 1e-9)
    return Verdict("H1", PASS if in_band >= need else FAIL, counts=counts)


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
    """No deployable point profitable at 8192 on the A100 or the H100, with
    at least half of them resolved (unprofitable).
    `points`: dicts with band, card, deployable, s_lower, s_upper, f."""
    pts = [p for p in points if p["band"] == 8192 and p["card"] in ("A100", "H100")
           and p["deployable"]]
    if not pts:
        return Verdict("H2c", UNSCORABLE, "no deployable point at 8192")
    cls = [classify_profit(p["s_lower"], p["s_upper"], p["f"]) for p in pts]
    prof, unres = cls.count("profitable"), cls.count("unresolved")
    counts = {"points": len(pts), "profitable": prof, "unresolved": unres}
    labels = ("prior anchored on visible data",)
    if prof:
        return Verdict("H2c", FAIL, labels=labels, counts=counts)
    if 2 * unres > len(pts):
        # A noisy session must not pass a "none is profitable" prediction.
        return Verdict("H2c", UNRESOLVED, "more than half the points are unresolved",
                       labels=labels, counts=counts)
    return Verdict("H2c", PASS, labels=labels, counts=counts)


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
    model: str = "1.5B"            # "1.5B" or "7B"


def score_h3(cells: Sequence[H3Cell]) -> Verdict:
    """>= 16 of 24 cells with mean diff >= 0.05 and one-sided 97.5% lower
    bound > 0, of which at least 6 of the twelve 7B cells (the 1.5B direction
    was informed by T4). A cell outside the G7 density ratio 1 +/- 0.01 is
    indeterminate: if indeterminate cells could still decide it, unresolved."""
    ok = ind = ok7 = ind7 = 0
    for c in cells:
        is7 = c.model == "7B"
        if abs(c.density_ratio - 1) > H3_DENSITY_TOL:
            ind += 1
            ind7 += is7
            continue
        d = np.asarray(c.diffs, float)
        good = bool(d.mean() >= H3_MIN_DIFF and one_sided_lower(d) > 0)
        ok += good
        ok7 += good and is7
    counts = {"cells": len(cells), "pass_cells": ok, "pass_cells_7b": ok7,
              "indeterminate": ind}
    if ok >= H3_MIN_CELLS and ok7 >= H3_MIN_7B_CELLS:
        return Verdict("H3", PASS, counts=counts)
    if ok + ind >= H3_MIN_CELLS and ok7 + ind7 >= H3_MIN_7B_CELLS:
        return Verdict("H3", UNRESOLVED, "indeterminate cells decide it", counts=counts)
    return Verdict("H3", FAIL, counts=counts)


def score_h3b(best_r_at_010: Sequence[float]) -> Verdict:
    if not best_r_at_010:
        return Verdict("H3b", UNSCORABLE)
    k = sum(r < H3B_LIMIT for r in best_r_at_010)
    return Verdict("H3b", PASS if k * 3 >= 2 * len(best_r_at_010) else FAIL,
                   counts={"cells": len(best_r_at_010), "below": k})


def kept_128(p: int, d_nom: float) -> int:
    """Blocks kept in row p at b = 128 under the era-4 rule (sec. 2.1): the
    sink and the diagonal free, plus the budget over the p - 1 candidates,
    computed as the selector computes it (`masks.era_budget`)."""
    if p < 2:
        return p + 1
    return 2 + era_budget(d_nom, p - 1)


def matched_budget(i: int, b: int, d_nom: float) -> int:
    """The budget of row i at block size b < 128 that matches the realised
    density of its parent row p = floor(i * b / 128) at b = 128 (sec. 3, H7).
    The two free blocks are still granted, outside this budget."""
    if i < 2:
        return 0
    p = i * b // 128
    target = kept_128(p, d_nom) / (p + 1)
    return min(max(round(target * (i + 1)) - 2, 0), i - 1)


def matched_row_budgets(n: int, b: int, d_nom: float) -> list:
    """`matched_budget` for rows 0..n-1, as the selector's `row_budgets`."""
    return [matched_budget(i, b, d_nom) for i in range(n)]


def row_density_matches(i: int, b: int, d_nom: float) -> bool:
    """The per-row check: row i's realised density at b is within 0.01 of its
    parent row's at b = 128."""
    p = i * b // 128
    target = kept_128(p, d_nom) / (p + 1)
    got = (min(i + 1, 2) + matched_budget(i, b, d_nom)) / (i + 1)
    return abs(got - target) <= H7_DENSITY_TOL


def h7_rows(n_tokens: int, b: int) -> range:
    """The rows H7 scores at block size b: those whose parent row at b = 128
    is p >= 7. Below that, a row at b = 16 has too few blocks to match its
    parent's density within 0.01."""
    return range(H7_MIN_PARENT_ROW * 128 // b, n_tokens // b)


@dataclass(frozen=True)
class H7Cell:
    r_mp_16: float
    r_mp_128: float
    gap_16: float          # R~_XA8 - R~_MP at b = 16
    gap_128: float
    density_ratio: float   # realised density b=16 / b=128, over the scored rows
    rows_match: bool = True    # every scored row passed `row_density_matches`


def score_h7(cells: Sequence[H7Cell]) -> tuple[Verdict, Verdict]:
    """24 cells, or the 12 1.5B cells under cut 4. A cell whose density ratio
    is outside 1 +/- 0.01, or with a row failing the per-row check, is
    descriptive: reported, not counted. More than 1/6 descriptive: unresolved.
    Pass: at least 2/3 of ALL the cells (16 of 24, 8 of 12)."""
    n = len(cells)
    if n not in H7_CELL_COUNTS:
        v = (UNSCORABLE, f"{n} cells; H7 has 24, or 12 under cut 4")
        return Verdict("H7a", *v), Verdict("H7b", *v)
    det = [c for c in cells if c.rows_match and abs(c.density_ratio - 1) <= H7_DENSITY_TOL]
    desc = n - len(det)
    labels = ("1.5B only (cut 4)",) if n == 12 else ()
    if desc > math.floor(H7_MAX_DESCRIPTIVE_FRACTION * n + 1e-9):
        v = (UNRESOLVED, f"{desc} descriptive cells")
        return Verdict("H7a", *v, labels=labels), Verdict("H7b", *v, labels=labels)
    need = math.ceil(H7_MIN_FRACTION * n - 1e-9)
    a = sum((1 - c.r_mp_16) <= H7A_FACTOR * (1 - c.r_mp_128) for c in det)
    b = sum(c.gap_16 < c.gap_128 for c in det)
    counts = {"cells": n, "descriptive": desc, "a": a, "b": b, "need": need}
    return (Verdict("H7a", PASS if a >= need else FAIL, labels=labels, counts=counts),
            Verdict("H7b", PASS if b >= need else FAIL, labels=labels, counts=counts))


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
                           labels=("prior anchored on visible data",),
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
    return Verdict("P-T4", PASS if k == 0 else FAIL, counts={"certified": k},
                   labels=("prior close to guaranteed by the n = 100 bound",))


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


# The evaluation split uses indices no banked file has used (sec. 5). The
# registry is every (task, band) -> (first, last) example index found in a
# banked parquet on 2026-10-03; `tests/test_frontier_eval_ids.py` re-derives
# it from the parquets and fails if they disagree.
BANKED_INDEX_RANGES = {
    ("niah_multikey", 2048): (0, 299), ("niah_multikey", 4096): (0, 299),
    ("niah_multikey", 8192): (0, 299), ("niah_multikey", 16384): (0, 99),
    ("niah_multikey_1", 16384): (0, 4), ("niah_multikey_1", 32768): (0, 4),
    ("niah_multiquery", 16384): (0, 49), ("niah_multiquery", 32768): (0, 49),
    ("niah_multivalue", 16384): (0, 49), ("niah_multivalue", 32768): (0, 49),
    ("niah_single", 2048): (0, 299), ("niah_single", 4096): (0, 299),
    ("niah_single", 8192): (0, 299), ("niah_single", 16384): (0, 99),
    ("niah_single", 32768): (0, 49),
    ("qa_1", 16384): (0, 99), ("qa_1", 32768): (0, 99),
    ("qa_2", 16384): (0, 4), ("qa_2", 32768): (0, 4),
    ("vt", 2048): (0, 299), ("vt", 4096): (0, 299), ("vt", 8192): (0, 299),
    ("vt", 16384): (0, 99),
}

# split -> (first index, how many), for every task. Seeds: sec. 5.
SPLIT_INDEX = {"evaluation": (3000, 300), "calibration": (2000, 48),
               "selection": (1000, 32), "t4_replication": (0, 100)}
EVAL_INDEX_OFFSET = SPLIT_INDEX["evaluation"][0]


def evaluation_indices(n: int) -> range:
    first, most = SPLIT_INDEX["evaluation"]
    if not 0 < n <= most:
        raise ValueError(f"evaluation n must be 1..{most}, got {n}")
    return range(first, first + n)


def check_evaluation_unbanked(eval_indices: Mapping[str, Iterable[int]],
                              banked: Mapping[tuple, tuple] = None) -> None:
    """No evaluation index of a task may appear in any banked file of that
    task, AT ANY BAND: the QA question is chosen by index alone."""
    banked = BANKED_INDEX_RANGES if banked is None else banked
    for task, idx in eval_indices.items():
        idx = set(idx)
        for (t, band), (lo, hi) in banked.items():
            both = sorted(i for i in idx if t == task and lo <= i <= hi)
            if both:
                raise SplitLeak(f"evaluation {task} indices {both[:3]} are banked at {band}")


def check_calibration_texts_disjoint(texts: Sequence[str], probes: Sequence[str],
                                     *, min_len: int = 40) -> None:
    """`text.json` calibrates every Qwen table and is multi-document QA, so it
    is held to the split firewall too: no evaluation or selection question, and
    no gold document's opening, may occur in any calibration text."""
    for p in probes:
        p = " ".join(p.split())
        if len(p) < min_len:
            continue
        for k, t in enumerate(texts):
            if p in " ".join(t.split()):
                raise SplitLeak(f"calibration text {k} contains {p[:60]!r}")


# What a cut (sec. 8.4) does to a hypothesis, fixed before any result.
CUT_NOT_RUN = {"9": ("H6b",), "11": ("H8",)}
# cut -> ((hypothesis, label), ...). A label never changes a pass threshold.
CUT_LABEL = {
    "0a": (("H2", "A100 escalation not run"), ("H5", "A100 escalation not run")),
    "0b": (("H2", "H100 escalation not run"),),
    "0c": (("H4", "without SL"), ("H5", "without SL")),
    "1": (("H8", "without the 65536c band"),),
    "4": (("H7a", "1.5B only (cut 4)"),),
    "6": (("H2", "H100 near-parity cells on 3 sessions"),),
    "10": (("H4", "without VS"), ("H5", "without VS")),
}
# What the paper states when a cut removes something no hypothesis reads.
CUT_STATEMENT = {"0": "full-set table not made"}
# Cuts 0, 2, 3, 5, 7 and 12 label no hypothesis (sec. 8.4): Llama secondary cells
# are in no test, H6a's population is the <= 32K texts, strides 4 and 16 are
# descriptive, H2's 84 cells are 1.5B only, and I2c-B never changes H6a.

# The cut order (sec. 8.4), first cut first. Item 8 was removed (I4 is never
# cut). A later item has HIGHER priority: it is cut later.
CUT_ORDER = ("0", "0a", "0b", "0c", "1", "2", "3", "4", "5", "6", "7", "9", "10", "11", "12")
CUT_WITHHELD_LAST = ("0c", "10")
# Cut items that are sessions or arms of their own, as bracket lines. The
# others (0a, 0b, 4, 5, 10) are parts of a star line or have no line.
CUT_LINES = {"0": ("C-full",), "0c": ("SL",), "1": ("XL 65536",), "2": ("XL secondary",), "3": ("I2d",),
             "6": ("R H100#3",), "7": ("I2b",), "9": ("I2e",),
             "11": ("XL0", "XL primary"), "12": ("I2c-B",)}


# Cutting item 11 (the Llama dense probe and primary run) takes the other
# Llama accuracy items with it: they need the probe.
CUT_IMPLIES = {"11": ("1", "2")}


def cut_order(*, xa_withheld: bool = False) -> tuple:
    """Sec. 8.4. If publication of XAttention results is withheld at any
    point, SL (0c) and VS (10) are cut last: the fallback rests on them."""
    if not xa_withheld:
        return CUT_ORDER
    return tuple(c for c in CUT_ORDER if c not in CUT_WITHHELD_LAST) + CUT_WITHHELD_LAST


def cut_item_of(line_name: str) -> Optional[str]:
    return next((c for c, names in CUT_LINES.items() if line_name in names), None)


def apply_cuts(verdicts: Mapping[str, Verdict], cuts: Iterable[str]) -> dict:
    """A hypothesis whose inputs were cut is 'not run (cost stop)', whatever
    a scorer would return on what is left."""
    out = dict(verdicts)
    for c in cuts:
        for h in CUT_NOT_RUN.get(str(c), ()):
            out[h] = Verdict(h, NOT_RUN, f"cut-order item {c}")
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
    ex: tuple = (1.0, 1.6)   # the arm factor; ARM_FACTOR_UPPER is its upper end
    decode_s: tuple = (0.020, 0.035)
    block_mult: tuple = (1.1, 1.5)
    # Upper-bound generated tokens per row: the task's hard cap
    # (stopping.TASK_TOKEN_CAPS). Primary is qa_1 (42); secondary is
    # niah_multivalue and niah_multiquery (62); the T4 replication's prompts
    # are half of each (52). Used from the 2026-10-04 re-bracket.
    cap_tokens: Mapping[str, float] = field(default_factory=lambda: {
        "primary": 42, "secondary": 62, "rep": 52})
    # Priced, not adopted: SL's extrinsic arm as a never-cut line (sec. 8.4).
    sl_star: bool = False
    n_gen: Mapping[int, float] = field(default_factory=lambda: {16384: 28, 32768: 36.5, 65536: 40})
    p7_lo: Mapping[int, float] = field(default_factory=lambda: {16384: 1.85, 32768: 4.19})
    decode_llama_s: tuple = (0.025, 0.045)
    # The C line's text factors: sum over Qwen-counted text.json texts
    # <= 32768 of (n/16384)^1.6. Per-text counts are not committed. From
    # 2026-10-04 there are two tables: C makes the claim table on the 104
    # texts left once the 24 QA texts are out; C-full, a cuttable session
    # that runs last, makes the descriptive one on all 121.
    c_text_factor: float = 106.095433
    c_text_factor_full: float = 117.71112258763596
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
    # Upper-bound minutes of the line's largest single phase, where the line
    # is more than one phase. A rerun repeats one phase (sec. 7.3).
    rerun_unit_min: Optional[float] = None
    # Upper-bound minutes per session, where the line is more than one
    # session: {unit name: minutes}. None means the line is one session.
    units: Optional[Mapping[str, float]] = None

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

    def add(name, card, f, star, group, rerun_unit_min=None, units=None):
        lines.append(BracketLine(name, card, tuple(f(k) for k in K), star, group,
                                 rerun_unit_min, units))

    def split(f, n):
        """n equal sessions of a line."""
        return {str(i + 1): f(1) / n for i in range(n)}

    add("FA3", "CPU", lambda k: (60, 150)[k], True, "intrinsic")
    add("C", "A100", lambda k: p7[16384][k] * inp.c_text_factor * inp.mu[k] / 60
        + inp.overhead_min[k] + ext(k), True, "intrinsic")

    # Cut item 0, cut first: the full-set 7B table, descriptive. Its own
    # session, last in the run order (sec. 8.4).
    add("C-full", "A100", lambda k: p7[16384][k] * inp.c_text_factor_full * inp.mu[k] / 60
        + inp.overhead_min[k], False, "intrinsic")

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
    # Never cut from 2026-10-03: H1 rests on the L4-A100 cells.
    add("I4", "L4", lambda k: (3, 10)[k] + inp.overhead_min[k], True, "intrinsic")

    def rep(p):
        return lambda k: (0, 3 * (inp.overhead_min[1] + sum(15 * 23 * 2 * p[x] for x in p)
                                  * inp.ex[1] / 60))[k]
    add("R A100x3", "A100", rep(pa), True, "intrinsic", units=split(rep(pa), 3))
    add("R H100x2", "H100", lambda k: rep(ph)(k) * 2 / 3, True, "intrinsic",
        units=split(lambda k: rep(ph)(k) * 2 / 3, 2))
    add("R H100#3", "H100", lambda k: rep(ph)(k) / 3, False, "intrinsic")

    def row15(bd, k, kind):
        """One 1.5B row on the A100, seconds. Lower: measured dense prefill
        plus the mean answer length at the lower decode rate. Upper, from the
        2026-10-04 re-bracket: measured dense prefill x the arm factor's
        upper end, plus the task's token cap at the upper decode rate. (Until
        then the upper was a proxy: the 7B A100 row at 16384 and the 1.5B L4
        row at 32768.)"""
        return (pa[bd] + inp.n_gen[bd] * inp.decode_s[0],
                pa[bd] * inp.ex[1] + inp.cap_tokens[kind] * inp.decode_s[1])[k]

    def ext15(n, arms, kind, bands=(16384, 32768)):
        return lambda k: n * (1 + arms[k]) * sum(row15(b, k, kind) for b in bands) * inp.wall[k] / 60

    def x_primary(bands, k, extra_arms=0):
        return (ext15(300, tuple(a + extra_arms for a in (5, 8)), "primary", bands)(k)
                + 300 * sum(pa[b] for b in bands) * inp.mu[k] / 60
                + len(bands) * inp.overhead_min[k])

    def sl(k):
        """SL at 0.25, one more arm in X0, X primary and X secondary. It has
        no estimator and reuses no exact-mass pass."""
        one = (0, 0)
        return (12 * (row15(16384, k, "secondary") + row15(32768, k, "secondary")) / 2
                * inp.wall[k] / 60
                + ext15(300, one, "primary")(k) + ext15(100, one, "secondary")(k))
    add("X0", "A100", lambda k: 12 * (1 + (5, 8)[k]) * (row15(16384, k, "secondary")
        + row15(32768, k, "secondary")) / 2 * inp.wall[k] / 60 + inp.overhead_min[k] + ext(k),
        True, "extrinsic")
    # X primary is one session per band; a rerun repeats the larger band.
    add("X primary", "A100", lambda k: x_primary((16384, 32768), k), True, "extrinsic",
        rerun_unit_min=max(x_primary((b,), 1) for b in (16384, 32768)),
        units={str(b): x_primary((b,), 1) for b in (16384, 32768)})
    # Cut item 0c. As a star line it rides in X primary's sessions, so the
    # rerun unit is the 32768 band with one more arm.
    add("SL", "A100", sl, inp.sl_star, "extrinsic",
        rerun_unit_min=max(x_primary((b,), 1, extra_arms=1) for b in (16384, 32768))
        if inp.sl_star else None)
    add("X secondary", "A100", lambda k: ext15(100, (5, 8), "secondary")(k)
        + 100 * (pa[16384] + pa[32768]) * inp.mu[k] / 60, True, "extrinsic")
    # The T4 replication: dense + XA native on T4's own 200 ids per band, at
    # both bands (the evaluation split no longer shares T4's ids).
    add("X-rep", "A100", ext15(200, (1, 1), "rep"), True, "extrinsic")

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


def launch_check(*, spent: float, next_u: float, next_item: Optional[str],
                 remaining_star_u: Sequence[float], remaining_cuttable_u: Mapping[str, float],
                 s_res_inr: float, r_b: float = 0.0, cap: float = CAP_INR,
                 xa_withheld: bool = False) -> str:
    """The reservation check from 2026-10-04 (sec. 8.3). A session launches
    only if what is left still covers every never-cut session and every
    higher-priority cuttable item not yet run.

    `next_item` is the session's cut-order item, or None for a never-cut
    session. A never-cut session outranks every cuttable item, so it reserves
    none of them. A cuttable session reserves every remaining item that is
    cut after it, and the rest of its own item. Item 12 (I2c-B) is reserved through `r_b`, never here."""
    order = cut_order(xa_withheld=xa_withheld)
    unknown = set(remaining_cuttable_u) - set(order)
    if unknown or (next_item is not None and next_item not in order):
        raise ValueError(f"not a cut-order item: {sorted(unknown) or next_item}")
    # From the item itself on: its own later sessions (XL primary after XL0)
    # are reserved too, so a probe cannot launch without its main run.
    higher = () if next_item is None else order[order.index(next_item):]
    reserve = sum(u for c, u in remaining_cuttable_u.items() if c in higher and c != "12")
    need = spent + next_u + sum(remaining_star_u) + reserve + s_res_inr + r_b
    return "PROCEED" if need <= cap else "STOP"


def fund(lines: Sequence[BracketLine], run_order: Sequence[str], *, months: int = 2,
         reserve_higher: bool = True, xa_withheld: bool = False,
         skip: Sequence[str] = ()) -> dict:
    """Walk `run_order` (line names) with every session costing its upper
    bound. Returns what launched and what was cut. `reserve_higher=False` is
    the rule before 2026-10-04, kept so the difference can be tested. `skip`
    names conditional lines that were not triggered."""
    by = {l.name: l for l in lines}
    spent, ran, cut = 0.0, [], []
    todo = [n for n in run_order if n not in set(skip)]     # conditional lines not triggered
    while todo:
        name = todo.pop(0)
        line, item = by[name], cut_item_of(name)
        star = line.star
        rest = [by[n] for n in todo]
        cuttable = {}
        for l in rest:
            c = cut_item_of(l.name)
            if not l.star and c is not None:
                cuttable[c] = cuttable.get(c, 0.0) + l.inr[1]
        r_b = by["I2c-B"].inr[1] if "I2c-B" in todo else 0.0
        # S_res covers the disk of everything this launch reserves: itself,
        # the never-cut sessions left, and the higher-priority cuttable ones.
        order = cut_order(xa_withheld=xa_withheld)
        higher = () if (star or item is None or not reserve_higher) else order[order.index(item):]
        hours = (line.minutes[1] + sum(l.minutes[1] for l in rest
                                       if l.star or cut_item_of(l.name) in higher)) / 60
        verdict = launch_check(
            spent=spent, next_u=line.inr[1], next_item=None if star else item,
            remaining_star_u=[l.inr[1] for l in rest if l.star],
            remaining_cuttable_u=cuttable if reserve_higher else {},
            s_res_inr=s_res(months, hours), r_b=r_b, xa_withheld=xa_withheld)
        if verdict == "PROCEED":
            # What a run session leaves in the ledger: its time at the card's
            # rate, and its boot disk (which S_res reserved until it ran).
            spent += line.inr[1] + DISK_PER_HOUR_INR * line.minutes[1] / 60
            ran.append(name)
        elif star:
            return {"ran": ran, "cut": cut + [name] + todo, "spent": spent, "stopped_at": name}
        else:
            # A cut takes the whole item, and what depends on it.
            gone = (item,) + CUT_IMPLIES.get(item, ())
            also = [n for n in todo if cut_item_of(n) in gone and not by[n].star]
            cut += [name] + also
            todo = [n for n in todo if n not in also]
    return {"ran": ran, "cut": cut, "spent": spent, "stopped_at": None}


# Sec. 8.5, as line names. Replicates and I2c-B run only if triggered.
# C-full is last: by then every item ranked above it has run or been cut,
# so its launch check is simply whether the money is still there.
RUN_ORDER = ("FA3", "C", "I1", "I2a", "I2c", "I2c-B", "I2b", "I2d", "I2e", "I3", "I4",
             "R A100x3", "R H100x2", "R H100#3", "X0", "X primary", "X secondary",
             "SL", "X-rep", "XL0", "XL primary", "XL secondary", "XL 65536", "C-full")

# DP1's trigger on the arm factor (sec. 8.3): the bracket's upper end.
ARM_FACTOR_UPPER = 1.6


def arm_factor_break_even(inp: BracketInputs = BracketInputs()) -> float:
    """The arm factor's upper end at which the worst case equals the cap."""
    import dataclasses
    lo, hi = inp.ex[1], 10.0
    f = lambda x: worst_case(bracket(dataclasses.replace(inp, ex=(inp.ex[0], x))))["spare"]  # noqa: E731
    assert f(lo) > 0 > f(hi)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return lo


def dp1_arm_factor(measured: float, inp: BracketInputs = BracketInputs()) -> str:
    """'within' (<= 1.6: the bracket stands), 'rebracket' (above 1.6: the
    measured value replaces the upper end before I1), or 'cut_or_stop'
    (at or above the break-even: the worst case no longer fits)."""
    if measured <= ARM_FACTOR_UPPER:
        return "within"
    return "cut_or_stop" if measured >= arm_factor_break_even(inp) else "rebracket"


DP1_COLUMNS = ("band", "arm", "prefill_ms")
NOT_DEPLOYABLE = ("dense", "O")


def measured_arm_factor(frame) -> float:
    """DP1's arm factor (sec. 8.3): the largest ratio, over the deployable
    arms and the bands, of an arm's mean end-to-end prefill to the dense
    prefill of the same band. Reads timing columns only."""
    check_no_result_columns(frame.columns)
    missing = [c for c in DP1_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"DP1 needs {DP1_COLUMNS}; missing {missing}")
    means = frame.groupby(["band", "arm"])["prefill_ms"].mean()
    ratios = []
    for band in sorted({b for b, _ in means.index}):
        if (band, "dense") not in means.index:
            raise ValueError(f"no dense prefill at band {band}: the factor cannot be formed")
        arms = [a for b, a in means.index if b == band and a not in NOT_DEPLOYABLE]
        ratios += [means[(band, a)] / means[(band, "dense")] for a in arms]
    if not ratios:
        raise ValueError("no deployable arm was timed")
    return float(max(ratios))


def dp1(frame, inp: BracketInputs = BracketInputs(), *, i2cb_runs: bool = True) -> dict:
    """The DP1 decision on the arm factor, from the canary's timing rows.

    At or below 1.6 the bracket stands. Above it the measured value replaces
    the upper end and every U is recomputed. The verdict is STOP if the
    never-cut core, the storage reserve and I2c-B alone exceed the cap,
    'proceed with cuts' if the funding walk cuts anything, else 'proceed'."""
    import dataclasses
    factor = measured_arm_factor(frame)
    trigger = dp1_arm_factor(factor, inp)
    new = inp if trigger == "within" else dataclasses.replace(inp, ex=(inp.ex[0], factor))
    lines = bracket(new)
    w = worst_case(lines)
    walk = fund(lines, RUN_ORDER, skip=() if i2cb_runs else ("I2c-B",))
    core_alone = w["core"] + w["s_res"] + (w["i2cb"] if i2cb_runs else 0.0)
    if core_alone > CAP_INR or walk["stopped_at"] is not None:
        verdict = "STOP"
    else:
        verdict = "proceed with cuts" if walk["cut"] else "proceed"
    return {"arm_factor": factor, "trigger": trigger, "arm_factor_upper": new.ex[1],
            "verdict": verdict, "cut": walk["cut"], "funded": walk["ran"],
            "worst_case": w, "u_inr": {l.name: l.inr[1] for l in lines}}


def worst_case(lines: Sequence[BracketLine], *, months: int = 2) -> dict:
    t = totals(lines)
    core_hi, core_h_hi = t["core"][1], t["core"][3]
    # One rerun of the costliest star phase: a line's largest phase where it
    # has more than one, else the whole line. (Until 2026-10-04 X primary's
    # phase was priced at half the line; the 32768 band is more than half.)
    rerun = max(l.inr[1] if l.rerun_unit_min is None
                else l.rerun_unit_min / 60 * RATE[l.card] for l in lines if l.star)
    i2cb = next(l for l in lines if l.name == "I2c-B").inr[1]
    sres = s_res(months, core_h_hi)
    total = core_hi + sres + rerun + i2cb
    return {"core": core_hi, "s_res": sres, "rerun": rerun, "i2cb": i2cb,
            "total": total, "spare": CAP_INR - total}


# --------------------------------------------------------------------------
# The budget gate (sec. 8.3): the ledger, the plan state, and the verdict
# scripts/frontier_budget_gate.py prints before every launch.
# --------------------------------------------------------------------------

LEDGER_COLUMNS = ("date", "session", "instance", "card", "minutes", "est_inr", "status", "source")
LEDGER_STATUS = ("launched", "complete", "failed", "other")
# An arm inside other sessions, not a session: the gate decides its funding,
# and a launcher refuses it.
ARM_LINES = ("SL",)
# Lines that run only if triggered. Only these can be released unrun.
CONDITIONAL_LINES = ("R A100x3", "R H100x2", "R H100#3", "I2c-B")
STATE_KEYS = ("cut", "released", "xa_withheld", "h6a_passed", "months_remaining", "inputs")


class GateRefusal(ValueError):
    """The gate cannot give a verdict: a bad ledger, state or session name."""


def session_units(line: BracketLine) -> dict:
    """{session name: upper-bound minutes}. 'I1' for a one-session line,
    'X primary:32768' for a unit of a line with several."""
    if line.units is None:
        return {line.name: line.minutes[1]}
    return {f"{line.name}:{u}": m for u, m in line.units.items()}


def read_ledger(rows: Iterable[Mapping]) -> dict:
    """Spend and progress from ledger rows. A 'launched' row counts at its
    recorded upper bound until a 'complete' or 'failed' row for the same
    instance replaces it, so a launch whose teardown was never recorded
    stays reserved in full."""
    final, opened, direct, complete = {}, {}, 0.0, set()
    for r in rows:
        if set(r) != set(LEDGER_COLUMNS):
            raise GateRefusal(f"ledger row has columns {sorted(r)}, not {sorted(LEDGER_COLUMNS)}")
        status, inst = r["status"], r["instance"]
        if status not in LEDGER_STATUS:
            raise GateRefusal(f"ledger status {status!r} is not one of {LEDGER_STATUS}")
        try:
            inr = float(r["est_inr"])
        except (TypeError, ValueError):
            raise GateRefusal(f"ledger est_inr {r['est_inr']!r} is not a number") from None
        if inr < 0:
            raise GateRefusal("ledger est_inr is negative")
        if status == "other" or not inst:
            if status in ("launched", "complete", "failed"):
                raise GateRefusal(f"a {status!r} row needs an instance")
            direct += inr
        elif status == "launched":
            opened[inst] = (r["session"], inr)
        else:
            final[inst] = final.get(inst, 0.0) + inr
            if status == "complete":
                complete.add(r["session"])
    still_open = {i: v for i, v in opened.items() if i not in final}
    spent = direct + sum(final.values()) + sum(u for _, u in still_open.values())
    return {"spent": spent, "complete": complete,
            "open": {sess for sess, _ in still_open.values()}}


def gate(session: str, card: Optional[str], rows: Iterable[Mapping],
         state: Optional[Mapping] = None) -> dict:
    """The verdict for launching `session` now. `card` is the launcher's
    card, or None for a funding decision on an arm (ARM_LINES)."""
    state = dict(state or {})
    if set(state) - set(STATE_KEYS):
        raise GateRefusal(f"unknown state keys {sorted(set(state) - set(STATE_KEYS))}")
    try:
        inp = BracketInputs(**dict(state.get("inputs", {})))
    except TypeError as e:
        raise GateRefusal(f"state inputs: {e}") from None
    lines = bracket(inp)
    by = {l.name: l for l in lines}
    units = {u: (l, m) for l in lines for u, m in session_units(l).items()}
    cut, released = set(state.get("cut", ())), set(state.get("released", ()))
    for name in cut | released:
        if name not in by:
            raise GateRefusal(f"state names {name!r}, which is not a bracket line")
    if any(by[n].star for n in cut):
        raise GateRefusal(f"a never-cut line cannot be cut: {sorted(n for n in cut if by[n].star)}")
    if released - set(CONDITIONAL_LINES):
        raise GateRefusal(f"only {CONDITIONAL_LINES} can be released")
    if session not in units:
        raise GateRefusal(f"{session!r} is not a session of the plan; sessions are {sorted(units)}")
    line, minutes = units[session]
    if line.name in cut | released:
        raise GateRefusal(f"{line.name} is cut or released")
    if line.name in ARM_LINES:
        if card is not None:
            raise GateRefusal(f"{line.name} is an arm, not a session: nothing launches for it")
    elif card != line.card:
        raise GateRefusal(f"{session} runs on the {line.card}, not on {card!r}")
    led = read_ledger(rows)
    unknown = (led["complete"] | led["open"]) - set(units) - {""}
    if unknown:
        raise GateRefusal(f"the ledger names sessions the plan does not have: {sorted(unknown)}")
    if session in led["complete"]:
        raise GateRefusal(f"{session} is already complete; a complete phase is never rerun (sec. 7.3)")
    if session in led["open"]:
        raise GateRefusal(f"{session} has a launch with no teardown recorded")

    def rate(l, m):
        return m / 60 * RATE[l.card]
    gone = led["complete"] | led["open"] | {session}
    rest = {u: lm for u, lm in units.items()
            if u not in gone and lm[0].name not in cut | released}
    xa_withheld = bool(state.get("xa_withheld", False))
    star_u = [rate(l, m) for l, m in rest.values() if l.star]
    cuttable, hours = {}, minutes + sum(m for l, m in rest.values() if l.star)
    order = cut_order(xa_withheld=xa_withheld)
    item = None if line.star else cut_item_of(line.name)
    higher = () if item is None else order[order.index(item):]
    for l, m in rest.values():
        c = cut_item_of(l.name)
        if not l.star and c is not None and c != "12":
            cuttable[c] = cuttable.get(c, 0.0) + rate(l, m)
            if c in higher:
                hours += m
    i2cb_pending = ("I2c-B" in {l.name for l, _ in rest.values()}
                    and not state.get("h6a_passed", False))
    r_b = rate(by["I2c-B"], by["I2c-B"].minutes[1]) if i2cb_pending else 0.0
    if i2cb_pending:
        hours += by["I2c-B"].minutes[1]
    sres = s_res(state.get("months_remaining", 2), hours / 60)
    next_u = rate(line, minutes)
    verdict = launch_check(spent=led["spent"], next_u=next_u, next_item=item,
                           remaining_star_u=star_u, remaining_cuttable_u=cuttable,
                           s_res_inr=sres, r_b=r_b, xa_withheld=xa_withheld)
    reserve = sum(u for c, u in cuttable.items() if c in higher)
    halt = math.ceil(1.25 * minutes)
    return {"verdict": verdict, "session": session, "line": line.name, "star": line.star,
            "cut_item": item, "spent": led["spent"], "next_u": next_u,
            "remaining_star": sum(star_u), "reserved_cuttable": reserve, "s_res": sres,
            "r_b": r_b, "need": led["spent"] + next_u + sum(star_u) + reserve + sres + r_b,
            "cap": CAP_INR, "halt_minutes": halt, "max_run_minutes": halt + 10}


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
