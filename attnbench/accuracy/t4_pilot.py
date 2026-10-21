"""The T4 pilots' pre-registered design, as data.

One home for every number docs/t4_dense_pilot.md and docs/t4_sparse_pilot.md
fix in advance: `scripts/run_accuracy.py` plans from it,
`scripts/run_t4_noninferiority.py` analyses with it, and
`tests/test_t4_sparse_pilot_plan.py` holds the documents to it. A design
restated in three places drifts in one of them.
"""

from __future__ import annotations

from .ruler import T4_DENSE_PILOT_TASKS

__all__ = ["T4_DENSE_PILOT_TASKS", "PILOT_BANDS", "PROBE_N", "SELECTED_PILOT_N",
           "SPARSE_PILOT_TASKS", "SPARSE_PILOT_N", "SPARSE_PILOT_SCORE_SOURCES",
           "SPARSITY_SEQUENCE", "MARGIN_PTS", "ALPHA", "TIERS",
           "XATTN_SCORE_SOURCE", "XATTN_THRESHOLD_SEQUENCE", "XATTN_STRIDE",
           "XATTN_CALIBRATIONS", "XATTN_CALIBRATION_SEED", "XATTN_CALIBRATION_RULER_N",
           "XATTN_CALIBRATION_TABLES", "XATTN_CALIBRATION_INDEX_OFFSET",
           "XATTN_CALIBRATION_STATISTIC", "XATTN_CALIBRATION_DESCRIPTIVE_GAP",
           "XATTN_CALIBRATION_CARD"]

# ---- dense-only pilot (docs/t4_dense_pilot.md) ---------------------------
PILOT_BANDS = (16384, 32768)
PROBE_N = 5
SELECTED_PILOT_N = 50

# ---- sparse pilot (docs/t4_sparse_pilot.md) -------------------------------
# Selected by the 40-90 rule on the n=5 probe; not re-selected at n=50.
SPARSE_PILOT_TASKS = ("qa_1", "niah_multivalue", "niah_multiquery")

# Examples per (task, band). qa_1 carries the primary claim and runs at the
# grid's ceiling for 32768 (100); at n=50 a single dense-only failure already
# bounds the difference below -12 points, so the 10-point margin would be
# reachable only with no loss at all. Examples are a seeded prefix, so the
# first 50 are the dense pilot's examples.
SPARSE_PILOT_N = {"qa_1": 100, "niah_multivalue": 50, "niah_multiquery": 50}

# The arms, by the scorer stamped on their rows: the oracle, and the
# deployable estimator run inline. The two-pass cheap arm is not in this
# pilot. XAttention is a later phase (see the plan).
SPARSE_PILOT_SCORE_SOURCES = ("dense_softmax_fp32", "minference_meanpool_inline")

# Fixed-sequence order within each (arm, task, band): test 0.5, then 0.75,
# then 0.9, stopping at the first failure. Controls the family-wise error
# within the family at ALPHA with no adjustment.
SPARSITY_SEQUENCE = (0.5, 0.75, 0.9)

MARGIN_PTS = 10.0
ALPHA = 0.025          # one-sided

TIERS = {
    ("qa_1", 16384): "primary",
    ("qa_1", 32768): "primary",
    ("niah_multivalue", 16384): "secondary",
    ("niah_multivalue", 32768): "secondary",
    ("niah_multiquery", 16384): "secondary",
    ("niah_multiquery", 32768): "observational",
}

# ---- XAttention phase (docs/t4_xattention_pilot.md) ------------------------
# Same tasks, bands, n, margin, alpha and tiers as the sparse pilot above;
# only the arm differs. XAttention selects blocks per head by cumulative
# antidiagonal mass >= tau, so its "setting" is a threshold and the fraction
# of blocks kept is an outcome, recorded per row as `realised_density`.
XATTN_SCORE_SOURCE = "xattention_inline"

# Fixed-sequence order: the least aggressive threshold first. A lower tau
# keeps less mass and so fewer blocks; the sequence stops at the first tau
# that is not non-inferior, exactly as SPARSITY_SEQUENCE does.
XATTN_THRESHOLD_SEQUENCE = (0.95, 0.9, 0.8)

# The method's own evaluation setting (eval/LongBench/pred.py at XATTN_COMMIT).
XATTN_STRIDE = 8

# ---- calibrated XAttention (docs/t4_xattention_calibrated.md) --------------
# Per-(layer, head) thresholds from the method's own profiler. Two
# calibrations, each its own arm and its own single test per (task, band):
# "authors" profiles on the method's own text set and carries the
# pre-registered claim; "ruler_heldout" profiles on RULER examples of the
# pilot's tasks and bands from another seed, and is descriptive only.
# A8 (2026-10-04): "authors" leaves out the 24 QA texts of the authors' set,
# which hold test questions and gold documents; "authors_full" is the whole
# set, as first pre-registered, and is descriptive only.
XATTN_CALIBRATIONS = {"authors": "claim", "ruler_heldout": "descriptive",
                      "authors_full": "descriptive"}

# The held-out RULER calibration set: this seed (the pilot's is 0), and
# this many examples per (task, band).
XATTN_CALIBRATION_SEED = 1
XATTN_CALIBRATION_RULER_N = 8

# T4 amendments, 2026-10-03 (docs/t4_xattention_calibrated.md, "Amendments").
# A1: the held-out set's examples start at this index, so its qa_1 questions
#     are 2000-2007 and never the test's 0-99 (a seed alone changes only the
#     distractors: ruler.py `index=i`).
XATTN_CALIBRATION_INDEX_OFFSET = 2000
# A3: the table statistic, verbatim from the released profiler, no cap.
XATTN_CALIBRATION_STATISTIC = "max over used texts, no cap"
XATTN_CALIBRATION_DESCRIPTIVE_GAP = 0.05     # entries with max - p90 above this
# A4: calibration runs where the official estimator takes its Triton path.
XATTN_CALIBRATION_CARD = "A100"

# Where the committed tables live, by calibration name.
XATTN_CALIBRATION_TABLES = {
    name: f"configs/xattn_thresholds/qwen2.5-1.5b-instruct_{name}_stride8.json"
    for name in XATTN_CALIBRATIONS}

