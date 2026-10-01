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
           "SPARSITY_SEQUENCE", "MARGIN_PTS", "ALPHA", "TIERS"]

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
