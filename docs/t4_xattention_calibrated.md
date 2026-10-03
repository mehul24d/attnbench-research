# T4 calibrated XAttention phase

Status: pre-registered 2026-10-03, before any threshold is calibrated and
before any row exists. Nothing below changes once the calibration session
has run. The numbers are data in `attnbench/accuracy/t4_pilot.py`, and
`tests/test_t4_xattention_calibrated_plan.py` holds this document to that
module.

**Why this phase exists, and when it was decided.** It was decided after
the scalar phase (`docs/t4_xattention_pilot.md`) had certified nothing. That
phase stated one deviation from the method: a scalar threshold where the
method profiles one per (layer, head). The authors' own RULER script
(`x-attention/scripts/run_ruler.sh` at the pinned commit) passes no `--threshold`, so it
runs a profiled 32×32 table (`llama_fuse_8`). This phase removes that
deviation and changes nothing else.

**It is the last XAttention configuration this study tests on this model,
whatever it shows.** A third configuration chosen after seeing this one
would be a search for one that passes.

## Question

With per-(layer, head) thresholds calibrated by the method's own profiler,
is XAttention non-inferior to dense, within a fixed margin, on the pilot's
tasks?

## Calibration

**The procedure is the authors' code.** It is
`xattn/threshold/profile_threshold/profile_threshold.py` at `XATTN_COMMIT`,
imported, not copied. `attnbench/backends/xattention.py` (`ThresholdProfiler`)
calls its `xattn_prefill_profile` once per layer, with that layer's q and
KV-repeated k. For each text, layer and head, it finds the fewest key blocks
covering 90% of the exact attention mass, per query block. It then returns
the threshold on XAttention's own estimate that would select at least those
blocks. The table is the maximum over texts, as the official script's
`final_threshold` is.

The settings are stride 8 and block 128, as in the scalar phase. The layer
output during profiling is dense causal attention, so each layer is profiled
on the hidden states the real model produces.

`scripts/calibrate_xattn_thresholds.py` runs it, with two sources. Each
source is its own arm:

| calibration | source | role |
|---|---|---|
| `authors` | the method's own profiling set: `text.json` at the pinned commit, 156 multi-document QA prompts, with the Llama-3 chat markers stripped | claim |
| `ruler_heldout` | RULER examples of the pilot's three tasks at 16384 and 32768, seed 1, 8 per (task, band) | descriptive |

- **`authors`** is disjoint from RULER, so nothing about the test tasks
  tunes it. It carries this phase's claims.
- **`ruler_heldout`** answers the objection that the authors' texts
  (thousands of tokens) are much shorter than the test bands. It profiles
  in distribution and at the test lengths. The script refuses to run if any
  of its contexts equals a test example's. A table tuned on the benchmark's
  own distribution is reported with its bounds but never claims.

**Each table is committed before any test row uses it.** The tables are
written to the paths in `XATTN_CALIBRATION_TABLES`. Each carries the sha256
of its values, its source's sha256, and the commits it came from. Every
XAttention row records `xattn_calibration = "<name>:<sha256[:12]>"`. A
table whose recorded digest does not match its values cannot be loaded.

**If flash-attn is absent.** The official profiler imports flash-attn only
to return a layer output, which this study replaces. Where flash-attn is
not installed, a stand-in is registered for that import, and the table
records `flash_attn_stubbed`.

## The arm

The arm is identical to the scalar phase except for the threshold: the
official estimator, inline; Block-Sparse-Attention; block 128; stride 8;
`norm=1`; `select_mode="inverse"`; `keep_sink=True` and `keep_recent=True`;
the official chunk size; KV heads repeated; dense decode.

Each layer's estimate receives that layer's row of the table as a per-head
tensor. That is the form the authors' RULER path passes to
`Xattention_prefill` (`threshold[layer_idx]`).

`keep_sink` and `keep_recent` stay on. The authors' RULER script leaves them
off, but turning them off as well would change two things at once relative
to the scalar phase. They can only add blocks.

Every row records `realised_density` and `realised_density_by_layer`.

## Tasks, tiers, n and the test

These are the sparse pilot's, unchanged:
- **Tasks and n:** `qa_1` (primary, n=100); `niah_multivalue` and
  `niah_multiquery` (secondary, n=50); `niah_multiquery` at 32768
  observational.
- **Test:** the exact paired bound in
  `attnbench/analysis/exact_noninferiority.py`.

**Margin of 10 points; one-sided alpha 0.025.** A calibration is one
setting, so each (task, band, calibration) is one test. There is no
sequence.

The two calibrations are separate families. Only `authors` may claim.
`ruler_heldout` is tested and reported, and its `claim` is always false.

The dense reference is re-run at this phase's commit, in the same session.
It must reproduce the 400 dense predictions both earlier pilots share.

```bash
python scripts/run_t4_noninferiority.py \
    results/t4_xattn_cal_dense_<date>/accuracy.parquet \
    results/t4_xattn_cal_authors_<date>/accuracy.parquet \
    results/t4_xattn_cal_ruler_heldout_<date>/accuracy.parquet \
    --out results/t4_xattn_cal_analysis_<date>
```

## What may be claimed

- A claim needs `claim=True`, which only the `authors` calibration can
  have. It reads: "with per-(layer, head) thresholds calibrated by the
  method's own profiler on its own profiling set, XAttention is
  non-inferior to dense within 10 points on `<task>` at `<band>` (exact
  one-sided 97.5% bound *L*; mean realised density *d*)".
- **"No accuracy loss" is never claimed.** Failing to certify is not
  evidence of loss. The observational cell yields no statement, and the
  secondary tier no headline.
- **If both calibrations fail, the supported statement is narrow.**
  XAttention, calibrated by its authors' procedure on two different
  profiling sets, did not certify on these tasks for this model. That is
  not a statement about every possible calibration.
- **Comparisons with the scalar phase and the other arms are descriptive
  only.**

## Sessions

**Session A: calibration, about 45 minutes.** Install; CUDA gate; calibrate
`authors`; calibrate `ruler_heldout`.

The gate must pass first. It includes `tests/test_xattn_calibration.py`,
whose two CUDA tests check:
- a table row runs bitwise identical to `Xattention_prefill` given that row;
- the profiler records exactly what the official profile computes.

After the session, both `thresholds.json` files are copied to the
committed paths, committed and pushed. That commit is what session B
deploys.

```bash
bash scripts/install_xattention.sh
python scripts/calibrate_xattn_thresholds.py --source authors \
    --out results/xattn_calibration_authors_<date>
python scripts/calibrate_xattn_thresholds.py --source ruler_heldout \
    --out results/xattn_calibration_ruler_heldout_<date>
```

**Session B: the test, about 2.6 hours.** Install and gate; a 2-example
smoke run of each calibration into its own directory (never pooled); then
the dense reference and each calibration in full. The dense run is 400
cells, and each calibration run is 400 cells.

These must all hold before the result is read:
- every phase has `rc=0`;
- every row has `git_dirty=False` and one commit;
- every XAttention row carries its committed table's label;
- the dense predictions equal the earlier pilots' 400.

```bash
python scripts/run_accuracy.py --t4-xattn-pilot --only-backends sdpa_flash \
    --out results/t4_xattn_cal_dense_<date>
for CAL in authors ruler_heldout; do
  python scripts/run_accuracy.py --t4-xattn-pilot --only-backends xattention \
      --xattn-calibration configs/xattn_thresholds/qwen2.5-1.5b-instruct_${CAL}_stride8.json \
      --out results/t4_xattn_cal_${CAL}_<date>
done
```

**Cost:** about 0.75 L4-hours for session A and 2.6 for session B, roughly
INR 270 together. The cap, hard-delete limit and artifact paths are stated
with each command before it launches.

## Amendments, 2026-10-03 (before Session A)

Session A has not run, so each of these is an amendment before data. They
come from the estimator-frontier pre-registration
(`docs/estimator_frontier_preregistration.md`, §12), which verified the
sources they rest on. Nothing above is edited. Where an amendment and the
text above disagree, the amendment governs. The code is in
`attnbench/accuracy/t4_pilot.py`, `scripts/calibrate_xattn_thresholds.py`
and `attnbench/accuracy/ruler.py`. `tests/test_t4_amendments.py`,
`tests/test_xattn_calibration.py` and this document's plan test check it.

**A1. `ruler_heldout` leaked `qa_1` questions.**

- **What was wrong.** RULER QA picks the SQuAD question by the example's
  **index**, not its seed (`ruler.py`, `index=i`). Seed-1 examples 0–7
  therefore asked test questions 0–7 with other distractors. The old check
  compared whole contexts, so it passed them.
- **Change, part 1.** `ruler_heldout` is drawn from seed 1 at index offset
  2000 (`XATTN_CALIBRATION_INDEX_OFFSET = 2000`). Its `qa_1` questions are
  2000–2007 per band, and its NIAH examples come from seed 1 at those
  indices.
- **Change, part 2.** `assert_disjoint` now compares four identities:
  - example id;
  - context sha256;
  - the task-qualified QA question index;
  - NIAH answer values.

  It refuses on any shared one.
- **Break-test.** A shared `qa_1` question under another seed and context is
  refused (`test_a1_a_shared_qa_question_under_another_seed_is_refused`). At
  offset 2000 the same seed passes.
- The table stays descriptive.

**A2. Over-length texts are excluded.**

- 35 of the 156 `authors` texts exceed Qwen2.5-1.5B's 32,768
  `max_position_embeddings`, up to 91,574 tokens. The median is 16,025, so
  "thousands of tokens, much shorter than the test bands" above is false.
- **Change.** A text is used if its token count is ≤ the model's
  `max_position_embeddings`, read from its config at run time. Longer texts
  are excluded, never truncated. That leaves 121. The excluded indices and
  their token counts are written into the table as `excluded`.

**A3. The threshold statistic.**

- **The table:** max over used texts, verbatim from the released profiler,
  with no cap (`XATTN_CALIBRATION_STATISTIC`).
- **Recorded, descriptive only, never used:**
  - the p90 table;
  - the number of entries where max − p90 > 0.05;
  - for each entry, the text that sets the max (`descriptive`).

**A4. The calibration card: Session A moves to the A100.**

- The profiler's `use_triton=True` takes the Triton path only where the
  device name contains "100". The script refuses any other card
  (`XATTN_CALIBRATION_CARD = "A100"`), and the table records
  `xattn_path="triton"`.
- Session B stays on the L4. Every XAttention row now carries `xattn_path`,
  which is `torch_fallback` there.
- Session A's cost becomes 15–48 A100-minutes (₹70–230), in place of about
  45 L4-minutes.

**A5. Calibration provenance.**

- The profiler is the authors' **released substitute** for the paper's
  unreleased DP method (issue #13). The table records this as `procedure`.
- This phase's claims read "calibrated by the authors' released profiler",
  not "by the method's procedure".

**A6. The replication link.** The estimator-frontier pre-registration's §3.9
(P-T4, R1–R4) is committed before Session A. This phase's `authors`
configuration is what that study replicates, and it is unchanged by it.

**A7. Positions past the limit.**

- 119 of the 200 dense rows at 32768 in the XA pilot decode past 32,768
  positions.
- New rows carry `positions_over_limit = max(0, prompt + generated −
  max_position_embeddings)`.
- No design change.
