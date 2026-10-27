# T4 calibrated XAttention phase

Status: pre-registered 2026-10-03, before any threshold is calibrated and
before any row exists. Nothing below changes once the calibration session
has run. The numbers are data in `attnbench/accuracy/t4_pilot.py`, and
`tests/test_t4_xattention_calibrated_plan.py` holds this document to that
module.

*Credit, added 21 Oct 2026: XAttention is Xu et al., "XAttention: Block Sparse Attention with Antidiagonal Scoring", ICML 2025 (arXiv:2503.16428); code at github.com/mit-han-lab/x-attention, used at commit `e379887`.*

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

## Amendment, 2026-10-04 (before Session A)

Session A has still not run, and no table exists. This is an amendment
before data, decided by the researcher on 2026-10-04. Nothing above is
edited. Where it and the text above disagree, it governs.

**A8. The authors' profiling set is a sample of RULER.**

- **What was wrong.** The text above calls `text.json` "156 multi-document QA
  prompts" and says `authors` "is disjoint from RULER, so nothing about the
  test tasks tunes it". Both are false. Gate G12 of the estimator-frontier
  pre-registration (§5, §6) was run on the file at `XATTN_COMMIT` on
  2026-10-04 and found:
  - its 156 texts are RULER prompts in RULER's own templates, 12 for each of
    13 tasks: 96 needle texts, 12 `qa_1`, 12 `qa_2`, 12 variable-tracking and
    24 word-frequency texts;
  - its `qa_1` texts ask SQuAD questions 0 and 10, which are two of this
    phase's 100 primary test questions;
  - those 12 texts carry the gold document of every one of the 100 primary
    test questions in their haystacks.
- **Change, part 1: `authors` leaves out the 24 QA texts.** They are the 12
  `qa_1` and 12 `qa_2` texts (file indices 0, 1, 18, 19, 26, 27, 44, 45, 52,
  53, 70, 71, 78, 79, 96, 97, 104, 105, 106, 107, 130, 131, 148, 149).
  - That leaves 132 texts. With A2's length rule, 104 are used (17 of the
    QA texts were within the limit and 7 over it).
  - `authors` still carries this phase's claims.
  - The table records the excluded indices and its G12 record.
- **Change, part 2: `authors_full` is added, descriptive.** It is the whole
  set as first pre-registered: 156 texts, 121 used under A2. It is tested and
  reported like `ruler_heldout`, and its `claim` is always false. Its table
  records that G12 fails on it.

  | calibration | source | role |
  |---|---|---|
  | `authors` | `text.json` at the pinned commit with its 24 QA texts left out | claim |
  | `authors_full` | every text of `text.json` at the pinned commit | descriptive |
  | `ruler_heldout` | unchanged (A1) | descriptive |

- **G12 on `authors`: passes.**
  - No test question and no gold document's opening occurs in the 132
    texts.
  - Every needle in the 96 needle texts (18,361 distinct) was compared with
    every needle this harness generates for `niah_single`,
    `niah_multikey_1`, `niah_multivalue` and `niah_multiquery`, at 16384 and
    32768, for the test ids and the calibration ids. No (key, value) pair is
    shared. Two 7-digit values recur under other keys among the test ids
    (one is an answer of `niah_multiquery` at 32768, index 5), against 2.0
    expected by chance.
  - The calibration script refuses any other `text.json`.
- **What `authors` still shares with the test.** Its needle texts use the
  same templates as `niah_multivalue` and `niah_multiquery`. So `authors` is
  profiled on the test tasks' format, though on none of their needles,
  questions or documents. The sentence "disjoint from RULER" is withdrawn.
  `authors` shares no question, gold document or needle with the test, and
  that is all that is claimed for it.
- **The claim reads** "calibrated by the authors' released profiler on their
  profiling set with its 24 question-answering texts left out".
- **Nothing banked used either table.** The six banked XAttention files are
  the scalar phase (thresholds 0.8, 0.9 and 0.95). No calibrated table has
  been made.
- **Sessions.** Session A makes three tables, so its `authors` passes
  roughly double: 30–96 A100-minutes (₹140–460), in place of A4's 15–48.
  Session B runs three calibrations, 400 cells more: about 3.5 L4-hours
  (about ₹280), in place of 2.6.

```bash
for SRC in authors authors_full ruler_heldout; do
  python scripts/calibrate_xattn_thresholds.py --source ${SRC} \
      --out results/xattn_calibration_${SRC}_<date>
done
for CAL in authors ruler_heldout authors_full; do
  python scripts/run_accuracy.py --t4-xattn-pilot --only-backends xattention \
      --xattn-calibration configs/xattn_thresholds/qwen2.5-1.5b-instruct_${CAL}_stride8.json \
      --out results/t4_xattn_cal_${CAL}_<date>
done
```

**A9. One test example whose answer value recurs in the profiling set
(2026-10-04, before Session A).**

- `niah_multiquery` at 32768, index 5: one of its four answer values also
  occurs in an `authors` text, under another key. It is a 7-digit random
  number, and two such coincidences are expected by chance among this
  phase's test ids. No (key, value) pair is shared, which is the rule the
  gate passes on (estimator-frontier pre-registration, §5, where the rule is
  justified).
- **It stays in the test.** n is not changed after the test was fixed.
- **No claim rests on it.** Its cell is this phase's observational cell.
- **Reporting.** That cell is reported on all 50 examples and with index 5
  left out. Both are descriptive.
- `niah_multikey` is not a task of this phase, so it was not compared.

## Note, 2026-10-21: A6's link points to an archived study (T0.16)

This note is not an amendment. It changes no calibration, task, n, test or
claim of this phase, and Session A has still not run.

- **The study A6 links to is archived.** The estimator-frontier study was
  superseded by `docs/RESEARCH_PLAN.md`. Its pre-registration moved to
  `docs/archive/estimator_frontier_preregistration.md` on 21 Oct 2026
  (T0.16), and the citations of `docs/estimator_frontier_preregistration.md`
  above now resolve there. It was never locked, and it will not run as
  pre-registered.
- **What A6 committed stays committed before Session A.** That document's
  §3.9 (P-T4, R1–R4) was first committed in `7bb40ac` (3 Oct 2026). The
  file was last changed on `frontier-prereg-2026-10-03` in `65b8f71`
  (4 Oct 2026). Both commits are reachable from `main` through the T0.16
  merge, and both predate any Session A row.
- **What this phase still uses from it** moved unchanged to live code: the
  four-identity split check behind A1 and G12 (`SplitLeak`,
  `check_splits_disjoint`), now in `attnbench/analysis/splits_and_budgets.py`.
- **Scoring.** Whether P-T4 and R1–R4 are scored against this phase's rows,
  and how, is the researcher's decision. Nothing here scores them.

## Results, 2026-10-25: Session B

Session B ran on 25 Oct 2026 on a GCP L4 (g2-standard-8, `asia-northeast1-a`,
on-demand), from `main` at `c0d9358` with a clean tree, on the torch fallback
path. The install and the CUDA gate passed first (32 of 32), then a 12-row
smoke for the dense reference and for `authors`. The phases ran from 06:46 to
10:00 UTC, all `rc=0`: dense 46 minutes, each calibration 49 to 50 minutes.
The session billed 210 minutes, about INR 380 at the planning rate.

There are four files of 400 rows each, all `git_dirty=False` and at the one
commit, under `results/t11b_session_20261025/results/` locally and under the
instance's prefix in the results bucket. Every row carries
`positions_over_limit` (A7), and every XAttention row carries
`xattn_path = torch_fallback` and its table's label: `authors:7b2429d2c49a`,
`ruler_heldout:0497d47f9260` or `authors_full:195da1d20d54`.

**The dense reference reproduced exactly.** All 400 dense predictions are
identical to the scalar phase's dense rows (`7490ee4`, 3 Oct 2026) for the
same example ids, so all three tables are compared against the same
reference as the scalar phase.

The analysis is the pre-registered command, run once per table against the
dense rows and written to `results/t4_xattn_cal_analysis_20261025/`. Each
cell is `correct of n: difference, lower bound (mean realised density)`,
with differences and bounds in points. **Bold** marks a claim. Only
`authors` can claim; `ruler_heldout` and `authors_full` are descriptive
(the calibration table above, and A8).

| tier | task | band | dense | `authors` (claim) | `ruler_heldout` (descriptive) | `authors_full` (descriptive) |
|---|---|---:|---:|---|---|---|
| primary | `qa_1` | 16384 | 68/100 | **69: +1.0, −4.3 (0.59)** | 66: −2.0, −10.7 (0.47) | 69: +1.0, −4.3 (0.60) |
| primary | `qa_1` | 32768 | 46/100 | **50: +4.0, −6.0 (0.51)** | 54: +8.0, −4.4 (0.39) | 50: +4.0, −6.0 (0.53) |
| secondary | `niah_multivalue` | 16384 | 16/50 | 16: 0.0, −17.2 (0.54) | 17: +2.0, −16.4 (0.42) | 16: 0.0, −14.9 (0.56) |
| secondary | `niah_multivalue` | 32768 | 17/50 | 1: −32.0, −50.7 (0.15) | 0: −34.0, −50.8 (0.12) | 0: −34.0, −50.8 (0.16) |
| secondary | `niah_multiquery` | 16384 | 13/50 | 11: −4.0, −18.2 (0.54) | 10: −6.0, −20.9 (0.42) | 10: −6.0, −18.2 (0.56) |
| observational | `niah_multiquery` | 32768 | 3/50 | 0 (0.15) | 0 (0.12) | 0 (0.16) |

**What Session B licenses, in this phase's wording:**

- **`authors` is non-inferior to dense on both primary cells,** at a
  10-point margin and one-sided alpha 0.025: `qa_1` at 16384 (bound −4.3,
  1 discordant pair, to XAttention) and at 32768 (bound −6.0, 6 pairs to
  XAttention and 2 to dense). The table keeps 59% and 51% of blocks there.
  This is XAttention calibrated by the authors' released profiler (A5), on
  the L4's torch path, for this model.
- **Its profiling set is a sample of RULER** (A8). The 24 QA texts were left
  out of `authors`, but the table is not independent of the test
  distribution.
- **No secondary cell certifies.** On `niah_multivalue` at 32768 every
  table collapses: 1 or 0 of 50 against dense's 17, keeping 12% to 16% of
  blocks. It is the same cell, and nearly the same count, as the scalar
  phase's collapse at tau 0.95 (0 of 50, 11% of blocks).
- "No accuracy loss" is not claimed. Nothing from `ruler_heldout` or
  `authors_full` is claimed, and the observational cell yields no
  statement.

**Read descriptively, not as a test.**

- `authors_full`, which adds the 24 QA texts, gives the same `qa_1`
  outcomes as `authors` at slightly higher density.
- `ruler_heldout` keeps fewer blocks, and certifies nothing as a
  descriptive table. Its `qa_1` bound at 16384 is −10.7, just past the
  margin. At 32768 it answers 8 more of 100 than dense.
- No timing was measured or used.

**The pre-registered predictions** (§3.9 of the estimator-frontier
pre-registration, committed before Session A through A6; scored on the
researcher's decision of 25 Oct 2026, plan Appendix A).

- **P-T4 failed.** It predicted that `authors` certifies 0 of 2 primary
  cells. It certifies 2 of 2.
- **The reasoning P-T4 was labelled "close to guaranteed" by is wrong.**
  The final text (`65b8f71`) says that at n = 100 the exact bound certifies
  only when the discordant pairs are at most 4 each way, citing §7.5, which
  states no such rule. The bound is limited by the pairs dense wins, not by
  the pairs XAttention wins. At n = 100, 3 dense-only pairs certify with any
  number of XAttention-only pairs, and 4 certify once XAttention wins at
  least 4. The 32768 cell certified with 6 and 2. The archived text is not
  edited.
- **R1 to R4: not scored: comparison run not made.** Each compares an A100
  run on the same 400 ids (the `authors` table on the Triton path, and
  `sdpa_flash` dense) with these L4 rows. That run was not made and is not
  scheduled.

**Positions past the limit (A7), re-derived 26 Oct 2026.** Counted from
`positions_over_limit` in the four banked files under
`results/t11b_session_20261025/results/` (`t4_xattn_cal_<table>_20261025/accuracy.parquet`,
force-committed on 26 Oct 2026). The field equals
max(0, prompt + generated − 32,768) on every row. Each cell is rows over the
limit of n, then the largest overshoot in positions. The decoding maximum is
each task's generation cap (`attnbench/accuracy/stopping.py`): 42 tokens for
`qa_1` and 62 for both NIAH tasks, and every cell reaches it.

| file | 16384, any task | `qa_1` 32768 | `niah_multivalue` 32768 | `niah_multiquery` 32768 | 32768, all tasks |
|---|---|---|---|---|---|
| dense | 0 of 200 | 19 of 100, 42 | 50 of 50, 62 | 50 of 50, 62 | 119 of 200 |
| `authors` | 0 of 200 | 20 of 100, 42 | 50 of 50, 62 | 50 of 50, 62 | 120 of 200 |
| `authors_full` | 0 of 200 | 20 of 100, 42 | 50 of 50, 62 | 50 of 50, 62 | 120 of 200 |
| `ruler_heldout` | 0 of 200 | 21 of 100, 42 | 50 of 50, 62 | 50 of 50, 62 | 121 of 200 |

- The overshoot comes from decoding, not from the prompt. The `qa_1` 32768
  prompts are 32,383 to 32,768 tokens and the NIAH 32768 prompts 32,767 to
  32,768. At 16384 the longest prompt is 16,384 tokens, so no row there is
  near the limit.
- The dense count, 119 of 200 at 32768, is the XA pilot's count in A7.
- Of the `qa_1` 32768 rows past the limit, 7 of 19 are correct for dense,
  7 of 20 for `authors` and for `authors_full`, and 9 of 21 for
  `ruler_heldout`.
- `tests/test_t4_session_b_positions.py` derives this table from the banked
  files and checks it against this section.

**The 32768 NIAH cells are confounded by the position limit (27 Oct 2026,
researcher's decision; plan Appendix A).** Every NIAH row at 32768 (50 of 50
on `niah_multivalue` and on `niah_multiquery`, in every file) and about a
fifth of the `qa_1` rows there (19, 20, 20 and 21 of 100) decode past the
model's 32,768 positions. So the `niah_multivalue` collapse at 32768 (1 or
0 of 50 against dense's 17) is confounded and is not a finding. The same
holds for the scalar phase's collapse in that cell: all 50 of its rows there
at every tau are flagged too (`results/positions_over_limit/`, which is not
tracked). The "No secondary cell certifies" bullet above stands as a
statement that the cell did not certify, not as evidence of a collapse. The
`qa_1` 32768 claim keeps its "[32K-positions caveat]" (`docs/claims.md`).
Future accuracy runs follow the plan's grid rule (Section 5.3 step 7): a
cell's prompt plus its task's generation cap must be within 32,768.
`tests/test_t4_session_b_positions.py` checks these counts against the banked
files.
