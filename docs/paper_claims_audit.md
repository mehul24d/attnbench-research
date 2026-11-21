# Paper claims audit (22 Nov 2026)

The adviser's task of 22 Nov 2026, item 1. Five claims were checked against `docs/RESEARCH_PLAN.md` (Sections and Appendix A, cited by date) and the code on `main` at `12ff1fd`. For each claim the audit gives what the repo shows, then **what the paper must say**. Nothing here changes a scored verdict. Where a number comes from the exploratory residual analysis (`docs/exploratory/residual_20261122.md`), it is marked EXPLORATORY.

---

## (a) The timed region, and whether estimator work is inside it

**What the repo shows.**
- `attnbench/ground_truth.py:timed_block` (lines 303–337) runs in this order:
  1. a synchronise;
  2. the sampler's opening read (untimed);
  3. the clock starts;
  4. `run_measured(input_ids, backend, logits_to_keep=1)`, which is one prefill forward returning the last position's logits, with no decode;
  5. a closing synchronise, inside the clock;
  6. the clock stops.
- Outside the clock are tokenisation, model load, the warm-up rounds, the sampler reads and the profiler. Ground-truth rounds are untraced. The Kineto traces come from a separate traced pass, which runs after the timed rounds (`time_with_profiler_overhead`; Appendix A, 10 Nov 2026).
- **For every deployable arm, the estimator is inside the clock.** This covers mean-pool scoring, XAttention's `estimate` and the no-assert subclass's own `estimate`, together with the mask build, the copies and the synchronisations (Section 5.3 step 7; Appendix A, 26 Oct 2026). The rule is enforced, not just stated. `ScoringProbe` logs every call to `SCORE_TARGETS` and `MASK_TARGETS` (lines 132–145). `check_timed_region` then refuses two cases (lines 340–378):
  - any score or mask-build call made in setup;
  - any timed rep with no score event, which catches scores computed earlier and reused, and a probe pointed at the wrong function.
- **For the oracle, the importance scores are outside the clock.** The oracle may cache its scores, but not its masks. Its mask is built from the cached scores on the host, inside the clock, so its builder kind is "cpu" (Appendix A, 31 Oct 2026). The oracle is labelled a "non-deployable reference (scoring cost outside the timed path)" (Appendix A, 26 Oct 2026, label corrected 27 Oct 2026). It is an attention-mass reference, not a quality ceiling (Appendix A, 4 Oct 2026). The exploratory dispatcher leaves the oracle out of its choices (`safe_dispatch.NOT_DEPLOYABLE`).
- **Figures from before T2.13 timed scoring outside the clock.** `scripts/run_vectorised_endtoend.py` computes `compute_importance_scores` (line 302) and only then times `run_measured`. `attnbench/accuracy/phase_timing.py:19` says scoring is "EXCLUDED from every latency number". So every end-to-end ratio in Section 2's "earlier oracle-mask study" table excludes the oracle's scoring. That covers the A100's 0.75x and 0.63x, the L4's ~1.3x at 32768 and the vectorised 1.201x. The same table prices that scoring at ~35x the saving at 32768.
- The P2/P4 timed path is pinned: `0e0fb012…` at `83b2edf` (`docs/lock_values.json`, `timed_path_identity_for_p2_and_p4`; Appendix A, 10 Nov 2026). The compiled groups were dropped from P2 and P4 (Appendix A, 12 Nov 2026), so every scored row is eager.

**The paper must say:**
1. The timed region is one batch-b prefill forward to the last position's logits, from a synchronise to a closing synchronise inside the clock. Decode, tokenisation and warm-up are outside it.
2. For mean-pool and both XAttention arms, the estimator, mask build, copies and synchronisations are inside the timed region, and the harness refuses a measurement otherwise.
3. Oracle rows exclude the importance-score pass. They are a non-deployable reference, never a deployable speedup, and the paper must say so wherever an oracle number appears.
4. Every Section 2 end-to-end ratio is an oracle figure with scoring excluded. The paper must not set those ratios beside P2/P4 deployable rows on one curve (Section 6, rule 2).

---

## (b) Historical 32768-token accuracy rows past the model's position limit

**What the repo shows.**
- **The counts.** Qwen2.5 has 32,768 positions. At 32768:
  - every NIAH row (50 of 50 per task per file) decodes past the limit;
  - 19 to 21 of 100 `qa_1` rows do too (dense 19, `authors` 20, `authors_full` 20, `ruler_heldout` 21; overshoot up to 42 positions, which is the generation cap).
  - The scalar-XAttention pilot's dense file has 119 of 200 past the limit.

  The source is `docs/t4_xattention_calibrated.md`, section "Positions past the limit (A7)", and `tests/test_t4_session_b_positions.py` derives those counts from the banked parquets.
- **They are labelled, not quarantined.** No analysis code filters on `positions_over_limit`. The field is written on new rows (`accuracy/schema.py:129`, `generation.py:95`) and is null on older banked rows. `scripts/flag_positions_over_limit.py` recomputes it, into `results/positions_over_limit/`, which is untracked. Three things handle the rows instead:
  - **Labels.** The `niah_multivalue`/32768 collapse is "confounded and not a finding" (Appendix A, 27 Oct 2026; Section 2; `docs/t4_xattention_calibrated.md`). Every 32768 accuracy figure in `docs/claims.md` carries "[32K-positions caveat]" (defined at `docs/claims.md:16`, added 3 Oct 2026).
  - **A rule for new runs.** Prompt plus generation cap must fit within 32,768. At 32768, admissibility comes from a cell sized down, while the timing cell stays at 32768 (Section 5.3 step 7; Appendix A, 27 Oct 2026). Probe lengths are capped at 32768 (Appendix A, 31 Oct 2026).
  - **The timing side.** Prefill only, at a realised 32,700 tokens, so no timing row reaches the limit.
- **T1.1's certified `qa_1`/32768 cell includes the rows past the limit.** It reads 50 vs 46 of 100, bound −6.0, density 0.51 (Section 2), and the bound is computed over all 100 rows.
- **EXPLORATORY sensitivity, 22 Nov 2026 (adviser's task, item 5; it does not replace the certified −6.0).** `scripts/t11_qa1_within_limit.py` reruns the certified path (`run_t4_noninferiority.load`, `run_exact_ni`, margin 10 points, one-sided alpha 0.025) on the rows inside the limit, excluding for each arm the union of ids past the limit in the dense file or the arm's file. For `authors`: 20 rows excluded, n = 80: dense 39, `authors` 43, 4 pairs to XAttention and 0 to dense, bound −4.2 (difference +5.0); still non-inferior. The all-rows rerun reproduces −6.0 (−5.97). Descriptive arms: `authors_full` the same as `authors`; `ruler_heldout` 21 excluded, n = 79, 38 vs 45, bound −3.9. `tests/test_t11_qa1_within_limit.py` checks these numbers against the banked files.

**The paper must say:**
1. In the historical 32768 accuracy rows, every NIAH row and about a fifth of `qa_1` rows decoded past the trained positions.
2. The NIAH 32768 cells are excluded as findings.
3. The `qa_1`/32768 certification includes 19 to 21 rows past the limit and is reported with that caveat. A sensitivity on the rows within the limit is owed (decision 1 below).
4. No study row ran past the limit, by the grid rule of 27 Oct 2026.

The supported claims in `docs/claims.md` that carry the caveat and still cite 32768 accuracy must keep it in the paper. These include "1.321x at 32768 with no accuracy loss" and "at ceiling at 32768".

---

## (c) Why P1's margin was zero before the exploratory margin 0.02264

**What the repo shows.**
- **The primary rule.** It is the smallest margin at which no P1 batch-1 point chooses a deployable sparse configuration that measured slower than dense (`safe_dispatch.margin_from`; Appendix A, 17 Nov 2026).
- **Why it gave 0.0.** On P1, the recursion's dispatcher picks sparse at two of its five batch-1 points, both mean-pool 0.25: at 16370 (predicted advantage 0.1112, measured 13.1% faster) and at 32700 (0.2329, measured 23.1% faster). It picks dense at 1951, 4068 and 8075. With no false-profitable choice on P1, the rule has nothing to bound and returns 0.0. A zero margin is degenerate: it would leave the abstaining dispatcher identical to the plain one.
- **The fallback.** As a plan default, the margin became the largest P1 overstatement of a deployable configuration's predicted advantage over dense: **0.02264**, mean-pool 0.5 at 8075 (`fallback_margin`; 20 configurations at 5 points).
- **Disclosure.** P2's and P4's per-cell predicted and measured times had been displayed before the margin was fixed. No P2 or P4 dispatch choice had been computed (Appendix A, 17 Nov 2026).
- **Sensitivity (EXPLORATORY, this wave).** The dispatcher's choices are identical for every margin in [0, 0.1112), so 0.02264 changed nothing. Any margin large enough to remove P4's one loss (≥ 0.1153, mean-pool 0.25 at 16370, 13.9% slower than dense) first drops the 16370 win on P1 (at 0.1112) and on P2 (at 0.1124).

**The paper must say:**
1. The pre-registered zero-false-profitable margin was 0 on P1, because no P1 sparse choice lost.
2. 0.02264 is a plan-default fallback, fixed from P1 after P2/P4 cell times had been seen.
3. The safe-dispatch result is EXPLORATORY.
4. No margin in [0, 0.111) changes any choice, and no margin removes P4's loss without giving up a P1 and a P2 win.
5. That loss follows from P4's dense FA3 misprice, item (d).

---

## (d) `fill_` folded into FA3; the short-length trace inflation

**What the repo shows.**
- **Where the extra operations come from.** P4's ground truth ran FlashAttention-3 for dense, but its prediction priced `sdpa_flash`. This is an implementation fault in the committed prediction (2.52x at 16370). H1 is not rescored (Appendix A, 17 Nov 2026). Q2 shows the misprice is the backend mismatch alone (Appendix A, 17 Nov 2026, Q2 line). FA3 issues one `aten::fill_` per layer before its forward kernel, so 28 per prefill. They appear in every dense P4 trace and are absent from the priced sequence.
- **How the exploratory alignment handles them.** `exploratory/traces.align` (`UNPRICED`) drops each `fill_` and adds its device time to the next operation, the FA3 forward. So the traced "attention" component includes the 28 `fill_` kernels. The 28 host launches of `fill_` are not in the prediction's host term; at ~5 µs each that is ~0.14 ms, an estimate not measured on its own. This only applies to P4 dense; P2 folds none.
- **Trace inflation.**
  - **The ~7 µs per kernel figure** (the 16 Nov report, `breakdown.py` docstring) holds only for P2's dense cell at 1951 tokens, where traced minus priced device time is 7.25 µs per GEMM or other-device kernel. P2's other arms at 1951 show 2.7–4.4 µs, and at 4068 the figure is 0.1–5.0 µs.
  - **P4 has essentially none:** 0.07–0.14 µs per kernel at 1951 and 4068 (EXPLORATORY, this wave).
  - **On P2, the trace can exceed the untraced run.** At dense 1951, the traced device busy time (45.2 ms) is longer than the whole untraced measurement (39.5 ms).
  - **The traced host timeline is far more inflated.** P4's 1951 dense traced region is 65.7 ms against 17.3 ms untraced.
  - So traced host time is never used as a measurement, and short-length device substitution is an upper bound on P2 only.

**The paper must say:**
1. P4's dense prediction priced the wrong kernel (SDPA flash instead of FA3), a fault in the committed prediction that the paper reports, not corrects. H1 on P4 (0.2495) stands as scored.
2. In the exploratory per-component analysis, FA3's per-layer `fill_` is counted with attention, and its launches are missing from the priced host term.
3. Trace inflation is pairing-specific: up to ~7 µs per kernel on the A100 (P2) at 1951 tokens and under 0.2 µs on the H100 (P4). The 16 Nov report's single "~7 µs per kernel" must not be quoted for P4.

---

## (e) The null `power_target_and_n` field in the lock reader

**What the repo shows.**
- In the tag's `docs/lock_values.json`, `power_target_and_n` is `null`, typed not applicable: "H4's power rule needs T1.5's pilot, which never ran". H4 and H5 are not scored and T1.5 is waived (Section 4; Appendix A, 12 Nov 2026).
- The analysis commit's `Locked.from_lock` (`83b2edf`) reads that field for H4's alpha and n, so it refuses the real lock. The scoring driver's `locked_from` builds the same object without H4's fields. That is a flagged deviation from "every threshold read through `from_lock`" (Appendix A, 12 Nov 2026, wave 2a).
- The check (Appendix A, 16 Nov 2026, item 1; `docs/scoring/lock_reader_check_20261116.json`):
  - an offline rerun reproduced the report exactly;
  - no null field is read by scoring. The only reader of `power_target_and_n` is the tag-versus-`main` comparison (`differing_paths`);
  - the 24 fields that scoring reads are non-null and equal the tag's file (sha256 `e1addec5…`);
  - on a lock that differs only by a placeholder, `locked_from` and `from_lock` agree on every non-H4 field.

  Four other fields are null in the tag and also read only by `differing_paths`: `admissibility_holm_family`, `cross_card_dense_task_score_bound`, `head_type_criterion` and `primary_cell_n_and_admissibility_n`.

**The paper must say:**
1. H0–H3 were scored by a reader other than the analysis commit's `from_lock`, because `from_lock` requires H4's power target and the lock types it not applicable (H4 not scored).
2. The substitute reader was checked to read no null field and to agree with `from_lock` on every non-H4 value.
3. The paper should cite this as a deviation in the artifact appendix.

---

## Decisions needed

1. **(b)** Should the `qa_1`/32768 bound be recomputed on the rows inside the position limit, as a sensitivity in the paper? Excluding the union of over-limit ids across the paired files is the plan default if yes. It would be CPU only, from the banked T1.1 parquets. **Ruled 22 Nov 2026: recomputed as an exploratory sensitivity (above, section (b)); the certified value stands. Whether the paper's Appendix A carries it is held with the bundled Appendix A wave.**
2. **(a, c, d)** Should these "must say" lines also go into Section 7 / T7.x of the plan as dated Appendix A lines? Doing so moves `plan_sha256` and needs the PDF rebuild. This wave did not edit the plan.
