# Estimator cost/quality frontier — pre-registration (DRAFT)

Status: **unlocked draft, amended 2026-10-03 (sixth draft). Not locked.** No GPU
session has run for this study, and no row exists. §13 lists what changed in
each draft.

The draft becomes a pre-registration when the lock gates in §11.1 are met and
committed **with** this file. Until then it can change without a dated
amendment. After lock, nothing above "Results" changes, and corrections go in
dated sections at the end.

Every number from a banked artifact names its path. Every number that does not
is marked **assumed**.

## 0. Permission, publication, sequence

### 0.1 Permission to use XAttention's code (recorded 2026-10-03)

**Written permission received, dated 2026-10-03, from the XAttention
authors.** The sender is not named, at the researcher's choice. The email
is private: it stays outside the repository, and no file quotes it.

**Scope, as the researcher's summary, not a quote.** The authors have
permitted:

- using their code;
- reproducing results;
- publishing results and figures;
- reproducing their kernel lines, including those in
  `attnbench/backends/xattention.py` in the public repository. The authors
  were informed of that file and approved it. It is left as it is, and the
  earlier plan to rewrite it (§4.4) is dropped.

This summary is the only record of the scope in the project. Anything beyond
it is not permitted (§0.2).

**Superseded.** The professor's guidance ("researcher-reported institutional
guidance, no written artifact yet") is superseded by this written permission.
It was never the authors' permission.

**Use is not a licence.** These stay prohibited:

- vendoring, or copying their repository;
- redistribution;
- baking their code into an image.

The pinned commit is cloned at run time. The repository still has no
licence (§4.1).

**Request date: 2026-10-03.** Nothing else about the request is kept in the
repository.

**Two questions remain unanswered.** Both were asked in that request:

1. the provenance of the shipped `llama_fuse_*` thresholds;
2. whether the Triton and torch paths select the same blocks.

There is no waiting window. Threshold provenance stays unknown. H6b and R1
answer both questions empirically, so §4.3 and R1 do not change on the basis
of silence. A later reply that contradicts the plan is handled as a dated
amendment **before any affected run**, never after its results.

### 0.2 Permission gate and publication rule (binding; §8.6)

**The default is NOT PERMITTED.** Every action that depends on the authors'
permission is NOT PERMITTED until the researcher explicitly says yes, in
this conversation, to that exact action.

- **Asking.** Claude Code asks each time, naming the exact action: the files
  or results involved, and the destination.
- **No inference.** A yes is never inferred from an earlier yes, from the
  §0.1 summary, or from silence. **No reply means no.**

The actions that need a yes include, and are not limited to:

1. pushing any file that contains their code lines, including
   `attnbench/backends/xattention.py`, and any file reproducing a line of
   x-attention;
2. putting any XAttention-arm result or figure (native, calibrated, or any
   arm running their selector, estimator or thresholds) into a paper,
   report, PDF, slide or any other document that leaves the project;
3. any use beyond the §0.1 summary;
4. vendoring or copying their repository, or baking it into an image. **These
   are not permitted at all; a yes cannot make them so here**;
5. any use of their code in a new context, or by someone else.

**This file itself is gated for pushing.** It reproduces lines of x-attention: the authors' RULER call `Xattention_prefill(q, k, v, stride, norm=1, threshold=table[layer_idx], use_triton=True)` in §4.1, and settings copied from their code. Committing it locally is not gated. **Pushing it is action 1 above**, and needs a logged yes.

**The permission log.** Every answer is logged with its date and the exact
action in `/Users/mehuldahiya/Desktop/research/xattention_permission_log.md`,
outside the repository. Claude Code reads the log before any gated action,
and never writes any quote of the email into it.

**The publication rule** (no XAttention result leaves the project without
permission) holds alongside the gate. The §0.1 summary covers publication in
principle, but each act still needs its own logged yes.

### 0.3 Fallback if XAttention results are withheld

If permission proves narrower than needed, the study publishes from the arms
that run none of their code:

- the oracle;
- mean-pool (attnbench's implementation of MInference Alg. 3);
- vertical-slash-style scoring (MIT-licensed MInference code);
- sink + local window (SL, §4.2);
- the era-4 selector;
- Block-Sparse-Attention (BSD-3);
- FA3 (BSD-3).

**What the paper claims then:**

| survives without XAttention | how it changes |
|---|---|
| H1 | runs on MP and the selector |
| H2 | runs on MP, VS and SL |
| H3 | becomes VS against MP under one selector, as a new comparison; XA withheld |
| H3b | runs over MP, VS and SL |
| H4 | runs on the non-XA extrinsic arms |
| H5 | counts non-XA arms only |
| H7a | MP across block sizes |
| the frontier | plotted without XA points |

**Withheld:** H6, H7b, H8, R1–R4, P-T4, and every XA point and row.

**An option, not adopted.** A reimplementation of the published algorithm
from the paper's description (antidiagonal scoring, mass-threshold
selection), written without their code. If used, it is labelled **"our
implementation of the published algorithm"**, never "XAttention".

- Its results are not comparable to XAttention's official code until shown
  equivalent.
- **It is not to be used until the researcher has checked with the
  professor.** Any such use is a dated amendment before its first run.

### 0.4 Sequence (binding)

1. **Dated T4 amendments.** Append §12's amendments to
   `docs/t4_xattention_calibrated.md`, with the code they need and their
   tests, and commit.
2. **Lock this file.** The replication predictions in §3.9 and the T4 outcome
   prediction P-T4 are therefore committed **before any T4 calibrated row
   exists**.
3. **Run T4 calibrated as amended.**
   - Session A: calibration, on the A100 (§12 A4).
   - Session B: the test, on the L4.
   - Bank the results and score P-T4.
4. **Run this study.**
   - Its native-XAttention arm on Qwen2.5-1.5B is a **replication of T4's
     `authors` configuration, identical in every setting** (§4.3).
   - It loads T4's committed table by digest.

No step starts before the one above it is committed.

---

## 1. Question

> For training-free block-sparse prefill, take each importance estimator, run
> on the GPU with the same selector, mask format and kernel. Where does it sit
> on the plane of (estimator cost relative to the attention it saves) × (how
> much of the true attention mass its mask keeps)?
>
> - Which points are faster than the fastest correct dense kernel, end to
>   end?
> - Does the plane move between GPU generations as a roofline account
>   predicts?
> - Does the mass a mask keeps predict whether the answer survives?
> - Does XAttention reproduce, in a neutral harness, on its authors' own
>   model and configuration?

This is audit Part 16 item 3 and gap G2 (`peer_review_audit_2026-09-30.md`).
It replaces the oracle-cost ratio, which is tautological (T2), with a
saving-relative frontier.

**What is already known, so that it is not the contribution.** XAttention
reports its own selection time. Sparse Frontier v3 states the estimation-cost
barrier qualitatively. "Estimator X costs a ms" is incremental on its own.

The non-incremental parts are:

- a component crossover model tested against end-to-end time (H2);
- a mechanistic prediction of the shift across GPU generations (H1);
- scorer quality separated from selector policy (H3);
- the effect of block granularity on scorer quality (H7);
- the intrinsic-to-extrinsic link (H4);
- a neutral reproduction of the method in its authors' configuration (H8,
  R1–R4).

---

## 2. Definitions

**Geometry.**

| model | layers L | query heads H_q | KV heads H_kv | head dim |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 28 | 12 | 2 | 128 |
| Qwen2.5-7B-Instruct | 28 | 28 | 4 | 128 |
| Llama-3.1-8B-Instruct | 32 | 32 | 8 | 128 |

Block size b = 128 for every timed arm. Intrinsic recall also runs at
b ∈ {16, 32, 64} (§2.3). Query block i, key block j.

### 2.1 Masks, density, era 4, cost

**Era-4 selection rule (per-head common selector).** Per layer, **query head**
h and row i:

- **Free blocks:** the sink (j = 0) and the diagonal (j = i) are granted free,
  outside the budget.
- **Budget:** the candidates are 0 < j < i, so n_cand = i − 1. The budget is
  round(d_nom · n_cand), half-to-even, filled by descending score of **head h's
  own** scores.
- **Ties:** the era-3 seeded 1e-9 jitter, identical across heads.

That is the era-3 rule (`masks.importance_block_mask_device`) applied per head
instead of to the head mean. With head-mean scores broadcast to every head, it
reproduces era 3 bitwise (gate G3). The era is registered in §4.5.

**Density:**

- the nominal density d_nom ∈ {0.50, 0.25, 0.10, 0.05} sets the budget over
  candidates;
- the realised density d is (blocks kept, free blocks included) / (causally
  valid blocks), over rows, heads and layers, as `realised_density` in
  `backends/xattention.py`;
- d is slightly above d_nom, because the free blocks are extra;
- every comparison and plot uses the realised d.

**Per-layer component times.** CUDA-event medians, warm, 50 reps after 5
warm-up calls, on selection-split prompts:

- T_est: from (q, k) to scores, including KV repetition if needed.
- T_sel: from scores to the bool mask on the device.
- T_bsa(d): the kernel on that mask.
- T_dense: the **timing baseline** of §4.6.

Two ratios follow:

- c = (T_est + T_sel) / T_dense
- m = (T_est + T_sel + T_bsa) / T_dense

### 2.2 Profitable

A point is a (model, card, band, estimator, selector, budget). It is
**profitable** if and only if the speedup s = T_dense^e2e / T_arm^e2e
satisfies:

    one-sided 97.5% lower bound on s  >  1 + f

where f is the resolution floor (§7.4). It is **unprofitable** if the upper
bound is below 1 − f. Otherwise it is **unresolved**.

- **End-to-end prefill** is one forward at batch 1 over the whole prompt,
  through every layer, with the estimator, selection, mask build and kernel
  running inline on the device. The clock runs between device
  synchronisations, from the embedding's first kernel to the logits' last.
  Tokenisation and decode are excluded.
- **The dense arm uses the timing baseline (§4.6).** It runs in the same
  session and commit, interleaved per rep, and twice per rep as an A/A
  control.

Profitable is never asserted from components.

### 2.3 Intrinsic quality: attention-mass recall

For layer ℓ, query head h, row i and block size b:

    M_b(i, j) = (1/|B_i|) · Σ_{q ∈ B_i} Σ_{k ∈ B_j, k ≤ q} softmax_q(q·k/√d)[k]

- It is computed in fp32 from the dense forward's post-RoPE bf16 q and k, so it
  is teacher-forced (limitation F5).
- Each row sums to 1.
- M at b = 16 is computed once per chunk and summed into 32, 64 and 128.

Let S be the selected set and F ⊂ S the free blocks. Then:

- **Raw recall:** R = Σ_{j∈S} M(i, j).
- **Normalised recall:** R̃ = R / R*. R* is the largest mass of F plus the
  same number of non-free blocks, which is the oracle's top-k under the era-4
  rule.

**Aggregation:**

- rows i < 2 are excluded, because the free blocks fill them;
- **primary:** the per-example unweighted mean of R̃ over (ℓ, h, i), then the
  mean over examples, with a 95% percentile bootstrap over examples
  (B = 10,000, seed 20261003);
- **secondary, untested:** the per-example 5th percentile, per-layer curves,
  and raw R.

**Block sizes.** b = 128 carries H3, H4 and H5. b ∈ {16, 32, 64} is
**intrinsic only**: no kernel at those sizes is the same kernel, so there is no
cost or profitability there. It carries H7.

### 2.4 Extrinsic quality: task accuracy

- **Outcome:** per-example `correct` (score = 100). Decode is greedy, prefill
  sparse, decode dense, with the decode kernel matched across arms.
- **Test:** paired against the **accuracy reference** on the same ids, with
  the exact paired bound in `attnbench/analysis/exact_noninferiority.py`.
  Margin 10 points, one-sided alpha 0.025.
- **The accuracy reference is `sdpa_flash` dense, fixed.** That keeps it
  identical to T4 and to every banked dense prediction. The timing baseline
  (§4.6) may be a different kernel. Kernel choice can change bf16 rounding
  and therefore greedy outputs, so the reference used for accuracy is not
  allowed to vary with the speed contest.

---

## 3. Hypotheses and pre-stated predictions

Each one states its prediction, its **input ratio**, the tolerance on that
ratio, how a cell inside the tolerance is scored, and its pass rule. That is
the lesson of H100 P2, which failed at input-ratio differences of 2.6% and
6.8% with no tolerance stated.

**Tolerances, from banked spread.** Four L4 sessions measured the same
components (`results/t4_{sparse,xattn}_{canary,pilot}_session_*/…/estimator_cost.parquet`):

- the medians varied by 0.1–3.6%, except the 0.5 ms mask build at 6.1%;
- the ratios used here varied less: mean-pool/dense 3.2%; block-sparse/dense
  1.4% at 16384 and 1.6% at 32768.

So the **kernel-level tolerance is τ_k = 0.05**. The **end-to-end tolerance is
τ_e = 0.07**, which covers P2's 6.8% miss.

**What was visible when these predictions were written (stated 2026-10-03).**
The A100 crossover estimates were in view while H2 and H5 were drafted and
amended:

- the per-layer A100 kernels at (12, 2) (`results/s7_sweep_hl122`):
  cuDNN 1.165 ms against flash 1.570 at 8192, fa2 4.167 against 4.434 at
  16384, and block-sparse at each sparsity;
- the banked A100 end-to-end speedups against `sdpa_flash`
  (`results/s8_vec_endtoend`, `results/s12_a100_vec_endtoend`), including
  1.48× and 1.67× at 32768;
- the baseline-strength estimates in `claims.md`: at 8192 against cuDNN,
  0.962–0.995×, below 1 at every sparsity; at 16384 against fa2, 1.072–1.260×.

H2c (no profitable point at 8192), H2d (A100 32768 in [1.25, 1.80]) and H5a
(none at 16384) were written with these numbers in view, and quote them. They
are **priors anchored on visible data**, not blind predictions of the A100
crossover. What they test is whether the new arms (era-4 selector, per-head
MP, XA through `bsa_prefill`) behave as the banked arms predict. Each is
reported with this label. The H100 half of H2c is the same: the banked H100
figures at 8192 (0.923–1.036× against `sdpa_flash`, `limitations.md`) were
also visible. The predictions not anchored on banked crossover data are H1's
card ratios, H2a/H2b's per-cell model, H3, H4, H6, H7 and H8.

### H1: estimator cost moves across cards as the roofline predicts

- **Inputs:** D = T_dense^card1 / T_dense^card2, the timing baseline, and
  W = the bandwidth ratio card2/card1 from a 1 GiB device copy. Both are
  measured in each session. Under the null, c is independent of the card.
- **Prediction:** Q = (c^card2 / c^card1) / (D/W) ∈ [0.75, 1.33]. It applies to
  the mean-pool scorer and the era-4 selector, at 3 bands and 3 card pairs, so
  18 cells.
- **Anchor:**
  - c_MP(L4, 32768) = 0.0437 (pilot session);
  - D ≈ 3.3 (A100 4.434 against L4 14.689 ms per layer at 16384;
    `results/s7_sweep_hl122/sweep.parquet` and the pilot);
  - W ≈ 6.8 (specification);
  - so c_MP(A100, 32768) ≈ 0.021, band [0.016, 0.028].
- **Tolerance:** D and W are each ±τ_k, and the band on Q widens by
  ×/÷ 1.05².
- **Discrimination guard:** a card pair with |ln(D/W)| < 2 ln 1.05 is
  **indeterminate**. The expected D/W is ≈ 0.49 for L4–A100 and ≈ 1.34 for
  A100–H100.
- **Pass:** at least 15 of the determinate cells in band, and no more than 3
  indeterminate. Otherwise H1 is **unresolved**.
- **XAttention is excluded.** Its L4 rows are the torch fallback and never
  enter a cross-card ratio (§4.9). Its A100–H100 ratio is reported
  descriptively.

### H2: the component crossover model predicts end-to-end profitability

    S_pred = L · (T_dense − T_est − T_sel − T_bsa(d))  −  n_sync · t_sync

- n_sync is the arm's device-to-host syncs per forward. It is 0 except for
  official XA (§4.1).
- t_sync is the host's measured sync floor (§4.8).
- The input ratio is m.
- **Near-parity:** a cell with |1 − m| ≤ 0.07 is not scored in H2a or H2b and
  goes to replicates (§7.3).

**Predictions:**

- **H2a, sign:**
  - profitable versus unprofitable matches the sign of 1 − m in at least 90%
    of determinate cells;
  - no wrong sign where |1 − m| > 0.15;
  - an unresolved determinate cell counts as a miss.
- **H2b, magnitude:** |S_meas − S_pred| ≤ max(0.20·|S_pred|, f·T_dense^e2e) in
  at least 80% of determinate cells.
- **H2c, prior:** **no** deployable point is profitable at 8192 on the A100 or
  the H100.
  - On the A100 the 8192 timing baseline is cuDNN at 1.165 ms per layer
    (§4.6). Block-sparse is 1.544 / 1.230 / 1.133 ms at sparsity 0.5 / 0.75 /
    0.9 (`s7_sweep_hl122`), so the most any estimator could save is 0.03 ms
    per layer.
- **H2d, prior:** on the A100 at 32768, per-head mean-pool at d_nom ∈ {0.25,
  0.10} is profitable, with point speedups in [1.25, 1.80].
  - The vectorised CPU-builder arm measured 1.48× and 1.67× against
    `sdpa_flash` (`results/s12_a100_vec_endtoend`).
  - The faster fa2 baseline (§4.6) lowers this by an estimated 2–3%. Removing
    host exposure raises it.

### H3: under one selector, antidiagonal scores beat mean-pool scores

The T4 descriptive XAttention-over-mean-pool result crosses scorer with
selector:

- XAttention selects per head by mass;
- the T4 mean-pool arm uses one head-uniform top-k mask.

H3 holds the selector fixed at era 4.

- **Prediction:** at b = 128, the mean per-example R̃_XA8 − R̃_MP ≥ +0.05, with
  the one-sided 97.5% bootstrap lower bound above 0.
- **Cells:** d_nom ∈ {0.25, 0.10} × {1.5B, 7B} × 3 tasks × 2 bands = 24, on
  the evaluation split.
- **Pass:** at least 16 of 24.
- **Input ratio:** the realised-density ratio of the two arms, 1 ± 0.01,
  enforced by gate G7.
- **H3b:** the best deployable R̃ at d_nom = 0.10 is below 0.95 in at least
  2/3 of (model × task × band) cells.
- **Caveat:** the 1.5B direction was informed by T4. The 7B cells are a fresh
  replication, and H3 is also reported for 7B alone.

### H4: intrinsic recall predicts extrinsic accuracy

- **Population:** pooled over the 1.5B deployable extrinsic arms, stratified
  by (arm, task, band).
- **Groups:** per-example R̃ terciles, within each stratum.
- **Prediction:** the loss rate P(sparse wrong | dense right) in the lowest
  tercile is ≥ 2.0× that in the highest.
- **Test:** the one-sided Mantel-extension trend test, p < 0.025.
- **Tolerance:** examples within ±0.005 of a tercile boundary go to the lower
  tercile.
- **Power floor:** at least 30 dense-correct examples per tercile, else
  **underpowered**.

### H5: profitable and certified deployable points (1.5B, `qa_1`, A100, n = 300)

- **H5a:** the count at 16384 is 0. At 16384 profitability needs
  d ≲ 0.25 (kernel saving 1.06 / 2.17 / 2.64 ms per layer at sparsity
  0.5 / 0.75 / 0.9, against 4.434 for flash). The T4 deployable arms lost 7–24
  points at d ≈ 0.4–0.5.
- **H5b:** the count at 32768 is ≤ 1, and any certified point is an XA arm.
  The best T4 deployable cell was XA τ = 0.95 on `qa_1`/32768: −2.0 points at
  d = 0.33.
- **Input ratio:** profitability uses τ_e and f, and an unresolved point is
  not profitable.
- Any certified point is reported against every configuration tried across T4
  and this study.

### H6: positive control, XAttention on its authors' model and table

Model: Llama-3.1-8B. Configuration: §4.3, the authors' RULER configuration.

- **H6a:**
  - **Setup:** run that configuration with the shipped `llama_fuse_8` table on
    `text.json`.
  - **Prediction:** mean raw R ≥ 0.85, against the profiler's 0.90 target less
    0.05 tolerance.
  - **STOP at R < 0.70** (gate G6).
- **H6b:**
  - **Setup:** run the authors' released profiler on Llama over all 156 texts,
    since none exceeds Llama's limit (§4.9 G9 verifies this). Cap at 0.96 for
    the comparison only.
  - **Prediction:** Spearman with `llama_fuse_8` ≥ 0.80 over its 996 non-zero
    entries.
  - This tests the collaborator statement that the profiler is "compatible
    with the DP-based method" (§4.1). A failure is reported. It does not stop
    the study.

### H7: block granularity limits mean-pool, not antidiagonal scoring as much

`claims.md` already pre-states that the mean-pool gap at block 128 "is an upper
bound on the gap at their block size", because pooling dilution scales with
block size.

- **H7a:** (1 − R̃_MP) at b = 16 ≤ 0.75 × (1 − R̃_MP) at b = 128. The cells are
  d_nom ∈ {0.25, 0.10} × {1.5B, 7B} × 3 tasks × 2 bands = 24. Pass at least
  16 of 24.
- **H7b:** (R̃_XA8 − R̃_MP) at b = 16 < the same at b = 128, in at least 16 of
  24 cells. At b = 16 a stride-8 antidiagonal sees only 2 × 2 groups.
- **Input ratio:** realised density at b = 16 over b = 128, within 1 ± 0.05.
  Free blocks are a larger share at b = 128. A cell outside that is
  **indeterminate**, and more than 4 indeterminate makes H7 unresolved.

### H8: XAttention, in its authors' configuration, on Llama-3.1-8B accuracy

- **Prediction:** the native XA arm (§4.3, `llama_fuse_8`) is non-inferior to
  dense within 10 points in every primary Llama cell (n = 300) selected by the
  task rule (§4.9). Bands are 16384 and 32768, plus 65536 if run.
- **Input ratio:** none on the test; it is exact. The prediction is scored
  only if G1, G2 and G9 pass on Llama.
- If no primary cell is selected, H8 is **unscorable** and reported as such.

### 3.9 Before T4 runs: the T4 outcome and the replication

These are committed at lock, which comes before T4 Session A (§0).

**P-T4.** The amended T4 `authors` calibration certifies **0 of 2** primary
cells (`qa_1` at 16384 and 32768, n = 100, L4).

- The scalar phase's best bounds were −21.1 and −12.7.
- At n = 100 the exact bound certifies only when the discordant pairs are at
  most 4 each way (§7.5).

**The replication.** Same table digest, same settings, same 400 example ids.
The A100 runs the Triton path. T4 Session B runs on the L4 torch fallback.

- **R1, path agreement:** the per-example mean realised density of the A100
  run is within ±5% (relative) of the L4 run on at least 90% of the 400 ids.
  This is the input-ratio check for R3 and R4: if R1 fails, they are scored
  but labelled **path-divergent**.
- **R2, dense agreement across cards:** A100 `sdpa_flash` dense predictions
  are identical to T4's L4 dense predictions on at least 90% of the 400 ids.
  If R2 fails, R3 and R4 are labelled **card-confounded**.
- **R3, outcome agreement:**
  - per-example `correct` agrees between the two XA runs on at least 85% of
    ids in each primary and secondary cell;
  - |Δ accuracy| ≤ 5 points in each primary cell.
- **R4, claim agreement:** on the 100 shared `qa_1` ids per band, under T4's
  test, the A100 claim/no-claim matches T4's in both primary cells.

**Decoding in the replications (checked 2026-10-03).**

- **The T4 replication** runs in this harness on both sides: T4's L4 rows
  and the A100 rows use the same greedy argmax loop, stop rule and caps. So
  decoding matches by construction.
- **The XAttention authors' own RULER pipeline**, which H6 and H8 refer to,
  is also **greedy**: `config_models.sh` sets `TEMPERATURE="0.0"`, and
  `call_api.py` passes `do_sample=False`. It does not sample, so no
  replication here needs a sampling treatment (no seeds, no repeated draws).
- **Where it differs from this study:** chat-format prompts, a double BOS,
  no newline stop, and its own `tokens_to_generate`. These are recorded in
  §4.9 as confounds for the positive control and H8.

---

## 4. Estimators and implementation

### 4.1 Verified from source (2026-10-03)

Checked at `mit-han-lab/x-attention@e37988770b9d1bebd489eba011d615f35587ba08`
(`XATTN_COMMIT`, the tip of `main`, last pushed 2025-07-06).

| item | finding | where |
|---|---|---|
| **Licence** | **None.** There is no LICENSE file, and GitHub returns `license: null`. The only licences are those of vendored subtrees (`eval/RULER`, `eval/VLMEvalKit`, `eval/HunyuanVideo`). Default copyright applies. No licence issue exists on the tracker. Permission request: §14. | repo; `gh api` |
| **Block size** | 128. `Xattention_prefill` asserts it, and the profiler and the kernel hard-code it. | `Xattention.py`, `profile_threshold.py` |
| **Kernel** | `block_sparse_attn_func` (Block-Sparse-Attention, **BSD-3-Clause**). `head_mask_type` all 1, a **per-head** bool mask (1, H, n_qb, n_kb) on the device, `deterministic=True`, `is_causal`. | `Xattention_prefill` |
| **Estimator** | "Inverse" antidiagonal at stride s: softmax of the s-strided Q·Kᵀ, summed into groups of b/s × b/s. | `xattn_estimate` |
| **Selection** | Per row and head, blocks are kept until the cumulative estimate reaches τ × the row total. Sink and diagonal are always kept. Computed in float64. | `utils.find_blocks_chunked` |
| **Host syncs** | Two `assert <tensor>.all()` per chunk. That is 2 per layer at 16384 (one chunk) and 8 per layer at 32768 (four chunks of 8192). **Not sync-free.** | same |
| **Triton vs torch** | Triton runs only if the device name contains "100" (A100 and H100; not the L4). The paths are numerically different: `exp2` with log2(e) folded in, −1e6 against −inf, bf16 block sums. T4's "the fallback selects the same blocks" is untested. Gate G2 tests it. | `kernels.py` |
| **Calibration as released** | `profile_threshold.py`: <br>1. exact fp32 block mass per row; <br>2. the fewest blocks covering **0.90** of the exact mass; <br>3. the smallest estimated value among them; <br>4. per head, (estimated mass ≥ that value, summed over rows) / (total); <br>5. **max over texts**. <br>It calls the estimate with `use_triton=True`, so it follows the card's path. | `profile_threshold.py` |
| **Calibration as published** | The paper's thresholds come from a **dynamic-programming method (Sec. 2.3) that was never released**. Issue #13 (2025-06-05) asked for it. A repository collaborator (`xrorrim`, 2025-06-22, the day the profiler landed) answered that DP is expensive and that `profile_threshold.py` "produces results that are compatible with the DP-based method". Issue #19 (2025-07-13, open) repeats the question. | GitHub issues #13, #19 |
| **Shipped tables** | `llama_threshold.py` (2025-03-20) predates the profiler (2025-06-22). In `llama_fuse_8`, 114 of 1,024 entries are exactly 0.96 (a cap) and 28 are exactly 0, neither of which the profiler produces. They are consistent with DP output plus post-processing. That cannot be confirmed from source (§14 asks). | `git log -- xattn/threshold/` |
| **Authors' configurations** | **LongBench** (`eval/LongBench/pred.py`): stride 8, `norm=1`, `keep_sink=True`, `keep_recent=True`, a per-layer table. <br>**RULER** (`scripts/run_ruler.sh` → `xattn/src/load_llama.py`): every prefill layer runs `Xattention_prefill(q, k, v, stride, norm=1, threshold=table[layer_idx], use_triton=True)`, so `keep_sink` and `keep_recent` take their defaults (**False**), and the default chunk size applies. Strides 16, 8 and 4, each with its table. | as named |

**Licence policy.** x-attention is cloned and called at run time. It is never
vendored, never published in modified form, and never baked into an image. Any
public artefact can only point at the commit (F13).

**Other pins:**

- **MInference** `29ef1974fcb5a1440ccf92d7d7996900230fddcf` (**MIT**). The
  deployed `vertical_and_slash_kernel` in `minference/modules/minference_forward.py`:
  - `last_q = 64`;
  - an fp32 softmax of q_last·kᵀ/√d, causal;
  - the vertical score is the column sum, with the first 30 forced;
  - the slash score is the diagonal sum, with the last 100 forced.
  - Its kernel is `vertical_slash_sparse_attention` (block 64), not
    Block-Sparse-Attention.
- **FlashAttention-3:** `Dao-AILab/flash-attention`, `hopper/` (**BSD-3**),
  pinned at lock. `main` today is `e9515d5dee6ade134a33d6020d38d01ef0596996`.
  Requirements: H100/H800 and CUDA ≥ 12.3 (12.8 recommended). The image runs
  CUDA 12.9.
- FlexPrefill is Apache-2.0. It is not an arm.

### 4.2 Arms

Every arm produces per-head scores (H_q, n_qb, n_kb) in fp32 on the device,
then a selector, then a bool mask (1, H_q, n_qb, n_kb) on the device, then
`bsa_prefill` (§4.4).

| arm | scorer | source | selectors |
|---|---|---|---|
| **O** oracle | exact M_b, per head | attnbench, GPU-side, chunked | era 4. A reference ceiling, **never deployable**. |
| **MP** mean-pool | softmax(mean_b(q_h)·mean_b(k_g(h))ᵀ/√d + causal), **per query head** | `minference_meanpool_scores_on_device` without head averaging | era 4 |
| **XA**_s antidiagonal | `attn_sums` of official `xattn_estimate` | x-attention @ `e379887` | era 4, s ∈ {4, 8, 16} (s = 8 only at b < 128); **native** (§4.3) |
| **VS** vertical-slash-style | block mean over causal (q, k) ∈ B_i×B_j of v_h(k) + s_h(q−k), from the deployed MInference function | MIT, vendored with attribution (`attnbench/_vendor/minference/`) and NOTICE | era 4. **Not** MInference's vertical-slash method (F11). |
| **SL** sink + local window | no estimator: score(i, j) = −(i − j), so the nearest earlier blocks win | attnbench | era 4, so the budget is filled by the closest blocks (StreamingLLM-shaped). T_est = 0. The zero-cost anchor of the frontier. Runs none of XAttention's code (§0.3). |

At b < 128, XA uses `xattn_estimate(block_size=b, stride=8)` on the Triton path
where it runs. Otherwise it uses the torch path, recorded per row as
`xattn_path`, with gate G2 run at that b.

### 4.3 The native XAttention arms

**Qwen2.5-1.5B: replication of T4 `authors`, identical configuration.**

- **Table:** T4's committed `authors` table, loaded **by digest**. It is
  calibrated in T4 Session A on the A100 (§12 A4), by the rule below.
- **Settings:** stride 8, `norm=1`, `select_mode="inverse"`,
  `keep_sink=True`, `keep_recent=True`, the official chunk size, KV heads
  repeated, Block-Sparse-Attention through `bsa_prefill`, dense decode.
- **Examples and test:** the same 400 example ids and the same dense accuracy
  reference (`sdpa_flash`). The 200 extra `qa_1` ids per band (§5) are an
  **extension**, reported separately from the replication.
- **What differs is the card and the path.** T4 runs on the L4 torch
  fallback; this runs on the A100 Triton path. R1–R4 test exactly that
  difference.

**Qwen2.5-7B (intrinsic only).** The same rule, calibrated in this study's
run C on the A100.

**Llama-3.1-8B: the authors' RULER configuration, verbatim.**

- shipped `llama_fuse_8`;
- stride 8, `norm=1`, `use_triton=True`;
- `keep_sink=False`, `keep_recent=False`;
- default chunk size;
- every prefill layer;
- decoding in their pipeline: greedy (`TEMPERATURE="0.0"` →
  `do_sample=False`), no stop words, chat-format prompts with a double BOS
  (§4.9). This study keeps greedy and the kernel settings above. It changes
  the prompt format for accuracy and the BOS count everywhere, both stated as
  confounds (§4.9).

This is a deliberate difference from the Qwen arm, whose settings follow T4.
Both are authors' configurations, one per benchmark.

**The calibration rule (identical in T4 A2–A3, binding on every table).**

1. **Source:** `text.json` (156 prompts), with the Llama-3 chat markers
   stripped, for the Qwen tables (C, T4). **For Llama** (H6a, H6b) the chat
   format is kept and only the leading literal `<|begin_of_text|>` is
   removed, so the input has one BOS (§4.9). Under the Llama limit (131,072)
   all 156 are used: the maximum is 65,314 tokens.
2. **Over-length texts are excluded, not truncated.**
   - A text is used if and only if its token count under the target model's
     tokenizer is ≤ that model's `max_position_embeddings`, read at run time
     from the pinned revision's `config.json`.
   - For both Qwen models that is 32,768, which leaves **121 of 156**. The
     Qwen token counts are: median 16,025, 10th percentile 3,881, 90th
     percentile 65,484, maximum 91,574; 35 exceed the limit.
   - **Why not truncate:** in these multi-document QA prompts the question is
     at the end. Keeping the tail drops documents, keeping the head drops the
     question, and either changes the attention structure being profiled.
   - The excluded indices and their counts are written into the table file.
3. **Threshold statistic:**
   - **the maximum over the used texts** of the profiler's per-(layer, head)
     value, which is the released profiler's `final_threshold`, verbatim;
   - **no cap** is applied;
   - recorded descriptively and never used: the 90th-percentile table, the
     number of entries where max − p90 > 0.05, and for each entry the text
     that sets the max (to detect domination by one text).
4. **Path:** the profiler runs on an A100, so its estimate is the Triton path.
   The table records `xattn_path="triton"`, the source sha256, the commits,
   and the excluded indices.

**Stated deviation.** This is the authors' *released* procedure. The paper's
DP procedure is unreleased (§4.1), so no table here is the paper's. H6b
measures how far the released procedure is from the shipped tables.

**`ruler_cal` tables** (descriptive): from the calibration split (§5), under
the same rule. They never claim.

### 4.4 Two kernel call sites, proved equivalent

The plan to merge the call sites is **dropped** (2026-10-03).
`attnbench/backends/xattention.py` stays as it is, under the authors'
approval (§0.1). Any change to it is a §0.2-gated action.

- **Every non-XAttention arm** calls one new function,
  `attnbench.kernels.bsa_prefill(q, k, v, mask)`. It takes q, k and v with KV
  already repeated, and a contiguous bool mask (1, H_q, n_qb, n_kb) on q's
  device.
- **Its keyword arguments copy those `backends/xattention.py` passes:**
  `head_mask_type = ones(H_q)`, `streaming_info=None`, `is_causal=True`,
  `deterministic=True`, `exact_streaming=False`.
- **`backends/block_sparse.py`'s own call** (default `deterministic`, mask
  broadcast across heads) is left for the banked era-3 path. Frontier arms do
  not use it.
- **Gate G1b.** On the same q, k, v and per-head mask, `bsa_prefill` and the
  XA backend's kernel invocation are bitwise equal, at 8192 and 32768, on the
  A100. That makes "same downstream kernel and mask format" a tested fact
  rather than one call site.

### 4.5 Era 4: registration

Era 4 is defined in §2.1. The per-head selector is **added beside** the
head-uniform path, which T4 and the oracle arms still use. So an era cannot be
read from the commit alone, as eras 1–3 are (`analysis/eras.py`).

**A row-level column.** From the commit that introduces it, every accuracy,
recall, component and end-to-end row carries `mask_selector`:

| `mask_selector` | era |
|---|---|
| `per_head` | 4 |
| `head_uniform` | 3, by the existing commit rule |
| `xattn_native` | **native**: the method's own selection, not an attnbench era |

A row at or after that commit with no `mask_selector` is **refused**. Rows
before it resolve by commit, as now.

This differs from the reasoning in `eras.py` against a mask-rule column. That
reasoning was about a half-populated field. This one is fully populated from
its introducing commit, and absence before it is unambiguous.

**To register, all in one commit:**

1. Add an era-4 row to the era table in `docs/limitations.md` ("Which mask
   era each banked accuracy file belongs to"). Its file list stays empty until
   rows are banked.
2. In `eras.py`, add `ERA_LABELS[4]` and `PER_HEAD_SELECTOR_COMMIT`, and
   resolve era by `mask_selector` first, then by commit.
3. Extend `tests/test_eras.py` to derive era 4 from the doc table joined to
   each parquet's `mask_selector`.
4. Add a **break-test**: a per-head row with the column stripped must be
   refused, not silently read as era 3.

**Consequences:**

- The comparison scripts refuse era 3 against era 4, and native against
  either, without `--cross-era`.
- T4's head-uniform arms against this study's per-head arms are therefore
  descriptive only.

### 4.6 Dense timing baseline: the fastest kernel that runs correctly

**Candidates.** For each (card, band):

- `sdpa_flash`, `sdpa_cudnn`, `fa2`;
- on the H100 only, FA3 (`flash_attn_3`).

**"Runs correctly" means three things:**

1. it completes at the model's geometry (GQA, head dim 128, bf16, causal);
2. it passes the existing numerics gate against an fp32 reference;
3. a device-health check passes afterwards: a fixed matmul checksum, and no
   new Xid in the serial console.

**The rule.** The baseline is the fastest correct candidate, from the
session's component microbenchmark. It is chosen **before** any end-to-end
timing and recorded on every row as `dense_baseline`. It also carries T_dense
in H1 and H2.

**Switch points, reported per card:**

- the microbenchmark covers {4096, 8192, 12288, 16384, 24576, 32768}, plus
  65536 for Llama;
- each switch is attributed either to **speed** or to **exclusion**.
- From banked data, the A100 at (12,2) (`s7_sweep_hl122`):

  | band | cuDNN | fa2 | `sdpa_flash` | fastest |
  |---:|---:|---:|---:|---|
  | 4096 | 0.343 | 0.383 | 0.509 | cuDNN |
  | 8192 | 1.165 | 1.459 | 1.570 | cuDNN |
  | 16384 | excluded | 4.167 | 4.434 | fa2 |

  The cuDNN→fa2 switch between 8192 and 16384 is caused by **exclusion**,
  not speed. Every earlier end-to-end figure in this project used
  `sdpa_flash`, which is the slowest of the three.

**Run D: a same-session dense-kernel comparison at the real geometry.**
`docs/claims.md` now carries a baseline-strength caveat (2026-10-03): every
banked A100 speedup is against `sdpa_flash`. Its end-to-end corrections are
**estimates**, made by subtracting the per-layer kernel difference ×28. Run D
measures them. It runs in I1 (A100) and I3 (H100), before candidate timing:

- **What:** end-to-end dense prefill of Qwen2.5-1.5B at 8192, 16384 and
  32768.
- **Arms:** every correct candidate kernel (`sdpa_flash`, `fa2`, cuDNN at
  ≤ 8192 or where the H100 probe passes, FA3 on the H100), interleaved per
  rep. 20 reps after 3 warm-up calls, 2 selection-split prompts, A/A on
  `sdpa_flash`.
- **Alongside:** the vectorised block-sparse arm at 0.5, 0.75 and 0.9 with
  the `results/s12_*` builder, so the banked speedups can be recomputed
  against each kernel in one session.
- **Output:** `results/frontier_dense_kernels_<card>_<date>/`. When banked,
  its figures **replace** the caveat's estimates in `claims.md`, as a dated
  update with `tests/test_a100_baseline_caveat.py` re-pointed at the new
  parquet.
- **Cost:** about 5 arms × 23 forwards × 2 prompts × 3 bands, or 4–7 min on
  the A100 and 2–4 min on the H100, inside I1 and I3's brackets.

**The cuDNN fault at 16384 and above:**

- **A100 and L4.** Excluded above 8192 and recorded as
  `status="illegal_memory_access"`. The evidence (`limitations.md`, "cuDNN
  fused attention faults the device above 8192"): Xid 31, an MMU fault, on
  four machines across sm_89 and sm_80, all at driver 580.173.02 and torch
  2.9.1+cu129. *(Corrected 2026-10-03: the banked serial logs hold three
  real Xid 31 events, two on L4 and one on A100. Later rows that record the
  fault are guard-written; see `limitations.md`.)* They are not re-probed
  while driver and torch are unchanged.
  If either changes, they are re-probed as for the H100.
- **H100 (sm_90). A record exists, and it is not an observation. Checked
  2026-10-03.**
  - **What the files say.** Every H100 probe file records `sdpa_cudnn` at
    16384 and 32768 as `actual="illegal_memory_access"`, with
    `claim_mismatch=False`: 168 rows in each of
    `results/h100_20260916_stage0/results/probe/probe.parquet`,
    `results/h100_20260916_stage0/stage0_h100_merged.parquet`,
    `results/h100_20260916_stage01/results/probe/probe.parquet` and
    `results/h100_20260916_stage2/results/probe/probe.parquet`. There are 84
    at 32768 in `…/stage0/results/probe32k/sdpa_cudnn/probe.parquet`, and 84
    at 16384 in `results/h100_20260916_stage0_partial/probe.parquet`.
  - **What actually happened.** At those sessions' commits (`92c253f`,
    `795823a`, `9c05dfd`, `de0e8ed`), `gates.py` returns that status
    **before** `run_once` whenever the claim reason starts with
    `KNOWN DEVICE FAULT: `, and every one of these rows carries that prefix
    and the sm_89 note. **cuDNN was never launched above 8192 on an H100.**
    The `actual` column holds the guard's prediction, not a measurement.
  - **Two documents say otherwise, and are wrong about these rows.**
    `docs/retry_session_runbook.md` (the "isolation" table) says cuDNN's 84
    faults "raise at the call, `probe()` catches them". The `--exclude-backends`
    help in `scripts/run_probe.py` says cuDNN "took the 32768 band down with
    it", which the same runbook then disproves ("excluding it changed
    nothing"). Both were corrected on 2026-10-03 (`3f8a884`), and each
    keeps its old text. *(This said "Both are flagged (§13). Neither is
    corrected here." until 2026-10-03, after the correction had been made.)*
  - **The sm_80 evidence is real.** `results/a100/serial_console.log:2686`
    logs Xid 31 (MMU fault) at 2026-09-05 13:03.
  - **The probe runs LAST** (amended 2026-10-03). It is the final phase of
    the H100 session I3, **after all of I3's results are synced** to GCS.
    - It runs in an isolated subprocess with `CUDA_LAUNCH_BLOCKING=1`.
    - It **bypasses the guard explicitly** (`--launch-known-fault
      sdpa_cudnn`, refused anywhere but an isolated process), because the
      guard writes its status without launching.
    - It runs cuDNN at 16384 and 32768 at the model's geometry, records
      `launched=True`, and is followed by the health check.
    - A fault can then cost nothing already measured.
  - **So cuDNN is never a candidate in an H100 session above 8192.** The
    H100 timing baseline above 8192 is chosen in I3 from the other
    candidates, and stays fixed for every later H100 session. Changing it
    between sessions would break comparability.
  - **If the probe passes:** cuDNN's correctness and speed are reported.
    Later H100 sessions (the replicates) add it to run D as a **descriptive**
    dense arm. If it is faster, every H100 profitability claim above 8192 is
    stated as an upper bound against it.
  - **If it faults:** it is recorded as observed on the H100. The scope then
    becomes "observed on L4, A100 and H100".
  - **If the health check fails:** the session is already over and its
    results are synced. The fault is recorded, and nothing is relaunched for
    the probe.

**FA3 on the H100: included.**

- **Why:** Block-Sparse-Attention is FA2-derived. Against an FA2-class dense
  kernel on Hopper, sparse savings would be overstated.
- **How:** a wheel is built from `hopper/` at the pinned commit, against the
  image's torch 2.9.1+cu129 and CUDA 12.9, on a CPU-only VM, so no GPU time is
  spent on the compile. It is installed on the H100 and must pass "runs
  correctly".
- **If the build or a check fails:** H100 rows carry
  `dense_set_incomplete="fa3:<reason>"`. Every H100 profitability claim then
  reads "against the fastest correct dense kernel available, FA3 excluded
  (reason)", and is stated as an upper bound.
- **FlashAttention-4** (CuTeDSL, `pip install flash-attn-4`) is not a
  candidate. Its README recommends CUDA 13, and the image runs 12.9. Recorded
  as limitation F2.

**The accuracy reference stays `sdpa_flash`** (§2.4).

### 4.7 GPU-side and synchronisation

- **On the device:** every scorer, selector and mask build, with masks never
  leaving the device. The oracle's exact mass is GPU-side and chunked; today's
  CPU-resident oracle path is not used.
- **Sync check:** each arm's forward runs once under
  `torch.cuda.set_sync_debug_mode("error")`. MP, VS, O and the era-4 selector
  must raise nothing.
- **Official XA** syncs by construction and runs **as shipped**. `n_sync` is
  recorded and enters H2's model (F9).

### 4.8 Host provenance on every row: CPU model and launch floor

**Per row, every output** (component, end-to-end, recall, accuracy,
calibration):

- `cpu_model` and `cpu_count`, which `provenance.py` stamps already;
- `host`;
- the live `cpuPlatform` from the metadata server, as in the s12 sessions.

**Per-host launch floor,** measured at session start and end on that host and
stamped on every row from the start values:

| field | measurement |
|---|---|
| `launch_floor_us` | median per-launch host time over 10,000 back-to-back empty kernel launches |
| `sync_floor_us` | median of 2,000 round trips: a trivial kernel launch, then device synchronise, on the host clock |
| `h2d_floor_us` | a 1-byte host-to-device copy under double sync, the S11 floor (`floor_p50_us`) |

- **Drift:** start and end values both go in the session file. More than 20%
  drift between them flags the session in the report, but does not stop it.
- **Use:**
  - `sync_floor_us` is t_sync in H2;
  - all three are reported beside every cross-card comparison;
  - for H1, a host whose `launch_floor_us` differs by more than 25% from the
    other card's host is named in that cell's report.

### 4.9 Cards, models, bands, XA path labels, context limits

**Cards:**

- **A100** `a2-ultragpu-1g`: primary; the official Triton path; recall,
  calibration and all extrinsic runs.
- **H100** `a3-highgpu-1g`: the generation axis.
- **L4:** components only.

**XA path labels.**

- Every XA row carries `xattn_path ∈ {triton, torch_fallback}`.
- L4 XA rows are always `torch_fallback`. They are reported in their own
  table titled "XAttention, torch fallback (official code's behaviour on this
  card, not its official path)".
- They are **never on the generational curve**. H1, the cross-card frontier
  plots and every cross-card ratio refuse `torch_fallback` rows. A
  separately-labelled descriptive table needs `--allow-fallback`.

**Models and pinned revisions:**

| repo id | revision | `max_position_embeddings` | `rope_scaling` | how verified |
|---|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 32768 | none | local HF cache |
| `Qwen/Qwen2.5-7B-Instruct` | `a09a35458c702b33eeacc393d103063234e8bc28` | 32768 | none | HF |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249a16976918f6564b8830bc894c89659` | 131072 | `llama3`: factor 8.0, low 1.0, high 4.0, original 8192; `rope_theta` 500000.0 | **G9, 2026-10-03:** read by the researcher locally with their own token. `config.json` sha256 `29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e`, taken from the researcher's local HF cache, whose only snapshot is this revision. *(Until 2026-10-03 this row read "not verified: gated, 401 without a token".)* |

**G9 result, 2026-10-03 (lock gate L5): passed for the position limit
only.** The researcher read `config.json` at the pinned revision locally,
with their own token:

- `max_position_embeddings` 131072;
- `rope_scaling` {`rope_type` llama3, `factor` 8.0, `low_freq_factor` 1.0,
  `high_freq_factor` 4.0, `original_max_position_embeddings` 8192};
- `rope_theta` 500000.0;
- 32 layers, 32 attention heads, 8 KV heads, `hidden_size` 4096 (head_dim
  128), vocab 128256, bfloat16.

The geometry matches §2. Every Llama band below fits under 131,072 with any
generation cap. The pass covers the limit and nothing else. The checks below
were done on CPU on 2026-10-03, or are still open.

**Same model as XAttention's shipped thresholds (checked from source,
`e379887`).**

- **Repo id.** This study uses `meta-llama/Llama-3.1-8B-Instruct`. The
  authors' RULER run (`scripts/run_ruler.sh`, model `llama3.1-8b-chat`)
  resolves in `eval/RULER/scripts/config_models.sh` to
  `${MODEL_DIR}/Llama-3.1-8B-Instruct`, a local directory. The profiler's
  default is `meta-llama/Llama-3.1-8B-Instruct`
  (`profile_threshold.py:185`). LongBench names `Meta-Llama-3.1-8B-Instruct`,
  the repository's earlier name.
- **Revision.** Not recorded anywhere in their code (a local path), so the
  tables were calibrated on this model at an **unknown revision**.
- **Table shape.** `llama_fuse_4`, `llama_fuse_8` and `llama_fuse_16` are
  each 32 × 32. `load_llama.py` repeats K and V up to 32 heads before
  `Xattention_prefill`, which asserts `num_q_head == num_kv_head`, and passes
  `threshold[layer_idx]`. `find_blocks_chunked` (`xattn/src/utils.py:91-93`)
  broadcasts that 32-vector over the head axis. So the table is **32 layers ×
  32 query heads**, matching G9's 32 layers and 32 attention heads.

**RoPE `llama3` in the pinned transformers.**

- The GPU images pin transformers 4.46.0. It implements `llama3`
  (`modeling_rope_utils.py:310`, `_compute_llama3_parameters`, registered
  at line 362), as does the workstation venv's 5.18.0.
- **The logits gate was re-run** with llama3 scaling that triggers:
  `tests/test_llama3_rope_gate.py`, a tiny random Llama with the G9 rope
  values and `original_max_position_embeddings` scaled to 16, run to 64
  positions. Wrapped logits equal unwrapped logits within 1e-5 on both
  4.46.0 and 5.18.0. The installed inverse frequencies equal Meta's formula,
  implemented independently in the test, for the real G9 geometry too. A
  wrapper fed unscaled RoPE fails the gate (break-tested, watched red).
- **A trap, found doing this.** transformers 4.46.0 **silently ignores** a
  config key it does not read: given `rope_parameters` (the newer spelling)
  instead of `rope_scaling`, it builds default RoPE with θ = 10000 and no
  error. The real `config.json` uses `rope_scaling`, which both versions
  read. Step 4 below guards against this on the instance.

**Gate G9, on the instance, before any Llama row:**

1. Read `config.json` at the pinned revision with the project's token.
   Assert its sha256 equals the one in the table above, so the instance
   loads the file the researcher checked.
2. Assert `max_position_embeddings` ≥ every Llama band's budget +
   `stopping.token_cap(task, "meta-llama/Llama-3.1-8B-Instruct")`.
3. Assert that `text.json`'s maximum Llama token count is within the limit,
   for H6b. **Done locally 2026-10-03** with the pinned tokenizer, offline:
   maximum 65,314 with one BOS (65,315 as the authors encode it), against
   131,072. The instance repeats the count. (The Qwen count is 91,574.)
4. After loading the model, assert `model.model.rotary_emb.rope_type ==
   "llama3"` and that its `inv_freq` equals Meta's formula (the helper in
   `tests/test_llama3_rope_gate.py`). STOP otherwise.
5. Assert the stop set is {128001, 128008, 128009} and the newline and
   whitespace sets match the digests in the stop-rule block below. STOP
   otherwise.
6. Determinism: the dense arm decodes two selection-split prompts twice
   each. Anything but byte-identical output is a STOP.

If an assertion fails for one band, that band is **dropped for validity**
(not under §8.4).

**Bands:**

- Qwen: 8192, 16384, 32768 (the 32K band capped, below).
- Llama extrinsic: 16384, 32768c and 65536c (amended 2026-10-03):
  - **32768c:** budget 32,768 − the Llama cap for the task, the same rule as
    Qwen's (below). Llama's limit does not require it. It keeps
    prompt + generation ≤ 32,768, so the 32K band means the same number of
    positions on both models.
  - **65536c:** budget 65,536 − the Llama cap, so prompt + generation ≤
    65,536, below the 131,072 limit.
  - **16384:** uncapped, as for Qwen. 16,384 plus any cap is far below the
    limit.
  - **No 128K band is planned.** One would need its own cap at 131,072. At
    about 4× the 65536 band's cost per row (§8.2), it does not fit under
    §8.3, and the 65536 band is already cut first.
- **Llama sizing: per example, every task.** Every Llama generation calls
  `ruler.generate_examples(..., per_example_fit=True)` (added 2026-10-03,
  `tests/test_per_example_fit_flag.py`, break-tested). That sizes every
  example on its own and raises if one still lands above budget. The five
  candidate tasks are already in `_PER_EXAMPLE_FIT`, so the flag changes none
  of their prompts. It closes the gap for any other task. G11 holds Llama to
  it as well.
- **Llama decoding, stop rule and prompt encoding (lock gate L7, amended
  2026-10-03, sixth draft).**
  - **`generation_config.json`** at the pinned revision, as reported by the
    researcher and matching the copy in their local cache: `bos_token_id`
    128000; `eos_token_id` [128001, 128008, 128009]; `do_sample` true,
    `temperature` 0.6, `top_p` 0.9; `transformers_version` 4.42.3. The
    researcher also confirmed locally that 128000 is `<|begin_of_text|>`,
    128001 `<|end_of_text|>`, 128008 `<|eom_id|>` and 128009 `<|eot_id|>`;
    that `tokenizer.eos_token_id` is 128009 alone; and that the tokenizer
    prepends BOS by default ("hello" → [128000, 15339]).
  - **Greedy, explicitly.** The decode loop is argmax and never reads
    `generation_config`. Even so, `SwappableAttentionModel` **forces** greedy
    for this model id when it wraps the model (`do_sample` False;
    `temperature`, `top_p` and `top_k` None), and **asserts** it at every
    `generate` call. Every Llama arm goes through that wrapper. `generate`
    takes no sampling arguments.
    `tests/test_llama_decoding_and_prompts.py` fails if `do_sample`,
    `temperature` or `top_p` reaches a Llama arm. It also checks
    determinism: the same prompt twice gives byte-identical ids, stop and
    text. **On the instance:** before XL0, the dense arm decodes two
    selection-split prompts twice each, and anything but byte-identical
    output is a STOP.
  - **Stop set: all three ids.** `StopTokens.from_tokenizer(..., model_id=)`
    refuses any Llama stop set but {128001, 128008, 128009}. In particular it
    refuses the tokenizer's 128009 alone. The newline stop keeps its
    non-whitespace arming rule.
  - **Firing id recorded.** Every row carries `stop_token_id`, the id that
    ended an `eos` or `newline` stop, beside `stop_reason`. The category
    column keeps its three values, so banked rows and analyses read the same.
  - **Newline and whitespace sets** are rebuilt from the Llama vocab
    (128,256 entries). 2,255 newline ids (sha256 of the sorted list, first 16
    hex: `dc24e2c3056c70b4`) and 530 whitespace-only ids (`445dc415bf20b800`).
    No special id is in the newline set. They are **identical on
    transformers 4.46.0 and 5.18.0**. The instance asserts the same digests.
  - **Decoded text differs by version.** On 4.46.0 the tokenizer's
    `clean_up_tokenization_spaces=True` applies on decode ("x , y" → "x, y").
    5.18.0 ignores it for BPE. Scoring is substring match on answers that
    contain no space before punctuation, so the score is not expected to
    move, but `predicted` text is version-dependent. Llama rows are
    generated and decoded on the image's 4.46.0 only.
  - **Caps, measured 2026-10-03** offline (`HF_HUB_OFFLINE=1`, the cached
    pinned tokenizer, no token) by
    `scripts/measure_answer_lengths.py --model meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f…`,
    200 examples per task, same rule as Qwen (cap = 2 × max). Recorded in
    `stopping.LLAMA_31_8B_TASK_TOKEN_CAPS` and pinned to this table by
    `tests/test_stopping.py`:

    | task | n | min | median | p95 | max | cap | model |
    |---|---:|---:|---:|---:|---:|---:|---|
    | `niah_single` | 200 | 3 | 3.0 | 3 | 3 | **6** | Llama |
    | `niah_multikey` | 200 | 18 | 23.0 | 26 | 28 | **56** | Llama |
    | `vt` | 200 | 12 | 16.0 | 18 | 20 | **40** | Llama |
    | `niah_multikey_1` | 200 | 3 | 3.0 | 3 | 3 | **6** | Llama |
    | `niah_multivalue` | 200 | 15 | 15.0 | 15 | 15 | **30** | Llama |
    | `niah_multiquery` | 200 | 15 | 15.0 | 15 | 15 | **30** | Llama |
    | `qa_1` | 200 | 1 | 3.0 | 10 | 21 | **42** | Llama |
    | `qa_2` | 200 | 1 | 3.0 | 8 | 23 | **46** | Llama |

    Llama groups digits in threes, so a 7-digit answer is 3 tokens (Qwen: 7),
    and the number tasks' caps are less than half of Qwen's. **Flag:** a cap
    of 6 leaves 3 tokens for anything before or after a 3-token number.
    XL0 reports the cap-hit rate per task, and a `niah_single` or
    `niah_multikey_1` cap-hit rate above 5% on dense is reported as a
    truncation confound. The cap is not changed after XL0.
  - **Budgets.** 32768c and 65536c Llama budgets are band − the cap above:
    `qa_1` 32,726 and 65,494; `niah_multivalue` and `niah_multiquery` 32,738
    and 65,506.
  - **Encoding: the sizer counts exactly what the model is fed.** Sizing
    (`ruler.generate_examples`) and generation (`generation.generate_one`)
    both encode through `accuracy/prompting.py`. It uses
    `add_special_tokens=True`, stated rather than defaulted, which prepends
    one BOS on Llama. `generate_one` refuses an example whose sized
    `context_length` differs from the number of ids it feeds.
    `tests/test_llama_decoding_and_prompts.py` checks the two are equal per
    example, and that a sizer counting with `add_special_tokens=False` (one
    short) is caught. Qwen's counts are unchanged, because its tokenizer
    adds no BOS.
  - **Double-BOS guard.** `prompting.prompt_ids` refuses ids that begin with
    two BOS ids.
  - **Block 0 holds the BOS on Llama.** Every Llama prompt's position 0 is
    `<|begin_of_text|>` (128000), so key block 0, the sink every era-3 and
    era-4 mask grants free (§2.1), contains it. On Qwen, position 0 is
    ordinary text.

**What the authors' pipeline does (checked from source at `e379887` and
with the pinned tokenizer, 2026-10-03).**

- **Decoding: greedy.** `eval/RULER/scripts/config_models.sh` sets
  `TEMPERATURE="0.0"  # greedy`. `pred/call_api.py` passes
  `do_sample=args.temperature > 0`, so False, with `repetition_penalty=1`,
  `top_k` 32 and `top_p` 1.0, which greedy ignores. Generation is
  `model.generate` with no stop words (`config_tasks.sh`). So it ends on
  `generation_config`'s three EOS ids or on `max_new_tokens`, the task's
  `tokens_to_generate`. **It does not sample, so the replication needs no
  sampling treatment.** Our decoding is also greedy. Our stop rule adds the
  newline stop and uses our own caps, and that difference is the confound
  stated below.
- **Prompt format: chat, hand-written.**
  - **Calibration** (`profile_threshold.py` on `text.json`): every prompt is
    `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n…<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n Answer:`.
  - **RULER evaluation** (`run_ruler.sh` → `config_models.sh`): template
    `meta-llama3`, the same user/assistant wrapping (`data/template.py`).
  - Neither is `tokenizer.apply_chat_template`, which would add a system
    header. Neither has a system turn.
- **Double BOS: yes, in both.** Both strings start with a literal
  `<|begin_of_text|>` and are tokenized with the default
  `add_special_tokens=True` (`profile_threshold.py:212`,
  `model_wrappers.py`, `tokenizer(prompts, …)`). With the pinned tokenizer,
  **156 of 156** `text.json` prompts and the RULER template encode to
  `[128000, 128000, 128006, …]`. The shipped `llama_fuse_*` tables'
  calibration pipeline is unreleased (§4.1). If it used the same pipeline,
  they were calibrated on double-BOS inputs.

**Pre-registered prompt format for this study, and the confounds it
leaves.**

- **H6a / H6b (positive control on `text.json`):** the authors' chat format
  exactly, with the leading literal `<|begin_of_text|>` removed. That gives
  **one BOS**, the tokenizer's. Only the BOS count differs from the authors'
  input. **Confound, stated:** their double BOS shifts every position by one
  and puts two BOS keys in block 0. H6a's recall and H6b's profiler
  reproduction are both measured on single-BOS inputs, and the confound is
  reported beside their verdicts. A double-BOS sensitivity arm (I2c on
  double-BOS inputs, about ₹117–262) is **not** in the plan. It needs the
  researcher's yes and a dated amendment before I2c runs.
- **H8 and every Llama accuracy arm:** this study's completion-style RULER
  prompts (no chat wrapping, as for Qwen), one BOS, greedy, the stop rule
  above. **Confound, stated:** the authors evaluated in chat format with no
  newline stop. So H8 tests XAttention's kernel configuration and shipped
  table on this study's prompt format. It does not reproduce the authors'
  RULER scores, and is never compared with them numerically.
  `calibrate_xattn_thresholds.py`'s marker stripping is for Qwen only, and
  is not applied to Llama.

**Positions past Qwen's limit at 32768.**

- **How the overrun arises.** `context_length` is the prefill length:
  `generation.py:109` tokenises the context with no chat template, the same
  `len(tokenizer(text).input_ids)` that sized it. The decode loop feeds
  generated token k at position P + k − 1 and never feeds the last one.
- **Banked rows, flagged 2026-10-03** by `scripts/flag_positions_over_limit.py`
  into `results/positions_over_limit/` (`summary.csv`, `flagged_rows.csv`,
  `copies.txt`). Across **31 distinct accuracy parquets** (47 paths):
  - **1,638 of 34,582 rows** have P + n_generated > 32,768;
  - **1,636** ran at least one forward at a position ≥ 32,768;
  - every flagged row is at the 32768 band, in `stage3_32768` (era 1) and
    every T4 file (probe, pilots, canaries).
- **Prefill itself past the limit.** **12 rows** in `stage3_32768` (`niah_single`
  examples 16, 35 and 40, each under 4 arms) have prompts of **32,769**
  tokens, so their prefill also crossed the limit. This contradicts
  `ruler.py`'s docstring ("never above `token_budget`"). The filler count is
  fitted on example 0 per budget, and a later example's needle can add a
  token.

**Qwen at 32K: the capped band "32768c" (pre-registered 2026-10-03).**

- **Budget.** Every new Qwen run at the long band uses a per-task budget of
  **32,768 − `stopping.token_cap(task)`**:

  | task | `token_cap` | budget |
  |---|---:|---:|
  | `qa_1` | 42 | 32,726 |
  | `niah_multivalue` | 62 | 32,706 |
  | `niah_multiquery` | 62 | 32,706 |

  So prompt + generation ≤ 32,768.
- **Enforced:** gate G11 checks `context_length` ≤ budget on every generated
  example. This study's tasks are all in `_PER_EXAMPLE_FIT`, so they are
  sized per example and cannot overshoot. The check holds them to it anyway.
- **Never pooled.** Rows at 32768c are never pooled, compared in a test, or
  plotted on one axis with banked 32768 rows. The band label differs on every
  row (`band="32768c"`).
- **Primary cells.** All of this study's own 32K tests (H2–H5, H7, the
  frontier, the n = 300 primary cell) run at 32768c, on fresh prompts:
  `qa_1` indices 0–299, regenerated at the capped budget.
- **The one exception: the T4 replication.** R1–R4 compare against T4's
  banked rows, so they need T4's exact prompts. They run XA native and the
  dense reference on **T4's 200 example ids at the original 32768 band**,
  and the results are compared only with T4's 32768 rows.
  - These rows inherit the overrun: `positions_over_limit` is recorded, and
    R3 and R4 are also reported on the subset where it is 0.
  - **Extra cost:** 2 arms × 200 rows, about ₹70–290. It is a ★ line (§8.2).
  - At 16384 there is no overrun, so the T4 prompts serve both the
    replication and the frontier.
- **On every row:** `positions_over_limit = max(0, prompt + generated − limit)`.
- **Llama** is capped at 32768c and 65536c (above). G9 confirms both are
  far inside the 131,072 limit.

**Llama task selection.** A dense probe, n = 5 per task and band, on the T4
candidates (`niah_multikey_1`, `niah_multivalue`, `niah_multiquery`, `qa_1`,
`qa_2`). It applies T4's rule unchanged: retain a (task, band) whose dense
score is in [40, 90].

- **Primary:** `qa_1` at each band where it is retained, n = 300. If `qa_1` is
  not retained at a band, there is **no substitution**: that band has no
  primary cell.
- **Secondary:** the other retained tasks, n = 50. Nothing is re-selected
  after the probe.

**Llama arms:**

- dense (`sdpa_flash` as the accuracy reference);
- XA native (§4.3);
- XA_8, MP and VS at d_nom = 0.25 under era 4.

---

## 5. Splits

| split | used for | source |
|---|---|---|
| **calibration** | `ruler_cal` tables, and T4's amended `ruler_heldout` (§12 A1) | RULER seed 1. `qa_1` question indices **2000–2047**. NIAH from seed 1. 8 per (task, band). |
| **selection** | extrinsic budgets (§7.2); end-to-end timing prompts; Llama b* (unused) | RULER seed 2. `qa_1` indices **1000–1031**. 32 per (task, band). |
| **evaluation** | every tested quantity | Seed 0. **`qa_1` indices 0–299 (n = 300)**. At 16384, 0–99 are T4's prompts. At 32K they are regenerated at the capped band 32768c (§4.9), and only the T4 replication uses T4's 32768 prompts. `niah_multivalue` and `niah_multiquery` n = 50, which are T4's. 7B: the first 30/15/15. Llama: the same indices at its bands. |

**Seeds are not enough.** In `attnbench/accuracy/ruler.py:207`, `_render`
passes `index=i`. For QA, the SQuAD question is chosen by the example's
**position**, not its seed. The seed changes only the distractors and their
order (`_vendor/ruler/qa.py:114-128`). `qa_1` draws from 5,928 questions, so
indices up to 2047 exist.

**Disjointness.** Enforced in code, fail-closed, over every pair of splits and
models:

1. `example_id`;
2. the context sha256;
3. QA question index **and** gold-document set;
4. NIAH (key, value) needle pairs.

It needs a question-index offset in `_render` / `generate_examples`.
**Break-test**, kept permanently: a split that shares one `qa_1` index with
evaluation, under another seed and context, must be refused.

**Firewall order:**

1. The tables are committed.
2. The selection-split recall writes `budget_selection.json`, which is
   committed.
3. Only then is evaluation recall analysed. Its script refuses without that
   file at `HEAD` and its digest.

**n = 300 applies only to cells that carry non-inferiority** (the primary
tier, `qa_1`). It comes from the exact bound
(`exact_noninferiority.paired_lower_bound`, margin 10, alpha 0.025):

| n | bound at b = c = 0 | largest equal discordance that still certifies |
|---:|---:|---|
| 100 | −4.3 | b = c = 4 (4% each way) |
| 300 | −1.5 | b = c = 50 (16.7% each way) |

- At n = 100, certification is attainable only for near-identical outputs. At
  n = 300 it is attainable at realistic churn.
- The old "ceiling" of 100 is a Stage-3 budget value in
  `configs/accuracy/stage3_grid.yaml`, not a data limit. A frontier grid
  config sets 300.
- The secondary tier stays at 50. The observational cell stays
  observational.

---

## 6. Gates and STOP conditions (code, not prose)

| gate | condition | on failure |
|---|---|---|
| **G1** | The XA native arm through `bsa_prefill` is bitwise equal to `Xattention_prefill` (with the matching `keep_*` settings) for Qwen 1.5B, Qwen 7B and Llama-3.1-8B, at 8192 and 32768, on the Triton path. | STOP |
| **G1b** | `bsa_prefill` is bitwise equal to the XA backend's own kernel call on the same inputs and per-head mask (§4.4). | STOP |
| **G2** | The same q and k through `xattn_estimate(use_triton=True)` and `(False)` on the A100, at every b used: masks disagree on ≤ 0.5% of causal blocks, and \|ΔR\| ≤ 0.005. | Not a stop. Labels T4's L4 XA rows as not the official selections (in `claims.md`); scores R1 as indeterminate. |
| **G3** | The era-4 selector on head-mean scores is bitwise equal to `importance_block_mask_device`. | STOP |
| **G4** | Each scorer matches its reference (MP bitwise before averaging; VS fp32 at rtol 1e-5; O rows sum to 1 ± 1e-4, at every b). | STOP |
| **G5** | The sync-debug check is clean for MP, VS, O and the selector, and XA's `n_sync` is recorded. | STOP |
| **G6** | H6a's R ≥ 0.70. | STOP before any Qwen XA interpretation |
| **G7** | The realised-density ratio between era-4 arms is 1 ± 0.01. | STOP |
| **G8** | Every row has `git_dirty=False`, one commit per analysis, `dense_baseline`, `mask_selector`, `cpu_model`, `cpu_count`, `launch_floor_us`, `sync_floor_us`, `h2d_floor_us`, `positions_over_limit`, `xattn_path` and the calibration digest (on XA rows). | the analysis refuses |
| **G9** | Llama `config.json` verified (§4.9). | that band or model is dropped for validity |
| **G10** | The dense baseline "runs correctly" (§4.6), including the device-health check after every probe. | the candidate is excluded; on a failed health check the session ends |
| **G11** | Every new Qwen example at the capped 32K band satisfies `context_length` ≤ 32,768 − `token_cap(task)`; every new Llama example at 32768c and 65536c satisfies `context_length` ≤ band − `token_cap(task, "meta-llama/Llama-3.1-8B-Instruct")` and was generated with `per_example_fit=True`; and every new example at any band satisfies `context_length` ≤ its budget. Checked on the generated prompts before any row is written (§4.9). | STOP |

**Canary (A100, before I1).** Two examples per (task, band), every arm, every
gate. It measures μ (exact-mass pass / dense prefill, at b = 128 and with all
four b) and the launch floors. It then re-projects §8.

---

## 7. Analysis plan

### 7.1 The frontier

For each (model, card, band):

- the deployable points at (c, R̃), with marker shape for the selector;
- the affordability curve c_max(d) = 1 − T_bsa(d)/T_dense;
- each point labelled profitable, unprofitable or unresolved;
- the Pareto set over (s, R̃).

It is **descriptive**: claims come only from §3. Every point is reported.
Torch-fallback XA appears only in its own L4 table.

### 7.2 Budget selection (selection split, 1.5B)

- **b\*_e** is the smallest d_nom in {0.50, 0.25, 0.10, 0.05} with mean
  R̃ ≥ 0.95. If none qualifies, it is 0.50.
- **Extrinsic arms per band:**
  - each e at 0.25 and at b\*_e (deduplicated), XA native, and O at 0.25,
    which is 5–8 arms;
  - plus SL at 0.25, the zero-cost anchor (§0.3), which is cuttable (§8.4).
- **Fixed sequence:** within an e, the larger d is tested first. XA native and
  O are single tests.

### 7.3 Replicate sessions for near-parity cells

- **Unit:** a session, meaning a fresh instance and host.
- **Near-parity:** |1 − m| ≤ 0.07, or |1 − s| ≤ 0.07.
- **Replicates:** **3 sessions**, so 4 in total. They run only the
  near-parity cells, interleaved with dense A/A per rep.
- **Why 3:** the banked cross-session SD of the ratios is 0.6–1.4%. With
  k = 4 the half-width is 1.59σ, so 1.0–2.2%.
- **Escalation:** if σ̂ > 2.5%, run up to 6 sessions in total, then stop.
  What is left is **unresolved**. Escalation is subject to the cost stop
  (§8.3).

### 7.4 Resolution floor

    f = max( 0.02,  f_AA,  t₀.₉₇₅,k−1 · σ̂_s / √k )

- f_AA = |mean(dense_a)/mean(dense_b) − 1| + 2·SE over reps.
- No sign is ever claimed inside f.

### 7.5 Statistics and multiplicity

- **Recall:** percentile bootstrap over examples.
- **End to end:** per-rep paired ratios (20 reps after 3 warm-up calls,
  interleaved, 2 selection-split prompts per band), with a t-interval across
  sessions.
- **Accuracy:** the exact paired bound.
- **Hypotheses:** each H, R and P-T4 is one pass/fail under its count rule.
  No pooling beyond what each states, and no post-hoc margin, alpha, budget,
  τ, floor or baseline.

### 7.6 What may be claimed

- **"Profitable":** only from end-to-end time, against the named
  `dense_baseline`.
- **"Non-inferior":** only from the exact test. "No accuracy loss" is never
  claimed.
- **XA claims** name the path, table and digest, the configuration (T4 /
  authors' RULER) and the selector.
- **VS** is "vertical-slash-style scoring under block selection".
- **The oracle** is never deployable.
- **Block sizes below 128** carry no cost or profitability claim.
- **Every failed or unscorable hypothesis** is reported beside what it tests.
- **A hypothesis not run because of the cost stop** is reported as "not run
  (cost stop)", never as passed or failed.

---

## 8. Sessions, cost, and the hard stop

### 8.1 Inputs

Rates (`docs/spend_ledger.md`): A100 Flex Start ₹284/h, H100 Flex Start
₹425/h, L4 ₹80/h. A CPU VM is **assumed** at ₹130/h (c2-standard-30, about
$1.5/h).

| input | value | status | source |
|---|---|---|---|
| dense prefill, 1.5B | A100 0.190 / 0.436 / 1.102 s; H100 0.079 / 0.189 / 0.506 s at 8K / 16K / 32K | measured | `results/s12_{a100,h100}_vec_endtoend` |
| dense row, 1.5B, L4 | 2.59 / 5.40 s at 16K / 32K | measured | T4 pilot parquets |
| dense row, 7B, A100, 16K | 2.73 s | measured | `results/s7_7b_16384` |
| μ = exact-mass pass / dense prefill, used in §8.2 | **6.19–9.08** (sixth draft; was 2–11.3) | **measured**: this harness's oracle pass on the A100, Qwen2.5-1.5B, min and max over the three bands below. The new GPU-side pass (§4.7) is measured at the canary. | `results/s12_a100_vec_endtoend/logs/s12_vec_e2e.log` |
| μ of this harness's oracle pass, A100, 1.5B | 7.90 / 6.19 / 9.08 at 8K / 16K / 32K (1.5 / 2.7 / 10.0 s, one cold call each) | measured | same |
| μ on the L4 | 7.2 at 16K, 11.1–11.3 at 32K | measured, L4 card only; no longer used for A100 lines | `docs/t4_sparse_pilot.md`, `writeup_input.md` |
| Llama-8B oracle pass per text | 8.2–14.4 s at 16K, 30.5–53.3 s at 32K, 112–212 s at 65,314 tokens (the longest `text.json` text in Llama tokens) | **derived** from the measured 1.5B pass above, scaled by config geometry (§8.2 note) | `scripts/derive_llama_oracle_cost.py` |
| block-size recall multiplier | 1.1–1.5 | **assumed** | the canary measures it |
| wall / Σ row latency | 1.25–1.69 | measured | phase logs |
| session overhead | 8–15 min | measured range | ledger sessions |
| decode per token | 1.5B on A100 20–35 ms (upper measured on L4); Llama 25–45 ms | partly **assumed** | |
| 7B and Llama prefill, A100 | 7B 1.85–2.73 s (16K), 4.19–6.90 s (32K). Llama 16K 2.06 s, 32K 4.78 s and 65K 12.6 s at the lower end, upper = 1.6× lower | **assumed** from a FLOP split anchored to measured 1.5B times | |
| `text.json` | 156 texts. Qwen: 3.92M tokens, max 91,574, 121 ≤ 32K. **Llama (one BOS): 3.63M tokens, max 65,314, 130 ≤ 32K** | measured | tokenised locally; Llama offline with the pinned tokenizer, 2026-10-03 |

### 8.2 Bracket

The computation moves into `attnbench/analysis/frontier_prereg.py` before lock
(§11). ★ = never cut (§8.4).

**Re-bracketed 2026-10-03 (sixth draft) with the measured μ on every
oracle-pass line.** The fifth draft's figures are kept beside the new ones.

| run | card | min, fifth | min, now | ₹, fifth | ₹, now | what moved it |
|---|---|---:|---:|---:|---:|---|
| ★ FA3 wheel build | CPU | 60–150 | 60–150 | 130–325 | 130–325 | — |
| ★ C: XA calibration, Qwen 7B (121 texts) | A100 | 18–85 | 33–74 | 86–404 | 158–348 | μ (a) |
| ★ I1: 1.5B components, end to end at 3 bands, recall (192 selection + 800 evaluation, b 16–128) | A100 | 59–259 | 117–220 | 281–1,228 | 553–1,040 | μ (a) |
| ★ I2a: 7B recall, 60 per band, b 16–128 | A100 | 24–188 | 52–156 | 115–890 | 247–738 | μ (a) |
| I2b: 7B end to end at 32K | A100 | 14–36 | 14–36 | 64–170 | 64–170 | — |
| ★ I2c: Llama H6a, texts ≤ 32K | A100 | 22–49 | 25–55 | 103–233 | 117–262 | Llama token counts (b) |
| I2d: Llama H6a, texts > 32K | A100 | 62–137 | 47–104 | 295–649 | 224–491 | Llama token counts (b) |
| I2e: Llama H6b profiler reproduction | A100 | 81–176 | 69–149 | 384–834 | 327–706 | Llama token counts (b) |
| ★ I3: H100 cuDNN probe, components, end to end at 3 bands | H100 | 22–44 | 22–44 | 155–313 | 155–313 | — |
| I4: L4 components | L4 | 11–25 | 11–25 | 15–33 | 15–33 | — |
| ★ R: A100 replicates ×3 (0 if no near-parity) | A100 | 0–140 | 0–140 | 0–664 | 0–664 | — |
| ★ R: H100 replicates ×2 | H100 | 0–59 | 0–59 | 0–414 | 0–414 | — |
| R: H100 replicate #3 | H100 | 0–29 | 0–29 | 0–207 | 0–207 | — |
| ★ X0: 1.5B extrinsic canary | A100 | 13–37 | 13–37 | 62–177 | 62–177 | — |
| ★ X: 1.5B primary `qa_1`, n = 300, dense + 5–8 arms | A100 | 137–735 | 170–718 | 650–3,478 | 803–3,397 | μ (a) |
| ★ X: 1.5B secondary, 100 per band | A100 | 40–235 | 51–229 | 192–1,112 | 242–1,085 | μ (a) |
| ★ X-rep: T4 replication on T4's 32768 prompts, dense + XA native, 200 ids (§4.9) | A100 | 15–61 | 15–61 | 72–288 | 72–288 | — |
| XL0: Llama dense task probe | A100 | 22–50 | 22–50 | 106–238 | 106–238 | — |
| XL: Llama primary `qa_1`, n = 300, 16K + 32K, dense + 4 | A100 | 272–601 | 272–601 | 1,289–2,843 | 1,289–2,843 | — |
| XL: Llama secondary, 16K + 32K | A100 | 88–195 | 88–195 | 417–924 | 417–924 | — |
| XL: Llama 65536 band | A100 | 574–1,252 | 574–1,252 | 2,719–5,925 | 2,719–5,925 | — |
| **Never-cut core** | | **6.8–34.1 h** | **9.4–32.4 h** | **₹1,842–9,526** | **₹2,540–9,053** | |
| **Full plan** | | **25.7–75.7 h** | **27.7–73.0 h** | **₹7,133–21,347** | **₹7,704–20,589** | |

**Sources of the change.**

- **(a) μ, measured.** μ = 6.19–9.08 replaces 2.0 (assumed) – 11.3 (L4) on
  every oracle-pass line: C, I1, I2a and both X lines. It is the minimum and
  maximum over 8K / 16K / 32K of this harness's oracle scoring pass over
  dense prefill, **measured on the A100 for Qwen2.5-1.5B**: 1.5 / 2.7 /
  10.0 s against 0.190 / 0.436 / 1.102 s, giving 7.90 / 6.19 / 9.08
  (`results/s12_a100_vec_endtoend/logs/s12_vec_e2e.log`; one cold call per
  band). 11.3 was an L4 figure, from the L4's own pass, so it no longer bounds
  an A100 line. Lower ends rise and upper ends fall.
- **(b) Llama token counts, measured.** I2c–I2e now use `text.json` counted
  with the pinned Llama tokenizer, offline, with one BOS as this study feeds
  it: 3,628,025 tokens, maximum 65,314, and 130 of 156 texts ≤ 32,768. The
  Qwen proxy was 3,921,756, maximum 91,574, 121 ≤ 32K. The Llama pass itself
  is still the measured 1.5B pass scaled by config ratios (below).
  `scripts/derive_llama_oracle_cost.py` reproduces the three lines.
- **Still assumed:** the block-size recall multiplier (1.1–1.5), Llama
  prefill and decode for the XL lines, and 7B prefill (§8.1).
- **What μ still is not.** It is the **existing** oracle pass's cost. The
  study's GPU-side exact-mass pass (§4.7) is new code. The canary measures
  its μ (DP1), and `rebracket()` replaces these values with that
  measurement.

**The Llama oracle-pass lines, derivation (2026-10-03).** They start from
this harness's oracle pass on the A100, **measured** for Qwen2.5-1.5B (source
(a)). Two ratios from the configs scale it to Llama:

- **attention:** layers × query heads × head_dim, 3.048;
- **everything else:** non-embedding linear parameters, 6.979B against
  1.310B, 5.327.

The split between the two is not assumed. The lower bound puts the whole
pass at 3.048 and the upper at 5.327, and any split lies between them.

- **Interpolation:** a power law between the measured bands. Past 32K, the
  exponent is the 16K–32K one (1.89) or 2. Below 8K, the 8K point is scaled
  linearly or quadratically.
- **XA estimate pass:** up to one dense prefill more, which is at most
  1 / 6.19 of the oracle pass.
- **Token counts:** source (b).

*(In the fifth draft these lines used Qwen token counts, and the other
oracle lines kept μ = 2.0–11.3. Before that, all three Llama lines used a
FLOP-split prefill model × the assumed μ.)*

**Intrinsic-only and extrinsic, full plan** (X-rep excluded, as before):

- intrinsic: ₹1,992–5,714 (7.5–20.7 h), from ₹1,633–6,367 (6.2–23.0 h);
- extrinsic: ₹5,640–14,588 (19.9–51.4 h), from ₹5,440–14,700, of which
  Llama is ₹4,530–9,930 (unchanged).

**T4 calibrated, as amended,** is a separate pre-registration, not under this
cap:

- Session A on the A100: 15–48 min, ₹70–230;
- Session B on the L4: about 2.6 h, about ₹210.

**For scale:** the project's ledgered spend to 2026-10-01 is ₹8,409 over 38
sessions.

**Not in the session figures above; checked 2026-10-03 with read-only
`gcloud` listings.** Prices are **assumed** list prices at ₹88/$.

| item | measured size | cost |
|---|---|---|
| custom images `attnbench-env-v5-20260905`, `attnbench-env-v6-20260917` | 22.0 GiB archive each | about ₹194/month, billed whether or not a session runs |
| results bucket `gs://attnbench-results-research-507316` | 5.19 GiB, about 10 GiB after the study | about ₹18/month |
| 200 GB boot disk per session | (no disks exist now) | about ₹2.4/h of session: ₹23–78 over the core, ₹67–176 over the full plan |
| FA3 wheel build | — | already a ★ line above (₹130–325) |

**Share of the core that depends on XAttention arms.** About **30–31%**:
₹762–2,810 of the ₹2,540–9,053 core. That includes the X-rep line, which
is XA-only. *(Fifth draft: 30–33%, ₹612–2,876 of ₹1,842–9,526. Before that:
31%, ₹560–3,000 of ₹1,790–9,650.)*

| component | XA-dependent ₹ |
|---|---:|
| X0 + X, the XA share of the 1.5B extrinsic rows (2–3 of 5–8 arms) | 333–1,397 |
| X-rep, the T4 replication at 32768 (XA only) | 72–288 |
| C, the 7B calibration (XA only) | 158–348 |
| I2c, the Llama positive control (XA only) | 117–262 |
| replicates, assuming a third of near-parity cells are XA | 0–359 |
| I3's XA end-to-end arms | 47–94 |
| I1's XA end-to-end arms and components | 35–62 |
| the recall passes (dominated by the shared exact-mass pass) | ≈ 0 |

So the non-XA core is about ₹1,778–6,243.

**Running non-XA parts first is mostly possible, with two exceptions:**

- **The extrinsic dense reference is shared.** Splitting XA arms into later
  sessions adds a session overhead each.
- **The 1.5B XA-native replication is ★.** Deferring it changes nothing about
  what is reserved.

### 8.3 Hard cost stop

- **Cap:** **₹12,000** for this study: all sessions above plus the storage
  reserve, T4 calibrated excluded.
  - **Confirmed by the researcher on 2026-10-03**, in conversation, after it
    was stated back. It cannot be raised after lock.
  - It covers, at worst-case inputs, the never-cut core (₹9,053) plus the
    storage reserve (about ₹500) plus one rerun of the largest ★ session
    (₹1,698), which is about ₹11,250, leaving about ₹750.
  - Anything cuttable is paid for only from what the measured μ (DP1) frees
    up.
- **Ledger:** `results/frontier_spend.csv`. Each session's teardown appends
  one row from its `session_cost.txt`. If that file is missing, the row is
  rebuilt from the audit log, as `docs/spend_ledger.md` does. Spend is never
  restated in prose.
- **Reservation check, before every launch:**
  `scripts/frontier_budget_gate.py` must pass, refusing with `STOP` otherwise:

      spent + U(next session) + Σ U(every remaining ★ session) + S_res  ≤  ₹12,000

  - **S_res is the storage reserve:**

        S_res = ₹212 × (months remaining to the planned end, rounded up)
              + ₹2.4 × (upper-bound hours of every remaining session)

    The ₹212 is the two images plus the bucket. Storage is counted in
    `spent` monthly from the billing export.

  - **U** is the session's upper bound, re-projected after the canary with
    measured μ and the measured block-size multiplier.
  - **If it fails and the next session is cuttable:** cut it, and continue
    down §8.4.
  - **If it fails and the next session is ★:** cut the remaining cuttable
    items in §8.4 order until it passes.
  - **If no cuttable item remains and it still fails:** **the study stops**.
    Nothing more launches, what is banked is analysed, and every hypothesis
    left without inputs is reported as "not run (cost stop)".
- **In-session:** every launch sets `GCP_MAX_RUN` and the in-guest halt to
  ⌈1.25 × U⌉ minutes, with a hard delete 10 minutes later.
  - A session that hits its halt banks what it has.
  - It is relaunched only if the reservation check passes for the remainder.
- **Health-check relaunches** (§4.6) are ordinary sessions under the same
  check.
- **The cap is not raised under this pre-registration.** More spend needs a
  new, dated pre-registration covering only the sessions not yet run,
  committed before the next launch.
- **Nothing is cut because of a result.** Cuts follow only from the check, in
  the order below.

**What the cap covers (checked 2026-10-03):**

| item | covered? |
|---|---|
| **FA3** | **Yes.** A ★ line (₹130–325). |
| **Storage** | **Yes, from this draft,** through S_res. It was not covered before. |
| **Repeats: base replicates** (A100 ×3, H100 ×2) | **Yes.** ★ lines. |
| **Repeats: escalation to 6 sessions** (§7.3) | **No.** Up to ₹440 A100 + ₹620 H100. Now cut-order items 0a and 0b, so the reservation check funds or cuts them first. |
| **Repeats: health-check relaunches and failed-session reruns** (preemption, stock-out, crash) | **No.** No reserve exists for them. They are ordinary sessions under the check. They consume cuttable work first, then trigger the stop. |

**The arithmetic at worst-case inputs:**

| | ₹ |
|---|---:|
| core, including the T4-replication line at 32768 | 9,053 (fifth draft 9,526) |
| S_res, two months | about 500 (₹424 storage + ₹2.4/h × 32.4 h) |
| one rerun of the largest ★ session (one band of X primary, half of ₹3,397) | 1,698 (fifth draft 1,740) |
| **total** | **about 11,250** (fifth draft about 11,780) |

That is **inside the ₹12,000 cap**, with about ₹750 to spare (fifth draft
₹220, fourth ₹100). Source of the change: the measured A100 μ (§8.2, (a))
and the Llama token counts ((b)). So:

- a single failed session at worst-case μ does not stop the study;
- a second one does, unless DP1 measures μ below its upper bound;
- the H100 cuDNN probe now runs last, so it needs no relaunch (§4.6).

**Re-bracketing decision points.** These are mechanical, committed before the
next launch, and read timings only:

- **DP1, after the A100 canary and before I1.**
  - **Inputs:**
    - measured μ at b = 128 and with all four b;
    - the block-size multiplier;
    - per-forward end-to-end times per arm;
    - launch and sync floors;
    - session overhead.
  - **Recompute:** `frontier_prereg.rebracket()` recomputes U for every
    remaining session, S_res and the reservation check.
  - **Output:** `rebracket_dp1.json`: inputs, U per session, the funded set,
    the cuts applied in §8.4 order, and the verdict.
  - **Verdict:** **proceed** (the core and the funded cuttables fit),
    **proceed with cuts**, or **STOP**. STOP means the core alone fails the
    check, with no study session run. Raising the cap is then a new
    pre-registration (above).
  - **Leak guard:** the function reads only timing, μ, and cost columns.
    It **refuses** any parquet with `correct`, `predicted`, `R`, `R̃` or
    density columns, so no canary result can enter a funding decision.
- **DP2, after X0 and before X.** The same procedure, with measured per-row
  times for every extrinsic arm. Output: `rebracket_dp2.json`.
- **Between them,** every launch runs the ordinary reservation check, using
  the latest DP's U values.

### 8.4 Cut order (first cut first)

0a. Replicate escalation beyond 4 sessions on the A100 (§7.3).
0b. Replicate escalation beyond 3 sessions on the H100.
0c. SL in the 1.5B extrinsic run: about ₹90–500, one more arm.
1. XL: the Llama 65536 band.
2. XL: Llama secondary cells.
3. I2d: Llama H6a on texts > 32K.
4. Block sizes 16–64 for 7B. 1.5B keeps them, so H7 runs on 1.5B only.
5. XA strides 4 and 16, intrinsic.
6. R: H100 replicate #3.
7. I2b: 7B end to end.
8. I4: L4 components.
9. I2e: H6b, the profiler reproduction.
10. VS in the 1.5B extrinsic run.
11. XL0 + XL primary: Llama accuracy entirely. H8 becomes "not run (cost
    stop)".

**Never cut (★):**

- gates G1–G10;
- FA3;
- C;
- I1, including 1.5B block sizes;
- I2a at b = 128;
- I2c, because G6 depends on it;
- I3;
- R: A100 ×3 and H100 ×2;
- X0;
- X: 1.5B primary at n = 300 and secondary, including the replication arm.

Item 1 is the only item addressing ⚑6, and item 11 removes ⚑3's mitigation.
§10 names both.

### 8.5 Order

**This study:** FA3 build ∥ local code. Then C. Then the canary and I1. Then
I2a and I2c (I2b, I2d and I2e if funded). Then I3. Then I4. Then R. Then
budget selection, committed. Then X0. Then X. Then XL0. Then XL.

- That order spends ★ items first, so a stop starves only cuttable work.
- The quota is GPUS_ALL_REGIONS = 1, so sessions run one at a time.
- The standing `scripts/gcp_*` rules apply.

### 8.6 Permission gate and publication rule (binding; cannot be amended after any result is seen)

§0.2 is part of the plan, and both halves are binding.

- **The gate.** Every action that depends on the authors' permission is
  **NOT PERMITTED** until the researcher says yes explicitly, in
  conversation, to that exact action.
  - Claude Code asks each time, naming the action.
  - A yes is never inferred.
  - Each answer is logged with its date in
    `/Users/mehuldahiya/Desktop/research/xattention_permission_log.md`,
    outside the repository. The log never contains a quote of the email.
- **The publication rule.** No result produced with XAttention code leaves
  the project — native, calibrated, or any arm running their selector,
  estimator or thresholds — except under the §0.1 scope **and** a logged yes
  for that act.
- **Status 2026-10-03.** Written permission was received, and the
  researcher's summary (§0.1) covers using the code, reproducing results,
  publishing results and figures, and reproducing their kernel lines,
  including `backends/xattention.py`. **No gated action has been approved
  yet.** The log has no entries.
- **Never permitted here,** whatever the answer: vendoring or copying their
  repository, redistribution, and baking their code into an image.
- **When it can change.** Only by a dated amendment committed before any
  XAttention result exists in the project. T4's XAttention pilot results
  already exist, so this section **cannot be loosened** under this
  pre-registration.
- **What is already public.** `mehul24d/attnbench-research` is a public
  repository. Since 2026-10-02 (`fdeae62`) it has carried T4's XAttention
  results (`docs/t4_xattention_pilot.md`, `docs/claims.md`) and
  `attnbench/backends/xattention.py`.
  - The researcher reports that the authors were informed of the file and
    approved it.
  - Those pushes predate this gate. Any **further** push of files containing
    XAttention results or code lines is a gated action.

---

## 9. Fatal-flaw review: the plan, read as a hostile referee

**C** = controlled, **P** = partly controlled, **L** = stated limitation.

| # | objection | verdict | how |
|---|---|---|---|
| F1 | *"Your XAttention isn't XAttention."* | **P** | Official code, pinned. G1 bitwise. The Triton path. G2. The released profiler, on the Triton path. H6 and H8 on the authors' model and configuration. **Open:** the paper's DP calibration is unreleased, and the shipped tables carry post-processing not in source. Only the released procedure is reproduced, and H6b measures the gap. |
| F2 | *"Weak dense baseline."* | **P** | The fastest correct kernel per (card, band), switch points reported, FA3 included. Run D measures the dense kernels against each other in one session. `claims.md` carries a tested caveat for the banked A100 figures. The H100 cuDNN record was checked and is a guard prediction, not an observation, so it is probed with the guard bypassed. **Open:** FA4 is not a candidate (CUDA 13), and cuDNN on sm_80/sm_89 is not re-probed at the same driver and torch. |
| F3 | *"Kernel time is not system time."* | **C** | Profitable is defined end to end. |
| F4 | *"Recall is an unvalidated proxy."* | **P** | H4, at n = 300 on the primary cells. Underpowered is a stated outcome. |
| F5 | *"Teacher forcing hides compounding."* | **L** | Compounding appears only extrinsically. |
| F6 | *"Scorers compared under different selectors."* | **C** | Era 4 for all. G3. G7. Native arms only at matched density. |
| F7 | *"Tuned on test; hypotheses written after T4."* | **P** | Content-disjoint splits, the firewall, 7B replication. The 1.5B H3 direction is T4-informed. |
| F8 | *"Near-parity signs are noise."* | **C** | τ_e, replicates, the floor, "unresolved". |
| F9 | *"Host effects, not estimators."* | **P** | Everything on the device. The sync gate. CPU model and launch, sync and H2D floors on every row. The XA sync term in H2. **Open:** XA's syncs are kept as shipped. |
| F10 | *"Toy model, one family."* | **P** | Llama-3.1-8B accuracy (H8), 7B intrinsic. **Open:** no Qwen 7B accuracy, and Llama accuracy is cuttable (item 11). |
| F11 | *"Block 128 and a non-MInference VS."* | **P** | Intrinsic recall at b = 16–128 (H7). **Open:** cost and profitability exist only at 128, and VS is VS-style. |
| F12 | *"A big grid guarantees a hit."* | **C** | Fixed hypotheses with count rules. The frontier is descriptive. Everything is reported. |
| F13 | *"Unlicensed code."* | **P** | Written permission, 2026-10-03, from the XAttention authors, recorded as the researcher's summary, not a quote (§0.1). Every dependent action sits behind the §0.2 gate: default no, asked each time, logged outside the repository. The code is called, never redistributed. **Open:** still no licence, so third-party reproduction depends on upstream staying up. |
| F14 | *"It stops below where sparsity pays."* | **P** | Llama accuracy at 65536. **Open:** it is cut first, there is no Llama end-to-end timing beyond 32K, and the L4 is off XA's curve. |
| F15 | *"One machine type per card."* | **L** | Recorded, not varied. |
| F16 | *"Batch 1, prefill only, HuggingFace stack."* | **L** | Audit items 8 and 10 stay open. |
| F17 | *"A fourth XAttention configuration after T4 said 'last'."* | **C** | The Qwen native arm is T4's configuration, replicated with identical settings and the table by digest. Its predictions (P-T4, R1–R4) are committed before T4 runs (§0). Era-4 XA arms are scorer positions, not XAttention configurations. Llama uses the authors' configuration, not a new one. |
| F18 | *"Calibration texts beyond the model's context."* | **C** | Excluded by a config-read rule (§4.3). T4 is amended to match (§12). |
| F19 | *"Decode at 32768 runs past Qwen's position limit."* | **P** | All banked rows are flagged: 1,638 of 34,582 have P + gen > 32,768, 1,636 with a forward at a position ≥ 32,768, and 12 era-1 rows with a 32,769-token prompt (§4.9). `positions_over_limit` goes on every new row, with a subset analysis. **Open:** the replication keeps T4's prompts on purpose. |
| F20 | *"Different dense kernels for speed and for accuracy."* | **C** | On purpose (§2.4). The accuracy reference is fixed for identity with T4. The timing baseline is the fastest correct kernel. Both are named on every row. |
| F21 | *"The cost stop will quietly drop whatever fails."* | **C** | Cuts come only from the reservation check, in a fixed order, never after a result. Uncut ★ items run first. Every cut item is reported as "not run (cost stop)". |
| F22 | *"You published results from code you had no permission to use."* | **P** | T4's XA results and `backends/xattention.py` were public on `origin` from 2026-10-02, before permission arrived. The authors have since permitted publishing results and reproducing their kernel lines, and were informed of that file and approved it (researcher's summary, §0.1). From now on every push or external use is gated (§0.2). **Open:** the record of that approval is the researcher's summary, not a citable document. |

---

## 10. Steps that weaken the methodology or settle for an incremental result

| ⚑ | step | status after this amendment |
|---|---|---|
| 1 | Qwen accuracy only at 1.5B. | **Open.** Qwen2.5-7B accuracy is not planned. Llama-3.1-8B partly answers model scale. |
| 2 | n = 100. | **Resolved** for the primary cells (n = 300, §5). The secondary tier stays at 50 and carries no headline. |
| 3 | One family for accuracy. | **Partly resolved** (Llama, H8). **Cut-order item 11 removes it** under cost pressure. |
| 4 | Block 128 only. | **Partly resolved:** intrinsic recall at 16–128 (H7). Cost and profitability stay at 128. |
| 5 | VS block-ified. | Open (F11). |
| 6 | Bands stop at 32K. | **Partly resolved** by Llama accuracy at 65536, **which is cut first**. No end-to-end timing beyond 32K. |
| 7 | L4 off XA's curve. | Kept on purpose. The torch fallback is labelled and refused cross-card. |
| 8 | No FA3. | **Resolved:** FA3 included and never cut. A failed build makes H100 claims upper bounds. |
| 9 | Descoping H2 or H4. | Forbidden. Their inputs are ★. |
| 10 | Teacher-forced recall. | Open (F5). |
| 11 | **New:** calibration is the authors' released substitute, not the paper's DP. | Open (F1). H6b measures it, and §14 asks the authors. |
| 12 | **New:** era 3 against era 4 is cross-era. | T4's head-uniform arms against this study's per-head arms are descriptive only. |
| 13 | **New:** the cap is ₹12,000 (confirmed 2026-10-03) against a ₹20,589 worst case for the full plan (sixth draft, measured μ). | At worst-case μ, only the core runs. The canary's μ decides how much of the cut list is funded. |
| 14 | **New:** H2c, H2d and H5a were written with the A100 crossover estimates visible (§3). | Stated. They are reported as priors anchored on visible data, not blind predictions. The blind tests of the crossover are H1 and H2a/H2b. |
| 16 | **New:** the authors' calibration and RULER pipelines feed a double BOS; this study feeds one (§4.9). | Stated as a confound for H6a/H6b and H8. A double-BOS sensitivity arm is not planned; adding one needs the researcher's yes and a dated amendment before I2c. |
| 17 | **New:** Llama accuracy uses this study's completion-style prompts and newline stop, not the authors' chat format. | Stated. H8 is not a reproduction of the authors' RULER scores, and is never compared with them numerically. |
| 18 | **New:** Llama number-task caps are 6 tokens (2 × a 3-token answer). | Kept by the rule. XL0 reports the dense cap-hit rate, and above 5% it is a stated truncation confound. The cap does not change after XL0. |
| 15 | **New:** the H100 in-model speedups are against `sdpa_flash`, and no banked H100 dense kernel exists at (12, 2). | Labelled **unmeasured** in `limitations.md` (tested, `tests/test_h100_baseline_caveat.py`). Run D in I3 measures it. Nothing is adjusted from a different geometry. |

---

## 11. Lock gates, and what must exist before the first session

### 11.1 Lock gates (only these; set 2026-10-03 by the researcher)

The hold on the authors' answers is lifted (§0.1). The file locks when all
seven of these are met and committed with it (L7 added 2026-10-03):

| # | gate | status 2026-10-03 |
|---|---|---|
| L1 | `attnbench/analysis/frontier_prereg.py`, its plan test `tests/test_frontier_prereg_plan.py`, and the §11.3 break-tests, each watched red | **not started** |
| L2 | The T4 amendment code (§12), with T4's own plan test updated, committed with the dated amendment sections | **not started** |
| L3 | The `mask_selector` column on every row type, era 4 registered (§4.5), and its stripped-column break-test | **not started** |
| L4 | The cuDNN-on-H100 record checked (§4.6) | **done.** The record exists and is guard-written: cuDNN was never launched above 8192 on an H100. §4.6 is corrected, and so is every doc that called these rows faults (runbook, `run_probe.py` help, `silent_failure_patterns.md` ×2, `limitations.md`, `a100_session_plan.md`): "observed on L4 and A100, H100 untested above 8192". |
| L5 | The Llama-3.1-8B `config.json` check (G9) at revision `0e9e39f…`: `max_position_embeddings`, `rope_scaling`, sha256 | **Passed 2026-10-03, for the position limit only.** The researcher read it locally with their own token (values and sha256 in §4.9). On CPU the same day: the repo id and table shape match the authors' model; transformers 4.46.0 and 5.18.0 implement `llama3`, and the logits gate passes with it (`tests/test_llama3_rope_gate.py`). What it does not cover is L7. |
| L7 | The Llama stop rule (added 2026-10-03 from the researcher's pre-lock list): `generation_config.json`'s stop ids recorded in §4.9; the Llama caps measured with the Llama tokenizer and committed to `stopping.LLAMA_31_8B_TASK_TOKEN_CAPS`; `text.json` re-counted with the Llama tokenizer (G9 step 3, done early) | **Done 2026-10-03 (sixth draft).** `generation_config.json` was reported by the researcher, with the token ids confirmed locally. The caps were measured offline with the cached tokenizer (no token) and committed, pinned to the §4.9 table by a test. The `text.json` maximum is 65,314. Also done: forced and asserted greedy decoding, the three-id stop set, `stop_token_id`, the newline set rebuilt from the Llama vocab, the shared encoder (sizer = fed ids) and the double-BOS guard, all tested (`tests/test_llama_decoding_and_prompts.py`). |
| L6 | Banked rows flagged for positions past 32,768 (§4.9) | **done.** `scripts/flag_positions_over_limit.py` → `results/positions_over_limit/`: 1,638 / 34,582 rows, plus the per-band sizing deltas. The root cause is sizing on example 0 (`ruler.py` docstring fixed). The script is committed. Its output sits in gitignored `results/` and is regenerated by the script. |

**Settled 2026-10-03, no longer open:** §0.1 records the researcher's
summary, the date and "XAttention authors"; §8.3 records the ₹12,000 cap.

### 11.2 Before the first session (after lock is fine)

1. Provenance: `launch_floor_us`, `sync_floor_us`, `h2d_floor_us` and
   `cpuPlatform` in `provenance.py`, stamped onto every row type.
   `positions_over_limit` and `xattn_path` on accuracy rows.
2. The dense baseline: the "runs correctly" check, the isolated cuDNN probe
   with the guard bypassed (`--launch-known-fault`), the health check,
   run D, and the `dense_baseline` column. The FA3 build script for a CPU
   VM, pinned.
3. Implementation, each with its gate test:
   - `bsa_prefill`;
   - the era-4 selector;
   - per-head MP;
   - VS (vendored MIT code and NOTICE);
   - SL;
   - the GPU-side multi-b oracle;
   - the recall script;
   - extensions to `measure_estimator_cost.py` and
     `run_vectorised_endtoend.py`;
   - a frontier grid config at n = 300.
4. `results/frontier_spend.csv`, `scripts/frontier_budget_gate.py`, and
   `frontier_prereg.rebracket()` with its leak guard (§8.3).
5. Llama gated access on the project token, verified.
6. The era table: **done** (`65cd9c1`). The 16 T4 files are registered as era 3 by commit, with their XAttention rows marked as native selection in prose. `tests/test_eras.py` passes, and its key regex was fixed and break-tested. Era-4 registration (L3) still has to add `mask_selector`.

### 11.3 Break-tests (part of L1), permanent and watched red

- One fail case and one indeterminate case per scorer (H1–H8, R1–R4, P-T4).
- The split break-test (§5).
- The era-4 stripped-column break-test (L3).
- A reservation-check break-test: a ledger and plan over the cap must print
  `STOP`.
- A `rebracket()` refusal of any parquet carrying a result column.
- A cross-card refusal of `torch_fallback` rows.
- The A100 caveat test (`tests/test_a100_baseline_caveat.py`) is **done**:
  red on a mutated figure, green after a byte-identical restore, 2026-10-03.

---

## 12. T4 amendments, to append (dated) to `docs/t4_xattention_calibrated.md` before this file locks

Session A has not run, so each is an amendment before data.

- **A1, `ruler_heldout` leaks `qa_1` questions.**
  - Its seed-1 examples 0–7 ask SQuAD questions 0–7, which are test questions
    0–7, because QA picks the question by index (§5). `assert_disjoint`
    compares whole contexts, so it passes.
  - **Change:** draw `ruler_heldout` from this study's calibration split (seed
    1, `qa_1` indices 2000–2007 per band, NIAH from seed 1), and replace
    `assert_disjoint` with the §5 content check and its break-test.
  - The table stays descriptive.
- **A2, over-length texts.**
  - 35 of the 156 `authors` texts exceed Qwen2.5-1.5B's 32,768-token
    `max_position_embeddings`, up to 91,574. The plan's premise that the texts
    are "thousands of tokens, much shorter than the test bands" is false: the
    median is 16,025.
  - **Change:** the exclusion rule of §4.3, which leaves 121 texts, with the
    excluded indices recorded in the table.
- **A3, the threshold statistic.** Pre-register the max over used texts,
  verbatim, with no cap, plus the descriptive p90 and argmax records (§4.3).
- **A4, the calibration card.** Session A moves to the **A100**, so the
  profiler's `use_triton=True` takes the Triton path it was written for. The
  table records `xattn_path="triton"`. Session B stays on the L4, with rows
  labelled `torch_fallback`. Cost: Session A becomes 15–48 min (₹70–230)
  instead of about 45 L4-minutes.
- **A5, calibration provenance.** State that the profiler is the authors'
  released substitute for the paper's unreleased DP method (issue #13). This
  phase's claims therefore read "calibrated by the authors' released
  profiler".
- **A6, the replication link.** This study's §3.9 (P-T4, R1–R4) is committed
  before Session A. T4's `authors` configuration is the one replicated, and
  is unchanged by this study.
- **A7, positions past the limit.** Note that 119 of 200 dense rows at 32768
  in the XA pilot decode past 32,768 positions, and add
  `positions_over_limit` to new rows. No design change.

Also correct, as a dated note in `docs/t4_xattention_pilot.md`, the untested
sentence "the fallback selects the same blocks". G2 tests it.

---

## 13. Changes since the first draft (2026-10-03)

*(Until 2026-10-03 each later table had been inserted inside the previous
table's separator row, so the tables rendered nested and out of order. They
are now in draft order. Their contents are unchanged, except that the first
table's "now" column is labelled "second draft".)*

**Second draft (2026-10-03):**

| area | first draft | second draft |
|---|---|---|
| Sequence | unspecified | T4 amendments, then lock, then T4, then this study (§0) |
| XA native on Qwen 1.5B | a new calibration in run C | a replication of T4 `authors`, identical settings, table by digest, with P-T4 and R1–R4 pre-stated (§3.9, §4.3) |
| Selector rule | free blocks counted inside the budget, which would have failed G3 by construction | the era-3 rule, per head; registered as **era 4** with a row-level `mask_selector` (§2.1, §4.5) |
| Calibration | "≤ 32K texts", statistic unstated | the config-read exclusion rule, max over texts with no cap, descriptive p90 and argmax, A100 Triton path, the DP provenance stated (§4.3) |
| Dense baseline | "fastest installed" | fastest **correct**: health check, switch points with cause, cuDNN excluded above 8192 on sm_80/sm_89 and probed in isolation on the H100, FA3 included and built on a CPU VM, a fixed `sdpa_flash` accuracy reference (§2.4, §4.6) |
| Host provenance | CPU in provenance | CPU model, `cpuPlatform`, launch, sync and H2D floors on every row; the sync term in H2 (§4.8) |
| XA on the L4 | excluded from the frontier | labelled `torch_fallback`; refused by every cross-card analysis (§4.9) |
| n | 100 | 300 on primary cells, with attainable bounds (§5) |
| Llama-3.1-8B | positive control only | plus accuracy at 16K, 32K and 65K, in the authors' RULER configuration, with task selection and gate G9; `config.json` **not verifiable here** (gated) (§4.9, H8) |
| Block size | 128 only | intrinsic recall at 16, 32, 64 and 128 (H7) |
| Positions past the limit | not noticed | 119 of 200 T4 rows at 32768; recorded and analysed as a subset (F19) |
| Cost | a bracket and a soft re-projection | a ₹10,000 hard cap, a reservation check, an 11-step cut order and a ★ core (§8.3–8.4) |
| Licence | policy only | a permission request drafted (§14) |
| Hypotheses | H1–H6 | plus H7 and H8; H5 restated for n = 300 |

**Third draft (2026-10-03):**

| area | second draft | third draft |
|---|---|---|
| Permission | a request drafted | written permission received; verbatim scope, date and sender to be pasted (§0.1); the professor's-guidance note superseded |
| Publication | a sentence in §14 | a binding rule, §0.2 and §8.6: no waiting window; it cannot be loosened after any result; met only for the quoted scope; the public `origin` already carries T4 XA results (F22) |
| Fallback | none | §0.3: what survives without XA; "our implementation of the published algorithm" named as an option, not used until the professor is consulted |
| SL arm | none | sink + local window, the zero-cost anchor (§4.2); its extrinsic arm is cuttable (0c) |
| cuDNN on H100 | "never tested" | a record exists in 6 H100 files, but it is guard-written (`gates.py` returns before `run_once`); the probe now bypasses the guard explicitly; two docs flagged as wrong about these rows |
| A100 baseline | a §4.6 note | a tested caveat in `claims.md` (`tests/test_a100_baseline_caveat.py`, break-tested); run D plans the same-session dense-kernel comparison at the real geometry |
| Positions past 32,768 | the XA pilot dense file only | all banked rows flagged (1,638 / 34,582; 12 rows with a 32,769-token prompt) |
| Cost | the cap, the cut order | DP1 and DP2 re-bracketing points with a leak guard; S_res storage reserve; replicate escalation made cuttable (0a, 0b); worst-case core + storage + one rerun ≈ ₹11,600 against a ₹10,000 cap (open decision) |
| XA share | — | 28–29% of the core (₹490–2,710 of ₹1,720–9,360) |
| Lock | gated on all §11 items, held for the authors | the hold lifted; gated on L1–L6 only (§11.1); L4 and L6 done |

**Fourth draft (2026-10-03):**

| area | third draft | fourth draft |
|---|---|---|
| Permission | placeholders for a verbatim scope, date and sender | the researcher's summary (not a quote), 2026-10-03, "XAttention authors"; the email stays private; `backends/xattention.py` is approved and left as is, so the call-site merge is dropped (§4.4, gate G1b) |
| Gate | a publication rule | a binding permission gate (§0.2, §8.6): default NOT PERMITTED, asked per exact action, never inferred, logged outside the repository |
| Cap | ₹10,000, open decision | **₹12,000**, confirmed by the researcher; worst case about ₹11,900 |
| cuDNN | the H100 record corrected in §4.6 | also corrected in 6 places across docs and code; the probe runs **last**, isolated, `CUDA_LAUNCH_BLOCKING=1`, after sync; cuDNN is never an H100 candidate above 8192 |
| 32K band | T4 prompts kept | the capped band 32768c (budget = 32,768 − `token_cap`), never pooled with banked 32768; the replication alone keeps T4's prompts (X-rep, ★); gate G11 |
| Sizing | 12 prompts at 32,769 noted | root cause (example-0 sizing for UUID/number-needle tasks), measured overshoot at every band (up to +82 at 16384), `ruler.py` docstring fixed |
| Docs | — | 32K-positions caveat on every 32768 accuracy citation in `claims.md`; A100 baseline caveat in README, `writeup_input.md` and `limitations.md` |

**Fifth draft (2026-10-03):**

| area | fourth draft | fifth draft |
|---|---|---|
| G9 (L5) | pending with the researcher | **passed for the position limit only** (values and `config.json` sha256 in §4.9); repo id and the 32 × 32 (layers × query heads) table shape match the authors' model; the revision they used is unknown |
| RoPE `llama3` | unchecked | implemented in transformers 4.46.0 and 5.18.0; logits gate re-run with triggering llama3 scaling (`tests/test_llama3_rope_gate.py`, break-tested); 4.46.0's silent `rope_parameters` trap; G9 step 4 on the instance |
| Llama bands | 16384, 32768, 65536, uncapped | 16384, 32768c, 65536c (band − Llama cap); no 128K band |
| Llama sizing | example-0 sizing possible for a task outside `_PER_EXAMPLE_FIT` | `per_example_fit=True` on every Llama generation (code and tests, break-tested); G11 extended |
| Llama stop rule | Qwen's caps by default | its own caps, measured with the Llama tokenizer; lookups raise until measured (break-tested); stop ids from `generation_config.json` (pending); lock gate **L7** |
| Llama oracle-pass cost | FLOP split × assumed μ (lower 2.0) and an assumed 1.6× | the measured 1.5B A100 oracle pass × config ratios 3.048–5.327 (`scripts/derive_llama_oracle_cost.py`); core ₹1,842–9,526; worst case about ₹11,780 |
| Visibility | unstated | §3: H2c, H2d and H5a were written with the A100 crossover estimates visible, and are reported as anchored priors |
| H100 baseline | not caveated | labelled **unmeasured** in `limitations.md` (tested, `tests/test_h100_baseline_caveat.py`); run D in I3 measures it |
| Licence request | a draft in `docs/` (untracked) | moved out of the repository; only the request date (2026-10-03) is recorded (§0.1, §14) |
| Sizing wording | "token-exact", "never above" | corrected in `limitations.md`, `writeup_input.md`, `sizing.py`, `ruler.py` and `reestimate_stage3.py`; 182 distinct over-budget examples (655 file-example pairs) |
| §4.6 | "Neither is corrected here"; "four machines" | stale: the docs were corrected in `3f8a884`, and three Xid 31 events are banked |
| §13 | later drafts' tables nested inside earlier tables' separator rows | repaired, in draft order |

**Sixth draft (2026-10-03):**

| area | fifth draft | sixth draft |
|---|---|---|
| Push | none | the 8 commits `dbabef7`..`469fed2` pushed to `origin` as branch `frontier-prereg-2026-10-03`, with the researcher's logged yes; no merge, no tag |
| μ in §8 | 2.0 (assumed) – 11.3 (L4) on C, I1, I2a, X | **6.19–9.08**, measured on the A100 (1.5B oracle pass); old and new side by side (§8.2) |
| Llama oracle lines | Qwen token counts | Llama token counts of `text.json` (3.63M, max 65,314, 130 ≤ 32K) |
| Cost | core ₹1,842–9,526; full ₹7,133–21,347; worst case about ₹11,780 | core **₹2,540–9,053**; full **₹7,704–20,589**; worst case **about ₹11,250** against ₹12,000 |
| L7 | pending | done: `generation_config.json` recorded; caps measured (6 / 56 / 40 / 6 / 30 / 30 / 42 / 46); `text.json` recounted |
| Decoding | argmax only | greedy forced and asserted on Llama, refusal test, determinism test, on-instance determinism STOP (G9 step 6) |
| Stop set | union of tokenizer and generation_config | Llama must be exactly {128001, 128008, 128009}; the tokenizer's 128009 alone is refused; `stop_token_id` on every row |
| Encoding | two call sites agreeing by default | one encoder (`prompting.py`, `add_special_tokens=True`), fed length checked against sized length, double-BOS guard |
| Authors' pipeline | not checked | greedy; chat format; **double BOS** in calibration and RULER (156/156); recorded with the confounds it leaves (§4.9, §3.9) |
| Block 0 | — | holds the BOS on Llama |

---

## 14. Licence and provenance request to the XAttention authors

- **The request.** Dated 2026-10-03 (§0.1). Its text is not kept in the
  repository (moved out 2026-10-03, never committed).
- **The reply.** Dated 2026-10-03, from the XAttention authors. It is
  recorded only as the researcher's summary (§0.1). The email is private and
  is not quoted anywhere in the project.
- **Unanswered:** threshold provenance, and Triton-versus-torch block
  selection. Both are in §0.1, with no waiting window.
- **Every dependent action** is gated by §0.2 and §8.6.

---

## Results

*(Empty until lock and the first session.)*
