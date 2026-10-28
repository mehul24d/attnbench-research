# Audit of `docs/RESEARCH_PLAN.md` (task C13)

**Date:** 28 Oct 2026. **Scope:** read-only audit of the plan and `CLAUDE.md`. No edit was made to the plan. Every edit below is a proposal for the researcher to accept or decline.

**What was checked.** The plan on branch `cloud/C13-2026-10-28` at `9d2d8b1` (the tip of `cloud/C5-2026-10-28`), and the code and plan on `origin/main` at `ada59c0`. Line numbers are those of the plan at `9d2d8b1`. Where `origin/main` differs, the finding says so.

**Severity.** *High*: the plan cannot be followed as written, or a gate or hypothesis cannot be evaluated. *Medium*: two parts of the plan, or the plan and the code, disagree in a way that changes what is done or reported. *Low*: wording, a stale status or a label, with no effect on results.

---

## A. Dependency cycles and ordering

### A1. High. T2.4 cannot be Done before T3.1, and T3.1 needs it Done
- **Where:** Section 8, T2.4 (line 398, "Done when") and T3.1 (line 441, "Depends").
- **Text:** T2.4: "Replay vs trace check passes on P1, and on the first card of every new code path". T3.1: "Every Phase 2 task Done, except T2.14's AWS half, T2.22 and T2.23 to T2.26".
- **Problem:** P1 is the T3.1 session, and "every new code path" includes P2 and P4 (Phase 6). T2.4 can only be Done after T3.1 and T6.x, so T3.1 can never start.
- **Proposed edit:** T2.4 "Done when": "Replay vs trace check passes end to end in the L4 shakedown (T2.27 step 3)". Move "passes on P1, and on the first card of every new code path" to Gate A (Section 7), which already holds it.

### A2. High. The status word "Done" requires use in the study, so no Phase 2 task can be Done before T3.1
- **Where:** Section 0, status words (line 20); T3.1 "Depends" (line 441).
- **Text:** "*Done*: built, tested, used. *Built*: code and tests exist, not yet used for the study."
- **Problem:** T3.1 is the first session that uses Phase 2 code for the study. Under the definition, every Phase 2 task is at most Built until T3.1 runs. The rows also apply the words unevenly: T2.1, T2.7, T2.8, T2.9, T2.10, T2.20 and T2.22 say Done on tests alone, while T2.11, T2.15, T2.19 and T2.28, in the same state, say Built.
- **Proposed edit:** Either change T3.1 "Depends" to "every Phase 2 task Built or Done, and T2.27 Done", or redefine *Done* as "meets its Done-when column". Then make the rows consistent (one word for "tests pass, not yet used").

### A3. High. Gate A is read before the values it needs are fixed
- **Where:** Section 7, Gate A (line 340); T3.3 (line 443); T4.1 (line 449).
- **Text:** Gate A: "within a bound fixed at lock" (replay-versus-trace) and "within a bound of 1.0 fixed at lock" (delay slope). T3.3 depends on T3.1; T4.1 depends on T3.2, which depends on T3.1.
- **Problem:** Gate A (T3.3) is decided before lock (T4.1), so its two bounds do not exist when it is read. Gate A also says "beats the sum and the KernelSight-style step form" with no statistic.
- **Proposed edit:** Add a pre-lock step: "Gate A bounds (replay-versus-trace, delay slope) and its comparison statistic (median absolute relative error over P1's cells) are fixed and hashed before T3.1's ground truth runs", as a new task T3.0 or as part of T2.27's output. Remove those two items from T4.1, or mark them "carried from T3.0".

### A4. High. The ground-truth density set is fixed at lock but must include densities known only after the accuracy runs
- **Where:** Section 5.3 step 7 (line 176); Section 5.7 primary cell size (line 271); T4.1 (line 449).
- **Text:** "density (a set fixed at lock that includes every admissible density ...)"; "Admissibility n is fixed at lock."
- **Problem:** Admissibility is measured with an n fixed at lock, so it is known only after lock, but the density set that must contain every admissible density is also fixed at lock. No task schedules the admissibility runs (see B1).
- **Proposed edit:** Fix the density set at lock as "every density in the accuracy grid of each arm" (a superset), so it does not depend on the outcome; or schedule the admissibility runs before T4.1 and state that the density set is read from them.

### A5. Medium. T2.28 waits on the mask gate, which runs after lock
- **Where:** T2.28 status (line 422); T2.23 (line 417).
- **Text:** "Still to come: the gate's output feeding `deltas`".
- **Problem:** The gate's output comes from the P2 session (T6.1, after Gate B). If "Still to come" means T2.28 is not finished, T3.1 (which needs it) waits on T6.1. Its "Done when" ("Tested") is already met.
- **Proposed edit:** Mark T2.28 as meeting its Done-when, and move "feeding the gate's output" to T2.23's "Done when".

### A6. Medium. The H4 power rule feeds T4.1 but T4.1 does not depend on T1.5
- **Where:** T4.1 "Depends" (line 449); T1.5 (line 385); Section 5.7 H4 power rule (line 279).
- **Text:** T4.1 fixes "power target and n, primary cell n"; n needs "p ... measured on the banked pilot rows, T1.5".
- **Proposed edit:** T4.1 "Depends": "T3.2, T1.5, T1.3" (T1.3 for the head-type criterion, see C3).

### A7. Medium. Steps needed before T3.1 are not in the shakedown checklist
- **Where:** T2.27 checklist (lines 424 to 435); T2.12 (line 406), T2.18 (line 412), T2.28 (line 422).
- **Text:** T2.12: "the same on a GPU with the real kernel (L4 dry run, T2.27)"; T2.18 "Done when": "Tested on the L4 against a full trace"; T2.28: "the real replays at several densities (T2.27, T3.1)".
- **Problem:** The checklist names T2.2 to T2.7, T2.13, T2.16, T2.17 and T2.21 only. T2.12 and T2.18 have L4 steps that no session runs, and T3.1 needs both Done. (On `origin/main`, T2.12 mentions a built "Shakedown 2" `t212` step, which this branch's plan does not have.)
- **Proposed edit:** Add checklist steps for T2.12 (precomputed-mask replay on the real kernel) and T2.18 (probe against a full trace), and add T2.12 and T2.18 to T2.27's "Depends".

---

## B. Missing tasks and acceptance criteria that cannot be tested

### B1. High. No task runs the confirmatory accuracy and recall sessions
- **Where:** Section 5.7 "Where measured" (line 281); Section 8 phases 1 to 6; rule 14 (line 326).
- **Text:** "once, on the L4 (the primary GPU), with the dense control repeated on the A100".
- **Problem:** H4 (n of at least 200 on `qa_1`/16384 at block sizes 32, 64 and 128, with the random control), admissibility for every arm and density, and the A100 cross-card dense control have no task, no session in rule 14's GCP list and no line in the funding paragraph. T1.4, T1.8 and T1.9 build code only.
- **Proposed edit:** Add a Phase 6 task (for example T6.7, "L4 accuracy and recall session: H4 and admissibility; A100 dense control; GCP, launch by 9 Dec 2026"), depending on Gate B, T1.4, T1.7 to T1.9. Add it to rule 14 and the funding paragraph.

### B2. Medium. H5 names "estimators with host-side steps", but no arm has one
- **Where:** H5 (line 107); Section 5.5 arms table (lines 219 to 229).
- **Text:** "(a) ... a faster GPU moves the crossover to longer contexts for estimators with host-side steps"; "(b) ... an estimator whose selection runs entirely on the GPU".
- **Problem:** Every timed arm builds its mask on the GPU or uses a fixed pattern; the CPU reference builder is not an arm. H5(a) has no subject, and (a) and (b) cannot be contrasted.
- **Proposed edit:** Redefine the contrast as "arms with synchronisations inside the step (calibrated XAttention as shipped) against arms without (mean-pool, asserts removed)", or add the CPU reference builder as an arm.

### B3. Medium. H1 says "all four baselines", but Section 5.4 defines six
- **Where:** H1 test column (line 103); Section 5.4 baselines (lines 195 to 202).
- **Text:** "recursion error below all four baselines' committed predictions".
- **Problem:** H1's text lists the fitted sum, sum, step form and learned predictor. Section 5.4 also defines the constant-host-cost and nearest-pairing baselines ("a naive baseline the model must beat"), and T2.9 adds the fitted sum + s_0 and sum + s_0. Which ones H1 scores is unclear, and "beat" has no statistic (median, mean or per cell).
- **Proposed edit:** "recursion's median absolute relative error below that of each of the fitted sum, sum, step form and learned predictor ... ; the constant-host-cost, nearest-pairing and + s_0 baselines are reported, not scored."

### B4. Medium. H1 and H6 decision rules use the resolution floor before it exists
- **Where:** Section 5.9 (line 291); T2.13 (line 407); Section 5.6 (line 246); T4.1.
- **Text:** "dense unless a sparse configuration is predicted faster by more than the resolution floor"; T2.13 flags a round "by more than the resolution floor"; "fixed at lock".
- **Problem:** T3.1 (before lock) runs T2.13, which needs the floor to flag rounds. P1 is development, but the session's flags need a value.
- **Proposed edit:** State a provisional floor for P1 (for example "set from the shakedown's run-to-run spread and recorded in the T3.1 config"), with the locked floor applying from P2.

### B5. Medium. Admissibility at 32768 conflicts with the grid rule
- **Where:** Section 5.3 step 7 grid rule (line 176); H6 (line 108); Section 5.7 position limit (line 282).
- **Text:** "a cell that does not fit is sized down, never run past the limit"; H6 "at 16384 and 32768 tokens (where admissibility is measured)".
- **Problem:** For the Qwen2.5 models (32,768 positions), a 32768-token prompt plus any generation cap does not fit, so every accuracy cell called 32768 is shorter than the 32768 timing cell it labels. The plan does not say which accuracy length gives admissibility at 32768.
- **Proposed edit:** "At 32768, admissibility is taken from the accuracy cell sized to 32,768 minus the task's generation cap; the timing cell stays at 32768. Each such pair is named at lock."

### B6. Medium. Batch-4 cells for XAttention are not stated in the body
- **Where:** Section 5.3 step 7 (line 176); Section 5.4 Batch (line 210); T2.12 (line 406).
- **Text:** T2.12: "Batch 1 only (the kernel call's limit, as XAttention's)".
- **Problem:** The body says batch 4 runs at 16384 on L4 pairings and at 32768 on A100 and H100, for "every arm". If XAttention is batch 1 only, the XAttention pair has no batch-4 cells, and H1's XAttention-pair test and H6 at batch 4 change. Only the precomputed-mask arm has an Appendix A line.
- **Proposed edit:** In Section 5.3 step 7, list the arms that run at batch 4, and add a dated Appendix A line if XAttention is excluded. Also state whether the A100 and H100 run batch 4 at 16384.

---

## C. "At lock" values missing from T4.1

T4.1 (line 449) is meant to list every **[lock]** value. These are fixed "at lock" elsewhere but are not in it.

| # | Sev. | Where | Text | Proposed addition to T4.1 |
|---|---|---|---|---|
| C1 | Medium | Section 5.4, learned baseline (line 198) | "features and hyperparameters fixed at lock" | "LightGBM features and hyperparameters" (T2.10 says features are frozen in code; hyperparameters are not) |
| C2 | Medium | Section 5.5, K2 (line 233) | "within a tolerance fixed at lock"; "runs N repeats without a device fault" | "K2 correctness tolerance (the gate's `TOL`) and the repeat count N" (T2.16: "the tolerance ... is not yet fixed at lock") |
| C3 | Medium | Section 5.7, recall by head type (line 274) | "criterion fixed at lock" | "head-type criterion (T1.3)" |
| C4 | Low | H1 (line 103) | "the XAttention pair's predicted and measured differences agree within **[lock]**" | Name it separately from "H0 to H3 tolerances": "H1 XAttention-pair difference tolerance" |
| C5 | Low | H4 power rule (line 279) | "α is the one-sided level ... 1-β is the power, all fixed at lock" | "H4 α" beside "power target" |
| C6 | Low | H5 (line 107) | "tolerance stated on the input ratio" | "H5 input-ratio tolerance", or say it is fixed before data, outside lock |
| C7 | Low | Section 5.1 (line 121) | strict and looser "same sequence" definitions | "the strict and looser same-sequence definitions" |
| C8 | Low | T2.13 (line 407) | "rerun within a stated budget"; "too few clean rounds" | "flagged-round rerun budget and minimum clean rounds" |

---

## D. Contradictions between sections

### D1. Medium. P4 is a new code path, so its masks are recorded, contradicting "only if the gate fails"
- **Where:** Section 5.1 operation sequence (line 121); Section 5.2 run order (line 140).
- **Text:** "Recorded per code path: GPU architecture, library versions, and which implementation runs"; "masks are recorded on the target only for the first card of a new code path, and only if that gate fails"; "P4's recorded Triton masks are compared with P2's"; "P4 needs only its own components and P1's operation sequence".
- **Problem:** By the definition, the H100 is a new GPU architecture, so P4 is a new code path whether or not P2's gate passes. Its masks are recorded (T2.24 needs them), and its operation sequence must come from its own structural probe, not from P1 (an L4 PyTorch path).
- **Proposed edit:** "P4 is a new code path (Hopper, Triton): its structural probe and mask recording run on P4, untimed, whatever the P2 gate's result." Run order: "P4 needs only its own components and its own structural probe."

### D2. Medium. Section 5.4 does not carry the 22 Oct 2026 recursion decisions
- **Where:** Section 5.4 recursion (lines 183 to 189); Appendix A, 22 Oct 2026 (line 636).
- **Text:** Appendix A: "h_i = 0 there and the recursion refuses anything else"; "A prefill sequence ends with an explicit closing synchronise". Section 5.4: "H_i = H_(i-1) + h_i ... end-to-end prefill time = D_N".
- **Problem:** Rule 4 (Section 0) says a decision gets a body change and an Appendix A line. The code (`attnbench/predict/recursion.py`, lines 32 to 58 and 163 to 181, both commits) applies both rules; the body does not.
- **Proposed edit:** Section 5.4: "For a synchronisation h_i = 0 (s_fixed holds its host cost). Every sequence ends with a closing synchronise, so end-to-end time = D_N = H_N."

### D3. Medium. HDBI: one threshold in the plan, two in the code
- **Where:** Section 5.4 HDBI (line 199); Appendix A 25 Oct (lines 639 and 648); `attnbench/predict/hdbi.py`, `classify` (lines 53 to 70, both commits).
- **Text:** "Its boundedness is one threshold, fitted on the training pairings by sign agreement". Code: `classify(value, *, host_bound_below, device_bound_above)`, three classes.
- **Proposed edit:** Either change the code to one threshold (two classes), or restate Section 5.4 as "two thresholds (host-bound, balanced, device-bound), fitted ..." with a dated line.

### D4. Medium. The confirmatory H4 family has more than three comparisons
- **Where:** Section 5.7, H4 power rule and family (lines 279 to 280).
- **Text:** "Bonferroni over three comparisons gives about 1,630 × p"; family: "the planned estimator pairs at block sizes 32, 64 and 128 ..., plus the random-selection control against each estimator".
- **Problem:** With k estimators, the family has 3 × k(k-1)/2 pair tests plus at least k control tests. Three is too few even for two estimators (3 pairs + 2 controls = 5). The 1,630 × p figure understates n.
- **Proposed edit:** Replace "three comparisons" with the count from the declared family, or write "about (z(1-α/m) + z(0.95))² / δ² × p for a family of m tests, m fixed at lock".

### D5. Medium. K statuses in Section 5.6 are stale against Section 8
- **Where:** Section 5.6, K1 to K9 (line 264).
- **Text:** "K2 ... (not built)"; "K9 predictions hashed before the workload (not built)".
- **Problem:** T2.16 (K2) is Partly built and T2.11 (K9) is Built.
- **Proposed edit:** "K2 ... (Partly built, T2.16)"; "K9 ... (Built, T2.11)".

### D6. Medium. Section 10 says "contexts above 65K"; the grid stops at 32768
- **Where:** Section 10 out of scope (line 510); Section 5.3 step 7 (line 176).
- **Text:** "contexts above 65K" against "nothing above 32768".
- **Proposed edit:** Section 10: "contexts above 32768".

### D7. Medium. The launch sequence in the plan and in `CLAUDE.md` differ
- **Where:** Section 6 rule 11 (line 323); `CLAUDE.md`, Cloud sessions.
- **Text:** Plan: "... preflight, run, teardown". `CLAUDE.md`: "... preflight, the GCS write gate, the bootstrap of a fresh instance (`scripts/gcp_bootstrap_instance.sh` ...), run, then teardown".
- **Problem:** The plan wins (Section 0 rule 1), so the write gate and bootstrap are, by the plan, not required steps.
- **Proposed edit:** Add "the GCS write gate, the bootstrap of a fresh instance" to rule 11.

### D8. Low. The push and merge rule is in `CLAUDE.md` only
- **Where:** Section 6 (lines 303 to 332); Appendix A 21 Oct (line 632); `CLAUDE.md` rule 11.
- **Text:** "a new `CLAUDE.md` rule 11 says any push or merge needs the researcher's explicit yes".
- **Problem:** `CLAUDE.md` says the full list is in plan Section 6, but Section 6 has no such rule, and the two lists number their rules differently (permissions are rule 10 in `CLAUDE.md`, rule 15 in the plan).
- **Proposed edit:** Add rule 16 to Section 6 with the push and merge text; in `CLAUDE.md`, cite plan rule numbers.

### D9. Low. Rule 5 against Section 10 and Section 1
- **Where:** Section 6 rule 5 (line 311); Section 10, first rejected row (line 514); T7.2 (line 478).
- **Text:** Rule 5: "anything else is marked 'from abstract'". Section 10: "Already explored (FlexPrefill, Twilight, AB-Sparse)". T7.2 lists FlexPrefill, Twilight, MInference, XAttention and DuoAttention as known from abstracts only.
- **Problem:** Section 10 and the arm rows (MInference-style, DuoAttention-style) characterise them without the mark. XAttention's code was read; its paper was not.
- **Proposed edit:** Add "(from abstract)" after FlexPrefill and Twilight in Section 10; note in the XAttention rows that statements are from its code at `e379887`.

### D10. Low. The glossary's crossover contradicts Section 5.4
- **Where:** Appendix B (line 679); Section 5.4 Crossover (line 211).
- **Text:** "the context length above which sparse beats dense" against "the bracket between adjacent grid lengths ... two crossovers are possible".
- **Proposed edit:** Glossary: "the grid bracket in which the sign of sparse minus dense time changes (Section 5.4); there may be none or two."

### D11. Low. Arm name not updated after the 12 Oct 2026 rename
- **Where:** H1 (line 103), Section 7, Section 9; Appendix A 12 Oct (line 608).
- **Text:** "XAttention with its checks removed" against the renamed "XAttention, Python asserts removed".
- **Proposed edit:** Use "Python asserts removed" throughout.

### D12. Low. Fallback zones the scripts do not have
- **Where:** Section 5.2 pinned machines (lines 146 to 148); rule 12 (line 324).
- **Text:** P1 "(fallback `asia-northeast1-c`)", P4 "(fallback `asia-southeast1`)"; rule 12 "fallback zones as in the launch scripts".
- **Problem:** `scripts/gcp_launch_l4.sh` (line 35) and `scripts/gcp_launch_h100.sh` (line 52) at `9d2d8b1` take one zone, overridable by `GCP_ZONE`, with no fallback. The P4 fallback is a region, while the column is "Zone"; T0.17 found the machine only in `asia-southeast1-b` and `-c`.
- **Proposed edit:** "Fallback by setting `GCP_ZONE` by hand: `asia-northeast1-c` (P1); `asia-southeast1-b` or `-c` (P4)." Rule 12: "fallback zones as in Section 5.2".

---

## E. Terms used but not defined

| # | Sev. | Term (where) | Proposed edit |
|---|---|---|---|
| E1 | Medium | **Resolution floor** and **noise floor** (Sections 4, 5.4, 5.6, 5.9; T2.13, T2.22). Defined only as "fixed at lock from the measured run-to-run spread", not how they differ or what statistic sets them | Glossary: "Noise floor: between-session spread of end-to-end time (P1 repeats). Resolution floor: the smallest sparse-minus-dense difference interpreted, set from the within-session spread of that difference." |
| E2 | Medium | **"gate G3"** (T2.12, line 406) clashes with gap G3 | Rename the bitwise selector gate, for example "selector gate (Section 5.5)" |
| E3 | Low | **Mandatory blocks** (random control row, line 229), **forced blocks** (Appendix A, 25 Oct) | Define once in Section 5.1 (sink and diagonal blocks every arm keeps) |
| E4 | Low | **"X% faster"** (Section 2, line 62). 0.95x and 1.84x follow only if "25.8% faster" means 25.8% less time (1.28 × 0.742 = 0.95), not 1.258x the speed (which gives 1.02x) | "cuDNN takes 25.8% less time at 8192 and FA2 6.0% less at 16384" |
| E5 | Low | **DWS**, **v7 image** (T2.7), **A8** (Section 2, line 78), **Session A/B**, **P-T4**, **R1 to R4** | Expand DWS as Dynamic Workload Scheduler; name the image's source; cite `docs/t4_xattention_calibrated.md` for A8, the sessions, P-T4 and R1 to R4 |
| E6 | Low | **"P-T4 (0 of 2) failed"** (line 78) reads as the outcome; the outcome was 2 of 2 | "P-T4 (predicted 0 of 2 certified; 2 of 2 certified) failed" |
| E7 | Low | **Density** (Appendix B) omits "causally valid blocks times query heads" (Appendix A, 25 Oct) | Add it to the glossary |

---

## F. Units and numbers

- **F1. Low. GB against GiB.** Section 5.2 (line 152): "the 200 GB balanced boot disk" priced "per GiB-month". Proposed: "200 GiB" (or state the conversion). The rupee and USD columns, the planning rates and T1.1's two session costs (₹273 for 55 min at ₹297.47/h; ₹380 for 210 min at ₹108.64/h) were all recomputed and agree to within ₹0.01/h of rounding.
- **F2. Low. Funding paragraph omits T2.27.** Section 5.2 Funding (line 164) lists T3.1, T3.2, T6.1 and T6.4 as unpriced GCP work; rule 14 (line 326) also lists T1.1 (Done) and T2.27. Proposed: list T2.27 (and B1's new task) in the funding paragraph; drop T1.1 from rule 14.
- **F3. Low. Section 5.2 roles under-list H5.** H5(a) uses P3, P5 and P6; H5(c) uses P1 against P3 and P4 against P6. The Role column names H5 only for P5 and P6. Proposed: add "H5a, H5c" to P3 and "H5c" to P1 and P4.
- **F4. Low. Grid rule hard-codes 32,768.** Section 5.3 step 7 applies 32,768 to every model; Llama-3.1-8B (K4) has a longer limit. Proposed: "within the model's position limit (32,768 for Qwen2.5)".
- **F5. Low. Header date.** Line 3 says "Last updated: 19 October 2026"; Appendix A runs to 27 Oct (28 Oct on `origin/main`). Also, the 25 Oct T2.6 line (line 656) sits after the 26 Oct lines. Proposed: update the header on each commit; leave the Appendix A lines in place and note the order.

---

## G. Task statuses against the code

Checked on `9d2d8b1` and `ada59c0`. The modules each Phase 2 row names exist on both commits.

| # | Sev. | Task | Plan says | Code shows | Proposed edit |
|---|---|---|---|---|---|
| G1 | Medium | T6.4 and others | This branch: T6.4 "Not started"; no 28 Oct Appendix A line | `origin/main` (16 commits ahead of the merge base `275faac`) has `attnbench/backends/fa3.py`, T6.4 "Partly built (27 Oct 2026)", a 28 Oct FA3 line, and newer T2.2, T2.12, T2.13 and T2.16 text. This branch's plan lags `main` | Merge `main` into the C-branches before any plan edit, so the plan is not forked |
| G2 | Medium | T2.13 | This branch: strict first-clocked-block stamp "still to come" | `first_clocked_block_started_at` is in `attnbench/ground_truth.py` on `origin/main` (4 hits), absent on `9d2d8b1`. `main`'s plan row matches `main`'s code | Same as G1 |
| G3 | Low | T0.11 | "Partly built (... rate re-check not done)" | T0.17 (Done, 20 Oct) says it is "also T0.11's open rate re-check for P2 and P4"; Section 5.2 has all three rates; the launch scripts carry the 20 Oct rates (`gcp_launch_h100.sh` lines 9 and 97) | T0.11: "Done (20 Oct 2026, through T0.17)" |
| G4 | Low | T0.6 | "Not started" | `docs/archive/` holds the superseded plan PDF, the pre-registration and its gate (moved under T0.10 and T0.16) | "Partly built (T0.10, T0.16 moved ...; remaining: ...)" |
| G5 | Low | T0.5 | "Not started" | No `legacy/` directory on either commit | Consistent; no edit |
| G6 | Low | T0.8 | "Not started" | `.github/copilot-instructions.md` points at `docs/copilot_review_plan.md`, not the plan | Consistent; no edit |
| G7 | Low | T2.26 | "Not started" | `scripts/gpu_suite_record.sh` exists but no launch script calls it | Consistent; no edit |
| G8 | Low | T2.5 | five mechanisms built, two to come | `SYNC_MECHANISMS` in `recursion.py` lists five; the blocking host-to-device copy and event synchronise are named as to come | Consistent; no edit |
| G9 | Low | T1.7, T1.8, T1.9, T2.23 | "Not started" | No window or vertical-slash backend in `attnbench/backends/`; no recall-matching or random-control module; `use_triton` appears only in the XAttention backends and the calibration script, with no gate code | Consistent; no edit |
| G10 | Low | Section 5.2 teardown | "From 25 Oct 2026 the teardown script has no default" | `scripts/gcp_teardown_session.sh` lines 38 to 40 refuse without `GCP_RATE_INR_HR` | Consistent; no edit |

---

## Summary of proposed edits by priority

1. Break the T3.1 cycles: A1, A2, A3, A4, A7.
2. Schedule the accuracy and recall sessions: B1.
3. Bring the body in line with Appendix A and the code: D2, D3, D5, B6.
4. Fix the H4 family count: D4.
5. Complete T4.1: C1 to C3, then C4 to C8.
6. Merge `main` into the C-branches before editing the plan: G1, G2.
