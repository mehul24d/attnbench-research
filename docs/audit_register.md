# Audit register

**What this is.** One row per audit item, with what it asked, what it looked
at, what it concluded, and where the evidence is. It exists because items were
being cited by number — "audit item S1a", "item S5", "item S6" — across
`claims.md`, `limitations.md` and `silent_failure_patterns.md` with **no list
anywhere of what the items were**. A finding referenced by an identifier that
resolves to nothing is not auditable: a reader cannot check the disposition,
and cannot tell an item that was examined and cleared from one that was never
run.

**Written 2026-09-20, retrospectively, by an external audit.** The 2026-09-19
single-measurement audit and the 2026-09-16→20 review that preceded it were
conducted without a register. Everything below is **reconstructed from commits,
result directories, the spend ledger and the prose that cites each item** —
not transcribed from a plan, because no plan file exists. Rows are marked
accordingly. Reconstruction is why the gaps in §3 are gaps rather than
omissions: they may be items that never existed.

**The rule going forward.** An item gets a row here *before* it is run, with
its question written down, and the disposition filled in after. An item cited
in prose and absent here is a bug in this file.

---

## 1. Items with a recorded disposition

| item | question | evidence | disposition |
|---|---|---|---|
| **S1a** | The 2048–8192 accuracy was measured on masks that did not force the attention sink. Does forcing it move the published cells? | `results/s1a/` (n=100, 3 bands, decode pinned to `sdpa_math`, dense canary 300/300); sessions `attnbench-l4-s1a-20260919-1939` (₹52) and `attnbench-l4-s1a2-20260920-0437` (₹143); commits `e739002`, `179c894`, `4d1c6bd`, `44ab65c`, `ba58cb4`, `9b336c8` | **Confirmed, large.** Cells moved by up to 44 points. Forced Stages 4/6/7 to be rebuilt (`a349d9b`, `28fa00a`). Withdrew the `vt` sparse-above-dense margin at 0.75, the "multikey breaks at 0.5" sentence, and the convergence claim. Produced patterns 43 and 44. |
| **S1a-control** | Is the S1a gain the *sink*, or just the extra block the sink fix also grants? | `results/sink_control/` (arbitrary free block at identical per-row block count, same examples, same pinned decode, dense canary 300/300); commits `b1afd85`, `33598b4`, `4feab9b`; ₹33 | **Resolved: mostly the sink.** Of the 34 / 44 / 44.6-point gain at 2048/0.9, an arbitrary block accounts for 6 / 9 / 13.2 and the sink's identity for 28 / 35 / 31.4. |
| **S5** | Arms within one run rank from different score precisions (cold-cache fp32 for the 0.5 arm, fp16 for 0.75/0.9). Does that break nesting? | `scripts/check_score_dtype_asymmetry.py`; `results/diagnostics/20260919_score_dtype/` (bands 2048/4096/8192, both mask rules); commits `edafe1a`, `120ad64` | **Cleared as asked, and the question was too narrow.** fp32/fp16 masks differ in 0.07–0.19% of active blocks and nesting held in every sampled layer-mask. **16384 was not sampled and a structural argument stood in for it — that argument was wrong.** See §2. |
| **S6** | Could the banked 1.5B cheap-estimator run have loaded score tensors written by the "cheap scorer returns zeros" bug (`562374a`)? | GCS object creation times against run windows; `silent_failure_patterns.md` #43 | **Cleared, but the intended evidence was invalid.** GCS creation time records when an object was *copied*, not computed. The conclusion stands on other grounds; the method does not, and is recorded as pattern 43. |
| **(unnumbered) 2026-09-19 fixes** | Four statements the audit found wrong or unmarked: sink density, clock status, the 8192 claim, the pre-fix era | commit `f7560aa` | **Corrected in place.** Sink fix does change block counts (+2.3% at the headline cell, +53.8% at 2048/0.9); `clocks_locked` splits by stage rather than being False everywhere. |
| **S10** | `score_source="dense_softmax_fp32"` is a label; is it enforced? | `attnbench/numerics.py`; guard in `compute_importance_scores`; `tests/test_numerics_precision.py` | **Closed 2026-09-20, by enforcing rather than measuring.** It was not enforced: `model.py:201` is a TF32-eligible fp32 CUDA matmul and nothing set `allow_tf32`. Banked runs were almost certainly true fp32 — by torch's default, not by this code, and `pyproject.toml` permits versions on both sides of that default. Now pinned by every result-producing script and asserted where the claim is applied, so a row carrying that label **cannot** be produced under TF32. No `Provenance` field added: it could only be RECORDED, which is instance 10. |
| **S?-external** | Adversarial external audit of the whole repository, 2026-09-20 | `masks.py` fix + 9 new tests; `silent_failure_patterns.md` #45; corrections across `claims.md`, `limitations.md`, `README.md` | **Found instance 45** (sparsity-dependent jitter stream), withdrew the timed-region coverage claim, and corrected six documentation defects. See §2. |

## 2. What the register itself surfaced

Two items were recorded as cleared and were not, and both were found only
because someone re-derived the disposition instead of reading it:

**S5 cleared the question it asked, not the question it was standing in for.**
It measured fp32-versus-fp16 nesting at three bands and found none broken.
`limitations.md` then extended that to 16384 with a *structural* argument,
which assumed the sparsity arms break score ties identically. They did not
(instance 45). The measurement was sound; the extrapolation from it was the
defect, and it lived in the gap between "what was measured" and "what was
claimed" — exactly the gap a register exists to keep visible.

**S6 reached the right answer by an invalid route.** Recorded as pattern 43 at
the time, which is the correct handling — but the row above is the first place
the *disposition* and the *defect in its method* appear together.

## 3. Numbers cited nowhere, and one inference this file corrects

**There is no S1, S2, S3 or S4 in any document, commit or result path.** The
numbering starts at `S1a`. Either items S1–S4 existed and left no trace, or the
scheme was never sequential. **This cannot be resolved from the repository**,
and it is recorded as unresolved rather than guessed.

**The `results/s7_*`, `s8_*`, `s9_*` directories are not audit items.** The
2026-09-20 external audit initially read them as implying a register S1–S9.
They are session/step numbering from the A100 work — `s7_7b_16384` (7B oracle
arm), `s8_vec_endtoend` (vectorised builder), `s9_7b_cheap_16384` (7B cheap
arm) — and they do not line up with the spend ledger's own "Session 10/11/12"
headings either. The prefix is ad hoc and drifted: `run_vectorised_endtoend.py`
still defaults `--out` to `results/s7_vec_endtoend` while the banked data is in
`results/s8_vec_endtoend`.

**A directory prefix is not an identifier.** Two schemes (`S*` for audit items,
`s*_` for sessions) differing only in case, neither registered, is how a reader
infers a register that was never there.

## 4. Closed by the 2026-09-20 L4 session (`attnbench-l4-s7-20260920-2158`)

One session, three items. Boot 16:28Z, last phase 17:51Z, Rs 172 (of which
Rs 64 was idle — see `spend_ledger.md`). The suite phases ran first, which is
why S8 and S9 have answers at all: they cost minutes and they found more than
the measurement did.

| item | question | disposition |
|---|---|---|
| **S7** | Did instance 45 move any published accuracy number? | **Yes, one, by 6 points.** 16384 re-measured under the fixed tie-break, cold cache, decode pinned to `sdpa_flash` (the kernel those rows carry — *not* the `sdpa_math` the 2048–8192 bands used). Two gates passed: dense canary **200/200 identical**, score canary **200/200 score tensors bit-identical**, so the scoring pass is unchanged and every flip is the mask rule's. `niah_single` holds at **100.0** at all three sparsities — the headline cells survive — but 14 / 2 / 2 of its *texts* changed, so it is stable because it is at ceiling, not because the masks agree. `niah_multikey` moves **66/59/20 → 65/53/17**; at 0.75 that is **−6.0, 95% CI [−12.0, −1.0]**, a CI excluding zero. The fixed builder scores **lower**, so the banked figure was flattered. `vt` not re-measured (stopping confound). Evidence: `results/s7_jitter/`, `gs://…/l4-s7-jitter-16384/`. |
| **S8** | Do `block_sparse` and `sdpa` rebuild setup inside the timed region? | **Yes for `block_sparse`; `sdpa` cannot be observed at all.** First capture of this test's GPU output. `block_sparse` reports `setup ops per call [1, 1, 1]` — [`backends/block_sparse.py:104`](../attnbench/backends/block_sparse.py#L104) calls `mask.to_block_sparse_attn_mask()` on **every forward**, and `BlockSparseMask` requires `active` to be a **CPU** tensor (`masks.py:41,58`), so each call does a host-to-device copy of the block grid. Stage 2 times every call. Same shape as the `create_block_mask` defect this test was written for, which cost a 10× error. `sdpa` reports `not runnable here: UnsupportedConfig` on the `causal` config **on a CUDA machine**, so its timed region is checked nowhere. Magnitude is **not** measured — opened as S11. |
| **S9** | Is `results/stage5/phases.parquet` covered by any test-suite run? | **The record now exists, and it is not green: 4 failed, 977 passed, 20 skipped.** Two findings, both invisible on the workstation. (a) **`procmatch.sh` is broken on Linux** — three of its own tests fail there and pass on macOS; reproduced directly: `status` against a pattern matching nothing answered `RUNNING pids=10710`, and `kill` printed `killing -TERM: 3697`. Both exclusion filters fail **open** on a pid that has exited. Fixed in `4ab382f` with tests that construct the phantom deterministically. No call site, so no measurement is affected. (b) **`test_cheap_scoring_pass_propagates_faithful_hidden_states` errors on the instance** (`AttributeError: 'NoneType' object has no attribute 'to_legacy_cache'`, transformers 4.46.0) while passing on the workstation's newer transformers — opened as S13. |

**What this session actually demonstrates about the register.** S7 was the
expensive item and it confirmed the headline. S8 and S9 were the free ones
bolted onto the same instance, and they found a defect in a timed region and a
guard that reproduces the incident it was written to prevent. **Both had been
"open" for sessions because nobody had kept the output of a test that was
already failing.**

## 5. Open

| item | question | why it is open |
|---|---|---|
| **S11** | How much does `block_sparse`'s per-call `to_block_sparse_attn_mask()` add to its measured time? | **Bounded from banked data, 2026-09-21, at no GPU cost; the bound rests on one assumed constant.** The transferred object is a bool grid of (n_blocks x n_blocks) bytes: **0.06 KB at 1024/128 up to 4 KB at 8192**, and 16 KB at 16384. At those sizes a pageable host-to-device copy is dominated by fixed launch cost, not bandwidth, so the tax is roughly **constant per call** and its fraction is set by how fast the call is. Against the banked `latency_ms_p50` and an assumed **15-30 us** per copy: worst case **1.9-3.7%** at the fastest Stage 2 cell (1024, 0.801 ms), **0.8-1.5%** at 8192, and **0.8-1.7%** at the A100 16384 cells that carry the 1.090x / 1.201x / 1.282x claim (block_sparse p50 3.373 / 2.266 / **1.795 ms**). So 1.282x is a **floor understated by at most ~1.7%**, and no published claim is overstated -- the tax makes block_sparse look slower, so every block_sparse-over-dense ratio is low. **The 15-30 us is assumed, not measured, and it is the whole bound.** Assuming a cost across regimes is instance #15's exact shape, so this is an estimate with a named weak point, not a disposition. Measuring it is ~2 minutes of any CUDA machine (time `active.to('cuda')` for a small CPU bool tensor), so it should **ride along on the next session** rather than book one. The hoist is worth doing regardless, and re-timing to benefit from it is a separate, larger question. |
| **S12** | What is the oracle-versus-estimator gap at 16384 under the fixed mask builder? | S7 re-measured the **oracle** arm at that cell (`niah_multikey` 66/59/20 → 65/53/17). The estimator arm is still era 2, and the jitter fix is in the builder **both** arms use, so the published +32/+40/+19 cannot be recomputed from one side. Needs the cheap arm re-run at the same cell. The gap's *direction* is not in question; its magnitudes are. See `claims.md`. |
| **S13** | Why does the cheap-scoring-path test error on transformers 4.46.0? | It passes on the workstation and errors on the image the measurements run on, so the cheap scoring path has **no working test where it executes**. `pyproject.toml` permits both versions. Same family as S9(a): green here, broken there. |
| **S14** | Is `sdpa`'s timed region observable anywhere? | S8 found it `UnsupportedConfig` even on CUDA, so the study's dense reference arm has never had its timed region checked. Fix the config that stops it running rather than accepting the skip. |
