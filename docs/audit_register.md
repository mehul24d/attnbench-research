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

## 4. Open

| item | question | why it is open |
|---|---|---|
| **S7** | Did instance 45 move any published accuracy number? | The fixed mask builder changes 5.6% of (layer, sparsity) masks and 0.23% of active blocks. Whether that moves a score is unmeasured. Needs one GPU band at 16384 — the headline band, and the one where 76 of 100 examples have a ragged final query block. Until it runs, **no banked file is era 3**: see `limitations.md`, "Which mask era each banked accuracy file belongs to". |
| **S8** | Do `block_sparse` and `sdpa` rebuild setup inside the timed region? | `tests/test_timed_region_setup.py` reaches neither on CPU, the two session logs that ran it on a GPU both record it **failing**, and no log records it passing. Needs one instance session capturing the full failure output per backend. See `limitations.md`, the withdrawn coverage claim. |
| **S9** | Is `results/stage5/phases.parquet` covered by any test-suite run? | Its session log (`results/gpu_session_20260907_stage5/stage5.log`) contains no suite record at all, and it is the source of every `normalized_ms` — the latency axis of Stage 6 and all of Stage 7. |

