# Threats to validity (draft, 28 Nov 2026)

The paper-quality wave of 28 Nov 2026, item 6. This is a draft list for the paper's threats section. Each threat names where the repository already records it. Numbers marked EXPLORATORY come from `docs/paper/exploratory/` or `docs/exploratory/`, never from the scored results.

## Construct validity (does the study measure what it claims?)

1. **The timed region is one prefill forward to the last position's logits, at batch 1 or 4.**
   - Decode, tokenisation, chunked prefill and serving-engine scheduling are outside it, so "end to end" means end of prefill.
   - Sources: `docs/paper_claims_audit.md` (a); `docs/limitations.md` "SCOPE, before anything else".
2. **"Dense" is not one kernel across pairings.**
   - P4's ground truth ran FA3 while its prediction priced `sdpa_flash`, a fault in the committed prediction (H1 not rescored). P3's selection chose `sdpa_flash` at short lengths and FA2 at long ones.
   - So the reference dense time differs in kind across pairings.
   - Sources: A(17 Nov 2026), A(26 Nov 2026).
3. **The oracle is a non-deployable reference.** Its scoring is outside the clock, so oracle rows are not deployable speedups (`docs/paper_claims_audit.md` (a)).
4. **Admissibility rests on one card's code path.**
   - Calibrated XAttention is certified only on the L4 PyTorch path. The Triton path fails the mask gate on P2.
   - So no pairing has a certified profitable set, which is why H6 is not scored (A(14 Nov 2026)).
5. **The accuracy rows behind the 32768 certification include rows decoded past the model's position limit.**
   - 19 to 21 of 100 `qa_1` rows. The post-hoc sensitivity without them is reported beside, not instead of, the certified bound (`docs/paper_claims_audit.md` (b)).
6. **The tasks follow RULER's algorithm, not RULER's benchmark.** Scoring is more permissive than Sparse Frontier's (`docs/limitations.md`).

## Internal validity (could something else explain the result?)

7. **Host components from the null-kernel pass overstate host cost.**
   - Priced host alone exceeds the whole measured time in 17 of 35 P2 cells and 23 of 35 P4 cells (EXPLORATORY).
   - The H100 diagnostic's Q4 puts measured / priced host at 0.60–0.65. The cause is unconfirmed; the S4 null-pass / recorded ratio of 2.1–2.5x is a candidate.
8. **Device busy and idle times come from traced runs.**
   - Traced device time inflates by up to about 7 µs per kernel on P2 at 1951 tokens (under 0.2 µs on P4).
   - No untraced host timeline exists, so measured idle time cannot be split into host, queue and synchronisation (`docs/exploratory/residual_20261122.md`; `docs/paper/exploratory/paper_tables.md`).
9. **Round-level uncertainty is within one session.**
   - Per-cell intervals (median width 0.5% on P2, 0.9% on P4) and the 58-cell paired sign check describe noise within a session. They do not cover session, instance or provider variance (`docs/paper/counts_and_uncertainty.md`).
10. **Clocks are not locked on every host**, and the host's turbo and frequency state is not controlled (`docs/limitations.md` "Clocks are not locked").
11. **P2 and P4 time eager attention only.** The compiled groups were dropped, so nothing scored tests what compilation leaves on the GPU (A(12 Nov 2026)).
12. **Decisions after data, all disclosed:**
    - the 0.02264 dispatcher margin, fixed after P2/P4 cell times were seen;
    - the position-limit exclusion rule (22 Nov 2026, after the 25 Oct certification);
    - this wave's knockouts, FA3 repricing and host scaling.

    All are labelled post hoc and none replaces a scored value.

## External validity (does it generalise?)

13. **One model family at one size scores the H1 cells** (Qwen2.5-1.5B-Instruct, GQA layout (12,2)), with batch 1 for the sparse arms.
14. **Two scored pairings for H1 (P2 and P4).**
    - H0, H2 and H3 could not be scored as pre-registered.
    - P3 is exploratory and was run after the lock; P5 and P6 never ran.
    - Two cloud providers, one GPU per pairing.
15. **Three host CPU families at most** (Cascade Lake, Sapphire Rapids, AMD EPYC on P3), each from one cloud machine type. Host provisioning (vCPU count, Spot or on-demand) is not varied.
16. **One serving stack.** PyTorch eager on one CUDA stream. Results need not hold for vLLM or SGLang, CUDA graphs end to end, continuous batching or multi-GPU tensor parallelism.
17. **Estimators as shipped.** XAttention's released code (upstream `e379887`) and in-house mean-pool and oracle builders. Synchronisation counts are properties of these implementations, not of the methods.

## Conclusion validity (are the inferences supported?)

18. **No significance test is pooled across pairings**, by the plan. With two scored pairings, H1's failures are per-pairing verdicts, not a population claim.
19. **The step form and the fitted sum beat the recursion on both scored pairings.**
    - A paper must not present the recursion's ordering of synchronisations as the source of accuracy.
    - The knockouts (EXPLORATORY) show waits and queue move P4's error by under 0.001.
20. **Citations:**
    - Three brief references could not be verified on 28 Nov 2026 and must be dropped or confirmed (`docs/paper/related_work_check.md`).
    - Read-in-full status (rule 5) is still open for most entries.
