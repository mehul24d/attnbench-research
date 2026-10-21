# Erratum for `methodology.pdf`

The PDF is the 30-page deliverable at
`/Users/mehuldahiya/attnbench_deliverables/methodology.pdf`, outside the
repository. **It has not been edited.** This file lists what it states that
the project's own data now contradicts or qualifies. Page numbers are the
PDF's. Items 1–8 were listed on 2026-10-03 (fourth pre-registration draft);
items 9–10 were added the same day (fifth draft).

## A100 figures quoted against `sdpa_flash`

The A100's fastest correct dense kernel at the model's (12, 2) geometry is
cuDNN at 8192 and fa2 at 16384, not `sdpa_flash` (`results/s7_sweep_hl122`).
Corrected figures below are **estimates** until run D
(`docs/estimator_frontier_preregistration.md` §4.6) measures the kernels in
one session. Derivation and test: `docs/claims.md`,
`tests/test_a100_baseline_caveat.py`.

1. **pp. 1, 19, 21.** The end-to-end 1.201× / 1.282× (and 1.090×) are against
   `sdpa_flash`. Against the fastest correct kernel the estimates are
   1.180× / 1.260× (1.072×).
2. **pp. 11, 19, 21.** The kernel "beats flash" figures 1.28× / 1.96× /
   2.47×. Against cuDNN at 8192 (sparsity 0.75) the kernel is 0.948×. Against
   fa2 at 16384 it is 1.839× / 2.321×.
3. **pp. 13, 19.** "`sdpa_flash` … the strongest dense baseline in this
   study" is false. It is the slowest of the three correct candidates at
   8192 and 16384.
4. **p. 15.** "End-to-end speedups at 8,192 are 1.03–1.06× while the kernel
   gives 1.28×". Against cuDNN, the estimates are 0.962–0.995× end to end, below
   1 at every sparsity, and 0.948× for the kernel.
5. **pp. 7, 10, 19, 21.** The 0.475× is against `sdpa_flash`.
6. **pp. 2, 13.** Figure 5 and `data/vectorised_builder_counterfactual.csv`
   are against `sdpa_flash`.

## Accuracy at 32768, and context lengths

7. **pp. 7, 9, 19, 21.** "1.321× at 32,768 with no accuracy cost (100.0 vs
   100.0)" needs the positions caveat:
   - 148 of the 200 `stage3_32768` rows decode past Qwen2.5's 32,768-position
     limit;
   - 12 of them have 32,769-token prompts, so prefill itself crossed it.

   Source: `results/positions_over_limit/`, from
   `scripts/flag_positions_over_limit.py`. Every 32768 accuracy citation in
   `claims.md` carries this caveat.
8. **The PDF predates** the A100 32768 results of 2026-10-01.
9. **p. 1, "Context 2,048 to 32,768 exact tokens"** (added 2026-10-03). Each
   row's context length is the exact *measured* count. It is not the band,
   and not always at or under it.
   - **Over budget:** 182 distinct banked examples land above their band
     (655 file-example pairs across 31 parquets), up to **+82** tokens
     (`niah_multikey` at 16384). `niah_single` reaches +1 at 32768, which
     gives the 32,769-token prompts.
   - **Cause:** tasks outside `ruler._PER_EXAMPLE_FIT` are sized once per
     budget on example 0. Later examples whose UUID or number needles
     tokenize longer overshoot.
   - **Not an erratum:** p. 5's "random token ids at exactly the band
     length" is correct. The phase decomposition does not use RULER prompts.
   - **The same wording in the repository,** checked 2026-10-03 for
     "token-exact" and "never above budget":

     | where | wording | disposition |
     |---|---|---|
     | `docs/limitations.md`, "Context lengths are exact token counts" | "a grid length means what it says, to within ~1%" | dated note added |
     | `docs/writeup_input.md`, scope table | "exact token counts" | dated note added |
     | `scripts/reestimate_stage3.py` docstring | "token-exact (0.9996–0.9999× of budget)" | dated note added |
     | `attnbench/accuracy/sizing.py`, `fit_units_to_budget` | "never above" | clarified: it holds only for the render the fit was given |
     | `attnbench/accuracy/ruler.py`, `RulerExample` and `generate_examples` | "never above" / "each example's filler" | corrected, dated |
     | `docs/retry_session_runbook.md:171`, `docs/limitations.md:797` | "token-exact sizing" | names the method, accurate; unchanged |
     | `tests/test_ruler_t4_tasks.py:12` | "never exceed their budget" | true of the per-example tasks it tests; unchanged |

## H100

10. **p. 1** (added 2026-10-03). "H100 sessions appear in the spend ledger
    for kernel-level work, and no claim rests on them" is out of date. The
    pre-registered H100 overlap test ran on 2026-10-01, and `limitations.md`
    now reports H100 in-model speedups. Those are against `sdpa_flash`, and
    their baseline strength is **unmeasured**: no banked H100 file times
    another dense kernel at (12, 2). The caveat is `limitations.md`, "The H100
    test", tested by `tests/test_h100_baseline_caveat.py`. The PDF quotes no
    H100 speedup itself, so nothing else in it needs correcting for this.
