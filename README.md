# attnbench

**Do sparse attention kernels' speedups survive to end-to-end latency at
matched accuracy?** A measured answer, and the failure patterns found getting
there.

Published sparse-attention speedups are kernel numbers. This study measures
what happens to them when the kernel is put inside a real model, on real
long-context tasks, at operating points that preserve accuracy — and prices
the thing every such speedup leaves out.

## The result, and its ceiling

> **On an NVIDIA A100, training-free block-sparse prefill beats `sdpa_flash`
> at 16384 context — 1.201× at 0.75 sparsity, 1.282× at 0.9 — but only with a
> vectorised mask builder, which the reference implementation does not
> have.** At 8192 the vectorised builder reaches parity; 32768 was not
> measured with it. With the reference builder the same
> configurations run at 0.633× and 0.956×. The model's outputs are bitwise
> identical under both builders; the difference is one function, ~91% of whose
> cost is Python interpreter overhead.

That is the fifth version of this sentence. The first four each fell to a
second point on an axis the previous version had sampled once — card, model
scale, kernel mechanism, and a disjunctive "weak baseline or fast builder" —
and [`docs/writeup_input.md`](docs/writeup_input.md) leads with that history
because it is the reason to trust version five.

**The ceiling is someone else's, and it is high.** Native Sparse Attention
([arXiv:2502.11089](https://arxiv.org/abs/2502.11089)) reports 9.0× forward
and 6.0× backward against FlashAttention-2 at 64k on A100. Any reading of this
repo as "sparse attention doesn't pay" is wrong. NSA is natively trainable;
everything here is training-free, applied to weights trained for dense
attention. This study sits between *The Efficiency Misnomer*
([arXiv:2110.12894](https://arxiv.org/abs/2110.12894)), which established that
theoretical and realised efficiency diverge, and NSA's ceiling — and locates
the divergence for one family: not in the kernel, but in mask construction.

**The oracle is still the binding constraint.** Every accuracy number above
uses a dense-softmax importance oracle that costs ~35× the latency it saves.
The deployable alternative was measured — MInference's mean-pool estimator
([arXiv:2407.02490](https://arxiv.org/abs/2407.02490), Algorithm 3) — and on
multi-key retrieval at 16384 it trails the oracle by **+32 / +40 points** at
0.5 / 0.75 sparsity on Qwen2.5-1.5B, and by **+39 / +67** on Qwen2.5-7B. The
gap *widens* with scale. So the speedup is real, available with a small
engineering fix, and bought with a mask nobody has yet shown how to afford.

**On an L4 with the reference builder**, the same pipeline reaches **1.321×
end-to-end at 32K context at zero accuracy cost** (100.0 vs 100.0):

| context | best speedup at no accuracy cost | oracle cost ÷ saving |
|---|---|---|
| 2048 | 0.984× (sparsity loses) | undefined — nothing is faster |
| 4096 | 0.997× | undefined |
| 8192 | 1.044× | **49×** |
| 16384 | 1.186× | **36×** |
| 32768 | **1.321×** | **35×** |

The oracle ratio flattens near 35 rather than heading toward 1 — scoring and
the saving grow at similar rates, so the gap is structural, not a small-scale
artefact.

*The 2048 row read "0.993× (sparsity loses)" until 2026-09-21. That is the 0.75-sparsity cell,
which scores 99.0 — so it is the best speedup at a SMALL accuracy cost, in a
column headed "at no accuracy cost". The only 2048 cell at 100.0 is 0.5
sparsity, at 0.984×. Forcing the sink does not change which cell qualifies:
the 0.90 row moves from 62.0 to 97.0 and still does not reach 100. The
correction makes the row worse and the column's own rule is what requires it.*

## The five gaps this targets

1. **Kernel speedup vs end-to-end speedup.** 1.24× at 90% sparsity as a
   kernel; ≤1.06× and usually <1.0× end-to-end at accuracy-matched points
   below 8K. Both ends measured, on the same hardware, in the same repo.
2. **Random masks vs importance-derived masks.** Timing under a random mask
   says nothing about accuracy. Stage 2 uses random masks to isolate the
   kernel; Stage 3 uses real importance-derived ones. Every row records which.
3. **The estimator's cost, which published speedups exclude.** Priced here as
   a first-class result rather than a limitation.
4. **Accuracy and latency measured apart vs at matched accuracy.** A speedup
   at an operating point that loses accuracy is not a speedup — and on two of
   three tasks the honest comparison turns out to be unreachable at any
   affordable sample size, which is a finding in its own right.
5. **One card vs two architectures.** Every ratio is remeasured against a
   baseline on the same machine; nothing is carried across hosts.

## Scope and cost, stated up front

Measured on **rented NVIDIA L4 (24 GB) and A100-SXM4 (80 GB)** instances —
two architectures, not a survey. H100 sessions also appear in the spend
ledger (kernel-level Stage 0–2, and one session lost entirely), but no row
in `docs/claims.md` rests on them. **One model family at two sizes**
(Qwen2.5-1.5B-Instruct throughout; 7B for the accuracy arms at 16384), **one
task family** (RULER-style NIAH and variable tracking), **batch 1**,
**prefill-only sparsity**, **inference only**. Context tops out at **32768
tokens**, which is a hardware ceiling, not a design choice.

Total rented GPU time: **₹8,279 itemised** (≈ US$94 at the ₹88/$ the ledger
prices its own audit-log rows at), across 34 priced sessions in
[`docs/spend_ledger.md`](docs/spend_ledger.md), plus one early validation
session recorded only in prose. The figure is summed from the ledger's table,
not restated here — this line previously said "roughly ₹6,000, of which ₹2,413
is itemised", and by 2026-09-19 the itemised part exceeded the total it was
said to be part of. The ledger says which rows come from an instance's own
boot clock or the audit log and which are reconstructed, and records ₹218 that
a green test spent by creating real instances on every suite run.

[`docs/limitations.md`](docs/limitations.md) is over 2,300 lines and is not
decoration. Two of this study's questions are **permanently unanswerable on
this hardware**, and it says which and why.

---

## Stages

| Stage | What it produces | Hardware |
|---|---|---|
| 0 | Capability matrix: what each backend *claims* vs actually supports (the claimed/actual split follows CAB, [arXiv:2210.07661](https://arxiv.org/abs/2210.07661)) | any CUDA GPU |
| 1 | Correctness gate vs a float64 reference | any CUDA GPU |
| 2 | Kernel microbenchmarks (synthetic, random masks) | locked clocks, exclusive |
| 3 | End-to-end accuracy on RULER-style tasks | 24 GB+ |
| 4 | Matched-accuracy operating points (non-inferiority + bootstrap) | derived |
| 5 | Phase decomposition: prefill, decode step, scoring | locked clocks, exclusive |
| 6 | Pareto frontiers per grid cell | CPU only |
| 7 | Decision map | CPU only |

*Stages 4, 6 and 7 were rebuilt on 2026-09-20 from forced-sink accuracy
(audit item S1a) and now occupy `results/stage4|6|7`. The versions published
before that date are correct measurements of a mask the current code does not
build, and are kept under `results/_superseded/` — see `docs/claims.md`, "The
derived stages rebuilt on forced-sink accuracy".*

Only Stages 2 and 5 need rented, clock-locked, exclusive GPUs. Everything
else runs on free-tier hardware or a laptop.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q     # 1109 passed, 35 skipped, ~65s, no GPU
```

The `[dev,eval]` extras are required, not optional: four test modules import
`transformers` at collection time, so a bare `pip install -e .` cannot even
collect the suite. The `kernels` extra is deliberately separate — those
compile from source and are architecture-gated, and the harness must import
on a machine where a kernel is unavailable.

The suite runs on CPU and needs no GPU, no model download, and no
credentials. Verified from a clean clone into a fresh virtualenv on
2026-09-12, resolving dependencies from scratch.

The 35 skips are the honest part, and they split four ways: **3** need CUDA,
**24** read banked result files that `results/` correctly keeps out of git,
**4** are scripts `test_script_call_sites.py` has nothing to check because
they import nothing from `attnbench`, and **4** are the same scripts skipped
again by `test_import_resolution.py`, which only has something to say about a
script that imports the package. The 24 validate real measured data —
including the check that the cross-arm decode guard actually fires on the
confounded Stage 3 rows, and the check that the two comparison scripts refuse
the banked era-2/era-3 pair, and the check that segment 1's 2026-09-03 probe
still carries the eager-flex fingerprint — so a fresh clone is green **without** running
them. Each skips with a message naming the file it wanted, rather than passing
silently.

*That third number was 6 until 2026-09-21. It is 4 now because
`run_scale_comparison.py` and `run_scorer_comparison.py` acquired the mask-era
check and therefore import `attnbench` for the first time — two scripts moved
out of the "nothing to check" bucket by being given something to check.*

Those counts are for a fresh clone. **With `results/` present the suite reads
1133 passed, 11 skipped**, because the 24 banked-file tests run instead of
skipping. Measured on transformers 5.17.0 and again on **4.46.0**, the version
the 2026-09-20 instance ran: identical, test for test. *(This paragraph said "The 11 skips … 2 need CUDA, and 9 read banked
result files" against a code block saying 10 skipped, while the real numbers
were 23 and 9. Three figures for one quantity, none of them measured;
`tests/test_doc_derived_numbers.py` now asserts the two here agree with each
other and with their own breakdown.)*

Then, on any CUDA GPU:

```bash
.venv/bin/python scripts/run_probe.py --out results/probe   # Stages 0 and 1
```

## The documents that matter

- **[`docs/claims.md`](docs/claims.md)** — the ledger every write-up is
  drafted from. One row per claim, pairing the sentence the data supports
  with the near-paraphrase it must not become. If a sentence is not in the
  supported column, this study does not license it.
- **[`docs/limitations.md`](docs/limitations.md)** — what the measurements
  cannot answer, including two questions that are permanently out of reach on
  this hardware and the variance calculation proving it.
- **[`docs/writeup_input.md`](docs/writeup_input.md)** — what was measured,
  on what, and what each number licenses, organised by the five gaps. Drafted
  from `claims.md` with every claim's boundary attached.
- **[`docs/silent_failure_patterns.md`](docs/silent_failure_patterns.md)** —
  52 confirmed incidents, each one a plausible number produced by machinery
  that looked like it was working. No crash, no failed test. Several changed
  a published figure. Each entry records the detection method, which is the
  transferable part. #45 is the first found by someone who did not write the
  code, and it had survived the full suite plus a diagnostic written to test
  the exact property it broke.
- **[`docs/withdrawn_figures.md`](docs/withdrawn_figures.md)** — every
  figure this study has withdrawn or superseded, paired with what replaced it.
  `tests/test_no_stale_figures.py` reads it and fails if one of them appears
  anywhere as a live statement. Added 2026-09-21 because five separate defects
  in that audit were one defect: a correction applied in `claims.md` and
  propagated nowhere else.
- **[`docs/audit_register.md`](docs/audit_register.md)** — one row per audit
  item: the question, the evidence, the disposition. Added 2026-09-20 because
  items were cited by number in three documents with no list of what they
  were, which meant a reader could not tell an item that was examined and
  cleared from one that was never run. Two rows in it record dispositions
  that did not survive being re-derived.

Also: [`docs/hardware_constraints.md`](docs/hardware_constraints.md),
[`results/stage3_s1/INVALID_ROWS.md`](results/stage3_s1/INVALID_ROWS.md).

## Measurement discipline

- No result row is written without a provenance stamp. Measured rows carry
  the GPU, driver, clock state and commit; derived rows carry `analysis_*`
  fields naming the tool and checkout that produced them. The two are
  deliberately not the same columns.
- **Clocks locked on every Stage 5 run; NOT on any Stage 2 run.** This line
  previously claimed both, and the provenance column said otherwise on every
  row: `clocks_locked` is `True` for all Stage 5 data (L4 1740 MHz, A100
  1200 MHz — 85% of each card's maximum, by policy) and `False` for every
  Stage 2 sweep ever run, on all three cards, because `run_sweep.py` has no
  clock-lock flag. Stage 2 ratios are within-host and the canary bounds the
  variance, so the data stands; the claim did not. GPU exclusivity is verified
  for both.
- Speedups are recomputed against a baseline **remeasured on the same
  machine**. Ratios are never carried across hosts.
- **Replication on the same architecture confirms a measurement and says
  nothing about its scope.** The 1.321× headline was replicated on a second L4
  to within 0.19% — and that replication was *structurally incapable* of
  detecting that the claim was L4-specific, which Stage 5 on an A100 later
  showed it was (0.475× at the same configuration). Two measurements agreeing
  is evidence about precision, not generality, and the tighter the agreement
  the more confident the wrong conclusion looks. A result is scoped by the
  axes it was **varied** across, never by the number of times it was repeated
  along one.
- Grid cells run in randomised order, so thermal drift cannot correlate with
  backend identity.
- Repeats live in separate sessions, not separate loops in one process.
- A backend that cannot run a config raises `UnsupportedConfig`. Nothing
  silently falls back to another implementation.
- Qualifiers travel as columns, not prose: `haystack_mode`, `score_source`,
  `gate_source`, `decode_backend`, `clocks_locked`. A summary drops a caveat;
  a column does not.

## Two things worth knowing before comparing these numbers to anything

**Absolute accuracy here is not comparable to published RULER or Sparse
Frontier numbers.** Stage 3 uses RULER's task-construction algorithm with a
dependency-free filler-text haystack in place of real prose, and numeric/UUID
needles rather than word needles; the scoring is also more permissive.
Filler text is measurably easier to search, so these numbers read high. They
remain valid for what the study does — comparing backends against each other
on identical inputs. Every row records `haystack_mode` so the substitution
travels with the data. See [`NOTICE`](NOTICE) and
`attnbench/_vendor/ruler/VENDORED.md`.

**The masks are an oracle.** Importance scores come from a full dense
attention pass, so accuracy figures are an upper bound and the estimator's
cost is excluded from every latency number — the same methodological move
this study criticises elsewhere, made deliberately to isolate the kernel, and
priced in the table above rather than left implicit. Every row records
`score_source`.

## Adding a backend

Subclass `AttentionBackend`, declare a `Capability`, implement `forward`, and
decorate with `@register`. Inputs and outputs are always
`(batch, n_heads, seq_len, head_dim)`. Any transposition or KV expansion the
kernel needs happens inside `forward`, and its cost counts toward that
backend — deliberate: a kernel with no native GQA path genuinely does cost
more on a GQA workload.

## Licence

Apache 2.0 — see [`LICENSE`](LICENSE). Third-party code and adopted
approaches, with commits and modifications, are recorded in
[`NOTICE`](NOTICE).
