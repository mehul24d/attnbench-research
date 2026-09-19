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
| 2048 | 0.993× (sparsity loses) | undefined — nothing is faster |
| 4096 | 0.997× | undefined |
| 8192 | 1.044× | **49×** |
| 16384 | 1.186× | **36×** |
| 32768 | **1.321×** | **35×** |

The oracle ratio flattens near 35 rather than heading toward 1 — scoring and
the saving grow at similar rates, so the gap is structural, not a small-scale
artefact.

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

Total rented GPU time: **₹7,879 itemised** (≈ US$85 at the rate implied by
the ledger's own A100 pricing), across 30 priced sessions in
[`docs/spend_ledger.md`](docs/spend_ledger.md), plus one early validation
session recorded only in prose. The figure is summed from the ledger's table,
not restated here — this line previously said "roughly ₹6,000, of which ₹2,413
is itemised", and by 2026-09-19 the itemised part exceeded the total it was
said to be part of. The ledger says which rows come from an instance's own
boot clock or the audit log and which are reconstructed, and records ₹218 that
a green test spent by creating real instances on every suite run.

[`docs/limitations.md`](docs/limitations.md) is over 950 lines and is not
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

Only Stages 2 and 5 need rented, clock-locked, exclusive GPUs. Everything
else runs on free-tier hardware or a laptop.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q     # 900 passed, 2 skipped, ~50s, no GPU
```

The `[dev,eval]` extras are required, not optional: four test modules import
`transformers` at collection time, so a bare `pip install -e .` cannot even
collect the suite. The `kernels` extra is deliberately separate — those
compile from source and are architecture-gated, and the harness must import
on a machine where a kernel is unavailable.

The suite runs on CPU and needs no GPU, no model download, and no
credentials. Verified from a clean clone into a fresh virtualenv on
2026-09-12, resolving dependencies from scratch.

The 11 skips are the honest part: 2 need CUDA, and 9 read banked result files
that `results/` correctly keeps out of git. Those 9 validate real measured
data — including the check that the cross-arm decode guard actually fires on
the confounded Stage 3 rows — so a fresh clone is green **without** running
them. Each skips with a message naming the file it wanted, rather than
passing silently.

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
  38 confirmed incidents, each one a plausible number produced by machinery
  that looked like it was working. No crash, no failed test. Several changed
  a published figure. Each entry records the detection method, which is the
  transferable part.

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
