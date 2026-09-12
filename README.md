# attnbench

**Do sparse attention kernels' speedups survive to end-to-end latency at
matched accuracy?** A measured answer, and the failure patterns found getting
there.

Published sparse-attention speedups are kernel numbers. This study measures
what happens to them when the kernel is put inside a real model, on real
long-context tasks, at operating points that preserve accuracy — and prices
the thing every such speedup leaves out.

## The result, and its ceiling

> **Block-sparse attention with oracle-derived masks reaches 1.321×
> end-to-end at 32K context at zero accuracy cost (100.0 vs 100.0), and
> computing the oracle costs roughly 35× the latency it saves.**

The second clause is not a caveat on the first. **It is the finding.** A
measured advantage bought with a mask nobody can afford to compute is not an
advantage any deployed system has. To make the operating point profitable, a
real importance estimator would have to produce a good-enough ranking for
under ~3% of the scoring pass's cost. This study implements no such estimator
and evaluates none.

| context | best speedup at no accuracy cost | oracle cost ÷ saving |
|---|---|---|
| 2048 | 0.993× (sparsity loses) | undefined — nothing is faster |
| 4096 | 0.997× | undefined |
| 8192 | 1.044× | **49×** |
| 16384 | 1.186× | **36×** |
| 32768 | **1.321×** | **35×** |

The speedup grows with context. So does the accuracy sparsity can tolerate.
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
4. **One card vs two architectures.** Every ratio is remeasured against a
   baseline on the same machine; nothing is carried across hosts.
5. **Accuracy and latency measured apart vs at matched accuracy.** A speedup
   at an operating point that loses accuracy is not a speedup.

## Scope and cost, stated up front

Measured on **rented NVIDIA L4 (24 GB) and A100-SXM4 (80 GB)** instances —
two architectures, not a survey. **One model** (Qwen2.5-1.5B-Instruct), **one
task family** (RULER-style NIAH and variable tracking), **batch 1**,
**prefill-only sparsity**, **inference only**. Context tops out at **32768
tokens**, which is a hardware ceiling, not a design choice.

Total rented GPU time: **roughly ₹6,000 (~US$70)**, of which ₹2,413 is
itemised per session in [`docs/spend_ledger.md`](docs/spend_ledger.md) — the
ledger says which figures come from an instance's own boot clock and which
are reconstructed from prose, and records ₹218 that a green test spent by
creating real instances on every suite run.

[`docs/limitations.md`](docs/limitations.md) is over 950 lines and is not
decoration. Two of this study's questions are **permanently unanswerable on
this hardware**, and it says which and why.

---

## Stages

| Stage | What it produces | Hardware |
|---|---|---|
| 0 | Capability matrix: what each backend actually supports | any CUDA GPU |
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
uv venv && uv pip install -e .
python -m pytest tests/ -q            # 856 tests, no GPU required
python scripts/run_probe.py --out results/probe   # Stages 0 and 1, any CUDA GPU
```

The test suite runs on CPU and needs no GPU, no model download, and no
credentials. Tests that read banked result files skip rather than fail when
those files are absent, so a fresh clone is green.

## The three documents that matter

- **[`docs/claims.md`](docs/claims.md)** — the ledger every write-up is
  drafted from. One row per claim, pairing the sentence the data supports
  with the near-paraphrase it must not become. If a sentence is not in the
  supported column, this study does not license it.
- **[`docs/limitations.md`](docs/limitations.md)** — what the measurements
  cannot answer, including two questions that are permanently out of reach on
  this hardware and the variance calculation proving it.
- **[`docs/silent_failure_patterns.md`](docs/silent_failure_patterns.md)** —
  30 confirmed incidents, each one a plausible number produced by machinery
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
- Clocks locked and GPU exclusivity verified before any Stage 2 or 5 run.
- Speedups are recomputed against a baseline **remeasured on the same
  machine**. Ratios are never carried across hosts.
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
