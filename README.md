# attnbench

Controlled cross-method benchmark of hardware-aware attention mechanisms.

Measures accuracy and wall-clock cost for the same attention implementations on
the same configurations, across hardware generations, and fits a decision map
predicting which mechanism is Pareto-optimal under which conditions.

## Why the two tracks are separate

Kernel microbenchmarks take synthetic tensors and no dataset. Accuracy runs take
a real model and real long-context data with the attention implementation
swapped underneath. These are never mixed, because published speedup figures
that conflate them differ by close to an order of magnitude.

The same split governs sparse masks. Stage 2 uses **randomly generated** block
masks to isolate a kernel's ability to convert sparsity into speed. Stage 3 uses
**real importance-derived** masks, because accuracy under a random mask is
meaningless. Every result row records which.

## Stages

| Stage | What it produces | Hardware needed |
|---|---|---|
| 0 | Capability matrix: what each backend actually supports | any CUDA GPU |
| 1 | Correctness gate vs float64 reference | any CUDA GPU |
| 2 | Kernel microbenchmarks (synthetic, random masks) | locked clocks, exclusive |
| 3 | End-to-end accuracy on RULER / LongBench-v2 | 24 GB+ |
| 4 | Matched-accuracy operating points | derived |
| 5 | End-to-end model latency at those points | locked clocks, exclusive |
| 6 | Pareto frontiers per grid cell | CPU only |
| 7 | Decision tree, leave-one-architecture-out validated | CPU only |

Stages 0, 1, 3, 6 and 7 run on free-tier hardware. Only 2 and 5 need rented,
clock-locked, exclusive GPUs.

## Stage 3 uses RULER's task construction, not RULER's benchmark

Stage 3 generates examples with RULER's own task-construction algorithms
(needle-in-a-haystack, variable tracking) and scores them with RULER's own
metric functions, vendored from the official repo -- see
`attnbench/_vendor/ruler/VENDORED.md` for exactly what's verbatim versus
adapted. But RULER's generator scripts aren't an importable library (they're
argparse CLIs with module-level global state), and their most realistic
haystack (real prose, via NLTK + a downloaded essay corpus) and word-based
needles (via the `wonderwords` package) both pull in dependencies this
project isn't adding. Stage 3 substitutes a dependency-free filler-text
haystack and numeric/UUID needles instead.

This means Stage 3's absolute accuracy numbers are **not comparable to
published RULER results** -- filler text is measurably easier to search than
real prose, so numbers here will likely read higher. They remain valid for
this study's actual purpose: comparing attention backends against each other
on identical inputs. Every row records this explicitly
(`AccuracyResult.haystack_mode`), rather than leaving it as a caveat someone
has to go find.

That pattern generalises, and `docs/claims.md` is where it is written down:
one row per claim, pairing the sentence this study's data supports with the
near-paraphrase it must not become. **Any write-up drawn from these results
should be drafted from that file.** Three columns exist so the qualifiers
travel with the data rather than only in prose -- `haystack_mode`,
`score_source`, and `gate_source` -- because a summary drops a caveat and a
column does not.

## Quick start

```bash
uv venv && uv pip install -e .
python scripts/run_probe.py --out results/probe
```

Stages 0 and 1 run anywhere with a CUDA device and need no special permissions.

## Adding a backend

Subclass `AttentionBackend`, declare a `Capability`, implement `forward`, and
decorate with `@register`. Inputs and outputs are always
`(batch, n_heads, seq_len, head_dim)`. Any transposition or KV expansion the
kernel needs happens inside `forward`, and its cost counts toward that backend,
which is deliberate: a kernel with no native GQA path genuinely does cost more
on a GQA workload.

Raise `UnsupportedConfig` for configs the kernel cannot run. Never silently fall
back to another implementation.

## Measurement discipline

- No result row is written without a provenance stamp (`versions.json`).
- Clocks locked and exclusivity verified before any Stage 2 or 5 run.
- Speedups recomputed against a PyTorch baseline **remeasured on the same
  machine**. Ratios are never carried across hosts.
- Grid cells executed in randomised order so thermal drift cannot correlate
  with backend identity.
- Three repeats per cell in separate sessions, not three loops in one process.
- Medians with IQR, never means.
