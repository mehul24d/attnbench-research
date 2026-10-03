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
> vectorised mask builder in place of this harness's own reference builder**
> (`masks.importance_block_mask` — attnbench code, not an upstream
> implementation). At 8192 the vectorised builder reaches parity; at 32768,
> measured 2026-10-01 in one session, it wins by 1.250–1.666×. With the reference builder the same
> configurations run at 0.633× and 0.956×. The model's outputs are bitwise
> identical under both builders; the difference is one function, ~91% of whose
> cost is Python interpreter overhead.

> **A100 baseline caveat (2026-10-03).** These A100 ratios are against `sdpa_flash`, which is not the fastest correct dense kernel on that card. At the model's `(12,2)` geometry, fa2 is 6.0% faster at 16384, and cuDNN is 25.8% faster at 8192 (cuDNN faults above 8192).
>
> - **16384:** against fa2, 1.090× / 1.201× / 1.282× become an **estimated** 1.072× / 1.180× / 1.260×.
> - **8192:** against cuDNN, the vectorised 0.968× / 1.023× / 1.058× become an **estimated** 0.910× / 0.962× / 0.995×, a loss at every sparsity.
> - **32768:** unadjusted, an upper bound.
>
> These are estimates until run D (`docs/estimator_frontier_preregistration.md` §4.6) measures the kernels in one session. Derivation and test: `docs/claims.md`, `tests/test_a100_baseline_caveat.py`.

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

The oracle ratio flattens near 35 rather than heading toward 1. That shape is
expected: scoring and the saving are both quadratic in context, and an oracle
that computes dense attention costs more than the attention it saves by
construction. The level, ~35, is set by this harness's fp32 chunked scoring
pass (11× a dense prefill at 32768); a one-dense-prefill oracle would sit near
3×. *(This paragraph ended "the gap is structural, not a small-scale artefact"
until 2026-10-01.)*

*The 2048 row read "0.993× (sparsity loses)" until 2026-09-21. That is the 0.75-sparsity cell,
which scores 99.0 — so it is the best speedup at a SMALL accuracy cost, in a
column headed "at no accuracy cost". The only 2048 cell at 100.0 is 0.5
sparsity, at 0.984×. Forcing the sink does not change which cell qualifies:
the 0.90 row moves from 62.0 to 97.0 and still does not reach 100. The
correction makes the row worse and the column's own rule is what requires it.*

## Research gaps

The study set out against five gaps. Where each stands, using only what
[`docs/claims.md`](docs/claims.md) supports:

| # | original gap | status | evidence (sections of `docs/claims.md`) |
|---|---|---|---|
| 1 | Kernel speedup vs end-to-end speedup | **partly closed** | On the L4: 1.24–1.26× as a whole-model prefill ratio at 16384/0.9, 1.186× end-to-end at the same cell, and ≤1.06× and usually <1.0× end-to-end at accuracy-matched points at 8K and below ("End-to-end speedup, which is the point of the whole study"). The kernel at the model's `(12,2)` geometry was measured on the A100 only ("The same comparison at the model's real head geometry"). |
| 2 | Random masks vs importance-derived masks | **partly closed** | Kernel cells are timed under random masks and accuracy under importance-derived ones ("Block-sparse accuracy"; "The speedup does not survive a change of card"). No row of `claims.md` states the separation as a supported claim, so it is not marked closed. |
| 3 | The estimator's cost, which kernel benchmarks exclude | **partly closed** | The oracle's cost is priced at ~35× the latency it saves ("THE STUDY'S CONCLUSION"). One deployable estimator, MInference's mean-pool, was run inline on an L4: it costs 0.06–0.31 of the attention time it saves and is non-inferior to dense at no sparsity on that pilot's tasks, at 16384 and 32768 ("The deployable estimator, run inline"; the 32768 rows carry that file's 32K-positions caveat). Latency at matched accuracy is unmeasured. Training-free prefill methods such as MInference and XAttention already count their own estimator ("Where this study sits"). |
| 4 | Accuracy and latency measured apart vs at matched accuracy | **partly closed** | Matched operating points exist: 15 of 34 are dominated by dense and the best is 1.059× ("End-to-end speedup…"). On two of three tasks the unconfounded end-to-end comparison is not obtainable at any affordable sample size ("The unconfounded end-to-end comparison is not obtainable on two of three tasks"). |
| 5 | One card vs two architectures | **partly closed** | The end-to-end speedup is hardware-conditional: with the reference builder, 1.344× on the L4 against 0.480× on the A100 at 32768/0.75, in one mask era ("The speedup does not survive a change of card"). The A100 figure is against `sdpa_flash` and falls under the A100 baseline caveat above, whose adjusted figures are estimates until run D. The backend ranking across architectures rests on 11 cells ("Cross-architecture timing"). |

*Gap 1 called 1.24× a kernel figure, with "both ends measured, on the same
hardware", until 2026-10-01; it is a prefill ratio. Gap 3 said "which published
speedups exclude" until 2026-10-01.*

**The current primary question.** For training-free block-sparse prefill:
where is the frontier between an importance estimator's cost and the quality
of its ranking, how does it scale with context length, and where does each
point on it become profitable across GPU generations? The study that asks it
is written up in
[`docs/estimator_frontier_preregistration.md`](docs/estimator_frontier_preregistration.md)
(§1). That file is an **unlocked draft**. The study has not run: no GPU
session and no measurement exists for it.

**What the earlier study did not establish** (from
[`docs/limitations.md`](docs/limitations.md)):

- Every accuracy figure uses oracle masks, from a full dense attention pass
  whose cost is excluded from the latency ("Importance scores are an upper
  bound, and their cost is excluded").
- It covers one model family (Qwen2.5, at two sizes) and one sparse kernel
  family (block-sparse at block size 128), in prefill only ("SCOPE, before
  anything else").
- The host CPU was an uncontrolled variable: no GPU session recorded it in
  its stamp until 2026-10-01, when it was measured ("Host CPU provenance").

## Scope and cost, stated up front

Measured on **rented NVIDIA L4 (24 GB) and A100-SXM4 (80 GB)** instances —
two architectures, not a survey. H100 sessions also appear in the spend
ledger (kernel-level Stage 0–2, and one session lost entirely), but no row
in `docs/claims.md` rests on them. **One model family at two sizes**
(Qwen2.5-1.5B-Instruct throughout; 7B for the accuracy arms at 16384), **one
task family** (RULER-style NIAH and variable tracking), **batch 1**,
**prefill-only sparsity**, **inference only**. Context tops out at **32768
tokens**, which is a hardware ceiling, not a design choice.

Total rented GPU time: **₹8,409 itemised** (≈ US$96 at the ₹88/$ the ledger
prices its own audit-log rows at), across 38 priced sessions in
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

## Methodology

### The eight stages

| Stage | What it measures | What it feeds | Hardware |
|---|---|---|---|
| 0 | Capability matrix: what each backend *claims* vs actually supports (the claimed/actual split follows CAB, [arXiv:2210.07661](https://arxiv.org/abs/2210.07661)) | which cells Stages 1 and 2 may run | any CUDA GPU |
| 1 | Correctness gate vs a float64 reference | which cells Stage 2 is licensed to time | any CUDA GPU |
| 2 | Kernel microbenchmarks (synthetic, random masks) | the kernel side of gap 1, and the dense-kernel comparison behind the A100 baseline caveat | exclusive; clocks **not** locked on any run |
| 3 | End-to-end accuracy on RULER-style tasks | Stage 4, and Stage 6's measured latency | 24 GB+ |
| 4 | Matched-accuracy operating points (non-inferiority + bootstrap) | Stages 6 and 7 | derived |
| 5 | Phase decomposition: prefill, decode step, scoring | Stage 6's `normalized_ms`, and the oracle-cost ratio | locked clocks, exclusive |
| 6 | Pareto frontiers per grid cell | Stage 7 | CPU only |
| 7 | Decision map | the write-up ([`docs/writeup_input.md`](docs/writeup_input.md)) | CPU only |

*Stages 4, 6 and 7 were rebuilt on 2026-09-20 from forced-sink accuracy
(audit item S1a) and now occupy `results/stage4|6|7`. The versions published
before that date are correct measurements of a mask the current code does not
build, and are kept under `results/_superseded/` — see `docs/claims.md`, "The
derived stages rebuilt on forced-sink accuracy".*

Only Stages 2 and 5 need rented, exclusive GPUs. Everything else runs on
free-tier hardware or a laptop.

**Clock state, from the `clocks_locked` column of the banked rows.**

| stage | files | dates | cards | `clocks_locked` |
|---|---|---|---|---|
| 2 | 6 `sweep.parquet` files, 1,027 rows | 2026-09-03 to 2026-09-17 | L4, A100, H100 | `False` on every row |
| 5 | 11 `phases.parquet` files, 162 rows | 2026-09-07 to 2026-09-16 | L4, A100 | `True` on every row |

*The stage table gave Stage 2's hardware as "locked clocks, exclusive", and
the sentence above it read "clock-locked, exclusive GPUs" for both stages,
until 2026-10-04. No Stage 2 run locked its clocks; see "Measurement
discipline" below.*

### Measurement integrity

The full list is under "Measurement discipline" below. In short:

- **Provenance on every row.** No result row is written without a stamp
  (GPU, driver, clock state, commit).
- **A correctness gate.** Stage 1 checks each cell against a float64
  reference before Stage 2 times it (`attnbench/gates.py`).
- **Resolution floors from measured variance.** A difference inside its
  cell's floor is not reported as a win (`docs/claims.md`, "End-to-end
  speedup…"; `docs/audit_register.md`, item S11).
- **Mask eras tied to commit ancestry.** A row's era is fixed by where its
  commit sits relative to the mask-rule fixes, and the comparison scripts
  refuse a cross-era pair (`attnbench/analysis/eras.py`,
  `tests/test_eras.py`; `docs/limitations.md`, "Which mask era each banked
  accuracy file belongs to").
- **A canary inside the comparison.** A re-measurement re-runs the dense arm
  in the same session and requires it to reproduce the banked run before any
  difference is attributed to the change under test: 300/300 for S1a,
  200/200 for S7 (`docs/audit_register.md`).
- **A check is not trusted until it has been made to fail.** Guards are
  break-tested: the defect is put back and the test is watched going red
  (`tests/test_no_stale_figures.py`; `docs/audit_register.md`, item S14).

### The planned frontier design

Pre-registration drafted (unlocked), in
[`docs/estimator_frontier_preregistration.md`](docs/estimator_frontier_preregistration.md).
Nothing below has run. What the draft plans:

- Every estimator runs on the GPU, through one kernel call site and one mask
  format (§4.2, §4.7). The arms are an oracle as a reference ceiling,
  mean-pool, XAttention's antidiagonal scoring as a baseline under study,
  vertical-slash-style scoring, and a sink + local window with no estimator.
- Calibration, budget-selection and evaluation splits are disjoint, and the
  evaluation split shares no example with any banked run (§5).
- Each hypothesis states a numeric prediction, the tolerance on its input
  ratio (0.05 at the kernel level, 0.07 end to end) and its pass rule (§3).
- A hard cost cap of ₹12,000, with a reservation check before every launch
  (§8.3).
- If the check fails, items are cut in a fixed order, and nothing is cut
  because of a result (§8.3, §8.4).

### The finding that changed the framing

The A100 reversal — block-sparse prefill faster than dense on the L4 and
slower on the A100 — was explained by host-side mask construction. The
builder costs the same on both hosts; the slower L4 hides it behind queued
GPU work and the faster A100 no longer can. The account was then tested on
an H100 under a pre-registration with three predictions: two passed, and one
(that the direction is set by the build-to-GPU time ratio) failed, with 7 of
9 cells where every cell was needed. H100 speedups in that section are
against `sdpa_flash`, and their baseline strength is unmeasured. The tables
are in [`docs/limitations.md`](docs/limitations.md): "The A100 reversal is CPU
mask construction", "Measured 2026-10-01" and "The H100 test
(pre-registered)". The prediction file is
[`docs/h100_overlap_preregistration.md`](docs/h100_overlap_preregistration.md).

## Remaining work and progress

As of 2026-10-04. Rows marked † describe code built that day in the commits
that follow this README change on the branch; before those commits the
files they name are not in the tree.

| item | status | source |
|---|---|---|
| Lock of the frontier draft | waiting on the researcher: the draft is unlocked, and whether to lock it is their call | draft §11.1 |
| T4 calibrated, sessions A and B (`docs/t4_xattention_calibrated.md`) | not started, after lock. Amended before any data on 2026-10-04 (amendment A8)† | §0.4 |
| Era table: T4 files registered as era 3, era 4 registered | done | §11.2 item 6 |
| `positions_over_limit` and the estimator-path label on accuracy rows; `rebracket()` with its leak guard | done | §11.2 items 1 and 4 |
| Host-floor provenance (`launch_floor_us`, `sync_floor_us`, `h2d_floor_us`, `cpuPlatform`) on every row type | not started | §11.2 item 1, §4.8 |
| Dense baseline: the "runs correctly" check, the health check, the `dense_baseline` column | not started | §11.2 item 2, §4.6 |
| FA3 build script for a CPU VM, pinned | not started | §11.2 item 2 |
| Era-4 per-head selector | done†: the selector is built, and with the same scores on every head it reproduces the head-uniform rule bitwise, checked on CPU. No arm calls it yet | §4.5; `attnbench/masks.py`, `tests/test_per_head_selector.py` |
| `bsa_prefill`, per-head mean-pool, vertical-slash-style scoring, sink + local window, the GPU-side multi-block-size oracle | not started | §11.2 item 3 |
| Recall script, the `split` column on recall rows, the selection script's refusal | in progress†: the row schema with `split`, the selection refusal and the firewall in front of the evaluation analysis are built; the recall pass that writes the rows is not started | §11.2 item 3; `attnbench/analysis/frontier_recall.py` |
| Extensions to `measure_estimator_cost.py` and `run_vectorised_endtoend.py` | not started | §11.2 item 3 |
| Frontier grid config at n = 300 | not started: `frontier_prereg.evaluation_indices` exists, and no config is built from it | §11.2 item 3 |
| Gate G12 wired into split generation | done†: split generation and the calibration script both stop on it. Run on the calibration texts as shipped, it stopped; a dated amendment (§5) changes which texts the calibration uses, and the gate passes on those | §5, §6; `attnbench/accuracy/frontier_splits.py` |
| `results/frontier_spend.csv` and `scripts/frontier_budget_gate.py` | in progress†, **verified only against a stand-in `gcloud`**: the gate script is built, and the A100, H100 and L4 launchers refuse to create an instance without it. Not wired: the compile-session launcher, the image boot-test and image-capture scripts, and the teardown's ledger row. The ledger file is created at first use | §8.3 |
| Llama-3.1-8B gated access | in progress: access granted; the G9 position-limit check passed, for the position limit only; weights not fetched | §11.1 L5 and L7, §11.2 item 5 |
| A100 canary, with the re-bracketing of the cost plan from its measurements | not started. The worst case was re-bracketed once already, before any session, from banked measurements (§8.2) | §6, §8.3 |
| Run D: dense kernels compared in one session at the model's geometry; replaces the A100 estimates and measures the H100 baseline | not started | §4.6 |
| H100 session (I3) | not started | §8.2, §8.5 |
| cuDNN-on-H100 isolated probe, the last phase of I3 | not started | §4.6 |
| Llama answer-length caps: measured offline and pinned by `tests/test_stopping.py` | done | §4.9 |
| Llama answer-length caps: the cap-hit check in the dense task probe, which may double a cap once | not started | §4.9 |
| S12: the oracle-versus-estimator gap at 16384 under the fixed mask builder | not started: open, and the register recommends leaving it and labelling the rows era 2 | `docs/audit_register.md` §5 |
| Whether audit items S1–S4 ever existed | blocked: it cannot be resolved from the repository | `docs/audit_register.md` §3 |

**Verified only against a stand-in `gcloud`.** The budget gate's wiring into
the launchers has never run against the real project. Its tests run each
launcher with a fake `gcloud` that only logs its arguments, and check three
things:

- with no passing gate, no create call is made;
- with a passing gate, one create call is made, under the hard cap the gate
  set;
- the launch is written to the ledger.

The first real launch is the first real test of all three.

This table is a snapshot. It was derived from
`docs/estimator_frontier_preregistration.md` (§4, §5, §6, §8, §11),
`docs/audit_register.md` and the code, and it goes stale as they change. No session of the frontier study has run, and §0.4 of the
draft fixes the order in which they may start.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q     # 1696 passed, 96 skipped, ~75s, no GPU
```

The `[dev,eval]` extras are required, not optional: four test modules import
`transformers` at collection time, so a bare `pip install -e .` cannot even
collect the suite. The `kernels` extra is deliberately separate — those
compile from source and are architecture-gated, and the harness must import
on a machine where a kernel is unavailable.

The suite runs on CPU and needs no GPU, no model download, and no
credentials. Verified from a clean clone into a fresh virtualenv on
2026-09-12, resolving dependencies from scratch. The counts were re-measured
2026-10-03 from a clean clone of the commit, using the existing virtualenv
(not a fresh one), `$ATTNBENCH_RULER_DATA` pointed at an empty directory and,
from 2026-10-03, `HF_HOME`/`HF_HUB_CACHE` pointed at an empty directory with
`HF_HUB_OFFLINE=1`.
A clean clone is a *partial* tree: it carries the 38 force-committed files
under `results/` and none of the rest.

The 96 skips are the honest part, and they split six ways: **15** need CUDA,
**65** read banked result files that `results/` correctly keeps out of git,
**6** are scripts `test_script_call_sites.py` has nothing to check because
they import nothing from `attnbench`, and **6** are the same scripts skipped
again by `test_import_resolution.py`, which only has something to say about a
script that imports the package, **2** need the pinned RULER data
directory (`scripts/fetch_ruler_data.py`; the essays and QA sets are not
committed), and **2** need the pinned Llama-3.1-8B tokenizer in the local
HF cache (gated; a test never fetches it). The 65 validate real measured data —
including the check that the cross-arm decode guard actually fires on the
confounded Stage 3 rows, and the check that the two comparison scripts refuse
the banked era-2/era-3 pair, and the check that segment 1's 2026-09-03 probe
still carries the eager-flex fingerprint, and the fourteen that recompute audit
item S11's published tax percentages from the parquets they are derived from,
and the four that derive the mask builder's sparsity-dependent cost and the
L4-host bound in `limitations.md` —
so a fresh clone is green **without** running
them. Each skips with a message naming the file it wanted, rather than passing
silently.

*That second number was 24 until 2026-09-23. Fourteen of the seventeen added
are `tests/test_s11_tax_derivation.py`, which exists because S11's disposition
was written from the figures its measurement script printed rather than the
columns it banked — see `silent_failure_patterns.md` #54.*

*The third and fourth numbers were 4 until 2026-10-03, when
`scripts/flag_positions_over_limit.py` was added, and 5 until later that day,
when `scripts/derive_llama_oracle_cost.py` was; neither imports anything from
`attnbench`. The second number was 45 until 2026-10-03, when the 18 tests of
`tests/test_h100_baseline_caveat.py` were added; two of their inputs are not
tracked.*

*The CUDA number was 14 until 2026-10-04, when `tests/test_kernels.py` added
the check of `bsa_prefill` against its plain-torch reference, which needs the
real kernel.*

*The second number was 63 until 2026-10-03, when
`tests/test_frontier_eval_ids.py` added two: that the registry of banked
example indices is the whole banked tree, and that no estimator timing is
banked off the L4.*

*The RULER-data number was 1 until 2026-10-03, when the T4 amendment A1
break-test (`test_xattn_calibration.py`, a shared `qa_1` question under
another seed) was added; it needs the SQuAD set.*

*That third number was 6 until 2026-09-21. It became 4 then because
`run_scale_comparison.py` and `run_scorer_comparison.py` acquired the mask-era
check and therefore import `attnbench` for the first time — two scripts moved
out of the "nothing to check" bucket by being given something to check.*

Those counts are for a fresh clone. **With `results/` present the suite reads
1765 passed, 27 skipped**, because the 65 banked-file tests run instead of
skipping (and, on a workstation with the RULER data directory and the
cached Llama tokenizer, those four tests run too). Both totals are 1792, which is what `pytest --collect-only` reports —
asserted, not transcribed, by
`test_the_readme_suite_counts_match_the_collected_suite`.

*Cross-version equality was measured once, on 2026-09-21, against the suite as
it stood that day: transformers 5.17.0 and 4.46.0 agreed test for test. It has
not been re-measured since, so treat it as a result about that suite on those
two versions rather than a standing property of this one. The project
virtualenv on this workstation runs **5.18.0** (with torch 2.14.1), installed
2026-10-01 20:31 by its `dist-info` timestamps, so it is no longer the 5.17.0
the equality was measured at. *(This sentence said "runs 5.17.0, the version
it was measured at" until 2026-10-03, and the workstation "now runs 5.16.1"
until 2026-10-01; that is the system `python3`, which the suite does not
use.)* *(That note ended "register item D1" until 2026-10-04;
`docs/audit_register.md` has no item D1.)* *The Llama-3.1 RoPE gate (`tests/test_llama3_rope_gate.py`)
was run on both 5.18.0 and 4.46.0 on 2026-10-03, with 4.46.0 installed in a
scratch directory on this macOS host, not a Linux container. That is one file
on 4.46.0, not the suite.* Reproducing 4.46.0 needs a Linux container and this host
has no container runtime. The Linux/4.46.0 suite run on record is audit item S9
(`docs/audit_register.md` §4): 4 failed, 977 passed, 20 skipped. Its failures
were fixed afterwards (`4ab382f`, and item S13), and the register records no
Linux rerun since.* *(This sentence ended "it is the one item on the audit
register's could-not-verify list that is still open" until 2026-10-04; the
register has no such list.)* *(This paragraph said "The 11 skips … 2 need CUDA, and 9 read banked
result files" against a code block saying 10 skipped, while the real numbers
were 23 and 9. Three figures for one quantity, none of them measured. The
PASSED figures then went stale on their own: they read 1109 and 1133 from
2026-09-21 until 2026-09-23, three short, because two later commits added
tests and one of them recorded the correct 1136 in its own commit message while
editing this file for something else. `tests/test_doc_derived_numbers.py`
asserts the skip counts agree with each other and with their own breakdown, and
now also that both passed+skipped totals equal the number of tests pytest
actually collects -- which is the check that was missing, since internal
consistency held while both halves were wrong.)*

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
  56 confirmed incidents, each one a plausible number produced by machinery
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
  nothing about its scope.** The headline configuration (32768/0.75) was
  replicated on a second L4 — prefill 1.373× to within 0.01% — and that
  replication was *structurally incapable* of detecting that the claim was
  L4-specific, which Stage 5 on an A100 later showed it was (0.475×
  prefill-only at the same configuration). *(This said "The 1.321× headline
  was replicated … to within 0.19%" until 2026-10-01; 1.321× is end-to-end and
  was not the replicated quantity.)* Two measurements agreeing
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
