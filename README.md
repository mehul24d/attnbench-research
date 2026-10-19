# BUNGA

**Block-sparse Utility Navigator for GPU Attention**

Does training-free block-sparse prefill make the whole prefill faster on a
given host CPU and GPU, at accuracy non-inferior to dense attention, and can
that be predicted before the workload is timed there?

This README summarises the research plan,
[`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md)
([PDF](docs/research_plan.pdf)). The plan is the source of truth and wins
wherever the two differ. `attnbench` remains the name of the measurement
harness (the `attnbench/` package, and this repository).

## Why a faster kernel does not mean a faster prefill

Papers report sparse attention kernel speedups on one or two machines. Our
own measurements show the kernel speedup often does not survive end to end.
On an A100 with our reference mask builder, prefill became slower (0.63x to
0.75x) although the sparse kernel beat PyTorch's SDPA flash kernel. These are
end-to-end prefill ratios against flash at sparsity 0.75, 0.75x at 8192
tokens and 0.63x at 16384, from one A100 run on 17 Sep 2026 at commit
`ae736b4` (plan Sections 1 and 2). Flash is not the fastest correct dense
kernel on that card; the plan gives the figures against faster kernels too.
Most of that builder's cost was Python overhead in an unvectorised loop, so
this is a finding about one implementation. Host-side cost is therefore not
fixed: it depends on how an estimator is implemented.

The deeper cause is that sparse attention keeps different blocks for every
input. An implementation then has three choices. It can read sizes or checks
back to the host, which makes the CPU wait for the GPU (a synchronisation). It
can pad to a worst case, which wastes GPU work. Or it can work in fixed-size
buffers, which fixes the number of blocks. Released research code still reads
GPU values from Python: XAttention runs `assert` checks on GPU tensors in every
layer, and the FlexPrefill implementation bundled with it sizes index tensors
with `.max().item()`. Compilation does not remove these by default. Under
`torch.compile` with default settings, an `assert` on a GPU tensor, `.item()`
and `nonzero()` each break the graph, while a fixed-size `topk` does not
(checked on CPU with PyTorch 2.8, 6 Oct 2026).

So whether sparse prefill pays depends on the host, the GPU, how their work
overlaps, which synchronisations an implementation makes, and whether the
sparse masks keep accuracy.

## How BUNGA works

![Pipeline diagram. Two inputs, "Estimator, model, task" and "Host-GPU
pairing", feed five processes. A structural probe (2048 and 4096 tokens,
durations discarded) gives the operation sequence and synchronisation map.
Selection recording (grid lengths, untimed) gives the estimator's selection,
its masks per code path. Kernel replay (a CUDA graph using the recorded masks)
gives device costs. Microbenchmarks on the target (seconds, workload not
timed) give host components, link costs, s_0 and the launch queue Q. Accuracy
and recall runs (once, primary GPU) give admissible densities per estimator,
from recall if H_4 holds. The first four feed the BUNGA model, a recursion over
H_i and D_i, which gives a predicted time D_N per setting. The BUNGA dispatcher
combines it with the admissible densities. Predicted times and the chosen
setting go into a hashed and synced prediction file before the timed run, and
the output is the chosen setting: dense, or a sparse arm (eager or compiled) at
an admissible density. A validation path, never an input, scores the file
against an untraced ground-truth run made after the hash: error against
baselines, and regret against the best measured admissible choice, with the
kernel-speedup rule as comparator.](docs/images/bunga.png)

*Figure 1. The BUNGA pipeline. A short structural probe records the operation
sequence, and the estimator's selection is recorded at the grid lengths,
untimed. Both run once per new code path, not on every pairing: a traced code
path is reused, and masks are recorded on the target only for the first card
of a new code path, and only if the mask gate fails. Device costs come from
replaying the kernel sequence as a CUDA graph with those masks; host, link and
synchronisation costs and the launch queue come from seconds of
microbenchmarks on the target, with the workload not timed. Each
synchronisation's cost s_0 is computed from its mechanism's fixed cost s_fixed
and the copy bandwidth B, which the microbenchmarks give. In the transfer tests
(H2), host components come from another pairing on the same host platform and
device components from another pairing with the same GPU; link and
synchronisation costs are always measured on the target. Accuracy is measured
once, on the primary GPU, and is not re-measured per pairing. Admissibility
carries over to another code path only if the mask gate passes. Predicted
times and the dispatcher's choice are hashed and synced before the timed run
that tests them.*

BUNGA is made of the G2 critical-path model (components defined in plan
Section 5.1, the recursion in Section 5.4) and the profitability dispatcher
(Section 5.9), as named in Section 1 of the plan.

- **The critical-path model** predicts end-to-end prefill time from measured
  components. It follows the host and the GPU through the prefill step,
  operation by operation, and keeps the order and position of each
  estimator's data-dependent host work and GPU-to-CPU synchronisations.
- **The profitability dispatcher** chooses, for each pairing, context length and batch, one configuration:
  dense, or a sparse arm (eager or compiled) at one of its admissible
  densities. It picks the admissible configuration with the shortest
  predicted time, and stays dense unless a sparse configuration is predicted
  faster by more than the resolution floor. The choice is written into the
  same hashed prediction file as the time prediction, before the workload
  runs. It is an offline decision, not a runtime scheduler.

### The recursion

For one CUDA stream, operation *i* has host cost h_i = c_host(i), device cost
d_i = c_dev(i) (GPU execution only) and link cost l_i = c_link(i) (copy time;
zero for a kernel). H_i is when the host has issued operation *i*; D_i is when
the device finishes it.

```
H_i = H_(i-1) + h_i
if Q launches are pending:   H_i = max(H_i, D_(i-Q))          (finite launch queue, Q measured per pairing)
if operation i is a synchronisation:  H_i = max(H_i, D_(i-1)) + s_0(m_i, payload_i);  D_i = H_i   (l_i = d_i = 0)
otherwise:  D_i = max(H_i, D_(i-1)) + l_i + d_i                  (l_i at the loaded bandwidth where the operation sequence puts a copy alongside host-side estimator work)
end-to-end prefill time = D_N
```

The definitions, from plan Section 5.1:

- **Host component** c_host(i): host time to issue operation *i* (framework
  dispatch, library front end, launch floor) plus host-side compute such as
  CPU mask building. Measured by adapting TaxBreak's null-kernel protocol to
  the whole operation sequence, and by running host-side compute directly at
  target sizes. The adaptation is not claimed as a contribution.
- **Device component** c_dev(i): GPU execution time of operation *i*, from
  replaying the arm's whole kernel sequence in order on the target GPU as a
  CUDA graph. Isolated per-operation replay is the cross-check.
- **Link component** c_link(i): copy time of a non-synchronising operation,
  from microbenchmarks on the pairing.
- **Synchronisation**: any point where the host waits for the GPU (an explicit
  synchronise, a copy to the host, or Python reading a GPU value). Each is one
  operation, costed once by its mechanism m:
  s_0(m, payload) = s_fixed(m) + payload / B, where s_fixed(m) is measured at
  the smallest payload with the device idle and B is the measured copy
  bandwidth. The wait itself is never a measured constant: the recursion
  produces it.
- **Q** is the launch-queue depth, measured per pairing. The host runs up to Q
  launches ahead of the GPU and waits only at synchronisations.

Only device costs transfer between hosts. Link and synchronisation costs are
always measured on the target pairing. Letting the host run ahead of the
device up to a queue limit is a standard form of host-device overlap and is
not claimed as new. What is new is placing data-dependent synchronisations
within the step and predicting their cost on pairings where the workload has
not been timed.

## What the project tests

These are hypotheses and plans, not results. Each test's tolerance is fixed at
a pre-registration lock that has not happened yet. Failures will be reported
as failures.

- **Prediction where the workload has not been timed** (RQ1; H1, H2, H3).
  Can component measurements predict end-to-end prefill time, and the
  sparse-versus-dense crossover, on a pairing on which the workload has not
  been timed? The inputs are seconds of microbenchmarks on the target, plus
  the operation sequence and the estimator's selection recorded per code path
  with every duration discarded. Each prediction is hashed and synced before
  the workload is timed. P1 is the development pairing; P2 to P7 are the
  confirmatory ones. H2 builds predictions from components measured on other
  pairings (P6 is the full-transfer test), and H3 forecasts the crossover on a
  host platform used nowhere else (P4).
- **What order and position add** (RQ2, H1). The model is compared with the
  committed predictions of the **fitted sum** (the primary comparison), the
  **sum**, the **KernelSight-style step form** (a per-layer step formula in
  the style of KernelSight-LM) and a **CTFusion-style LightGBM** predictor
  (in the style of CTFusion; secondary). The fitted sum and LightGBM are
  trained only on pairings already run when each prediction is committed. The
  step form differs from the recursion only through operation order,
  synchronisation positions and the queue term, so the gap between the two
  measures what placing synchronisations adds. On arms with no
  synchronisation inside the step, the two should agree. A paired test runs
  the same XAttention estimator as shipped and with its Python `assert`
  checks removed, and the model predicts the time difference on each pairing
  before it is measured.
- **Whether host and device costs carry over between machines** (H0,
  separability). Are host components measured with one GPU equal to those
  measured with another GPU on the same host platform? Are device components
  measured on one host equal to those on another host for the same GPU? If
  not, transfer (H2) is not tested and the scope narrows to components
  measured on the target pairing.
- **What the compiler removes.** Each arm also runs under `torch.compile`,
  with default settings and with the capture options for scalar outputs and
  data-dependent shapes. Graph breaks per layer and the waits that remain on
  the GPU are recorded, and the compiled variants are predicted the same way.
  On variants with no wait left, the model should add nothing over simpler
  forms.
- **Whether recall can stand in for accuracy** (RQ3, H4). At matched
  per-layer recall, does accuracy depend on which estimator chose the blocks?
  Tested at block sizes 32, 64 and 128, with a random-selection control held
  to the same recall. Recall is matched only through each estimator's own
  budget settings, never by editing masks. The sample size comes from a power
  rule fixed before the lock, and a result below that power is reported as
  inconclusive. If H4 holds, the lowest admissible density can be read from
  recall, which is cheap to measure and does not depend on the hardware.
- **Whether the dispatcher beats the kernel-speedup rule** (RQ4, H6). The
  kernel-speedup rule is the decision a practitioner would make from
  published results: use an estimator at its admissible density wherever its
  sparse kernel alone beats the selected dense kernel on that GPU. On
  held-out pairings, at 16384 and 32768 tokens, the dispatcher is scored by
  regret against the best measured admissible choice, and is also compared
  with always dense, always the sparse arm with the best kernel speedup, and
  the best choice on the nearest measured pairing.

**If sparse prefill never pays.** The set of profitable configurations may be
empty, on some pairings or on all of them. That is stated in advance as a
possible result, not a failure. The dispatcher can then at best tie always
dense, and its test becomes whether it avoids the kernel-speedup rule's
losses. The other claims stand: predicted against measured time per pairing
and arm, which operations and synchronisations keep each arm from paying, and
the recall result.

## Status

Copied from the task statuses in Section 8 of the plan (last updated 18
October 2026). Nothing below is a result.

| Part | Plan task | Status in the plan |
|---|---|---|
| Earlier oracle-mask study and pilots | Section 2 | Banked evidence; motivates the study, not its confirmatory evidence |
| One selector and kernel for every arm | Control K1 | Partly built (per-head selector not yet merged) |
| Calibrated XAttention sessions | T1.1 | Built, not run |
| Recall with both references | T1.2 | Partly built (intrinsic recall pass exists, not yet merged) |
| Reference masked path for block sizes 32 and 64 | T1.4 | Partly built |
| H4 dry analysis, recall matching, random control | T1.5, T1.8, T1.9 | Not started |
| Window and vertical-slash arms | T1.7 | Not started |
| Host-speed probe | T2.1 | Partly built |
| Link microbenchmarks | T2.5 | Partly built (launch, sync, 1-byte copy) |
| Untraced ground-truth harness | T2.13 | Partly built |
| Dense-baseline selection | T2.16 | Partly built (kernel sweep from the earlier study) |
| Host-component profiler, synchronisation map, device replay, queue depth, structural probe | T2.2, T2.3, T2.4, T2.6, T2.18 | Not started |
| The recursion (the critical-path model) | T2.8 | Not started |
| Sum, fitted sum and KernelSight-style baselines | T2.9 | Not started |
| LightGBM baseline | T2.10 | Not started |
| Prediction commit and hashing | T2.11 | Not started |
| XAttention arm with Python asserts removed | T2.17 | Not started |
| Compiled variants | T2.21 | Not started (the four graph-break patterns checked on CPU, 6 Oct 2026) |
| Dispatcher and its comparators | T2.22 | Not started |
| P1 session, repeat sessions, Gate A | T3.1 to T3.3 | Not started |
| Pre-registration lock | T4.1 to T4.4 | Not started (T4.2: Decision) |
| AWS tooling | T5.1 to T5.3 | Not started |
| Pairings P2 to P7 | T6.1 to T6.6 | Not started |
| Related work read in full | T7.2 | In progress |
| Analyses, predicted profitability, dispatcher evaluation | T7.1, T7.6, T7.7 | Not started |

## Scope and limits

- One GPU, one CUDA stream per forward pass, batch sizes 1 and 4, PyTorch
  eager and compiled, cloud machines. Claims do not extend to multi-GPU,
  continuous batching or multi-stream serving engines.
- Context lengths 2048 to 32768 tokens. 2048 and 4096 are model-validation
  points, not operating points; the dispatcher is scored only at 16384 and
  32768.
- Seven pairings is a small panel, and claims are limited to the measured
  host platforms and GPUs. There is no bare-metal control; the cloud noise
  floor is measured and reported as a number instead.
- Accuracy is reported as non-inferiority against dense at a stated margin,
  never as "no accuracy loss".
- Sparse prefill may never pay on some pairings. That is a stated possible
  result.

| Pairing | Host platform | GPU | Provider | Role |
|---|---|---|---|---|
| P1 | Intel Cascade Lake | L4 | GCP G2 | Development pairing; source of transferred components |
| P2 | Intel Cascade Lake | A100 | GCP A2 | Host-side separability; host transfer |
| P4 | Intel Sapphire Rapids | H100 | GCP A3 | Held-out host platform: forecast |
| P3 | AMD EPYC 3rd gen | L4 | AWS G6 | Device-side separability; device transfer |
| P5 | AMD EPYC 3rd gen | L40S | AWS G6e | Host-side separability; host transfer; GPU speed |
| P6 | AMD EPYC 3rd gen (7R13) | H100 | AWS p5.4xlarge | Device-side separability; full transfer |
| P7 (optional) | AMD EPYC 2nd gen | A10G | AWS G5 | Extra held-out point if time allows |

## Running the tests

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q     # 1375 passed, 68 skipped, no GPU
```

The `[dev,eval]` extras are required: four test modules import
`transformers` at collection time. The suite runs on CPU and needs no GPU, no
model download and no credentials.

The 68 skips are the honest part, and they split five ways: **14** need CUDA,
**45** read banked result files that `results/` keeps out of git, **4** are
scripts `test_script_call_sites.py` has nothing to check because they import
nothing from `attnbench`, **4** are the same scripts skipped again by
`test_import_resolution.py`, and **1** needs the pinned RULER data directory
(`scripts/fetch_ruler_data.py`, found through `$ATTNBENCH_RULER_DATA` or
`~/.cache/attnbench/ruler_data`). Each skip names the file or device it wanted.

Those counts are for a fresh clone on a machine without the RULER data.
**With `results/` present the suite reads 1421 passed, 22 skipped**, because
the 45 banked-file tests run instead of skipping and, on a machine with the
RULER data directory, the data test runs too. Both totals are 1443, which is what `pytest --collect-only`
reports, and `test_the_readme_suite_counts_match_the_collected_suite` asserts
it. At this commit, with the full `results/` tree present, one of those tests
fails: `tests/test_eras.py::test_the_register_agrees_with_the_documented_era_table`,
because an era listed in `docs/limitations.md` is not yet registered in
`attnbench/analysis/eras.py`.

## Earlier work in this repository

The repository started as a benchmark of sparse attention speedups at matched
accuracy, with oracle masks, on L4, A100 and H100 machines. That work is
banked, commit-stamped evidence that motivates this study; it is not the
study's confirmatory evidence. Its audit history is kept and annotated in
place, never rewritten:

- [`docs/claims.md`](docs/claims.md): one row per claim the earlier
  measurements support, with the paraphrase each must not become.
- [`docs/limitations.md`](docs/limitations.md) is over 2,700 lines: what the
  earlier measurements cannot answer, and the mask-rule eras that must not be
  pooled.
- [`docs/silent_failure_patterns.md`](docs/silent_failure_patterns.md):
  56 confirmed incidents of a plausible number produced by machinery that
  looked like it was working, each with how it was detected.
- [`docs/withdrawn_figures.md`](docs/withdrawn_figures.md) and
  [`docs/audit_register.md`](docs/audit_register.md): withdrawn figures and
  audit dispositions.

## References

Works the plan names as nearest work, baselines or arms. Full entries are in
[`docs/research_brief.pdf`](docs/research_brief.pdf). Works the plan records
as known from the abstract only are marked so.

- Host and device costs, and latency prediction: TaxBreak; KernelSight-LM;
  CTFusion; SKIP (TKLQT); CloserToMe; LLMServingSim 2.0; APEX (Fan et al.);
  APEX simulator (Lin et al.); HiSparse; nn-Meter (from abstract); HELP (from
  abstract).
- Sparse prefill and recall: VSPrefill; The Sparse Frontier; Less Is More;
  oracle-guided sparse prefill; AB-Sparse; CompactAttention; DFSAttn;
  XAttention (from abstract); MInference (from abstract); FlexPrefill (from
  abstract); DuoAttention (from abstract).

## Licence

Apache 2.0, see [`LICENSE`](LICENSE). Third-party code and adopted approaches
are recorded in [`NOTICE`](NOTICE).
