# BUNGA

**Block-sparse Utility Navigator for GPU Attention**

BUNGA is a research project about running language models on long prompts. It
asks whether skipping part of the attention computation really makes a model
read a long prompt faster on a given machine, and whether that can be
predicted before the work is run there.

## The question

When a language model reads a prompt, its first pass over the whole prompt is
called *prefill*. At long prompt lengths most of prefill's time goes into
attention, which compares every token with every earlier token. *Block-sparse
prefill* cuts the attention computation into square blocks of tokens and
computes only the blocks that an *estimator* selects. An estimator is a cheap
method for guessing which blocks matter. The share of blocks kept is the
*density*; computing every block is called *dense* attention.

Papers usually report that block-sparse attention *kernels* (the GPU functions
that compute attention) run faster than dense ones, measured on one or two
machines. This project asks two questions:

1. On a given machine, does block-sparse prefill make the whole prefill
   faster, with accuracy no worse than dense attention by more than a stated
   margin?
2. Can that be predicted before the workload is timed on that machine, from
   short, cheap measurements taken there?

## Why a faster kernel does not mean a faster prefill

A sparse kernel can win while the whole prefill loses. In an earlier study in
this repository, on an NVIDIA A100, prefill became slower with block-sparse
attention than with dense attention, running at 0.63x to 0.75x of dense speed,
although the sparse kernel itself beat PyTorch's flash attention kernel. These
figures are for 75% of blocks skipped, at 8,192 and 16,384 tokens, from a run
on 17 September 2026. They are recorded in the
[research plan's summary of earlier results](docs/RESEARCH_PLAN.md#2-what-is-already-established-banked-evidence).

The extra time came from building the list of kept blocks (the *mask*) on the
CPU, and most of that work was Python overhead in one slow loop, so this is a
finding about one implementation. In the same run, a vectorised version of the
mask-building code made prefill 1.20x faster than dense flash attention at
16,384 tokens, but only reached parity (1.02x) at 8,192 tokens. Flash attention is also not the
fastest correct dense kernel on that card. Against the fastest one, the sparse
kernel's own advantage shrinks from 1.96x to 1.84x at 16,384 tokens, and at
8,192 tokens it becomes a small loss (0.95x).

The deeper reason is that a sparse method keeps different blocks for every
input. Code that handles this has three options. It can read the selected
sizes back from the GPU, which makes the *host* (the CPU that drives the GPU)
stop and wait; such a wait is called a *synchronisation*. It can reserve room
for the worst case, which wastes GPU work. Or it can fix the number of blocks
in advance. Released research code often takes the first option. XAttention,
for example, runs Python `assert` checks on GPU tensors, at least two for each
chunk of the prompt, in every layer. In a check on a CPU with PyTorch 2.8 on
6 October 2026, `torch.compile` with default settings stopped compiling and
fell back to Python at an `assert` on a GPU tensor, at `.item()` and at
`nonzero()`, but not at a fixed-size `topk`. Which waits remain on a GPU after
compilation is one of the things this project measures.

So whether sparse prefill pays depends on the host, the GPU, how their work
overlaps, which waits the code makes, and whether the sparse masks keep
accuracy.

## How BUNGA works

![Diagram of the BUNGA pipeline. Short measurements on a host and GPU, and
traces of each estimator's operations, feed a time model. Its predicted times,
with the lowest safe density per estimator from one-off accuracy runs, feed a
dispatcher that picks dense attention or a sparse estimator and density. Timed
runs made later check the predictions.](docs/images/bunga.png)

*The BUNGA pipeline. Short measurements on the target machine and recorded
traces feed the time model. Accuracy is measured once. Predictions are saved
with a checksum before the timed runs that check them.*

BUNGA has two parts.

- **A time model.** It predicts how long a whole prefill takes on a given
  host and GPU. It follows the host and the GPU through the prefill one
  operation at a time, using costs measured separately: how long the host
  takes to issue each operation, how long the GPU takes to run it, how long
  copies take, and what each wait costs. Because it keeps the order of
  operations, it shows where each wait sits and what removing it would save.
- **A dispatcher.** For each machine, prompt length and batch size, it
  chooses dense attention or a sparse estimator at a given density. It uses
  the predicted times and, for each estimator, the lowest density at which
  accuracy holds. It decides before anything is timed on that machine.

Each prediction is written down, with a checksum, before the timed runs that
test it, so it cannot be adjusted afterwards.

## What the project tests

These are open questions. No answers are available yet.

- **Prediction on machines where the workload has not been timed.** Does the
  time model predict whole-prefill time, and the prompt length at which
  sparse starts to beat dense, on a host-GPU *pairing* it has not timed? A
  pairing is one host CPU platform with one GPU model on one cloud provider.
  Some predictions use costs measured on other pairings.
- **Whether the order of operations matters.** The model is compared with
  simpler predictors built from the same measurements: a weighted sum of the
  parts (the main comparison), a plain sum, a step form in the style of
  KernelSight-LM (one formula per layer that ignores the order of
  operations), and a learned predictor (LightGBM) in the style of CTFusion. On code with no waits inside the prefill, the model should do no
  better than these. One direct test runs XAttention twice, as released and
  with its Python checks removed, and the model predicts the time difference
  before it is measured.
- **Whether measured costs carry over between machines.** Are host costs
  measured with one GPU the same with another GPU on the same host? Are GPU
  costs the same on another host?
- **What the compiler removes.** Each estimator also runs under
  `torch.compile`, with default settings and with the options that keep
  data-dependent sizes inside the compiled code. The places where compilation
  still breaks, and the waits that remain, are recorded.
- **Whether recall can stand in for accuracy.** *Recall* is the share of the
  full attention weight that the kept blocks capture. At the same recall in
  every layer, does accuracy depend on which estimator chose the blocks? A
  control picks blocks at random until it reaches the same recall. If recall
  is enough, the lowest safe density can be read from recall, which is cheap
  to measure and does not depend on the hardware.
- **Whether the dispatcher helps.** It is compared with the rule
  practitioners use today: use sparse attention wherever the sparse kernel
  alone beats dense. It is scored by how much time it loses against the best
  measured choice.

## Scope and limits

- One GPU per machine, one CUDA stream (a single ordered queue of GPU work)
  per forward pass, batch sizes 1 and 4, PyTorch with and without its
  compiler.
- Prompts of 2,048 to 32,768 tokens. The dispatcher is scored at 16,384 and
  32,768 tokens; the shorter lengths only test the time model.
- Seven host-GPU pairings on rented cloud machines. On Google Cloud: an Intel
  Cascade Lake host with an L4 GPU, the same host platform with an A100, and
  an Intel Sapphire Rapids host with an H100. On AWS: AMD EPYC hosts with an
  L4, an L40S and an H100, and, if time allows, an older AMD EPYC host with an
  A10G. Claims are limited to these.
- Three models are planned: Qwen2.5-1.5B and Qwen2.5-7B, with Llama-3.1-8B
  held out of all fitting. Nothing beyond the Qwen2.5 and Llama-3.1 families
  is in scope. The accuracy pilots so far used one model, one GPU and
  synthetic tasks.
- Not covered: serving across several GPUs, engines that batch many requests
  together, sparsity during token generation, and sparse attention that is
  trained into the model.
- Accuracy is reported as "no worse than dense attention by more than a
  stated margin", never as "no accuracy loss".
- Cloud machines are noisy. The noise is measured and reported, and
  prediction errors smaller than the noise are not interpreted.
- Sparse prefill may never pay on some machines. That is a stated possible
  outcome, not a failure. The project would then report what keeps it from
  paying, and whether the dispatcher avoids the losses that the kernel-only
  rule would make.

## Status

Results of this study are not available yet. As of October 2026:

| Part | State |
|---|---|
| Earlier study (masks chosen from the full attention scores, on L4, A100 and H100 GPUs) and accuracy pilots on an L4 | Finished; its results motivate this study |
| Accuracy runs for XAttention with calibrated thresholds | Prepared, not yet run |
| Tools for the time model's measurements (host speed, copy timings, timed ground-truth runs, choice of dense kernel) and recall measurement | Early pieces exist; not finished |
| The time model, the comparison predictors, the profilers, the compiled variants and the dispatcher | Not started |
| Runs on the seven pairings, analysis and write-up | Not started |

## Getting started

You need Python 3.10 or later. The tests run on a CPU, with no GPU, model
download or credentials.

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,eval]"
.venv/bin/python -m pytest tests/ -q
```

The `dev` and `eval` extras are needed to run the tests. Some tests are
skipped on a fresh clone because they need a GPU or result files that are not
kept in git; each one says what it needs. The GPU kernels are a separate
extra, `kernels`, which compiles from source.

On a machine with an NVIDIA GPU, this checks which attention backends run and
whether they give correct results:

```bash
.venv/bin/python scripts/run_probe.py --out results/probe
```

## Repository layout

- `attnbench/`: the measurement package that BUNGA is built on (timing, a
  record on every result of the code and machine that produced it, block
  masks, attention backends, the accuracy pipeline, analysis).
- `scripts/`: runners for measurement sessions, and cloud tooling.
- `tests/`: the test suite. Many tests also check the documents.
- `docs/`: the research plan, a short research brief, the audit history of
  the earlier study, and [developer notes](docs/development.md).
- `tools/`: builders for the plan and brief PDFs.
- `configs/`: run configurations.
- `results/`: measured results. Most are kept out of git.

## Related work

XAttention (Xu et al., ICML 2025); MInference; FlexPrefill; DuoAttention;
TaxBreak; KernelSight-LM; CTFusion. Full references, and the rest of the
related work, are in [`docs/research_brief.pdf`](docs/research_brief.pdf).

The full design is in [`docs/RESEARCH_PLAN.md`](docs/RESEARCH_PLAN.md).

## Licence

Apache 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
