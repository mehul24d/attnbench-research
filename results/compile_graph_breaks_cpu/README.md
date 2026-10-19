# Where `torch.compile` splits the graph, on a CPU

Written by [`scripts/check_compile_graph_breaks.py`](../../scripts/check_compile_graph_breaks.py)
at commit `7f4be0b` with `git_dirty` False, on 19 Oct 2026, on macOS 15.5
arm64 (Apple M2) with Python 3.12.1. One JSON file per PyTorch version:

- [`torch-2.8.0_cpu.json`](torch-2.8.0_cpu.json)
- [`torch-2.14.1_cpu.json`](torch-2.14.1_cpu.json)

Each pattern is a short function written for this check, using PyTorch only,
with work on both sides of the pattern. `torch.compile` runs with every
setting at its default. A split is counted from Dynamo's graph count after
that compile and from `torch._dynamo.explain`; the two agree for every
pattern in both files, and the compiled output matches eager output.

| Pattern | 2.8.0 | 2.14.1 |
|---|---|---|
| `assert (y >= 0).all()` | no split | no split |
| `assert (y >= 0).all(), "message"` | no split | no split |
| `assert not (y < 0).any()` | split | split |
| `assert bool((y >= 0).all())` | split | split |
| `.item()` | split | split |
| `nonzero()` | split | split |
| `torch.topk(y, k=4)` (fixed size) | no split | no split |

**The two versions agree on every pattern.** A plain `assert` on a tensor
does not split the graph because Dynamo rewrites it into `torch._assert`
(`rewrite_assert_with_torch_assert`, on by default in both versions). An
`assert` whose condition first turns the tensor into a Python bool cannot be
rewritten that way, and splits.

The versions differ only under `fullgraph=True`, which is not a default
setting and is recorded in the files as `fullgraph_eager`, not counted as a
split. PyTorch 2.8.0 raises there at `.item()`, `nonzero()` and
`assert bool(...)`; 2.14.1 compiles those three without raising. Both raise
at `assert not ...`.

What this does not show: anything about a GPU. It counts graph splits on a
CPU. Whether a wait on the GPU remains after compilation, and what a
rewritten `torch._assert` costs there, are for the compiled arms to measure.
The `host` field in each file is a label, not the machine's network name
(`host_is_label` is true).
