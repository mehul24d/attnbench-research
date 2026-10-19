# Where `torch.compile` splits the graph, on a CPU

**This check ran on a CPU only.** No GPU was used, and nothing here says
whether compiled code makes a GPU wait.

Written by [`scripts/check_compile_graph_breaks.py`](../../scripts/check_compile_graph_breaks.py)
on 19 Oct 2026, on macOS 15.5 arm64 (Apple M2) with Python 3.12.1, with
`git_dirty` False. One JSON file per PyTorch version and commit:

| File | PyTorch | Commit | Patterns |
|---|---|---|---|
| [`torch-2.8.0_cpu_1a6f995.json`](torch-2.8.0_cpu_1a6f995.json) | 2.8.0 | `1a6f995` | (a) to (f) |
| [`torch-2.14.1_cpu_1a6f995.json`](torch-2.14.1_cpu_1a6f995.json) | 2.14.1 | `1a6f995` | (a) to (f) |
| [`torch-2.8.0_cpu.json`](torch-2.8.0_cpu.json) | 2.8.0 | `7f4be0b` | (a) to (d) |
| [`torch-2.14.1_cpu.json`](torch-2.14.1_cpu.json) | 2.14.1 | `7f4be0b` | (a) to (d) |

The first run (`7f4be0b`) covered patterns (a) to (d); the second
(`1a6f995`) added (e) and (f). On every pattern the two runs share, they
agree.

Each pattern is a short function written for this check, using PyTorch only,
with work on both sides of the pattern. `torch.compile` runs with every
setting at its default. A split is counted from Dynamo's graph count after
that compile and from `torch._dynamo.explain`; the two agree for every
pattern in every file, and the compiled output matches eager output.

| Pattern | 2.8.0 | 2.14.1 |
|---|---|---|
| (a1) `assert (y >= 0).all()` | no split | no split |
| (a2) `assert (y >= 0).all(), "message"` | no split | no split |
| (a3) `assert not (y < 0).any()` | split | split |
| (a4) `assert bool((y >= 0).all())` | split | split |
| (b) `.item()` | split | split |
| (c) `nonzero()` | split | split |
| (d) `torch.topk(y, k=4)` (fixed size) | no split | no split |
| (e) `assert (y >= 0).all()` inside `try`/`except AssertionError` | no split | no split |
| (f) `assert (y >= 0).all() and (y <= 2).all()` | split | split |

**The two versions agree on every pattern.** A plain `assert` on a tensor
does not split the graph, with or without a message and inside a
`try`/`except`, because Dynamo rewrites it into `torch._assert`
(`rewrite_assert_with_torch_assert`, on by default in both versions). An
`assert` whose condition first turns a tensor into a Python bool cannot be
rewritten that way, and splits: `not`, `bool()`, and `a and b`, which tests
`a` in Python.

The versions differ only under `fullgraph=True`, which is not a default
setting and is recorded in the files as `fullgraph_eager`, not counted as a
split. PyTorch 2.8.0 raises there at `.item()`, `nonzero()` and
`assert bool(...)`; 2.14.1 compiles those three without raising. Both raise
at `assert not ...` and `assert a and b`.

What this does not show: anything about a GPU. It counts graph splits on a
CPU. Whether a wait on the GPU remains after compilation, and what a
rewritten `torch._assert` costs there, are for the compiled arms to measure
(T2.21). The `host` field in each file is a label, not the machine's network
name (`host_is_label` is true).

`tests/test_compile_graph_breaks.py` checks the pattern table above against
the files.
