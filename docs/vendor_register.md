# Vendor register

Third-party code is installed and called, never copied (plan rule 8), except
the entries below. An entry counts only once the researcher has approved it
and its date is in the "Approved" column. Code from a repository with no
licence is never vendored; its lines are reproduced only where
`docs/vendor_register.md` lists them and the researcher's approval is
recorded in `docs/permission_log.md`.

Status on 18 Oct 2026: drafted. No approval dates yet.

Status on 21 Oct 2026: every row states its licence status and a credit line
(authors or project, and code location). The five vendored rows are approved
by the researcher on that date. The reproduced-lines table has no approval
column, and its permission column is unchanged.

Later on 21 Oct 2026 (researcher's decision): the `xattention.py` row's
permission cell points to the researcher's approval of 21 Oct 2026 in
`docs/permission_log.md`.

27 Oct 2026 (researcher's decision): `XAttentionBackend.forward` now builds
the kernel's cumulative-length tensors exactly as `Xattention_prefill` does
at `e379887`: three device tensors from Python lists (query lengths, key
lengths, head-mask type), where it built one shared tensor and a
`torch.ones` fill before. The outputs are bitwise identical
(`tests/test_xattention_backend.py`); the AST fingerprint of the
`xattention` and `xattention_noassert` timed regions changes, so the T1.1
results carry the earlier fingerprint.

## Vendored code

| Path | Upstream file | Commit | Licence | Credit | Verbatim or adapted | Licence and notice | Why not installed and called | Approved |
|---|---|---|---|---|---|---|---|---|
| `attnbench/_vendor/ruler/scoring.py` | NVIDIA/RULER, `scripts/eval/synthetic/constants.py` | `c3f5e3b` | Apache-2.0 | NVIDIA Corporation, the RULER project; code at github.com/NVIDIA/RULER | Verbatim. Code lines match upstream (checked 18 Oct 2026) | Upstream header kept; top-level `LICENSE` is the Apache 2.0 text; `NOTICE` item 1 | RULER is a set of scripts, not an installable package | 21 Oct 2026 |
| `attnbench/_vendor/ruler/niah.py` | NVIDIA/RULER, `scripts/data/synthetic/niah.py` | `c3f5e3b` | Apache-2.0 | NVIDIA Corporation, the RULER project; code at github.com/NVIDIA/RULER | Adapted: argparse and global RNG removed, essay and word seams injected | Upstream header kept; modification notice at the top of the file (Apache-2.0 section 4(b)); `LICENSE`; `NOTICE` item 1 | Upstream parses its command line at import | 21 Oct 2026 |
| `attnbench/_vendor/ruler/qa.py` | NVIDIA/RULER, `scripts/data/synthetic/qa.py` | `c3f5e3b` | Apache-2.0 | NVIDIA Corporation, the RULER project; code at github.com/NVIDIA/RULER | Adapted: readers take parsed JSON, local RNG | As for `niah.py` | Upstream parses its command line and loads data at import | 21 Oct 2026 |
| `attnbench/_vendor/ruler/variable_tracking.py` | NVIDIA/RULER, `scripts/data/synthetic/variable_tracking.py` | `c3f5e3b` | Apache-2.0 | NVIDIA Corporation, the RULER project; code at github.com/NVIDIA/RULER | Adapted: noise-haystack branch only, local RNG, few-shot machinery dropped | As for `niah.py` | Upstream parses its command line at import | 21 Oct 2026 |
| `attnbench/accuracy/ruler.py`, the `_NIAH_PARAMS` presets marked verbatim | NVIDIA/RULER, `scripts/synthetic.yaml` | `c3f5e3b` | Apache-2.0 | NVIDIA Corporation, the RULER project; code at github.com/NVIDIA/RULER | Verbatim configuration values | `NOTICE` item 1 | Configuration values, not code | 21 Oct 2026 |

`attnbench/_vendor/ruler/__init__.py` and `VENDORED.md` are this project's
own files. `VENDORED.md` records file by file what is verbatim and what is
adapted.

## Reproduced lines from repositories with no licence

| Path | Upstream | Commit | Licence | Credit | What is reproduced | Permission |
|---|---|---|---|---|---|---|
| `attnbench/backends/xattention.py` | `mit-han-lab/x-attention` | `e379887` | None (upstream has no licence) | Xu et al., "XAttention: Block Sparse Attention with Antidiagonal Scoring", ICML 2025 (arXiv:2503.16428); code at github.com/mit-han-lab/x-attention | Reproduced lines from upstream commit `e379887`; upstream has no licence. From 27 Oct 2026 these include `Xattention_prefill`'s three list-built tensors for the kernel call (`q_cu_seq_lens`, `k_cu_seq_lens`, `head_mask_type`) | Researcher's approval recorded in `docs/permission_log.md`, 21 Oct 2026 |

## Pending

- `attnbench/_vendor/minference/` (microsoft/MInference, MIT) is on branch
  `frontier-prereg-2026-10-03`, not on this branch. Its entry is added when it
  arrives (T0.16). *(21 Oct 2026, T0.16: it did not arrive. It is in commit
  `ab94c81`, which exists only on the researcher's local copy of that branch
  and was left out of the merge. Nothing from MInference is on `main`.)*
