# Vendor register

Third-party code is installed and called, never copied (plan rule 8), except
the entries below. An entry counts only once the researcher has approved it
and its date is in the "Approved" column. Code from a repository with no
licence is never vendored; its lines are reproduced only under a written
permission recorded in `docs/permission_log.md`.

Status on 18 Oct 2026: drafted. No approval dates yet.

## Vendored code

| Path | Upstream file | Commit | Licence | Verbatim or adapted | Licence and notice | Why not installed and called | Approved |
|---|---|---|---|---|---|---|---|
| `attnbench/_vendor/ruler/scoring.py` | NVIDIA/RULER, `scripts/eval/synthetic/constants.py` | `c3f5e3b` | Apache-2.0 | Verbatim. Code lines match upstream (checked 18 Oct 2026) | Upstream header kept; top-level `LICENSE` is the Apache 2.0 text; `NOTICE` item 1 | RULER is a set of scripts, not an installable package | |
| `attnbench/_vendor/ruler/niah.py` | NVIDIA/RULER, `scripts/data/synthetic/niah.py` | `c3f5e3b` | Apache-2.0 | Adapted: argparse and global RNG removed, essay and word seams injected | Upstream header kept; modification notice at the top of the file (Apache-2.0 section 4(b)); `LICENSE`; `NOTICE` item 1 | Upstream parses its command line at import | |
| `attnbench/_vendor/ruler/qa.py` | NVIDIA/RULER, `scripts/data/synthetic/qa.py` | `c3f5e3b` | Apache-2.0 | Adapted: readers take parsed JSON, local RNG | As for `niah.py` | Upstream parses its command line and loads data at import | |
| `attnbench/_vendor/ruler/variable_tracking.py` | NVIDIA/RULER, `scripts/data/synthetic/variable_tracking.py` | `c3f5e3b` | Apache-2.0 | Adapted: noise-haystack branch only, local RNG, few-shot machinery dropped | As for `niah.py` | Upstream parses its command line at import | |
| `attnbench/accuracy/ruler.py`, the `_NIAH_PARAMS` presets marked verbatim | NVIDIA/RULER, `scripts/synthetic.yaml` | `c3f5e3b` | Apache-2.0 | Verbatim configuration values | `NOTICE` item 1 | Configuration values, not code | |

`attnbench/_vendor/ruler/__init__.py` and `VENDORED.md` are this project's
own files. `VENDORED.md` records file by file what is verbatim and what is
adapted.

## Reproduced lines under written permission (no licence)

| Path | Upstream | Commit | What is reproduced | Permission |
|---|---|---|---|---|
| `attnbench/backends/xattention.py` | | | | |

## Pending

- `attnbench/_vendor/minference/` (microsoft/MInference, MIT) is on branch
  `frontier-prereg-2026-10-03`, not on this branch. Its entry is added when it
  arrives (T0.16).
