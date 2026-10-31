# Artifact appendix: how to reproduce the study

This guide says how to rebuild the study's results from the repository. It
names no private paths, cloud project identifiers or credentials. Where a
value is private, the guide names the environment variable that holds it,
and that variable is set outside the repository.

`docs/RESEARCH_PLAN.md` is the source of truth for what the study measures
and why. If this guide disagrees with it, the plan wins. The paper's
appendix (`paper/sections/appendix_artifact.tex`) summarises this file.

Status: written before any confirmatory pairing has run. Sections that
depend on later tasks say which task fills them.

## 1. What the artifact contains

| Path | What it is |
|---|---|
| `attnbench/` | The measurement harness: provenance, timing, masks, backends, the accuracy pipeline, the cost model and its baselines (`attnbench/predict/`), and analysis |
| `scripts/` | Session runners, figure scripts (`scripts/figures/`) and the GCP tooling |
| `tests/` | The test suite, including checks on the docs and on call sites |
| `configs/` | Run configurations |
| `results/` | Banked, commit-stamped evidence. Never edited |
| `docs/` | The plan, pre-registrations, audit history and permission log |
| `paper/` | The paper sources (see `paper/README.md`) |

Third-party code is installed and called, not copied, except for what
`docs/vendor_register.md` lists.

## 2. Software environment

1. Python as required by `pyproject.toml`.
2. Install the harness from the repository root:

       pip install -e .

3. Optional extras, by purpose, are listed in `pyproject.toml`: `kernels`
   (GPU kernels, installed per machine), `eval` (the accuracy pipeline, with
   exact pins for the RULER word lists and sentence splitter) and
   `ruler-data` (building the RULER data directory only).
4. XAttention is installed on a GPU instance by
   `scripts/install_xattention.sh`, at the upstream commit the plan pins. The
   script checks that the import resolves to that commit, unedited.

## 3. Checks that need no GPU

From the repository root:

    python -m pytest -q

On a CPU-only machine, five tests in `test_generation_wiring.py`,
`test_inline_estimator.py` and `test_swappable_attention_model.py` report as
expected failures, because PyTorch's CPU flash SDPA kernel rejects
grouped-query attention shapes. The mark is conditional on a probe of that
call and strict (`tests/_sdpa_env.py`), so any other failure in those tests
still fails the suite.

The torch.compile graph-break check runs on a CPU:
`scripts/check_compile_graph_breaks.py`, with its banked output in
`results/compile_graph_breaks_cpu/`.

## 4. Data

The RULER tasks need a data directory built the way RULER builds it:

    pip install -e '.[ruler-data]'
    python scripts/fetch_ruler_data.py --out DIR

The default output directory is `$ATTNBENCH_RULER_DATA`. On a GPU instance,
`scripts/gcp_bootstrap_instance.sh` installs the data and checks it against
its hash pins before any accuracy run.

## 5. GPU sessions

Each pairing in plan Section 5.2 is one or more sessions on a rented
instance. The machine types and zones are pinned there. Every session
follows the same order:

1. Cleanup check (`scripts/gcp_cleanup_check.sh`), a cost cap and a
   hard-delete limit.
2. Quarantine of stale results.
3. Deploy the clean commit (`scripts/gcp_deploy_source.sh`). The script
   refuses unless the instance describes exactly that commit.
4. Preflight (`scripts/gcp_preflight_instance.sh`): refuses an instance whose
   `results/` tree is not empty.
5. The storage write gate (`scripts/gcp_verify_gcs_writable.sh`).
6. Bootstrap a fresh instance (`scripts/gcp_bootstrap_instance.sh`).
7. Run each phase through `scripts/run_phase.sh`, whose last action is
   always a sync of everything on disk to the results bucket.
8. Teardown through `scripts/gcp_teardown_session.sh`.

Configuration is through environment variables set outside the repository:

| Variable | Used by | Meaning |
|---|---|---|
| `ATTNBENCH_GCP_PROJECT` | every `scripts/gcp_*.sh` | The cloud project. The scripts refuse while the active project differs, and never print it |
| `ATTNBENCH_BUCKET` | `scripts/run_phase.sh` | Optional. The results bucket; otherwise named after the instance's own project, read from the metadata server |
| `GCP_RATE_INR_HR` | teardown and the A100 launcher | The planning rate of the card used (plan Section 5.2) |
| `ATTNBENCH_RULER_DATA` | the accuracy pipeline | The RULER data directory |

`scripts/_launch_policy.sh` enforces the launch-date and region rules in
every launch script.

A reproducer on another cloud account sets these variables to their own
values. Nothing in the repository depends on the original account.

## 6. Provenance and what counts as a result

- Every row is stamped with the commit that produced it, and counts only
  with `git_dirty = False` (`attnbench/provenance.py`).
- Results from different mask-rule eras or code paths are never pooled. The
  code path is recorded on every row.
- Oracle, deployable-estimator, CPU-builder and GPU-builder results are kept
  apart.
- Accuracy is reported as non-inferiority at a stated margin.

## 7. Predictions before data

Each pairing's predictions are written, hashed and synced before its timed
workload starts (`attnbench/predict/commit.py`, task T2.11). The analysis
refuses a prediction stamped later than the rule in plan Section 5.1 allows.
A reproducer checks each prediction file against its recorded sha256 before
reading any score.

## 8. Figures and the paper

- Figure scripts are in `scripts/figures/`. Each figure is made from banked
  results and its `meta.json` (`scripts/figures/figure_meta.py`); no figure
  is made from a script's demo mode.
- The paper's numbers come from one generated file, written by
  `scripts/paper_artifacts.py` (task C10, not yet built). The paper never
  builds without it.
- Build the paper from `paper/`:

      make check
      make NUMBERS=path/to/numbers.tex

  This needs TeX Live with the IEEEtran class. `paper/README.md` has the
  details.

## 9. Not included

- Credentials, cloud project identifiers and bucket names.
- The machine images used on the instances; the bootstrap script installs
  what a fresh instance lacks.
- The model weights, which are downloaded from their publishers under their
  own licences.
