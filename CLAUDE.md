# CLAUDE.md

## Read this first

**`docs/RESEARCH_PLAN.md` is the single source of truth for this project.** Read it before starting any task. It holds the research question, hypotheses, method, operating rules and the task list (Section 8). If anything else (an older doc, a code comment, memory of an earlier session) conflicts with it, the plan wins. Flag the conflict instead of following the other source.

The study: **predicting when training-free block-sparse prefill pays end to end on real host-GPU pairings** (a component-composed critical-path model, G1/G2) and **testing attention recall as an estimator-independent accuracy proxy** (G3). The repository started as a benchmark of sparse attention speedups at matched accuracy. That earlier work is banked evidence, not the current goal.

## How to work here

- Pick tasks from Section 8 of the plan by ID (for example "T2.8"). Name the task ID in commit messages.
- When a task's status changes, update its row in `docs/RESEARCH_PLAN.md` in the same commit. Then regenerate the PDF with `python tools/build_plan_pdf.py`, which needs `pip install markdown` and Playwright's Chromium.
- Do not change scope (gaps, hypotheses, pairings, arms, budget) on your own. Propose it. The researcher decides, usually in Claude chat, and the decision is then written into the plan with a dated line in Appendix A.
- `docs/code_inventory.md` records which existing code is kept, adapted or archived. Check it before reusing or deleting a module.

## Rules (full list in plan Section 6)

1. Never claim "no accuracy loss" or "zero accuracy cost". Report non-inferiority at a stated margin.
2. Keep oracle, deployable-estimator, CPU-builder and GPU-builder results distinct.
3. Measurements are commit-stamped with `git_dirty = False`. Never pool across mask-rule eras or code paths.
4. Never delete or rewrite historical audit findings (`docs/limitations.md`, `claims.md`, `withdrawn_figures.md`, `silent_failure_patterns.md`, `audit_register.md`). Annotate instead.
5. Cite only papers read in full. If a paper is needed, ask the researcher for the PDF.
6. Write and hash predictions before the data they predict.
7. Run the focused tests for what you changed, and keep the full suite green (`python -m pytest -q`). In a CPU-only container, 5 tests in `test_generation_wiring.py`, `test_inline_estimator.py` and `test_swappable_attention_model.py` report as xfail: PyTorch's CPU flash SDPA kernel rejects grouped-query attention shapes. The mark is conditional on a probe of that call and strict (`tests/_sdpa_env.py`), so any other failure in those tests still fails the suite.
8. No unrelated changes in a commit. Third-party code is installed and called, never copied, except as listed in `docs/vendor_register.md`. Each row there states the upstream, its licence status and a credit line, and counts only once the researcher has approved it.
9. No model identifiers in research artifacts.
10. Any use of results that needs the researcher's permission must be in `docs/permission_log.md` before the use. If it is not in the log, it is not approved.
11. Any push or merge needs the researcher's explicit yes to that exact action, naming files, branch and destination, recorded in the permission log before acting.

## Cloud sessions

- Before any GPU session, state the exact command, estimated cost, cost cap, hard-delete limit and artifact path, then wait for approval. The researcher runs all cloud commands from their Mac.
- The sequence is: cleanup check, cost cap, hard-delete limit, quarantine of stale results, deploy the clean commit, preflight, run, then teardown through `scripts/gcp_teardown_session.sh`. Record the cost in `docs/spend_ledger.md`.
- Every `scripts/gcp_*.sh` takes the GCP project from `ATTNBENCH_GCP_PROJECT`, set outside the repository, and refuses while the active gcloud project differs. None prints it. On the instance, `scripts/run_phase.sh` names the results bucket after the instance's own project, read from the metadata server, unless `ATTNBENCH_BUCKET` is set, and refuses the phase if it has neither. The teardown and the A100 launcher refuse without `GCP_RATE_INR_HR`, the planning rate of the card used.
- Machines and zones are pinned in plan Section 5.2. The L4 zone is `asia-northeast1-a`. Never use an Australia region.
- No GCP session launches after **9 December 2026** (India time; the date itself is allowed). `scripts/_launch_policy.sh` enforces this and the Australia ban in every launch script; do not bypass it.
- No AWS launches until task T5.1 is done.

## Layout

- `attnbench/` is the harness: provenance, timing, masks, backends, the accuracy pipeline and analysis.
- `scripts/` holds the runners and GCP tooling.
- `tests/` has many tests that also check docs and call sites. Moving a file means updating its test references in the same commit.
- `docs/` holds the plan, the inventory, the audit history, the pre-registrations and the permission log. `docs/archive/` holds superseded plans.
- `docs/brief/brief.html` is the professor brief's source. Rebuild `docs/research_brief.pdf` with `python tools/build_brief_pdf.py` whenever it changes. Brief style: plain language, short sentences, no em dashes.
- `results/` is banked, commit-stamped evidence. Do not edit it.
