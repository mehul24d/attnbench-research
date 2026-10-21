# Archive

Superseded planning documents, kept unedited for traceability. They record how
the current plan was reached, including every referee report and the response
to it. **None of them is authoritative.** The current plan is
`docs/RESEARCH_PLAN.md`.

| File | What it is |
|---|---|
| `estimator_frontier_plan_final_with_referee_responses.pdf` | Last version of the Estimator Frontier plan (4 Oct 2026), with responses to four referee reports and the novelty-search log. |
| `estimator_frontier_preregistration.md` | The estimator-frontier pre-registration, unlocked draft as last pushed on 4 Oct 2026 (`65b8f71`, branch head `459c892`). Superseded by the plan; moved here on 21 Oct 2026 (T0.16). Earlier documents cite it at its old path, `docs/estimator_frontier_preregistration.md`: `docs/claims.md`, `docs/limitations.md`, `docs/methodology_pdf_erratum.md`, `docs/writeup_input.md` and `docs/t4_xattention_calibrated.md`. Those citations now resolve here. |
| `frontier_prereg.py` | That pre-registration as code, formerly `attnbench/analysis/frontier_prereg.py`. The split firewall and matched row budgets that live code uses moved unchanged to `attnbench/analysis/splits_and_budgets.py`, and this file imports them from there. It is not on the import path. |
| `test_frontier_prereg_plan.py` | Its plan test, formerly in `tests/`. |
| `frontier_budget_gate.py`, `frontier_gate.sh`, `test_frontier_budget_gate.py` | The study's budget gate, the shell hook the GPU launchers sourced, and its test, formerly in `scripts/` and `tests/`. The launchers on `main` no longer source it. The gate amendment of 21 Oct 2026 (planning rates, ₹14,000 cap) was made on local branch `t11-run-rates-2026-10-20` and is not here. |

Files here are not collected by pytest (`testpaths` in `pyproject.toml`) and not imported by the harness.
