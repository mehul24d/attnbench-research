# Paper

IEEEtran journal skeleton. Its structure follows `docs/RESEARCH_PLAN.md`,
which wins over anything here. Every section is a stub: no results, no
numbers typed in, no claims.

## Layout

| Path | What it is |
|---|---|
| `main.tex` | Preamble, title block, the order of sections |
| `sections/*.tex` | One file per section, each a stub naming its plan section |
| `refs.bib` | Empty until a paper is read in full (rule 5) |
| `generated/numbers.tex` | Placeholder path for the numbers file from `scripts/paper_artifacts.py` (task C10, not yet built) |
| `figures/` | Generated figures (see `figures/README.md`) |
| `check_skeleton.py` | Static checks, no LaTeX needed |
| `Makefile` | Build and check targets |

## Rules the sources follow

- Every number is a macro from the numbers file. `check_skeleton.py` refuses
  a digit in typeset text, except in identifiers such as H1 or T7.1.
- If the numbers file is missing, both `make` and LaTeX itself stop with an
  error. There is no fallback value.
- A missing figure is a LaTeX error, except in `make skeleton`.
- A `refs.bib` entry needs `readinfull = {YYYY-MM-DD}`, the date the paper
  was read in full.
- No model identifiers and no assistant attribution (rule 9).
- Non-inferiority at a stated margin only (rule 1). No em dashes.

## Build

Needs TeX Live with the IEEEtran class (`pdflatex`, `bibtex`), and Python 3
for the checks. From this directory:

    make check                       # static checks
    make skeleton                    # layout only: empty numbers, boxed figures
    make NUMBERS=path/to/numbers.tex # full build to main.pdf
    make final                       # checks, and refuses any \stub left

Without `make`, a full build with the default numbers path is:

    pdflatex main.tex && pdflatex main.tex   # bibtex main in between once something is cited

or, with Tectonic, `tectonic main.tex`.
