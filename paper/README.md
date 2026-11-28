# Paper

The ISPASS 2027 submission, anonymised for double-blind review (paper
skeleton wave, 29 Nov 2026). Its structure follows `docs/RESEARCH_PLAN.md`,
which wins over anything here. The prose and the abstract are the author's;
every section holds a stub and, under it, the facts from
`docs/paper/claim_status.md` that belong there, each with its source file.

## Format (ISPASS 2027 submission page, fetched 29 Nov 2026)

- IEEE conference style: `\documentclass[conference]{IEEEtran}`. The page
  names "the IEEE conference style templates for Word and LaTeX provided by
  IEEE eXpress Conference Publishing" and no class file; IEEEtran in
  conference mode is that template's LaTeX class (the IEEE page returned no
  content when fetched, so this is an inference).
- Nine single-spaced, double-column pages, 10-point type, 8.5x11 inch paper;
  unlimited extra pages for references only. Figures and appendices count
  within the nine.
- Pages numbered (IEEEtran's conference mode omits numbers; `main.tex` turns
  them on). PDF only.
- Double-blind: no author names or other hints of identity. Cite one's own
  work in the third person; no tool or benchmark URL that reveals identity
  (state the intent to open-source it on acceptance).
- References list all authors.
- Substantial generative-AI use is disclosed in the acknowledgments.

## Layout

| Path | What it is |
|---|---|
| `main.tex` | Preamble, anonymous title block, section order |
| `sections/*.tex` | One file per section: a `\stub` and `\fact{... \src{file}}` drafting notes |
| `generated/` | Tables, figure environments and `numbers.tex`, from `scripts/paper_figures.py` |
| `figures/` | Figure PDFs and each figure's plotted values as CSV, from the same script |
| `refs.bib` | Empty until a paper is read in full (rule 5) |
| `check_skeleton.py` | Static checks, no LaTeX needed |
| `build.py` | Build, page count, identifier scan, identity hints |
| `Makefile` | The older pdflatex route (TeX Live with IEEEtran) |

`sections/appendix_artifact.tex` is kept for the camera-ready artifact
appendix and is not input: in the submission, appendices count within the
nine pages.

## Rules the sources follow

- Every number in the text is a macro from `generated/numbers.tex` or sits in
  a generated table or figure. `check_skeleton.py` refuses a digit in typeset
  text outside identifiers, `\stub` and `\fact`.
- Every caption states whether it is pre-registered or exploratory.
- Names a reviewer could trace (the lock tag, commit hashes) are macros in
  `numbers.tex`, so they can be blanked in one place.
- A `refs.bib` entry needs `readinfull = {YYYY-MM-DD}`.
- No model identifiers (rule 9). No em dashes.

## Build

Needs Tectonic (`brew install tectonic`) and Python 3 with pypdf and
matplotlib. From the repository root:

    python3 scripts/paper_figures.py --out paper   # regenerate tables and figures
    python3 paper/build.py                         # draft build and checks
    python3 paper/build.py --submission            # also refuses any \stub or \fact

`build.py` writes `paper/main.pdf` and `paper/build/BUILD.md`. Set
`ARTIFACT_SCAN_LITERALS` (one value per line, never printed) for the
identifier scan's literals; without it the scan checks patterns only and the
build fails. `scripts/paper_reproduce.py` regenerates the tables and figures
from banked data and compares them byte for byte.
