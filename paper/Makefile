# Build the paper. Run from this directory.
#
#   make            full build; needs the generated numbers file and every figure
#   make skeleton   layout-only build: empty numbers file, boxed figure placeholders
#   make check      static checks on the sources (no LaTeX needed)
#   make final      static checks that also refuse any remaining \stub
#   make clean
#
# NUMBERS is the numbers file from scripts/paper_artifacts.py (task C10).
# The default is a placeholder path until C10 fixes it.
#
# Engine: pdflatex and bibtex (TeX Live, with the IEEEtran class). Tectonic
# also works for a full build with the default NUMBERS path:
#   tectonic main.tex
# It does not take NUMBERS= or the skeleton mode.

NUMBERS  ?= generated/numbers.tex
BUILD    := build
PDFLATEX ?= pdflatex
BIBTEX   ?= bibtex
PYTHON   ?= python3
LATEXFLAGS := -interaction=nonstopmode -halt-on-error -file-line-error \
              -output-directory=$(BUILD) -jobname=main

SOURCES := main.tex refs.bib $(wildcard sections/*.tex)

.PHONY: all pdf skeleton check final clean need-numbers need-engine

all: pdf

pdf: check need-engine need-numbers
	@mkdir -p $(BUILD)
	$(call latex_run,$(NUMBERS),0)
	@cp $(BUILD)/main.pdf main.pdf
	@echo "built main.pdf"

skeleton: check need-engine
	@mkdir -p $(BUILD)
	@printf '%% Empty numbers file for the skeleton build. Not data.\n' > $(BUILD)/skeleton_numbers.tex
	$(call latex_run,$(BUILD)/skeleton_numbers.tex,1)
	@cp $(BUILD)/main.pdf skeleton.pdf
	@echo "built skeleton.pdf (layout only, no data)"

check:
	$(PYTHON) check_skeleton.py

final:
	$(PYTHON) check_skeleton.py --final

need-numbers:
	@test -f "$(NUMBERS)" || { \
	  echo "error: numbers file '$(NUMBERS)' is missing." >&2; \
	  echo "Run scripts/paper_artifacts.py (task C10) or pass NUMBERS=path." >&2; \
	  exit 1; }

need-engine:
	@command -v $(PDFLATEX) >/dev/null || { \
	  echo "error: $(PDFLATEX) not found. Install TeX Live with IEEEtran," >&2; \
	  echo "or run 'tectonic main.tex' for a full build." >&2; exit 1; }

# $(1) numbers file, $(2) skeleton flag. bibtex runs only once something is
# cited: with no \citation it exits with an error, and refs.bib starts empty.
define latex_run
	$(PDFLATEX) $(LATEXFLAGS) '\def\NumbersFile{$(1)}\def\SkeletonBuild{$(2)}\input{main.tex}'
	if grep -q '^\\citation' $(BUILD)/main.aux; then \
	  BIBINPUTS=.: $(BIBTEX) $(BUILD)/main; fi
	$(PDFLATEX) $(LATEXFLAGS) '\def\NumbersFile{$(1)}\def\SkeletonBuild{$(2)}\input{main.tex}'
	$(PDFLATEX) $(LATEXFLAGS) '\def\NumbersFile{$(1)}\def\SkeletonBuild{$(2)}\input{main.tex}'
endef

clean:
	rm -rf $(BUILD) main.pdf skeleton.pdf
