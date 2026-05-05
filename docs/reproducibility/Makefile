# Reproducibility report build.
#
#   make           # build reproducibility.pdf
#   make clean     # remove build artefacts
#   make distclean # also remove the PDF

LATEX  ?= pdflatex
BIBTEX ?= bibtex
TEX     = reproducibility.tex
BIB     = refs.bib
PDF     = reproducibility.pdf

.PHONY: all clean distclean

all: $(PDF)

$(PDF): $(TEX) $(BIB)
	$(LATEX)  -interaction=nonstopmode $(TEX)
	$(BIBTEX) $(basename $(TEX))
	$(LATEX)  -interaction=nonstopmode $(TEX)
	$(LATEX)  -interaction=nonstopmode $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz *.fls *.fdb_latexmk

distclean: clean
	rm -f $(PDF)
