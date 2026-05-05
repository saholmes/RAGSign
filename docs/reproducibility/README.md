# Reproducibility report

This directory contains the standalone reproducibility evaluation of
the RAG-Sign paper, suitable for inclusion as a chapter in a PhD
thesis or for direct compilation into a PDF.

## Files

| File | What it is |
|---|---|
| `reproducibility.tex` | Self-contained LaTeX article. Compiles standalone with `pdflatex` and `bibtex`; can also be `\input`-ed into a thesis chapter. |
| `refs.bib` | Bibliography. |
| `iacr_results.json` | Aggregate empirical results — pure numbers, no paper text. The numbers in `reproducibility.tex`'s tables are derived from this file. |
| `Makefile` | `make` to build, `make clean` to tidy, `make distclean` to also remove the PDF. |

## Compiling

```bash
cd docs/reproducibility
make            # produces reproducibility.pdf
```

Requires a TeX Live (or MacTeX) install with the standard packages:
`booktabs`, `siunitx`, `pgfplots`, `tikz`, `hyperref`, `microtype`.
All of these are in the default TeX Live distribution.

## Inclusion in a thesis

The body content (between `\begin{document}` and `\end{document}`) can
be lifted unchanged into a thesis chapter. Replace the standalone
`\title` / `\author` / `\maketitle` block with the thesis's own
chapter heading. The bibliography entries in `refs.bib` can be merged
into the thesis-wide `.bib` file.

## Source data

The numbers in `iacr_results.json` were produced by the experiment
harness in this repo:

```bash
.venv/bin/python -m scripts.iacr_warm_cache --workers 8
.venv/bin/python -m scripts.iacr_experiment \
    --years 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 \
    --disjoint-year 2024 \
    --limit 100000 \
    --drift-pct 0.5 1 2 5 7.5 10 15
```

The IACR PDFs themselves are **not** included in this repository — they
remain the copyright of their respective authors. To re-run the
experiment from scratch, set `RAG_SIGN_IACR_DATA` to a directory
containing `iacr_<year>_papers/<year>-<id>.pdf` files (the IACR ePrint
Archive's natural layout) and re-run the commands above.
