#!/bin/sh
# Rebuild the figures, the comparison tables (if the raw runs are given) and the PDF.
#   ./make_all.sh                 figures + PDF from data/
#   RAW=<folder> ./make_all.sh    also recompute data/metal_vs_reference.csv (see README)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY analysis/paper_figures.py
cd paper
pdflatex -interaction=nonstopmode LISFLOOD_8.4.tex >/dev/null
bibtex LISFLOOD_8.4 >/dev/null
pdflatex -interaction=nonstopmode LISFLOOD_8.4.tex >/dev/null
pdflatex -interaction=nonstopmode LISFLOOD_8.4.tex >/dev/null
echo "paper/LISFLOOD_8.4.pdf written"
