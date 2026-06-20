#!/usr/bin/env bash
# Regenerate every paper figure + the graphical abstract from the committed artefacts
# (Level B; no BOPTEST, no GPU). Thin wrapper around make_figures.py.
#   ./make_all_figures.sh            # all figures
#   ./make_all_figures.sh --list     # list scripts
set -euo pipefail
cd "$(dirname "$0")"
exec "${PYTHON:-python}" make_figures.py "$@"
