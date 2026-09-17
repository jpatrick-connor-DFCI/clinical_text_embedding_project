#!/usr/bin/env bash
# Render C-index panels and compiled Figures 1–4 (PNG/PDF).
# Usage: R/render_all_figures.sh [plot_figure_1.R ...]
# Honors CTEP_FIGURE_DATA_DIR and CLINICAL_FIGURES_OUT.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

# Override both legacy controls, including values inherited from the caller.
export MANUSCRIPT_METRIC=cindex
export MANUSCRIPT_METRICS=cindex
echo "C-index-only renderer: $(pwd)/R/render_all_figures.sh"

if [ "$#" -gt 0 ]; then
  scripts=()
  for a in "$@"; do scripts+=("R/$(basename "$a")"); done
else
  scripts=(R/plot_figure_*.R)
fi

fail=0
failed_runs=()
for sc in "${scripts[@]}"; do
  [ -f "$sc" ] || { echo "!! no such script: $sc" >&2; fail=1; failed_runs+=("$sc (missing)"); continue; }
  echo "=== $(basename "$sc") [cindex] ==="
  if ! Rscript "$sc"; then
    fail=1
    failed_runs+=("$(basename "$sc") [cindex]")
  fi
done

echo
if [ "$fail" -ne 0 ]; then
  echo "FAILED:"
  for r in "${failed_runs[@]}"; do echo "  - $r"; done
  exit 1
fi
echo "All figures rendered using C-index."
