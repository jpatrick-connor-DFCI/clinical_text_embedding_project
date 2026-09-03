#!/usr/bin/env bash
# Render every manuscript figure for BOTH metrics.
#
# The plot scripts pick their metric up from MANUSCRIPT_METRIC and default to
# "auc" when it is unset (see figure_utils.R). Metric-dependent panels are
# written to metric-tagged filenames (fig1b_cindex / fig1b_auc, ...), so running
# a script once renders only one of the two sets -- and running it with the
# variable unset renders the AUC set while looking like a complete run. This
# wrapper exists so that "render the figures" always means both.
#
# Usage:
#   R/render_all_figures.sh                 # every plot script, both metrics
#   R/render_all_figures.sh plot_figure_1.R # just one, both metrics
#
# Honors CTEP_FIGURE_DATA_DIR and CLINICAL_FIGURES_OUT as the plot scripts do.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

if [ "$#" -gt 0 ]; then
  scripts=()
  for a in "$@"; do scripts+=("R/$(basename "$a")"); done
else
  scripts=(R/plot_figure_*.R)
fi

fail=0
failed_runs=()
for metric in cindex auc; do
  for sc in "${scripts[@]}"; do
    [ -f "$sc" ] || { echo "!! no such script: $sc" >&2; fail=1; continue; }
    echo "=== $(basename "$sc") [$metric] ==="
    if ! MANUSCRIPT_METRIC="$metric" Rscript "$sc"; then
      fail=1
      failed_runs+=("$(basename "$sc") [$metric]")
    fi
  done
done

echo
if [ "$fail" -ne 0 ]; then
  echo "FAILED:"
  for r in "${failed_runs[@]}"; do echo "  - $r"; done
  exit 1
fi
echo "All figures rendered for both metrics."
