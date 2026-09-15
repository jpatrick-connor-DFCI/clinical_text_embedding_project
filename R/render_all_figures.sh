#!/usr/bin/env bash
# Render every manuscript figure on the primary metric (C-index).
#
# The manuscript reports Harrell's C-index as its single primary metric, so this
# wrapper renders the c-index set only. Mean AUC(t) is retained as an OPTIONAL
# sensitivity view: set MANUSCRIPT_METRICS="cindex auc" (or just "auc") to also
# render it. Metric-dependent panels are written to metric-tagged filenames
# (fig1b_cindex / fig1b_auc, ...), so the two sets never overwrite each other.
#
# Note the event-exclusion set is judged on the c-index in BOTH renders (see
# figure_utils.R::excluded_event_keys), so an AUC render shows the same
# endpoints as the c-index one and differs only in the metric plotted.
#
# Usage:
#   R/render_all_figures.sh                    # every plot script, c-index
#   R/render_all_figures.sh plot_figure_1.R    # just one, c-index
#   MANUSCRIPT_METRICS="cindex auc" R/render_all_figures.sh   # both sets
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
read -r -a metrics <<< "${MANUSCRIPT_METRICS:-cindex}"
for metric in "${metrics[@]}"; do
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
echo "All figures rendered for: ${metrics[*]}"
