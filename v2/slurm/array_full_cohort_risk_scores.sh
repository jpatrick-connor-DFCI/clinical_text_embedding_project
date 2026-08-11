#!/bin/bash

#SBATCH --job-name=coxnet_full_risk
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=v2/slurm/array_full_cohort_risk_scores/output/%A_%a.out
#SBATCH --error=v2/slurm/array_full_cohort_risk_scores/error/%A_%a.err

# Requires run_full_cohort_event.py (array_full_cohort_run.sh) to have already produced
# text_val.csv for each row — this script picks best hyperparameters from that CV grid and
# generates held-out risk scores (run_full_cohort_risk_scores.py). Same manifest format as
# array_full_cohort_run.sh (scheme<TAB>event), so full_cohort_tasks.tsv is reused by default.

# ANCHOR selects the time-zero anchor (see v2/anchors.py): "treatment" (default) or
# "sequencing". Forwarded to run_full_cohort_risk_scores.py as --anchor; non-default anchors
# read/write under <scheme_results_dir>/anchor_<anchor>/ (schemes.py scheme_results_dir).
ANCHOR=${ANCHOR:-treatment}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
V2_ROOT="$PROJECT_ROOT/v2"
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST=${MANIFEST:-$V2_ROOT/slurm/slurm_manifests/full_cohort_tasks${ANCHOR_SUFFIX}.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-20}

if [[ ! -d "$V2_ROOT" ]]; then
  echo "v2 root not found: $V2_ROOT"
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

cd "$V2_ROOT"
export PYTHONPATH="$V2_ROOT"

module load miniforge3
eval "$(conda shell.bash hook)"
conda activate clinical_notes_project || { echo "Failed to activate conda env clinical_notes_project"; exit 1; }
python -c "import config" 2>/dev/null || { echo "config not importable - check conda env / PYTHONPATH"; exit 1; }

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p slurm/array_full_cohort_risk_scores/output slurm/array_full_cohort_risk_scores/error

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

TOTAL_ROWS=$(wc -l < "$MANIFEST")
START_LINE=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK + 1))
END_LINE=$((START_LINE + ROWS_PER_TASK - 1))

if [[ "$START_LINE" -gt "$TOTAL_ROWS" ]]; then
  echo "No rows assigned to task ${SLURM_ARRAY_TASK_ID} (start=${START_LINE}, total=${TOTAL_ROWS})"
  exit 0
fi
if [[ "$END_LINE" -gt "$TOTAL_ROWS" ]]; then
  END_LINE="$TOTAL_ROWS"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing manifest rows ${START_LINE}-${END_LINE}"

export CTEP_DATA_PATH=${CTEP_DATA_PATH:-${DATA_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project}}
RESULTS_ROOT="$CTEP_DATA_PATH/time-to-event_analysis/results"
FAILED_ROWS=0

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  IFS=$'\t' read -r RAW_SCHEME EVENT EXTRA_FIELD <<< "${TASK_LINE}"
  if [[ -n "${EXTRA_FIELD:-}" ]]; then
    echo "Unsupported manifest row ${LINE_NUM}: expected 2 tab-separated fields, got more"
    exit 1
  fi

  case "$RAW_SCHEME" in
    icd3) SCHEME="icd3_post" ;;
    icd4) SCHEME="icd4_post" ;;
    phecode) SCHEME="phecode_post" ;;
    *) SCHEME="$RAW_SCHEME" ;;
  esac

  case "$SCHEME" in
    icd3_post) SCHEME_RESULTS_DIR="level_3_ICD_post_results" ;;
    icd4_post) SCHEME_RESULTS_DIR="level_4_ICD_post_results" ;;
    phecode_post) SCHEME_RESULTS_DIR="phecode_post_results" ;;
    death_met) SCHEME_RESULTS_DIR="death_met_results" ;;
    *)
      echo "Unsupported scheme in manifest row ${LINE_NUM}: $SCHEME"
      exit 1
      ;;
  esac

  if [[ "$ANCHOR" == "treatment" ]]; then
    ANCHOR_SUBDIR="$SCHEME_RESULTS_DIR"
  else
    ANCHOR_SUBDIR="$SCHEME_RESULTS_DIR/anchor_$ANCHOR"
  fi
  mkdir -p "$RESULTS_ROOT/$ANCHOR_SUBDIR/full_cohort_risk_scores/$EVENT"

  echo "Running row ${LINE_NUM}: scheme=${SCHEME}, event=${EVENT}, anchor=${ANCHOR}"
  if ! python -m pipelines.training.run_full_cohort_risk_scores \
    --scheme "$SCHEME" \
    --event "$EVENT" \
    --anchor "$ANCHOR" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --max-iter "${COXNET_MAX_ITER:-5000}" \
    --backend "${COXNET_BACKEND:-threading}" \
    ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
    echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}"
    FAILED_ROWS=$((FAILED_ROWS + 1))
  fi
done

conda deactivate
if [[ "$FAILED_ROWS" -gt 0 ]]; then
  echo "$FAILED_ROWS manifest row(s) failed"
  exit 1
fi
