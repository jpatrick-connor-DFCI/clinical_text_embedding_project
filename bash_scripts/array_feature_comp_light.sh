#!/bin/bash

#SBATCH --job-name=coxnet_feat_light
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --array=0-0%1
#SBATCH --output=output/array_feature_light/%A_%a.out
#SBATCH --error=error/array_feature_light/%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
MANIFEST=${MANIFEST:-$PROJECT_ROOT/bash_scripts/slurm_manifests/feature_comp_light_tasks.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-30}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

cd "$PROJECT_ROOT"

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output/array_feature_light error/array_feature_light

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

DATA_PATH=${DATA_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project}
RESULTS_ROOT="$DATA_PATH/time-to-event_analysis/results"

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  IFS=$'\t' read -r SCHEME EVENT MODALITY <<< "${TASK_LINE}"

  case "$SCHEME" in
    icd3) SCHEME_RESULTS_DIR="level_3_ICD_results" ;;
    icd4) SCHEME_RESULTS_DIR="level_4_ICD_results" ;;
    phecode) SCHEME_RESULTS_DIR="phecode_results" ;;
    death_met) SCHEME_RESULTS_DIR="death_met_results" ;;
    *)
      echo "Unsupported scheme in manifest row ${LINE_NUM}: $SCHEME"
      exit 1
      ;;
  esac
  mkdir -p "$RESULTS_ROOT/$SCHEME_RESULTS_DIR/feature_comps/$EVENT"

  echo "Running row ${LINE_NUM}: scheme=${SCHEME}, event=${EVENT}, modality=${MODALITY}"
  python "$PROJECT_ROOT/python_scripts/model_training/run_feature_comp_task.py" \
    --scheme "$SCHEME" \
    --event "$EVENT" \
    --modality "$MODALITY" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --max-iter "${COXNET_MAX_ITER:-2500}" \
    --backend "${COXNET_BACKEND:-threading}" \
    "${OVERWRITE_FLAG[@]}"
done

conda deactivate
