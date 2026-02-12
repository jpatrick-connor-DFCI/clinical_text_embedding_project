#!/bin/bash

#SBATCH --job-name=coxnet_full_event
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-0%1
#SBATCH --output=output/array_full_cohort/%A_%a.out
#SBATCH --error=error/array_full_cohort/%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/clinical_text_project/code}
MANIFEST=${MANIFEST:-$PROJECT_ROOT/bash_scripts/slurm_manifests/full_cohort_tasks.tsv}

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output/array_full_cohort error/array_full_cohort

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
if [[ -z "${TASK_LINE}" ]]; then
  echo "No manifest row ${LINE_NUM} in ${MANIFEST}"
  exit 1
fi

IFS=$'\t' read -r SCHEME EVENT <<< "${TASK_LINE}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

python "$PROJECT_ROOT/python_scripts/model_training/run_full_cohort_event.py" \
  --scheme "$SCHEME" \
  --event "$EVENT" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --max-iter "${COXNET_MAX_ITER:-3000}" \
  --backend "${COXNET_BACKEND:-threading}" \
  "${OVERWRITE_FLAG[@]}"

conda deactivate
