#!/bin/bash

#SBATCH --job-name=coxnet_feat_heavy
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-0%1
#SBATCH --output=output/array_feature_heavy/%A_%a.out
#SBATCH --error=error/array_feature_heavy/%A_%a.err

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/clinical_text_project/code}
MANIFEST=${MANIFEST:-$PROJECT_ROOT/bash_scripts/slurm_manifests/feature_comp_heavy_tasks.tsv}

source /PHShome/jpc91/.bashrc
module load miniforge3
conda activate clinical_notes_project

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p output/array_feature_heavy error/array_feature_heavy

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
if [[ -z "${TASK_LINE}" ]]; then
  echo "No manifest row ${LINE_NUM} in ${MANIFEST}"
  exit 1
fi

IFS=$'\t' read -r SCHEME EVENT MODALITY <<< "${TASK_LINE}"

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

python "$PROJECT_ROOT/python_scripts/model_training/run_feature_comp_task.py" \
  --scheme "$SCHEME" \
  --event "$EVENT" \
  --modality "$MODALITY" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
  --max-iter "${COXNET_MAX_ITER:-3000}" \
  --backend "${COXNET_BACKEND:-threading}" \
  "${OVERWRITE_FLAG[@]}"

conda deactivate
