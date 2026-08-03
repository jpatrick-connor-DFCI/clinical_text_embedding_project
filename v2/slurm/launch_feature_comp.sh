#!/bin/bash
# Launch wrapper for array_feature_comp.sh.
#
# Reads the manifest, computes the required array size, and submits the job.
# Any environment variables accepted by array_feature_comp.sh can be
# forwarded here (PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE,
# COXNET_MAX_ITER, COXNET_BACKEND).
#
# Usage:
#   bash launch_feature_comp.sh
#   OVERWRITE=1 bash launch_feature_comp.sh
#   ROWS_PER_TASK=5 bash launch_feature_comp.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/v2/slurm/slurm_manifests/feature_comp_tasks.tsv}"
ROWS_PER_TASK="${ROWS_PER_TASK:-7}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: manifest not found: $MANIFEST"
    exit 1
fi

N_ROWS=$(wc -l < "$MANIFEST")
if [[ "$N_ROWS" -eq 0 ]]; then
    echo "Error: manifest is empty: $MANIFEST"
    exit 1
fi

# Ceiling division: number of SLURM array tasks needed
N_TASKS=$(( (N_ROWS + ROWS_PER_TASK - 1) / ROWS_PER_TASK ))
MAX_TASK=$(( N_TASKS - 1 ))

echo "Manifest:      $MANIFEST"
echo "Rows:          $N_ROWS"
echo "Rows per task: $ROWS_PER_TASK"
echo "Array tasks:   $N_TASKS  (--array=0-${MAX_TASK})"

sbatch \
    --array="0-${MAX_TASK}" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK" \
    "$PROJECT_ROOT/v2/slurm/array_feature_comp.sh"
