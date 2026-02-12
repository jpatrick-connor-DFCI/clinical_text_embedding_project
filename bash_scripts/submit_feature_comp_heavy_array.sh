#!/bin/bash

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
MANIFEST=${MANIFEST:-$PROJECT_ROOT/bash_scripts/slurm_manifests/feature_comp_heavy_tasks.tsv}
MAX_CONCURRENT=${MAX_CONCURRENT:-12}
ROWS_PER_TASK=${ROWS_PER_TASK:-12}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "Project root not found: $PROJECT_ROOT"
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
mkdir -p "$PROJECT_ROOT/output/array_feature_heavy" "$PROJECT_ROOT/error/array_feature_heavy"

cd "$PROJECT_ROOT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

NUM_ROWS=$(wc -l < "$MANIFEST")
if [[ "$NUM_ROWS" -le 0 ]]; then
  echo "Manifest has no rows: $MANIFEST"
  exit 1
fi

NUM_TASKS=$(( (NUM_ROWS + ROWS_PER_TASK - 1) / ROWS_PER_TASK ))
ARRAY_SPEC="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}"
echo "Submitting feature-heavy array with ${NUM_TASKS} tasks for ${NUM_ROWS} rows (${ARRAY_SPEC}), ${ROWS_PER_TASK} rows/task"

sbatch \
  --array="$ARRAY_SPEC" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK" \
  "$PROJECT_ROOT/bash_scripts/array_feature_comp_heavy.sh"
