#!/bin/bash

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/clinical_text_project/code}
MANIFEST=${MANIFEST:-$PROJECT_ROOT/bash_scripts/slurm_manifests/full_cohort_tasks.tsv}
MAX_CONCURRENT=${MAX_CONCURRENT:-20}

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

NUM_TASKS=$(wc -l < "$MANIFEST")
if [[ "$NUM_TASKS" -le 0 ]]; then
  echo "Manifest has no rows: $MANIFEST"
  exit 1
fi

ARRAY_SPEC="0-$((NUM_TASKS - 1))%${MAX_CONCURRENT}"
echo "Submitting full-cohort array with ${NUM_TASKS} tasks (${ARRAY_SPEC})"

sbatch \
  --array="$ARRAY_SPEC" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST" \
  "$PROJECT_ROOT/bash_scripts/array_full_cohort_run.sh"
