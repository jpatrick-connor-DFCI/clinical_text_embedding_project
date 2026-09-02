#!/bin/bash
# Launch wrapper for array_full_cohort_run.sh.
#
# Reads the manifest, computes the required array size, and submits the job.
# Any environment variables accepted by array_full_cohort_run.sh can be
# forwarded here (PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE,
# COXNET_MAX_ITER, COXNET_BACKEND, ANCHOR, PARTITION, CPUS_PER_TASK, MEMORY).
#
# Usage:
#   bash launch_full_cohort.sh
#   OVERWRITE=1 bash launch_full_cohort.sh
#   ROWS_PER_TASK=5 bash launch_full_cohort.sh
#   ANCHOR=sequencing MANIFEST=.../full_cohort_tasks__sequencing.tsv bash launch_full_cohort.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
ANCHOR="${ANCHOR:-treatment}"
PARTITION="${PARTITION:-normal}"
SBATCH_PARTITION_ARGS=()
if [[ -n "$PARTITION" ]]; then
    SBATCH_PARTITION_ARGS=(--partition="$PARTITION")
fi
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST="${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/full_cohort_tasks${ANCHOR_SUFFIX}.tsv}"
ROWS_PER_TASK="${ROWS_PER_TASK:-20}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
COXNET_MAX_ITER="${COXNET_MAX_ITER:-1000}"
COXNET_BACKEND="${COXNET_BACKEND:-threading}"

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
echo "Resources:     ${CPUS_PER_TASK} CPUs, ${MEMORY} per array task"
echo "Cox fitting:   max_iter=${COXNET_MAX_ITER}, backend=${COXNET_BACKEND}"

# The array script's #SBATCH --output/--error are relative to the submission CWD, not to
# $PROJECT_ROOT (which the script only cd's into after the job starts) -- override them here
# with absolute paths so log location doesn't depend on where this launcher happens to be run
# from.
mkdir -p "$PROJECT_ROOT/slurm/array_full_cohort_run/output" "$PROJECT_ROOT/slurm/array_full_cohort_run/error"
sbatch \
    ${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"} \
    --array="0-${MAX_TASK}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --output="$PROJECT_ROOT/slurm/array_full_cohort_run/output/%A_%a.out" \
    --error="$PROJECT_ROOT/slurm/array_full_cohort_run/error/%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",ANCHOR="$ANCHOR",COXNET_MAX_ITER="$COXNET_MAX_ITER",COXNET_BACKEND="$COXNET_BACKEND" \
    "$PROJECT_ROOT/slurm/array_full_cohort_run.sh"
