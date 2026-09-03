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
#
# SCHEDULER SIZING. The binding constraint is the scheduler, not the node: many short jobs
# queue worse than a few long ones. A row here fits BOTH the text and base models on the
# full cohort (no common-modality-MRN restriction), so it costs more than a feature-comp
# row -- budget ~75m. Against the 24h wall that puts ROWS_PER_TASK at 16 (~20h, ~83%
# utilization) rather than 20, which would overshoot if rows run at the top of their range.
# The array script is deadline-guarded and measures actual per-row cost at runtime, so it
# stops dispatching and defers the rest rather than being killed mid-row; re-running the
# same array picks the deferred rows up, since completed rows are skipped. Raise
# ROWS_PER_TASK once you have real timings in the logs.

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
ROWS_PER_TASK="${ROWS_PER_TASK:-16}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
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
echo "Time limit:    $TIME_LIMIT"
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
    --time="$TIME_LIMIT" \
    --output="$PROJECT_ROOT/slurm/array_full_cohort_run/output/%A_%a.out" \
    --error="$PROJECT_ROOT/slurm/array_full_cohort_run/error/%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",ANCHOR="$ANCHOR",COXNET_MAX_ITER="$COXNET_MAX_ITER",COXNET_BACKEND="$COXNET_BACKEND" \
    "$PROJECT_ROOT/slurm/array_full_cohort_run.sh"
