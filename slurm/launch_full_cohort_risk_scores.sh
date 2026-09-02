#!/bin/bash
# Launch wrapper for array_full_cohort_risk_scores.sh.
#
# Reads the manifest, computes the required array size, and submits the job.
# Any environment variables accepted by array_full_cohort_risk_scores.sh can be
# forwarded here (PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE,
# COXNET_MAX_ITER, COXNET_BACKEND, ANCHOR, PARTITION, CPUS_PER_TASK, MEMORY).
# If PARTITION is unset, Slurm uses the cluster/account default partition.
#
# Memory: nested CV here runs a full inner grid per outer fold over the same cohort that
# array_full_cohort_run.sh fits once at 32G, so it is the heaviest step in the pipeline.
# The 8G this used to inherit was OOM-killing tasks; the default is now 48G. If a wide
# scheme still trips the limit, raise MEMORY, or lower CPUS_PER_TASK — peak memory scales
# with the number of in-flight fits, and the outer fold loop is serial regardless.
#
# Usage:
#   bash launch_full_cohort_risk_scores.sh
#   OVERWRITE=1 bash launch_full_cohort_risk_scores.sh
#   ROWS_PER_TASK=5 bash launch_full_cohort_risk_scores.sh
#   MEMORY=64G bash launch_full_cohort_risk_scores.sh
#   ANCHOR=sequencing bash launch_full_cohort_risk_scores.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
ROWS_PER_TASK="${ROWS_PER_TASK:-20}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-48G}"
ANCHOR="${ANCHOR:-treatment}"
PARTITION="${PARTITION:-}"
SBATCH_PARTITION_ARGS=()
if [[ -n "$PARTITION" ]]; then
    SBATCH_PARTITION_ARGS=(--partition="$PARTITION")
fi
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST="${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/full_cohort_tasks${ANCHOR_SUFFIX}.tsv}"

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

# See launch_full_cohort.sh for why --output/--error are overridden here with absolute paths
# rather than left to the array script's relative #SBATCH directives.
mkdir -p "$PROJECT_ROOT/slurm/array_full_cohort_risk_scores/output" "$PROJECT_ROOT/slurm/array_full_cohort_risk_scores/error"
sbatch \
    ${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"} \
    --array="0-${MAX_TASK}" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --output="$PROJECT_ROOT/slurm/array_full_cohort_risk_scores/output/%A_%a.out" \
    --error="$PROJECT_ROOT/slurm/array_full_cohort_risk_scores/error/%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",ANCHOR="$ANCHOR" \
    "$PROJECT_ROOT/slurm/array_full_cohort_risk_scores.sh"
