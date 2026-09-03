#!/bin/bash
# Launch wrapper for array_somatic_comp.sh -- somatic modality only.
#
# The feature-comp arrays run a whole modality class per row (MODALITY_CLASS=big is
# somatic + prs + text). Use this wrapper instead when only the somatic fits need to be
# (re-)run: it pins --modality somatic for every manifest row, so prs and text are not
# re-queued alongside it.
#
# Resource sizing matches the "big" class in launch_feature_comp.sh
# (--cpus-per-task=5 --mem=16G): somatic's design matrix is a wide gene-by-alteration
# panel of data-dependent width, so it is worth a cluster slot even though its penalized
# column count is modest.
#
# Skip behavior: run_feature_comp_task.py skips any scheme/event whose somatic outputs
# already exist unless OVERWRITE=1, so this is safe to run alongside (or after)
# array_feature_comp.sh -- each skips what the other already finished. Pass OVERWRITE=1
# to force recomputation of every row.
#
# Env vars (same names/meanings as launch_feature_comp.sh):
#   PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE, COXNET_MAX_ITER, COXNET_BACKEND,
#   ANCHOR, PARTITION, CPUS, MEM, THROTTLE
# If PARTITION is unset, Slurm uses the cluster/account default partition.
#
# Usage:
#   bash launch_somatic_comp.sh                     # somatic for every manifest row
#   OVERWRITE=1 bash launch_somatic_comp.sh         # force recompute
#   ROWS_PER_TASK=5 bash launch_somatic_comp.sh     # smaller, more numerous array tasks
#   THROTTLE=20 bash launch_somatic_comp.sh         # at most 20 array tasks running at once
#   ANCHOR=sequencing bash launch_somatic_comp.sh   # sequencing-anchored manifest/results

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
ROWS_PER_TASK="${ROWS_PER_TASK:-7}"
ANCHOR="${ANCHOR:-treatment}"
PARTITION="${PARTITION:-}"
CPUS="${CPUS:-5}"
MEM="${MEM:-16G}"
THROTTLE="${THROTTLE:-}"

SBATCH_PARTITION_ARGS=()
if [[ -n "$PARTITION" ]]; then
    SBATCH_PARTITION_ARGS=(--partition="$PARTITION")
fi
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST="${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/feature_comp_tasks${ANCHOR_SUFFIX}.tsv}"

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
ARRAY_SPEC="0-${MAX_TASK}"
if [[ -n "$THROTTLE" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${THROTTLE}"
fi

echo "Manifest:      $MANIFEST"
echo "Rows:          $N_ROWS"
echo "Rows per task: $ROWS_PER_TASK"
echo "Modality:      somatic (only)"
echo "Array tasks:   $N_TASKS  (--array=${ARRAY_SPEC})"

# See launch_full_cohort.sh for why --output/--error are overridden here with absolute paths
# rather than left to the array script's relative #SBATCH directives.
mkdir -p "$PROJECT_ROOT/slurm/array_somatic_comp/output" "$PROJECT_ROOT/slurm/array_somatic_comp/error"

echo "Submitting somatic-only feature comps (--cpus-per-task=${CPUS} --mem=${MEM})"
sbatch \
    ${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"} \
    --array="$ARRAY_SPEC" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --output="$PROJECT_ROOT/slurm/array_somatic_comp/output/%A_%a.out" \
    --error="$PROJECT_ROOT/slurm/array_somatic_comp/error/%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",ANCHOR="$ANCHOR" \
    "$PROJECT_ROOT/slurm/array_somatic_comp.sh"
