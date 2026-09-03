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
# Array sizing targets the scheduler, not the node. Observed cost is ~20-60m per row
# (mean ~45m), so ROWS_PER_TASK defaults to 28 -- roughly 21h of a 24h wall (~88%
# utilization) -- rather than the feature-comp default of 7, which would fill only ~5h and
# submit 4x as many array tasks. On the current 1978-row manifest that is 71 tasks instead
# of 283. array_somatic_comp.sh is deadline-guarded, so a task that runs slow stops
# dispatching and defers its remaining rows instead of being killed mid-row; re-running the
# same array picks them up, since completed rows are skipped.
#
# If your partition has a shorter wall than 24h, set TIME and ROWS_PER_TASK together, e.g.
#   TIME=12:00:00 ROWS_PER_TASK=14 bash launch_somatic_comp.sh
# The array script reads its actual limit from Slurm at runtime, so the guard adapts even
# if the two are set inconsistently -- it will just defer more rows.
#
# Skip behavior: run_feature_comp_task.py skips any scheme/event whose somatic outputs
# already exist unless OVERWRITE=1, so this is safe to run alongside (or after)
# array_feature_comp.sh -- each skips what the other already finished. Pass OVERWRITE=1
# to force recomputation of every row.
#
# Env vars (same names/meanings as launch_feature_comp.sh):
#   PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE, COXNET_MAX_ITER, COXNET_BACKEND,
#   ANCHOR, PARTITION, CPUS, MEM, THROTTLE, TIME, SAFETY_MIN, ROW_BUDGET_MIN
# If PARTITION is unset, Slurm uses the cluster/account default partition.
#
# Usage:
#   bash launch_somatic_comp.sh                     # somatic for every manifest row
#   OVERWRITE=1 bash launch_somatic_comp.sh         # force recompute
#   ROWS_PER_TASK=40 bash launch_somatic_comp.sh    # pack more rows per array task
#   TIME=12:00:00 ROWS_PER_TASK=14 bash launch_somatic_comp.sh   # shorter-wall partition
#   THROTTLE=20 bash launch_somatic_comp.sh         # at most 20 array tasks running at once
#   ANCHOR=sequencing bash launch_somatic_comp.sh   # sequencing-anchored manifest/results

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
ROWS_PER_TASK="${ROWS_PER_TASK:-28}"
ANCHOR="${ANCHOR:-treatment}"
PARTITION="${PARTITION:-}"
CPUS="${CPUS:-5}"
MEM="${MEM:-16G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"
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
echo "Time limit:    $TIME_LIMIT"
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
    --time="$TIME_LIMIT" \
    --output="$PROJECT_ROOT/slurm/array_somatic_comp/output/%A_%a.out" \
    --error="$PROJECT_ROOT/slurm/array_somatic_comp/error/%A_%a.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",ANCHOR="$ANCHOR" \
    "$PROJECT_ROOT/slurm/array_somatic_comp.sh"
