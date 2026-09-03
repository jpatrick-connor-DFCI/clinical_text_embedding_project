#!/bin/bash
# Launch wrapper for array_feature_comp.sh.
#
# Reads the manifest, computes the required array size, and submits the job(s).
# Any environment variables accepted by array_feature_comp.sh can be
# forwarded here (PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE,
# COXNET_MAX_ITER, COXNET_BACKEND, ANCHOR, PARTITION). If PARTITION is unset,
# Slurm uses the cluster/account default partition.
#
# Resource sizing: by default (MODALITY_CLASS=big) this submits ONE array job covering only
# the wide-design modalities, which are the ones that actually need a cluster slot —
#   "big"   (somatic, prs, text)              : --cpus-per-task=5 --mem=16G
#   "small" (stage, treatment, metburden)     : --cpus-per-task=1 --mem=4G
# run_feature_comp_task.py forces n_jobs=1 for any modality under 50 penalized columns, so the
# "small" three are single-core work that gains nothing from the cluster and only queues behind
# the heavy fits. notebooks/2_models/01_feature_comparison.ipynb runs them locally instead, and
# is the intended path — it skips whatever these arrays have already finished, and vice versa.
#
# somatic sits in "big" despite a modest penalized-column count because its design matrix is a
# wide gene-by-alteration panel of data-dependent width.
#
# Usage:
#   bash launch_feature_comp.sh                      # big only (somatic, prs, text)
#   MODALITY_CLASS=split bash launch_feature_comp.sh # big + small, if not using the notebook
#   MODALITY_CLASS=small bash launch_feature_comp.sh # small only
#   MODALITY_CLASS=all   bash launch_feature_comp.sh # all six in one process (legacy)
#   OVERWRITE=1 bash launch_feature_comp.sh
#   ROWS_PER_TASK=5 bash launch_feature_comp.sh
#
# SCHEDULER SIZING. The binding constraint is the scheduler, not the node: many short jobs
# queue worse than a few long ones. Observed per-modality cost on the "big" class is
# ~2.5-11m grid + ~16-49m held-out risk (~20-60m, mean ~45m); MODALITY_CLASS=big runs three
# modalities per row, so a big row is ~2.2h. ROWS_PER_TASK=28 was chosen for the somatic
# array (one modality per row, ~21h of a 24h wall); the same count here means a "big" task
# would need ~60h and be killed by the wall.
#
# So ROWS_PER_TASK is kept per-class below rather than taken from this one default:
#   big   -> 10 rows (3 modalities x ~45m = ~2.2h/row -> ~22h)
#   small -> 28 rows (single-core, minutes per row; the notebook usually covers these)
# Setting ROWS_PER_TASK explicitly overrides both. Unlike array_somatic_comp.sh, this array
# has no deadline guard, so these counts are deliberately conservative -- a task that runs
# long here is killed mid-row rather than deferring cleanly.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
# Captured before the default is applied: "was ROWS_PER_TASK set by the caller?" cannot be
# asked after the ${VAR:-default} assignment, because that always leaves it set.
ROWS_PER_TASK_SET="${ROWS_PER_TASK+set}"
ROWS_PER_TASK="${ROWS_PER_TASK:-28}"
MODALITY_CLASS="${MODALITY_CLASS:-big}"
ANCHOR="${ANCHOR:-treatment}"
PARTITION="${PARTITION:-}"
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

echo "Manifest:      $MANIFEST"
echo "Rows:          $N_ROWS"
# Rows-per-task and the resulting array size are per-class (see rows_for_class below), so
# they are reported by submit_class rather than here; MODALITY_CLASS=all uses these.
echo "Rows per task: $ROWS_PER_TASK (default; per-class values reported below)"

# See launch_full_cohort.sh for why --output/--error are overridden here with absolute paths
# rather than left to the array script's relative #SBATCH directives.
mkdir -p "$PROJECT_ROOT/slurm/array_feature_comp/output" "$PROJECT_ROOT/slurm/array_feature_comp/error"

# ROWS_PER_TASK defaults per class (see SCHEDULER SIZING); an explicit ROWS_PER_TASK in the
# environment wins. Recomputed per class because the array size depends on it.
rows_for_class () {
    if [[ -n "$ROWS_PER_TASK_SET" ]]; then echo "$ROWS_PER_TASK"; return; fi
    case "$1" in
        big)   echo 10 ;;
        small) echo 28 ;;
        *)     echo "$ROWS_PER_TASK" ;;
    esac
}

submit_class () {
    local class="$1" cpus="$2" mem="$3"
    local rows; rows=$(rows_for_class "$class")
    local n_tasks=$(( (N_ROWS + rows - 1) / rows ))
    local max_task=$(( n_tasks - 1 ))
    echo "Submitting MODALITY_CLASS=${class} (--cpus-per-task=${cpus} --mem=${mem}, ${rows} rows/task, ${n_tasks} tasks)"
    sbatch \
        ${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"} \
        --array="0-${max_task}" \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --output="$PROJECT_ROOT/slurm/array_feature_comp/output/%A_%a.out" \
        --error="$PROJECT_ROOT/slurm/array_feature_comp/error/%A_%a.err" \
        --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$rows",MODALITY_CLASS="$class",ANCHOR="$ANCHOR" \
        "$PROJECT_ROOT/slurm/array_feature_comp.sh"
}

if [[ "$MODALITY_CLASS" == "split" ]]; then
    submit_class big 5 16G
    submit_class small 1 4G
elif [[ "$MODALITY_CLASS" == "big" ]]; then
    submit_class big 5 16G
elif [[ "$MODALITY_CLASS" == "small" ]]; then
    submit_class small 1 4G
else
    sbatch \
        ${SBATCH_PARTITION_ARGS[@]+"${SBATCH_PARTITION_ARGS[@]}"} \
        --array="0-${MAX_TASK}" \
        --output="$PROJECT_ROOT/slurm/array_feature_comp/output/%A_%a.out" \
        --error="$PROJECT_ROOT/slurm/array_feature_comp/error/%A_%a.err" \
        --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",MODALITY_CLASS="$MODALITY_CLASS",ANCHOR="$ANCHOR" \
        "$PROJECT_ROOT/slurm/array_feature_comp.sh"
fi
