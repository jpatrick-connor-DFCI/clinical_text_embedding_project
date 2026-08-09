#!/bin/bash
# Launch wrapper for array_feature_comp.sh.
#
# Reads the manifest, computes the required array size, and submits the job(s).
# Any environment variables accepted by array_feature_comp.sh can be
# forwarded here (PROJECT_ROOT, MANIFEST, ROWS_PER_TASK, OVERWRITE,
# COXNET_MAX_ITER, COXNET_BACKEND, ANCHOR).
#
# Resource sizing (Phase-A A0): by default (MODALITY_CLASS=split) this submits TWO array jobs
# against the same manifest, sized per modality class, each running its modalities one process
# per modality (as before) —
#   "big"   (text, prs)              : --cpus-per-task=5 --mem=8G
#   "small" (stage, treatment, somatic) : --cpus-per-task=1 --mem=4G
# because run_feature_comp_task.py already forces n_jobs=1 for the small modalities
# (< 50 penalized columns), so they were previously requesting 6 CPUs and using 1.
# Set MODALITY_CLASS=all to submit a single unsplit job instead (legacy 6 CPU / 8G sizing),
# which uses the collapsed --modality all in-process loop over all five modalities (A2).
#
# Usage:
#   bash launch_feature_comp.sh
#   OVERWRITE=1 bash launch_feature_comp.sh
#   ROWS_PER_TASK=5 bash launch_feature_comp.sh
#   MODALITY_CLASS=all bash launch_feature_comp.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
MANIFEST="${MANIFEST:-$PROJECT_ROOT/v2/slurm/slurm_manifests/feature_comp_tasks.tsv}"
ROWS_PER_TASK="${ROWS_PER_TASK:-7}"
MODALITY_CLASS="${MODALITY_CLASS:-split}"
ANCHOR="${ANCHOR:-treatment}"

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

submit_class () {
    local class="$1" cpus="$2" mem="$3"
    echo "Submitting MODALITY_CLASS=${class} (--cpus-per-task=${cpus} --mem=${mem})"
    sbatch \
        --array="0-${MAX_TASK}" \
        --cpus-per-task="$cpus" \
        --mem="$mem" \
        --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",MODALITY_CLASS="$class",ANCHOR="$ANCHOR" \
        "$PROJECT_ROOT/v2/slurm/array_feature_comp.sh"
}

if [[ "$MODALITY_CLASS" == "split" ]]; then
    submit_class big 5 8G
    submit_class small 1 4G
else
    sbatch \
        --array="0-${MAX_TASK}" \
        --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MANIFEST="$MANIFEST",ROWS_PER_TASK="$ROWS_PER_TASK",MODALITY_CLASS="$MODALITY_CLASS",ANCHOR="$ANCHOR" \
        "$PROJECT_ROOT/v2/slurm/array_feature_comp.sh"
fi
