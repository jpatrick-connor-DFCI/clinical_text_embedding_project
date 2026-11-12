#!/bin/bash
# Launch wrapper for note_transformer_job.sh.
#
# Submits one job that prepares the note-level inputs and runs the note-survival
# transformer for overall survival. Environment variables accepted by the job script
# (ANCHOR, COHORT, TAG, NUM_TRIALS, STAGE, DEVICE, CHECKPOINT, OVERWRITE, RESUME,
# PROJECT_ROOT, CONDA_ENV_PREFIX, PYTHON_BIN) are forwarded.
#
# Usage:
#   bash slurm/launch_note_transformer.sh
#   NUM_TRIALS=5 TAG=cpu_trial bash slurm/launch_note_transformer.sh
#   ANCHOR=sequencing bash slurm/launch_note_transformer.sh
#   STAGE=test CHECKPOINT=/path/to/best_model.pt bash slurm/launch_note_transformer.sh
#
# GPU run. GPU_GRES adds --gres to the submission. The partition name is cluster
# specific, so there is no default for it here:
#   GPU_GRES=gpu:1 PARTITION=<gpu-partition> bash slurm/launch_note_transformer.sh
# With DEVICE=auto (the default) the runner uses CUDA when it is visible. If a CUDA
# kernel rejects deterministic mode, add --allow-nondeterministic by running the runner
# directly; the launcher does not expose it on purpose.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}"
ANCHOR="${ANCHOR:-treatment}"
COHORT="${COHORT:-text}"
TAG="${TAG:-default}"
NUM_TRIALS="${NUM_TRIALS:-20}"
STAGE="${STAGE:-all}"
DEVICE="${DEVICE:-auto}"
PARTITION="${PARTITION:-normal}"
GPU_GRES="${GPU_GRES:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-24:00:00}"

SBATCH_ARGS=()
if [[ -n "$PARTITION" ]]; then
    SBATCH_ARGS+=(--partition="$PARTITION")
fi
if [[ -n "$GPU_GRES" ]]; then
    SBATCH_ARGS+=(--gres="$GPU_GRES")
fi

echo "Anchor / cohort: $ANCHOR / $COHORT   tag: $TAG   stage: $STAGE"
echo "Trials:          $NUM_TRIALS   device: $DEVICE"
echo "Resources:       ${CPUS_PER_TASK} CPUs, ${MEMORY}, ${TIME_LIMIT}${GPU_GRES:+, gres=$GPU_GRES}, partition=${PARTITION:-default}"

# #SBATCH --output/--error in the job script are relative to the submission directory;
# absolute paths here make the log location independent of where this launcher is run.
mkdir -p "$PROJECT_ROOT/slurm/note_transformer"
sbatch \
    ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEMORY" \
    --time="$TIME_LIMIT" \
    --output="$PROJECT_ROOT/slurm/note_transformer/%j.out" \
    --error="$PROJECT_ROOT/slurm/note_transformer/%j.err" \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",ANCHOR="$ANCHOR",COHORT="$COHORT",TAG="$TAG",NUM_TRIALS="$NUM_TRIALS",STAGE="$STAGE",DEVICE="$DEVICE",CHECKPOINT="${CHECKPOINT:-}",OVERWRITE="${OVERWRITE:-0}",RESUME="${RESUME:-0}" \
    "$PROJECT_ROOT/slurm/note_transformer_job.sh"
