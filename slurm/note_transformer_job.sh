#!/bin/bash

#SBATCH --job-name=note_transformer_os
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/note_transformer/%j.out
#SBATCH --error=slurm/note_transformer/%j.err

# Note-level survival transformer for overall survival (death_met:death): prepare the
# per-note inputs, then run search / train / held-out test. Single job, not an array.
#
# Environment variables:
#   ANCHOR       treatment (default) or sequencing
#   COHORT       text (default; the Cox text cohort and its exact 20% test split) or all
#   TAG          run label under <scheme_results_dir>/note_transformer/<TAG>/ (default: default)
#   NUM_TRIALS   random-search trials (default 20; use 5 on CPU)
#   STAGE        search | train | test | all (default all = search then test)
#   DEVICE       auto (default) | cpu | cuda
#   CHECKPOINT   optional explicit checkpoint; only valid with STAGE=test
#   OVERWRITE    1 to replace existing inputs and stage outputs
#   RESUME       1 to resume a partly finished search or fit
#   PROJECT_ROOT, CONDA_ENV_PREFIX, PYTHON_BIN, CTEP_DATA_PATH as in array_full_cohort_run.sh
ANCHOR=${ANCHOR:-treatment}
COHORT=${COHORT:-text}
TAG=${TAG:-default}
NUM_TRIALS=${NUM_TRIALS:-20}
STAGE=${STAGE:-all}
DEVICE=${DEVICE:-auto}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "project root not found: $PROJECT_ROOT"
  exit 1
fi

cd "$PROJECT_ROOT" || { echo "Failed to cd into $PROJECT_ROOT"; exit 1; }
export PYTHONPATH="$PROJECT_ROOT"

CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-/data/gusev/USERS/jpconnor/conda/envs/clinical_notes_project}
PYTHON_BIN=${PYTHON_BIN:-$CONDA_ENV_PREFIX/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN"
  exit 1
fi
"$PYTHON_BIN" -c "import config" 2>/dev/null || { echo "config not importable - check PYTHON_BIN / PYTHONPATH"; exit 1; }

# Deliberately enabled only after validating the environment and imports above.
set -euo pipefail

export CTEP_DATA_PATH=${CTEP_DATA_PATH:-${DATA_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project}}
export PYTHONUNBUFFERED=1

# Torch gets its thread count from --num-threads below; keep BLAS libraries from oversubscribing.
THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
# Polars/Rayon otherwise size their pool to every core on the node, not the SLURM allocation.
export POLARS_MAX_THREADS=${POLARS_MAX_THREADS:-$THREADS}
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-$THREADS}

mkdir -p slurm/note_transformer

if [[ "$COHORT" != "text" && "$COHORT" != "all" ]]; then
  echo "COHORT must be text or all, got: $COHORT"
  exit 1
fi
if [[ -n "${CHECKPOINT:-}" && "$STAGE" != "test" ]]; then
  echo "CHECKPOINT is only valid with STAGE=test"
  exit 1
fi

# One inputs directory, resolved once, passed to both commands. The cohort=all bundle lives
# in inputs_all/ and is never picked up by the runner's default lookup.
INPUTS_DIR=$("$PYTHON_BIN" -c "
from pipelines.note_transformer.prepare_os_inputs import default_inputs_dir
print(default_inputs_dir('$ANCHOR', '$COHORT', None))
")
echo "Anchor=$ANCHOR cohort=$COHORT tag=$TAG stage=$STAGE device=$DEVICE trials=$NUM_TRIALS threads=$THREADS"
echo "Inputs: $INPUTS_DIR"

PREPARE_FLAGS=()
RUN_FLAGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  PREPARE_FLAGS+=(--overwrite)
  RUN_FLAGS+=(--overwrite)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  RUN_FLAGS+=(--resume)
fi
if [[ -n "${CHECKPOINT:-}" ]]; then
  RUN_FLAGS+=(--checkpoint "$CHECKPOINT")
fi

"$PYTHON_BIN" -m pipelines.note_transformer.prepare_os_inputs \
  --anchor "$ANCHOR" --cohort "$COHORT" --inputs-dir "$INPUTS_DIR" \
  ${PREPARE_FLAGS[@]+"${PREPARE_FLAGS[@]}"}

"$PYTHON_BIN" -m pipelines.note_transformer.run_os_transformer \
  --anchor "$ANCHOR" --inputs-dir "$INPUTS_DIR" --tag "$TAG" \
  --stage "$STAGE" --num-trials "$NUM_TRIALS" \
  --device "$DEVICE" --num-threads "$THREADS" \
  ${RUN_FLAGS[@]+"${RUN_FLAGS[@]}"}
