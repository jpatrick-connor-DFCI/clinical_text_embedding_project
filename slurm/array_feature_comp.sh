#!/bin/bash

#SBATCH --job-name=coxnet_feat_comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=slurm/array_feature_comp/output/%A_%a.out
#SBATCH --error=slurm/array_feature_comp/error/%A_%a.err

# Resource sizing is per-modality-class (see launch_feature_comp.sh). Only the wide-design
# modalities are worth a cluster slot; the narrow ones are single-core work that
# notebooks/2_models/01_feature_comparison.ipynb runs locally instead.
#   MODALITY_CLASS=big   -> somatic, prs, text  (--cpus-per-task=5 --mem=16G) -- the default
#   MODALITY_CLASS=small -> stage, treatment, metburden (--cpus-per-task=1 --mem=4G); the
#     notebook covers these, so submit this class only when running without the notebook.
#   MODALITY_CLASS=all -> all six modalities in one process (legacy behavior).
MODALITY_CLASS=${MODALITY_CLASS:-big}

# ANCHOR selects the time-zero anchor (see anchors.py): "treatment" (default) or
# "sequencing". Forwarded to run_feature_comp_task.py as --anchor; non-default anchors
# nest results under <scheme_results_dir>/anchor_<anchor>/ (schemes.py scheme_results_dir).
ANCHOR=${ANCHOR:-treatment}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST=${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/feature_comp_tasks${ANCHOR_SUFFIX}.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-7}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "project root not found: $PROJECT_ROOT"
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
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

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p slurm/array_feature_comp/output slurm/array_feature_comp/error

OVERWRITE_FLAG=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_FLAG=(--overwrite)
fi

TOTAL_ROWS=$(wc -l < "$MANIFEST")
START_LINE=$((SLURM_ARRAY_TASK_ID * ROWS_PER_TASK + 1))
END_LINE=$((START_LINE + ROWS_PER_TASK - 1))

if [[ "$START_LINE" -gt "$TOTAL_ROWS" ]]; then
  echo "No rows assigned to task ${SLURM_ARRAY_TASK_ID} (start=${START_LINE}, total=${TOTAL_ROWS})"
  exit 0
fi
if [[ "$END_LINE" -gt "$TOTAL_ROWS" ]]; then
  END_LINE="$TOTAL_ROWS"
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: processing manifest rows ${START_LINE}-${END_LINE}"

export CTEP_DATA_PATH=${CTEP_DATA_PATH:-${DATA_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project}}
RESULTS_ROOT="$CTEP_DATA_PATH/time-to-event_analysis/results"
FAILED_ROWS=0

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  IFS=$'\t' read -r RAW_SCHEME EVENT MANIFEST_MODALITY EXTRA_FIELD <<< "${TASK_LINE}"
  if [[ -n "${EXTRA_FIELD:-}" ]]; then
    echo "Unsupported manifest row ${LINE_NUM}: expected 2 or 3 tab-separated fields, got more"
    exit 1
  fi

  case "$RAW_SCHEME" in
    icd3) SCHEME="icd3_post" ;;
    icd4) SCHEME="icd4_post" ;;
    phecode) SCHEME="phecode_post" ;;
    *) SCHEME="$RAW_SCHEME" ;;
  esac

  case "$SCHEME" in
    icd3_post) SCHEME_RESULTS_DIR="level_3_ICD_post_results" ;;
    icd4_post) SCHEME_RESULTS_DIR="level_4_ICD_post_results" ;;
    phecode_post) SCHEME_RESULTS_DIR="phecode_post_results" ;;
    death_met) SCHEME_RESULTS_DIR="death_met_results" ;;
    *)
      echo "Unsupported scheme in manifest row ${LINE_NUM}: $SCHEME"
      exit 1
      ;;
  esac
  if [[ "$ANCHOR" == "treatment" ]]; then
    ANCHOR_SUBDIR="$SCHEME_RESULTS_DIR"
  else
    ANCHOR_SUBDIR="$SCHEME_RESULTS_DIR/anchor_$ANCHOR"
  fi
  mkdir -p "$RESULTS_ROOT/$ANCHOR_SUBDIR/feature_comps/$EVENT"

  if [[ -n "${MANIFEST_MODALITY:-}" ]]; then
    case "$MANIFEST_MODALITY" in
      stage|treatment|somatic|prs|text|metburden)
        MODALITIES=("$MANIFEST_MODALITY")
        ;;
      *)
        echo "Unsupported modality in manifest row ${LINE_NUM}: $MANIFEST_MODALITY"
        exit 1
        ;;
    esac
  else
    case "$MODALITY_CLASS" in
      big)   MODALITIES=(somatic prs text) ;;
      small) MODALITIES=(stage treatment metburden) ;;
      all)   MODALITIES=(stage treatment somatic prs text metburden) ;;
      *)
        echo "Unsupported MODALITY_CLASS: $MODALITY_CLASS (expected big|small|all)"
        exit 1
        ;;
    esac
  fi

  echo "Running row ${LINE_NUM}: scheme=${SCHEME}, event=${EVENT}, modalities=${MODALITIES[*]}, anchor=${ANCHOR}"

  # A2: when the full six-modality set is requested for this row (no per-row MANIFEST_MODALITY
  # override, MODALITY_CLASS=all), run one process that loads the base frame once and loops all
  # six modalities in-process (per-modality try/except lives inside run_feature_comp_task.py, so
  # one modality failing does not stop the rest). Otherwise (a single manifest-pinned modality,
  # or a big/small resource class) invoke run_feature_comp_task.py once per modality as before.
  if [[ -z "${MANIFEST_MODALITY:-}" && "$MODALITY_CLASS" == "all" ]]; then
    if ! "$PYTHON_BIN" -m pipelines.training.run_feature_comp_task \
      --scheme "$SCHEME" \
      --event "$EVENT" \
      --modality all \
      --anchor "$ANCHOR" \
      --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
      --max-iter "${COXNET_MAX_ITER:-1000}" \
      --backend "${COXNET_BACKEND:-threading}" \
      ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
      echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}, modality=all"
      FAILED_ROWS=$((FAILED_ROWS + 1))
    fi
  else
    for MODALITY in "${MODALITIES[@]}"; do
      if ! "$PYTHON_BIN" -m pipelines.training.run_feature_comp_task \
        --scheme "$SCHEME" \
        --event "$EVENT" \
        --modality "$MODALITY" \
        --anchor "$ANCHOR" \
        --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
        --max-iter "${COXNET_MAX_ITER:-1000}" \
        --backend "${COXNET_BACKEND:-threading}" \
        ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
        echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}, modality=${MODALITY}"
        FAILED_ROWS=$((FAILED_ROWS + 1))
      fi
    done
  fi
done

if [[ "$FAILED_ROWS" -gt 0 ]]; then
  echo "$FAILED_ROWS modality run(s) failed"
  exit 1
fi
