#!/bin/bash

#SBATCH --job-name=coxnet_feat_comp
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=v2/slurm/array_feature_comp/output/%A_%a.out
#SBATCH --error=v2/slurm/array_feature_comp/error/%A_%a.err

# Resource sizing is per-modality-class (see launch_feature_comp.sh), which submits this script
# twice with different --cpus-per-task/--mem overrides on the sbatch CLI (these win over the
# #SBATCH defaults above) plus MODALITY_CLASS set accordingly:
#   MODALITY_CLASS=big   -> text, prs                            (--cpus-per-task=5 --mem=8G)
#   MODALITY_CLASS=small -> stage, treatment, somatic, metburden  (--cpus-per-task=1 --mem=4G)
#   MODALITY_CLASS=all (default) -> all six modalities, unchanged legacy behavior.
MODALITY_CLASS=${MODALITY_CLASS:-all}

# ANCHOR selects the time-zero anchor (see v2/anchors.py): "treatment" (default) or
# "sequencing". Forwarded to run_feature_comp_task.py as --anchor; non-default anchors
# nest results under <scheme_results_dir>/anchor_<anchor>/ (schemes.py scheme_results_dir).
ANCHOR=${ANCHOR:-treatment}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
V2_ROOT="$PROJECT_ROOT/v2"
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST=${MANIFEST:-$V2_ROOT/slurm/slurm_manifests/feature_comp_tasks${ANCHOR_SUFFIX}.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-7}

if [[ ! -d "$V2_ROOT" ]]; then
  echo "v2 root not found: $V2_ROOT"
  exit 1
fi

mkdir -p "$(dirname "$MANIFEST")"
if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

cd "$V2_ROOT" || { echo "Failed to cd into $V2_ROOT"; exit 1; }
export PYTHONPATH="$V2_ROOT"

module load miniforge3 || { echo "Failed to load module miniforge3"; exit 1; }
eval "$(conda shell.bash hook)"
conda activate clinical_notes_project || { echo "Failed to activate conda env clinical_notes_project"; exit 1; }
python -c "import config" 2>/dev/null || { echo "config not importable - check conda env / PYTHONPATH"; exit 1; }

# Deliberately enabled only here, after the prologue above: module load / conda activate /
# `eval "$(conda shell.bash hook)"` commonly reference unset variables internally and can
# behave unpredictably under -u; every step above already has its own explicit failure guard,
# so nothing upstream of this line silently continues past a failure.
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
      big)   MODALITIES=(prs text) ;;
      small) MODALITIES=(stage treatment somatic metburden) ;;
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
    if ! python -m pipelines.training.run_feature_comp_task \
      --scheme "$SCHEME" \
      --event "$EVENT" \
      --modality all \
      --anchor "$ANCHOR" \
      --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
      --max-iter "${COXNET_MAX_ITER:-5000}" \
      --backend "${COXNET_BACKEND:-threading}" \
      ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
      echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}, modality=all"
      FAILED_ROWS=$((FAILED_ROWS + 1))
    fi
  else
    for MODALITY in "${MODALITIES[@]}"; do
      if ! python -m pipelines.training.run_feature_comp_task \
        --scheme "$SCHEME" \
        --event "$EVENT" \
        --modality "$MODALITY" \
        --anchor "$ANCHOR" \
        --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
        --max-iter "${COXNET_MAX_ITER:-5000}" \
        --backend "${COXNET_BACKEND:-threading}" \
        ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
        echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}, modality=${MODALITY}"
        FAILED_ROWS=$((FAILED_ROWS + 1))
      fi
    done
  fi
done

conda deactivate
if [[ "$FAILED_ROWS" -gt 0 ]]; then
  echo "$FAILED_ROWS modality run(s) failed"
  exit 1
fi
