#!/bin/bash

#SBATCH --job-name=coxnet_full_risk
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=slurm/array_full_cohort_risk_scores/output/%A_%a.out
#SBATCH --error=slurm/array_full_cohort_risk_scores/error/%A_%a.err

# Requires run_full_cohort_event.py (array_full_cohort_run.sh) to have already produced
# text_val.csv for each row — this script picks best hyperparameters from that CV grid and
# generates held-out risk scores (run_full_cohort_risk_scores.py). Same manifest format as
# array_full_cohort_run.sh (scheme<TAB>event), so full_cohort_tasks.tsv is reused by default.
#
# Memory: this step is the heaviest in the pipeline and was previously being OOM-killed at 8G.
# get_nested_heldout_risk_scores_CoxPH runs a *full inner grid search per outer fold* over the
# same cohort array that array_full_cohort_run.sh fits once at 32G, so it needs at least that
# much, plus headroom for the concurrently live inner-fold copies. Hence 48G / 4 CPUs (down
# from 6: threads multiply the per-fit copies without speeding up the outer loop, which is
# serial). Raise MEMORY in launch_full_cohort_risk_scores.sh if a wide scheme still trips it;
# lowering CPUS_PER_TASK is the other lever, since peak memory scales with in-flight fits.

# ANCHOR selects the time-zero anchor (see anchors.py): "treatment" (default) or
# "sequencing". Forwarded to run_full_cohort_risk_scores.py as --anchor; non-default anchors
# read/write under <scheme_results_dir>/anchor_<anchor>/ (schemes.py scheme_results_dir).
ANCHOR=${ANCHOR:-treatment}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST=${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/full_cohort_tasks${ANCHOR_SUFFIX}.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-16}

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

mkdir -p slurm/array_full_cohort_risk_scores/output slurm/array_full_cohort_risk_scores/error

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

# --- Deadline guard ------------------------------------------------------------------
# Packing rows to fill the wall only pays off if a task that runs long stops cleanly rather
# than being killed mid-row. Before each row the task checks elapsed time plus an estimate
# of the next row's cost against its limit; if it does not fit, it stops dispatching and
# reports the deferred rows. Re-running the array picks them up, since completed rows are
# skipped. TIME_LIMIT_MIN is read from Slurm at runtime, so the guard stays correct even if
# ROWS_PER_TASK and --time are set inconsistently.
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-}"
if [[ -z "$TIME_LIMIT_MIN" ]] && command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  _tl=$(squeue -h -j "$SLURM_JOB_ID" -O TimeLimit 2>/dev/null | tr -d ' ' || true)
  if [[ "$_tl" =~ ^[0-9]+-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
    _d=${_tl%%-*}; _rest=${_tl#*-}
    TIME_LIMIT_MIN=$(( _d * 1440 + 10#${_rest%%:*} * 60 + 10#$(echo "$_rest" | cut -d: -f2) ))
  elif [[ "$_tl" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
    TIME_LIMIT_MIN=$(( 10#${_tl%%:*} * 60 + 10#$(echo "$_tl" | cut -d: -f2) ))
  fi
fi
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-1440}"          # default: the 24h #SBATCH --time above
SAFETY_MIN="${SAFETY_MIN:-20}"
ROW_BUDGET_MIN="${ROW_BUDGET_MIN:-75}"          # prior estimate; see SCHEDULER SIZING
DEADLINE_MIN=$(( TIME_LIMIT_MIN - SAFETY_MIN ))
TASK_START=$SECONDS
ROWS_DONE=0
MAX_ROW_MIN=0          # slowest row seen so far; the guard budgets on this, not the mean
DEFERRED_ROWS=()

echo "Deadline guard: limit=${TIME_LIMIT_MIN}m safety=${SAFETY_MIN}m usable=${DEADLINE_MIN}m, initial per-row budget=${ROW_BUDGET_MIN}m"

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  ELAPSED_MIN=$(( (SECONDS - TASK_START) / 60 ))
  if [[ "$ROWS_DONE" -gt 0 ]]; then
    # Budget on the slowest row observed, not the mean: a mean dragged down by fast rows
    # would green-light a slow row that cannot finish inside the wall.
    EST_MIN="$MAX_ROW_MIN"
    [[ "$EST_MIN" -lt 1 ]] && EST_MIN=1
  else
    EST_MIN="$ROW_BUDGET_MIN"
  fi
  if [[ $(( ELAPSED_MIN + EST_MIN )) -gt "$DEADLINE_MIN" ]]; then
    for DEFER in $(seq "$LINE_NUM" "$END_LINE"); do DEFERRED_ROWS+=("$DEFER"); done
    echo "[deadline] ${ELAPSED_MIN}m elapsed + ~${EST_MIN}m for the next row exceeds ${DEADLINE_MIN}m usable;" \
         "deferring rows ${LINE_NUM}-${END_LINE} (re-run this array to pick them up)"
    break
  fi

  IFS=$'\t' read -r RAW_SCHEME EVENT EXTRA_FIELD <<< "${TASK_LINE}"
  if [[ -n "${EXTRA_FIELD:-}" ]]; then
    echo "Unsupported manifest row ${LINE_NUM}: expected 2 tab-separated fields, got more"
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
  mkdir -p "$RESULTS_ROOT/$ANCHOR_SUBDIR/full_cohort_risk_scores/$EVENT"

  echo "Running row ${LINE_NUM}: scheme=${SCHEME}, event=${EVENT}, anchor=${ANCHOR}"
  ROW_START=$SECONDS
  if ! "$PYTHON_BIN" -m pipelines.training.run_full_cohort_risk_scores \
    --scheme "$SCHEME" \
    --event "$EVENT" \
    --anchor "$ANCHOR" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --max-iter "${COXNET_MAX_ITER:-1000}" \
    --backend "${COXNET_BACKEND:-threading}" \
    ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
    echo "[error] row ${LINE_NUM} failed: scheme=${SCHEME}, event=${EVENT}"
    FAILED_ROWS=$((FAILED_ROWS + 1))
  fi
  ROW_MIN=$(( (SECONDS - ROW_START) / 60 ))
  [[ "$ROW_MIN" -gt "$MAX_ROW_MIN" ]] && MAX_ROW_MIN="$ROW_MIN"
  ROWS_DONE=$((ROWS_DONE + 1))
done

echo "Task ${SLURM_ARRAY_TASK_ID}: completed ${ROWS_DONE} row(s) in $(( (SECONDS - TASK_START) / 60 ))m"
if [[ "${#DEFERRED_ROWS[@]}" -gt 0 ]]; then
  _last_idx=$(( ${#DEFERRED_ROWS[@]} - 1 ))
  echo "[deadline] ${#DEFERRED_ROWS[@]} row(s) deferred to a future run: ${DEFERRED_ROWS[0]}-${DEFERRED_ROWS[$_last_idx]}"
fi

if [[ "$FAILED_ROWS" -gt 0 ]]; then
  echo "$FAILED_ROWS manifest row(s) failed"
  exit 1
fi
