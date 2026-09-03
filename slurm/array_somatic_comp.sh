#!/bin/bash

#SBATCH --job-name=coxnet_somatic_comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-0%1
#SBATCH --output=slurm/array_somatic_comp/output/%A_%a.out
#SBATCH --error=slurm/array_somatic_comp/error/%A_%a.err

# Somatic-only variant of array_feature_comp.sh.
#
# array_feature_comp.sh runs a *class* of modalities per manifest row
# (MODALITY_CLASS=big -> somatic, prs, text). This script pins --modality somatic for
# every row, so a somatic re-run does not drag prs and text along with it. Resource
# sizing matches the "big" class, which is where somatic sits: its design matrix is a
# wide gene-by-alteration panel of data-dependent width.
#
# Outputs, skip logic and result paths are identical to the feature-comp arrays --
# run_feature_comp_task.py writes the same
#   <feature_comps>/<event>/somatic_{test,val}.csv, somatic_ipcw_reference.csv.gz
#   <held_out_risk_scores>/<event>/somatic_risk_scores.csv
# files and skips a row whose outputs already exist unless OVERWRITE=1. So this job and
# array_feature_comp.sh are safe to interleave; each skips what the other finished.
#
# Manifest: two tab-separated fields per row (scheme, event) -- the same
# feature_comp_tasks.tsv the feature-comp array reads. A third "modality" field, if
# present, is ignored here: this script is somatic-only by construction.
#
# SCHEDULER SIZING. The binding constraint is the scheduler, not the node: many short jobs
# queue worse than a few long ones. Observed per-modality cost on the "big" class is
# ~2.5-11m grid search + ~16-49m held-out risk, i.e. ~20-60m per row, mean ~45m. Against
# the 24h wall that puts ROWS_PER_TASK at 28 rows (~21h, ~88% utilization), versus the
# feature-comp default of 7 (~5h, ~20%) -- and takes a 1978-row manifest from 283 array
# tasks down to 71.
#
# A fixed row count alone is unsafe at that utilization: a run of slow rows would overrun
# the wall and lose the in-flight row's work. So the loop below is deadline-guarded -- it
# stops dispatching once the elapsed time plus a running estimate of the next row's cost
# would exceed the task's time limit, and reports the rows it deferred. Re-running the same
# array picks those up, since run_feature_comp_task.py skips completed rows. Set
# ROW_BUDGET_MIN to override the per-row estimate used before any row has finished.
#
# BATCHING. run_feature_comp_task.py loads its data once per *process*. Invoking it once per
# row (as array_feature_comp.sh does) re-reads the embedding frame, cancer-type file, somatic
# file and re-computes the 4-way MRN intersection for every row -- the repeated
# "Common feature cohort: N patients" line in the logs. This script instead groups the task's
# rows by scheme and passes each group's endpoints to --events, so that load happens once per
# scheme-group rather than once per row. With ROWS_PER_TASK=28 and a manifest sorted by
# scheme, that is typically one load per task instead of 28.
#
# --events is optional and additive on the Python side: array_feature_comp.sh and any job
# already sitting in the queue still use --event and are completely unaffected. Set
# BATCH_EVENTS=0 here to fall back to one process per row (the old behaviour) if a batched
# task ever misbehaves.

# ANCHOR selects the time-zero anchor (see anchors.py): "treatment" (default) or
# "sequencing". Forwarded to run_feature_comp_task.py as --anchor; non-default anchors
# nest results under <scheme_results_dir>/anchor_<anchor>/ (schemes.py scheme_results_dir).
ANCHOR=${ANCHOR:-treatment}

PROJECT_ROOT=${PROJECT_ROOT:-/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project}
if [[ "$ANCHOR" == "treatment" ]]; then ANCHOR_SUFFIX=""; else ANCHOR_SUFFIX="__${ANCHOR}"; fi
MANIFEST=${MANIFEST:-$PROJECT_ROOT/slurm/slurm_manifests/feature_comp_tasks${ANCHOR_SUFFIX}.tsv}
ROWS_PER_TASK=${ROWS_PER_TASK:-28}

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

mkdir -p slurm/array_somatic_comp/output slurm/array_somatic_comp/error

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

echo "Task ${SLURM_ARRAY_TASK_ID}: processing manifest rows ${START_LINE}-${END_LINE} (modality=somatic)"

export CTEP_DATA_PATH=${CTEP_DATA_PATH:-${DATA_PATH:-/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project}}
RESULTS_ROOT="$CTEP_DATA_PATH/time-to-event_analysis/results"
FAILED_ROWS=0

# --- Deadline guard ------------------------------------------------------------------
# Stop starting new rows when the next one is unlikely to finish inside the wall clock.
# TIME_LIMIT_MIN is read from Slurm when available; SAFETY_MIN is held back for the final
# row's writes and the epilog. ROW_BUDGET_MIN seeds the per-row estimate, which is then
# replaced by the observed mean once rows start completing.
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-}"
if [[ -z "$TIME_LIMIT_MIN" ]] && command -v squeue >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # TimeLimit as raw minutes; squeue prints "UNLIMITED" for jobs without one.
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
ROW_BUDGET_MIN="${ROW_BUDGET_MIN:-45}"            # prior estimate; see SCHEDULER SIZING
BATCH_EVENTS="${BATCH_EVENTS:-1}"                 # 0 = one process per row (pre-batching)
DEADLINE_MIN=$(( TIME_LIMIT_MIN - SAFETY_MIN ))
TASK_START=$SECONDS
ROWS_DONE=0
MAX_ROW_MIN=0          # slowest row seen so far; the guard budgets on this, not the mean
DEFERRED_ROWS=()

echo "Deadline guard: limit=${TIME_LIMIT_MIN}m safety=${SAFETY_MIN}m usable=${DEADLINE_MIN}m, initial per-row budget=${ROW_BUDGET_MIN}m"

# --- Phase 1: parse + validate every assigned row, grouping consecutive rows by scheme --
# Grouping is what makes batching pay off: run_feature_comp_task.py loads per process and
# the load depends on the scheme (and anchor), not the event, so all endpoints sharing a
# scheme can be handed to one process via --events. Validation happens up front so a
# malformed manifest row fails the task before any compute is spent.
GROUP_SCHEMES=()      # scheme for each group
GROUP_EVENTS=()       # space-separated events for each group
GROUP_FIRST_ROW=()    # first manifest line number in each group (for deadline reporting)
GROUP_NROWS=()        # how many manifest rows each group covers
PREV_SCHEME=""

for LINE_NUM in $(seq "$START_LINE" "$END_LINE"); do
  TASK_LINE=$(sed -n "${LINE_NUM}p" "$MANIFEST")
  if [[ -z "${TASK_LINE}" ]]; then
    echo "Skipping empty manifest row ${LINE_NUM}"
    continue
  fi

  # Third field (modality), if the manifest carries one, is read and discarded: this
  # array is somatic-only regardless of what the manifest pins.
  IFS=$'\t' read -r RAW_SCHEME EVENT _IGNORED_MODALITY EXTRA_FIELD <<< "${TASK_LINE}"
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

  # BATCH_EVENTS=0 forces one group per row, i.e. one process per row as before.
  if [[ "$BATCH_EVENTS" == "1" && "$SCHEME" == "$PREV_SCHEME" ]]; then
    _gi=$(( ${#GROUP_SCHEMES[@]} - 1 ))
    GROUP_EVENTS[$_gi]="${GROUP_EVENTS[$_gi]} $EVENT"
    GROUP_NROWS[$_gi]=$(( ${GROUP_NROWS[$_gi]} + 1 ))
  else
    GROUP_SCHEMES+=("$SCHEME")
    GROUP_EVENTS+=("$EVENT")
    GROUP_FIRST_ROW+=("$LINE_NUM")
    GROUP_NROWS+=(1)
  fi
  PREV_SCHEME="$SCHEME"
done

N_GROUPS=${#GROUP_SCHEMES[@]}
if [[ "$N_GROUPS" -eq 0 ]]; then
  echo "No usable rows assigned to task ${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
echo "Task ${SLURM_ARRAY_TASK_ID}: $((END_LINE - START_LINE + 1)) row(s) in ${N_GROUPS} scheme-group(s); BATCH_EVENTS=${BATCH_EVENTS}"

# --- Phase 2: one process per scheme-group -------------------------------------------
for (( GI=0; GI<N_GROUPS; GI++ )); do
  SCHEME="${GROUP_SCHEMES[$GI]}"
  # shellcheck disable=SC2206  # deliberate word-split: events were joined on spaces above
  EVENTS=(${GROUP_EVENTS[$GI]})
  GROUP_ROWS=${GROUP_NROWS[$GI]}

  # The deadline guard budgets per ROW, so a group of N rows needs N row-budgets. Groups
  # are not split: a group is either started whole or deferred whole.
  ELAPSED_MIN=$(( (SECONDS - TASK_START) / 60 ))
  if [[ "$ROWS_DONE" -gt 0 ]]; then
    # Budget on the slowest row observed, not the mean: per-row cost spans ~20-60m, so a
    # mean dragged down by fast rows would green-light a slow row that cannot finish.
    EST_ROW_MIN="$MAX_ROW_MIN"
    [[ "$EST_ROW_MIN" -lt 1 ]] && EST_ROW_MIN=1
  else
    EST_ROW_MIN="$ROW_BUDGET_MIN"
  fi
  EST_MIN=$(( EST_ROW_MIN * GROUP_ROWS ))
  if [[ $(( ELAPSED_MIN + EST_MIN )) -gt "$DEADLINE_MIN" ]]; then
    FIRST_DEFERRED=${GROUP_FIRST_ROW[$GI]}
    for DEFER in $(seq "$FIRST_DEFERRED" "$END_LINE"); do DEFERRED_ROWS+=("$DEFER"); done
    echo "[deadline] ${ELAPSED_MIN}m elapsed + ~${EST_MIN}m for the next group (${GROUP_ROWS} row(s))" \
         "exceeds ${DEADLINE_MIN}m usable; deferring rows ${FIRST_DEFERRED}-${END_LINE}" \
         "(re-run this array to pick them up)"
    break
  fi

  echo "Running group $((GI + 1))/${N_GROUPS}: scheme=${SCHEME}, ${GROUP_ROWS} event(s)=${EVENTS[*]}, modality=somatic, anchor=${ANCHOR}"
  GROUP_START=$SECONDS

  if [[ "${#EVENTS[@]}" -gt 1 ]]; then
    EVENT_ARGS=(--events "${EVENTS[@]}")
  else
    # Single-endpoint groups use --event, the long-standing form, so the common case is
    # byte-identical to what array_feature_comp.sh runs.
    EVENT_ARGS=(--event "${EVENTS[0]}")
  fi

  if ! "$PYTHON_BIN" -m pipelines.training.run_feature_comp_task \
    --scheme "$SCHEME" \
    "${EVENT_ARGS[@]}" \
    --modality somatic \
    --anchor "$ANCHOR" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}" \
    --max-iter "${COXNET_MAX_ITER:-1000}" \
    --backend "${COXNET_BACKEND:-threading}" \
    ${OVERWRITE_FLAG[@]+"${OVERWRITE_FLAG[@]}"}; then
    echo "[error] group $((GI + 1)) failed: scheme=${SCHEME}, events=${EVENTS[*]}, modality=somatic"
    # run_feature_comp_task.py isolates each event internally, so a non-zero exit means at
    # least one endpoint in the group failed, not necessarily all of them.
    FAILED_ROWS=$((FAILED_ROWS + 1))
  fi

  GROUP_MIN=$(( (SECONDS - GROUP_START) / 60 ))
  ROW_MIN=$(( GROUP_MIN / GROUP_ROWS ))
  [[ "$ROW_MIN" -gt "$MAX_ROW_MIN" ]] && MAX_ROW_MIN="$ROW_MIN"
  ROWS_DONE=$(( ROWS_DONE + GROUP_ROWS ))
done

echo "Task ${SLURM_ARRAY_TASK_ID}: completed ${ROWS_DONE} row(s) in $(( (SECONDS - TASK_START) / 60 ))m"
if [[ "${#DEFERRED_ROWS[@]}" -gt 0 ]]; then
  _last_idx=$(( ${#DEFERRED_ROWS[@]} - 1 ))
  echo "[deadline] ${#DEFERRED_ROWS[@]} row(s) deferred to a future run: ${DEFERRED_ROWS[0]}-${DEFERRED_ROWS[$_last_idx]}"
fi

if [[ "$FAILED_ROWS" -gt 0 ]]; then
  echo "$FAILED_ROWS somatic group(s) failed"
  exit 1
fi
