"""Generate Mortality Trajectories script for model evaluation workflows."""

import concurrent.futures
import io
import multiprocessing
import os
import time

import numpy as np
import polars as pl
import zstandard as zstd

from config import FEATURE_PATH, NOTES_PATH, SURV_PATH
from pipelines.training.slurm_array_utils import (DEFAULT_ALPHAS, DEFAULT_L1_RATIOS,
                                                  adaptive_low_alphas_for)
from schemes import scheme_results_dir
from survival import (LandmarkCheckpoint, dataframe_fingerprint, fit_external_CoxPH_model,
                      generate_survival_embedding_df, get_nested_heldout_risk_scores_CoxPH,
                      run_grid_CoxPH_parallel, score_external_CoxPH)
from shared.polars_utils import filter_finite_rows


# Progress reporting.  This script runs for hours across 21 landmarks with two long silent
# phases each (re-pooling the embedding array, then a nested CV fit), so silence is
# indistinguishable from a hang.  Every line below is newline-delimited rather than a
# carriage-return redraw: the notebook driver (04d) runs this as a piped subprocess, where
# `\r` bars collapse into one unreadable line.
SHOW_PROGRESS = os.environ.get("TRAJECTORY_PROGRESS", "1") not in {"0", "false", "False"}

# Resume.  Each landmark's scores are checkpointed as soon as they exist, so an interrupted run
# picks up where it left off instead of refitting from month 0.  Set TRAJECTORY_RESUME=0 to force
# a full recompute, and TRAJECTORY_RETRY_FAILED=1 to refit landmarks a previous run recorded as
# failed (they are skipped by default, so a resume does not burn hours re-hitting the same error).
RESUME = os.environ.get("TRAJECTORY_RESUME", "1") not in {"0", "false", "False"}
RETRY_FAILED = os.environ.get("TRAJECTORY_RETRY_FAILED", "0") not in {"0", "false", "False"}

# Landmark-level parallelism.  The 20 post-baseline landmarks are mutually independent -- each is
# drawn from the full baseline cohort (see the `prev_mrns` note at the loop) and scored by the one
# month-0 model -- so they are computed in a process pool rather than one after another.
#
# Processes, not threads: the dominant per-landmark cost is the per-(patient, note-type) Python
# loop in survival.preprocessing.pool_embedding_series_vectorized, which holds the GIL, so threads
# would serialize exactly the part worth parallelizing.
#
# TRAJECTORY_WORKERS=1 restores the original strictly serial path (no pool, no memmap), which is
# the fallback if a worker is OOM-killed.
WORKERS_ENV = os.environ.get("TRAJECTORY_WORKERS", "").strip()

# Per-worker RAM budget for auto-sizing.  A worker holds one landmark's filtered note metadata and
# its pooled embedding frame (n_eligible x 3 note types x embed_dim, float64) on top of the shared
# read-only memmap, which is not counted here because it is paged, not copied.
WORKER_MEM_GB = float(os.environ.get("TRAJECTORY_WORKER_MEM_GB", "8"))

# Each worker is single-core by construction.  Without this every worker would size its BLAS and
# Rayon pools to the whole machine -- not just contending for cores, but exhausting RLIMIT_NPROC
# outright once N workers each ask for ~2 x n_cpu threads.  Exported by the parent immediately
# before the pool is built (see main()), because a spawned child builds these pools during its
# own imports, before any initializer of ours can run.
SINGLE_THREAD_ENV = {var: "1" for var in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "POLARS_MAX_THREADS", "RAYON_NUM_THREADS")}


def _stage(msg: str) -> None:
    """One timestamped progress line, flushed so a piped reader sees it immediately."""
    if SHOW_PROGRESS:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# === Landmark worker ==========================================================================
# State a worker needs on every landmark, set once per process by _worker_init rather than
# pickled per task: the note metadata and baseline frame are large, and the embedding array is
# opened as a read-only memmap so N workers share the OS page cache instead of each holding a
# private multi-GB copy.
_W: dict = {}


def _worker_init(embeddings_path: str, embeddings_shape: tuple, embeddings_dtype: str,
                 embeddings_offset: int, notes_meta: pl.DataFrame, events_data: pl.DataFrame,
                 cancer_type_df: pl.DataFrame, fitted_model: dict, spec: dict) -> None:
    """Per-process setup. Runs once per worker, not once per landmark.

    The embedding array is mapped read-only at the .npy payload offset the parent already
    parsed, so every worker shares one mapping through the OS page cache. `pool_embedding_
    series_vectorized` only fancy-indexes rows out of it, which copies into a fresh array --
    nothing writes through the mapping.
    """
    # No thread limiting here: the parent exports SINGLE_THREAD_ENV before creating the pool,
    # and by the time this runs the child's polars/numpy pools are already built from that
    # inherited environment.  Setting it at this point would be too late to have any effect.
    _W['embeddings'] = np.memmap(embeddings_path, dtype=np.dtype(embeddings_dtype), mode='r',
                                 shape=tuple(embeddings_shape), offset=int(embeddings_offset))
    _W['notes_meta'] = notes_meta
    _W['events_data'] = events_data
    _W['cancer_type_df'] = cancer_type_df
    _W['fitted_model'] = fitted_model
    _W['spec'] = spec


def _worker_init_serial(embeddings: np.ndarray, notes_meta: pl.DataFrame,
                        events_data: pl.DataFrame, cancer_type_df: pl.DataFrame,
                        fitted_model: dict, spec: dict) -> None:
    """Populate the same worker state in-process, for the TRAJECTORY_WORKERS=1 serial path.

    Uses the already-decompressed in-memory array: with one worker there is nothing to share,
    so spilling a multi-GB memmap would be pure cost.
    """
    _W['embeddings'] = embeddings
    _W['notes_meta'] = notes_meta
    _W['events_data'] = events_data
    _W['cancer_type_df'] = cancer_type_df
    _W['fitted_model'] = fitted_model
    _W['spec'] = spec


def _score_landmark(month_adj: int, eligible_mrns: list) -> dict:
    """Pool this landmark's notes and score them with the pre-fitted month-0 model.

    No model is fit here. `_W['fitted_model']` is the single bundle fit once in the parent; a
    worker only re-pools this landmark's notes and applies it.

    Returns a plain dict rather than raising, so one landmark's failure is recorded and the
    remaining landmarks still run -- matching the try/except the serial loop had. The parent
    owns every checkpoint write; a worker only computes.
    """
    started = time.time()
    spec = _W['spec']
    event = spec['event']
    landmark_day = month_adj * 30
    try:
        notes_meta_copy = _W['notes_meta'].filter(pl.col('DFCI_MRN').is_in(eligible_mrns))
        events_data_copy = _W['events_data'].filter(pl.col('DFCI_MRN').is_in(eligible_mrns))
        # continuous_window=False must match the month-0 pooling: the month-0 model is applied to
        # these features unchanged, so a landmark pooled under a different note-selection rule
        # would shift the feature distribution out from under the fixed model.
        monthly_data = (generate_survival_embedding_df(
            notes_meta_copy, events_data_copy, _W['embeddings'], note_types=spec['note_types'],
            pool_fx={key: 'time_decay_mean' for key in spec['note_types']},
            decay_param=spec['decay_param'], max_note_window=landmark_day,
            continuous_window=False)
            .select(['DFCI_MRN', event, f'tt_{event}'] + spec['base_vars'] + spec['embed_cols'])
            .join(_W['cancer_type_df'], on='DFCI_MRN'))
        # Complete-case on the model's columns, scoped the same way as the month-0 filter.  The
        # .select() above already dropped everything else (AGE_AT_SEQUENCING included), so this
        # is currently equivalent to filtering on dtype -- named explicitly so that widening the
        # select later cannot silently reintroduce an unrelated column as a drop criterion.
        # Every row already satisfies tt > 0 post-shift by construction of `eligible_mrns`;
        # this only drops rows with missing embeddings/covariates.
        landmark_model_cols = [c for c in
                              dict.fromkeys(spec['base_vars'] + spec['type_cols']
                                            + spec['embed_cols'] + [event, f'tt_{event}'])
                              if c in monthly_data.columns]
        monthly_data = filter_finite_rows(monthly_data, landmark_model_cols)

        # Inference only, with the ONE model fit at month 0 -- no refit here. Its imputation
        # means and continuous-variable scaling were derived from the baseline training frame
        # alone, so this landmark's own feature distribution never influences its transform and
        # every landmark's scores stay on one common scale. No outcome from this landmark is
        # used; monthly_data's event columns exist only to satisfy the frame's schema.
        risk_scores = pl.DataFrame({
            'DFCI_MRN': monthly_data['DFCI_MRN'],
            'risk_score': score_external_CoxPH(_W['fitted_model'], monthly_data),
        })
        return {'month': month_adj, 'ok': True, 'scores': risk_scores,
                'n_at_risk': len(eligible_mrns), 'n_scored': int(risk_scores.height),
                'n_complete': int(monthly_data.height),
                'elapsed_s': round(time.time() - started, 1)}
    except Exception as exc:
        return {'month': month_adj, 'ok': False, 'error': f"{type(exc).__name__}: {exc}",
                'n_at_risk': len(eligible_mrns),
                'elapsed_s': round(time.time() - started, 1)}


def _resolve_workers(n_tasks: int) -> int:
    """Auto-size the pool: leave a core free, and budget WORKER_MEM_GB per worker."""
    if WORKERS_ENV:
        return max(1, min(int(WORKERS_ENV), n_tasks))
    workers = max(1, (os.cpu_count() or 1) - 1)
    try:
        # Linux-only; on a cgroup-limited node this reflects the real allowance better than
        # total system memory. Skipped silently where unavailable (e.g. macOS).
        available_gb = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')) / 1024 ** 3
        workers = max(1, min(workers, int(available_gb // WORKER_MEM_GB)))
    except (ValueError, OSError, AttributeError):
        pass
    return max(1, min(workers, n_tasks))


def main() -> None:
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    run_started = time.time()
    trajectory_path = os.path.join(scheme_results_dir("death_met"), 'mortality_trajectories/')
    os.makedirs(trajectory_path, exist_ok=True)

    # Load datasets
    _stage("loading note metadata ...")
    notes_meta = pl.read_parquet(NOTES_PATH + 'full_clinical_notes_embeddings_metadata.parquet')
    _stage(f"  {notes_meta.height:,} notes")

    _stage("decompressing embedding array (large; this takes a few minutes) ...")
    with open(NOTES_PATH + 'full_clinical_notes_embeddings_as_array.npy.zst', 'rb') as f:
        embeddings_data = np.load(io.BytesIO(zstd.decompress(f.read())))
    embeddings_data = embeddings_data.astype(np.float32)
    _stage(f"  embeddings {embeddings_data.shape}, {embeddings_data.nbytes / 1e9:.2f} GB in memory")

    _stage("loading survival cohort and cancer types ...")
    events_data = pl.read_parquet(SURV_PATH + 'death_met_surv_df.parquet')
    cancer_type_df = pl.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'))
    _stage(f"  {events_data.height:,} patients in the survival cohort")

    # Drop the sequencing columns outright.  This is a treatment-anchored death model: it uses
    # AGE_AT_TREATMENTSTART, never AGE_AT_SEQUENCING, and nothing downstream reads
    # sequencing_date.  They ride along only because the survival parquet is written with the
    # whole of BASE_OUTPUT_COLS.  Both are null for every patient without a genomic specimen
    # (build_cohort derives them from a left-joined sequencing_date), so leaving them in the
    # frame made them a live drop criterion for any complete-case filter scoped by dtype -- which
    # is exactly what evicted tens of thousands of patients from a model that never reads them.
    # Removing them here, at the source, means no filter or feature list downstream can pick
    # them up again.
    _sequencing_cols = [c for c in ('AGE_AT_SEQUENCING', 'sequencing_date')
                        if c in events_data.columns]
    if _sequencing_cols:
        events_data = events_data.drop(_sequencing_cols)
        _stage(f"  dropped sequencing columns (unused by this model): "
               f"{', '.join(_sequencing_cols)}")

    event = 'death'
    # Hyperparameter grid matched to the full-cohort training setup
    # (pipelines/training/run_full_cohort_*), including its adaptive low-alpha refinement, so a
    # month-0 trajectory score is tuned over the same space as the full-cohort risk score.
    alphas_to_test = DEFAULT_ALPHAS
    l1_ratios = DEFAULT_L1_RATIOS
    low_alphas = adaptive_low_alphas_for(alphas_to_test)
    max_iter = 1000

    # regenerate predication dataframe at time 0 with new decay parameter
    decay_param = 0.1
    note_types = ['Clinician', 'Imaging', 'Pathology']
    # continuous_window=False matches the canonical text cohort built by
    # pipelines/preprocessing/generate_embedding_prediction_datasets.py, which pools every
    # pre-landmark note.  The default (True) routes through
    # find_continuous_records_to_analyze, which drops notes across gaps > 2 years -- a
    # different note set, hence different embeddings, and a month-0 column that could not be
    # compared against the full-cohort risk scores it is meant to reproduce.
    _stage(f"pooling baseline embeddings (decay_param={decay_param}, window=0) ...")
    pooled_baseline = generate_survival_embedding_df(
        notes_meta, events_data, embeddings_data,
        note_types=note_types, pool_fx={key: 'time_decay_mean' for key in note_types},
        decay_param=decay_param, max_note_window=0, continuous_window=False,
    )

    # Cohort attrition, reported step by step.  The cancer-type join is an INNER join, so a
    # patient with no cancer_type_df row leaves the cohort here -- before `cohort_mrns` is
    # taken, and therefore before the per-landmark `n_at_risk` denominators are computed.
    # Without this the coverage table downstream would show full coverage of a quietly
    # shrunken cohort, since it can only measure against a denominator this join already cut.
    n_pooled = pooled_baseline.height
    full_prediction_df = pooled_baseline.join(cancer_type_df, on='DFCI_MRN')
    n_typed = full_prediction_df.height

    # Feature columns, defined before the complete-case filter because the filter is scoped to
    # them (see below).  `base_vars` and the cancer-type indicators are the model's non-embedding
    # inputs; embed_cols is the pooled text block plus the two year-adjustment columns.
    base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
    embed_cols_all = [c for c in full_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
    type_cols = [c for c in full_prediction_df.columns if c.startswith('CANCER_TYPE_')]
    base_vars_all = base_vars + type_cols

    # Complete-case on the MODEL's columns only -- the pooled embeddings, the year-adjustment
    # columns, the two baseline covariates, the cancer-type indicators, and this endpoint's own
    # outcome pair.  Emphatically NOT every numeric column in the frame: `events_data` also
    # carries every OTHER endpoint's event/tt pair (tt_brain_met and friends), which are null for
    # patients who never had that event and would otherwise each become a drop criterion for a
    # death model.  The sequencing columns were the worst case and are now dropped at load, but
    # scoping the filter is the durable fix -- a dtype-based filter silently acquires any numeric
    # column a future change adds to the parquet.  Mirrors the canonical builder, which likewise
    # filters on the pooled feature columns rather than on dtype.
    model_cols = [c for c in
                  dict.fromkeys(base_vars_all + embed_cols_all + [event, f'tt_{event}'])
                  if c in full_prediction_df.columns]
    full_prediction_df = filter_finite_rows(full_prediction_df, model_cols)
    n_complete = full_prediction_df.height
    full_prediction_df = full_prediction_df.filter(pl.col(f'tt_{event}') > 0)

    attrition = [
        ('survival cohort', events_data.height),
        ('after embedding pooling', n_pooled),
        ('after cancer-type join', n_typed),
        ('after complete-case filter', n_complete),
        ('baseline cohort (tt>0)', full_prediction_df.height),
    ]
    _prev = None
    for _label, _n in attrition:
        _drop = '' if _prev is None else f'  ({_prev - _n:,} dropped)'
        _stage(f"  {_label + ':':<28}{_n:>7,}{_drop}")
        _prev = _n
    if n_typed < n_pooled:
        pct = 100.0 * (n_pooled - n_typed) / n_pooled
        _stage(f"  [warn] {pct:.1f}% of pooled patients had no cancer_type_df row and left the "
               "cohort before the landmark denominators were computed")

    # Persisted so the notebook's coverage cell can show what the at-risk denominators are a
    # denominator *of*: every count below is upstream of `cohort_mrns`, so a patient dropped
    # here is invisible to the per-landmark coverage table.
    pl.DataFrame({'step': [s for s, _ in attrition],
                  'n_patients': [n for _, n in attrition]}).write_csv(
        os.path.join(trajectory_path, f'cohort_attrition_w_decay_param_{decay_param}.csv'))

    # Column groups. base_vars, type_cols and embed_cols_all are defined above, before the
    # complete-case filter that is scoped to them.
    embed_cols = embed_cols_all
    continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

    ## Generate monthly data frames
    months_to_test = [i * 3 for i in range(1, 21)]

    cohort_mrns = full_prediction_df['DFCI_MRN'].unique().to_list()

    # === Checkpoint store (persists each landmark as it completes; enables resume) ===
    # Fingerprint the deterministic inputs. A mismatch on a later run means the cohort, the
    # feature set, the decay parameter, or the landmark grid changed, so stored landmarks are
    # stale and LandmarkCheckpoint starts fresh on its own rather than mixing score columns
    # computed under different definitions into one trajectory matrix.
    fingerprint = {
        'script': 'mortality_trajectories',
        # Bumped when landmarks stopped refitting per landmark and became inference from the
        # month-0 model: checkpoints written under the old scheme are not comparable.
        'scoring': 'fit_once_month0_infer_forward',
        # Note-selection rule. `data_hash` below already changes when this does, but naming it
        # makes the invalidation legible in landmark_meta.json rather than an opaque hash diff.
        'continuous_window': False,
        'alphas': [float(a) for a in alphas_to_test],
        'l1_ratios': [float(r) for r in l1_ratios],
        'max_iter': max_iter,
        'n_cohort': int(len(cohort_mrns)),
        'n_embed': int(len(embed_cols)),
        'n_base': int(len(base_vars)),
        'n_type': int(len(type_cols)),
        'decay_param': decay_param,
        'months': list(months_to_test),
        'seed': 1234,
        'data_hash': dataframe_fingerprint(
            full_prediction_df,
            ['DFCI_MRN', event, f'tt_{event}'] + base_vars + type_cols + embed_cols,
        ),
    }
    ckpt = LandmarkCheckpoint(os.path.join(trajectory_path, 'checkpoints'), fingerprint,
                              resume=RESUME, retry_failed=RETRY_FAILED)
    trajectory_predictions_df = pl.DataFrame(
        {'DFCI_MRN': cohort_mrns} |
        {f'plus_{month_adj}_months_data': [np.nan for _ in range(len(cohort_mrns))] for month_adj in months_to_test}
    )

    # === Month 0: select hyperparameters ONCE, on the baseline cohort ===
    # Every later landmark re-scores this same model rather than refitting its own. Refitting per
    # landmark made each column a different model -- different penalty, different scaling -- so a
    # per-patient slope across columns (figures.prep.figure4) mixed real risk change with model
    # drift. One fixed model makes the trajectory a statement about the patient.
    #
    # Tuning uses the full-cohort grid search; the reported month-0 column stays the nested-CV
    # held-out score, so no patient's own outcome informs their own month-0 value.
    _stage(f"landmark 0/{len(months_to_test)} (month 0, baseline): "
           f"selecting hyperparameters on {full_prediction_df.height:,} patients ...")
    _t0 = time.time()
    _, month0_val, month0_model = run_grid_CoxPH_parallel(
        full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
        max_iter=max_iter, adaptive_low_alphas=low_alphas,
    )
    if month0_model is None:
        raise RuntimeError(
            "Month-0 grid search did not converge on any hyperparameter pair, so there is no "
            "model to carry forward to the later landmarks."
        )
    _best = filter_finite_rows(month0_val, ['mean_auc(t)']).sort(
        'mean_auc(t)', descending=True
    ).row(0, named=True)
    best_l1, best_alpha = float(_best['l1_ratio']), float(_best['alpha'])
    _stage(f"  selected l1_ratio={best_l1}, alpha={best_alpha:.3e} "
           f"(mean_auc(t)={_best['mean_auc(t)']:.4f}) in {(time.time() - _t0) / 60:.1f} min")

    # Month 0's own column: nested-CV held-out scores under the same grid.
    if ckpt.landmark_done(0):
        risk_map = ckpt.load_landmark(0)
    else:
        _stage("  fitting nested CV for the month-0 column ...")
        _t0 = time.time()
        risk_scores = get_nested_heldout_risk_scores_CoxPH(
            full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
            l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
            id_col='DFCI_MRN', max_iter=max_iter, adaptive_low_alphas=low_alphas,
            show_progress=SHOW_PROGRESS,
        )
        _elapsed = time.time() - _t0
        ckpt.save_landmark(0, risk_scores,
                           meta={'n_at_risk': int(full_prediction_df.height),
                                 'n_scored': int(risk_scores.height),
                                 'elapsed_s': round(_elapsed, 1)})
        _stage(f"  month 0 done in {_elapsed / 60:.1f} min")
        risk_map = dict(zip(risk_scores['DFCI_MRN'].to_list(),
                            risk_scores['risk_score'].to_list()))

    trajectory_predictions_df = trajectory_predictions_df.with_columns(
        pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan).alias('plus_0_months_data')
    )

    # Landmark eligibility is defined on the ORIGINAL, unshifted event times.
    # generate_survival_embedding_df shifts every tt_* column back by the
    # landmark, so a post-shift `tt_death > 0` filter is not a statement about
    # who was at risk at the landmark -- and chaining the survivors of one
    # landmark into the next (the old `prev_mrns`) compounded that into a
    # progressively survivor-enriched cohort, making AUCs non-comparable across
    # the x-axis of the trajectory figure.  Each landmark is now drawn
    # independently from the full baseline cohort.
    baseline_tt = dict(
        zip(events_data['DFCI_MRN'].to_list(), events_data[f'tt_{event}'].to_list())
    )

    failed_landmarks = []
    landmark_sizes = []

    # Landmark eligibility and resume status are decided here, in the parent, before any work is
    # dispatched: the checkpoint store appends to a shared manifest CSV and mutates in-process
    # bookkeeping, neither of which is safe under concurrent writers. Workers only compute and
    # return scores; every save_landmark/mark_failed below happens in this process.
    pending = []
    for month_adj in months_to_test:
        landmark_day = month_adj * 30
        # At risk at the landmark: still under follow-up strictly beyond it.
        eligible_mrns = [
            mrn for mrn in cohort_mrns
            if baseline_tt.get(mrn) is not None and baseline_tt[mrn] > landmark_day
        ]
        landmark_sizes.append((month_adj, len(eligible_mrns)))

        # Resume: reload a completed landmark rather than refitting it. Done before dispatch, so
        # a resumed landmark costs a CSV read instead of a worker slot.
        if ckpt.landmark_done(month_adj):
            risk_map = ckpt.load_landmark(month_adj)
            trajectory_predictions_df = trajectory_predictions_df.with_columns(
                pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan)
                .alias(f'plus_{month_adj}_months_data')
            )
            continue

        # A landmark a previous run recorded as failed is left failed unless RETRY_FAILED is
        # set: without this a resume would spend hours re-hitting the same error every time.
        if ckpt.should_skip_failed(month_adj):
            reason = ckpt.failure_reason(month_adj) or 'previously failed'
            _stage(f"landmark month {month_adj}: skipping previously failed landmark "
                   f"({reason}); set TRAJECTORY_RETRY_FAILED=1 to refit")
            failed_landmarks.append((month_adj, f"previously failed: {reason}"))
            continue

        if not eligible_mrns:
            print(f"Trajectory landmark {month_adj} months: no patients at risk, skipping")
            ckpt.mark_failed(month_adj, "no patients at risk at landmark", meta={'n_at_risk': 0})
            failed_landmarks.append((month_adj, "no patients at risk at landmark"))
            continue

        pending.append((month_adj, eligible_mrns))

    # Dispatch the remaining landmarks. Longest first: the at-risk set shrinks monotonically with
    # the landmark, so the earliest months are the most expensive, and starting them first keeps
    # the pool from finishing with one long task running alone.
    pending.sort(key=lambda task: len(task[1]), reverse=True)

    # Trim what crosses the process boundary.  Workers are spawned (the default on macOS, and
    # what `python -m` gets on Linux under any start method here), so every object in `initargs`
    # is pickled once per worker.  Pooling with continuous_window=False touches only these five
    # columns, and only for cohort patients, so shipping the full multi-million-row note table
    # would be pure serialization cost.
    _worker_note_cols = [c for c in ('DFCI_MRN', 'NOTE_TYPE', 'EMBEDDING_INDEX',
                                     'NOTE_TIME_REL_FIRST_TREATMENT_START', 'NOTE_DATETIME')
                         if c in notes_meta.columns]
    notes_meta_for_workers = (notes_meta
                              .select(_worker_note_cols)
                              .filter(pl.col('DFCI_MRN').is_in(cohort_mrns)))
    events_data_for_workers = events_data.filter(pl.col('DFCI_MRN').is_in(cohort_mrns))

    # === The one model. Fit once, here, on the baseline cohort. ===
    # Every landmark below scores with this exact fitted object -- it is never refit. That is
    # what makes a trajectory a statement about the patient: the coefficients, the imputation
    # means and the feature scaling are all frozen at month 0, so a change in a patient's score
    # across landmarks can only come from a change in that patient's pooled notes.
    _stage(f"fitting the single month-0 model carried forward to every landmark "
           f"(l1_ratio={best_l1}, alpha={best_alpha:.3e}) ...")
    _t0 = time.time()
    fitted_model = fit_external_CoxPH_model(
        full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
        event_col=event, tstop_col=f'tt_{event}', l1_ratio=best_l1, alpha=best_alpha,
        max_iter=max_iter,
    )
    _stage(f"  fit on {fitted_model['n_train']:,} patients in {time.time() - _t0:.1f}s")

    spec = {
        'event': event, 'note_types': note_types, 'decay_param': decay_param,
        'base_vars': base_vars, 'type_cols': type_cols, 'continuous_vars': continuous_vars,
        'embed_cols': embed_cols, 'max_iter': max_iter,
    }

    n_workers = _resolve_workers(len(pending)) if pending else 1
    if pending:
        _stage(f"scoring {len(pending)} landmark(s) across {n_workers} worker(s), "
               f"1 thread each (month 0 and {len(months_to_test) - len(pending)} other "
               "landmark(s) already accounted for)")

    def _record_result(result: dict) -> None:
        """Apply one worker result in the parent: checkpoint it, then fold it into the matrix."""
        nonlocal trajectory_predictions_df
        month_adj = result['month']
        if result['ok']:
            ckpt.save_landmark(month_adj, result['scores'],
                               meta={'n_at_risk': result['n_at_risk'],
                                     'n_scored': result['n_scored'],
                                     'elapsed_s': result['elapsed_s']})
            risk_map = dict(zip(result['scores']['DFCI_MRN'].to_list(),
                                result['scores']['risk_score'].to_list()))
            trajectory_predictions_df = trajectory_predictions_df.with_columns(
                pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan)
                .alias(f'plus_{month_adj}_months_data')
            )
            _stage(f"  month {month_adj} done in {result['elapsed_s'] / 60:.1f} min "
                   f"({result['n_scored']:,}/{result['n_at_risk']:,} at-risk patients scored, "
                   f"{(time.time() - run_started) / 3600:.2f} h elapsed)")
        else:
            print(f"Trajectory landmark {month_adj} months failed: {result['error']}")
            ckpt.mark_failed(month_adj, result['error'],
                             meta={'n_at_risk': result['n_at_risk'],
                                   'elapsed_s': result['elapsed_s']})
            _stage(f"  month {month_adj} FAILED after {result['elapsed_s'] / 60:.1f} min")
            failed_landmarks.append((month_adj, result['error']))

    if pending and n_workers == 1:
        # Serial path, kept explicit: TRAJECTORY_WORKERS=1 gets no pool and no memmap, so a run
        # that OOMs under the pool has a fallback with the original memory profile.
        _worker_init_serial(embeddings_data, notes_meta_for_workers, events_data_for_workers,
                            cancer_type_df, fitted_model, spec)
        for month_adj, eligible_mrns in pending:
            _stage(f"landmark month {month_adj}: {len(eligible_mrns):,} patients at risk")
            _record_result(_score_landmark(month_adj, eligible_mrns))
    elif pending:
        # Workers are spawned, never forked, on every platform.  `main()` has already done
        # substantial polars work by this point, and forking a process whose Rayon threadpool is
        # warm copies the pool's mutexes without the threads holding them -- the child then hangs
        # on its first group_by, which is the very first thing a landmark does.  Verified: a fork
        # child deadlocks here; the spawn path below reproduces the serial scores exactly.
        #
        # Spawn has no copy-on-write inheritance, so the embedding array is spilled to a
        # memmap-able .npy once and the N workers share one read-only mapping through the OS page
        # cache rather than each pickling a private multi-GB copy.  Written next to the
        # checkpoints and removed on the way out.
        #
        # Single-threading is set HERE, in the parent, not in _worker_init: a spawned child
        # builds its Rayon and OpenBLAS pools while importing polars and numpy, which happens
        # in multiprocessing.spawn's run_module -- long before the initializer runs.  Setting
        # it there sized every pool to the whole machine anyway, and N workers x ~2 x n_cpu
        # threads blew past RLIMIT_NPROC on a 64-core node (ThreadPoolBuildError, then a
        # poisoned LazyLock and a failed numpy C-extension import).  Children inherit
        # os.environ across spawn, so assigning it before the pool is what actually lands.
        # The parent's own pools are already built and keep their threads, which is what the
        # month-0 fit above wants.
        for _var, _val in SINGLE_THREAD_ENV.items():
            os.environ[_var] = _val
        os.makedirs(os.path.join(trajectory_path, 'checkpoints'), exist_ok=True)
        embeddings_mmap_path = os.path.join(trajectory_path, 'checkpoints',
                                            f'_embeddings_shared_{os.getpid()}.npy')
        _stage(f"spilling embedding array to a shared memmap "
               f"({embeddings_data.nbytes / 1e9:.2f} GB) ...")
        _spill = np.lib.format.open_memmap(
            embeddings_mmap_path, mode='w+', dtype=embeddings_data.dtype,
            shape=embeddings_data.shape)
        _spill[:] = embeddings_data
        _spill.flush()
        del _spill
        # Offset, shape and dtype of the raw block inside the .npy, read once here so each
        # worker maps the payload directly instead of re-parsing the header.  Public header
        # readers only (no np.lib.format._read_array_header, which is private and has changed
        # signature across NumPy versions).
        with open(embeddings_mmap_path, 'rb') as _fh:
            _major, _minor = np.lib.format.read_magic(_fh)
            _read_header = (np.lib.format.read_array_header_1_0 if _major == 1
                            else np.lib.format.read_array_header_2_0)
            _mm_shape, _mm_fortran, _mm_dtype = _read_header(_fh)
            _mm_offset = _fh.tell()
        if _mm_fortran:
            raise RuntimeError("Spilled embedding memmap is Fortran-ordered; workers map it C-order")

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_worker_init,
                initargs=(embeddings_mmap_path, _mm_shape, str(_mm_dtype), _mm_offset,
                          notes_meta_for_workers, events_data_for_workers, cancer_type_df,
                          fitted_model, spec),
            ) as pool:
                in_flight = {pool.submit(_score_landmark, month_adj, eligible_mrns): month_adj
                             for month_adj, eligible_mrns in pending}
                n_done = 0
                try:
                    for future in concurrent.futures.as_completed(in_flight):
                        _record_result(future.result())
                        n_done += 1
                        _stage(f"  [{n_done}/{len(pending)} dispatched landmarks complete]")
                except KeyboardInterrupt:
                    # Without this, pool shutdown blocks on every queued task. Landmarks already
                    # recorded above are checkpointed, so a resume picks up from them.
                    for future in in_flight:
                        future.cancel()
                    print("\nInterrupted -- cancelled queued landmarks, waiting for in-flight ones.")
                    raise
        finally:
            # The spill file is a derived copy of an input, not a checkpoint: leaving it behind
            # would put a multi-GB stale artifact in the checkpoint dir that resume never reads.
            if os.path.exists(embeddings_mmap_path):
                os.unlink(embeddings_mmap_path)

    # Landmark order is dispatch order, not month order, so restore the month ordering the
    # figure tier expects before writing.
    landmark_sizes.sort()
    failed_landmarks.sort()
    trajectory_predictions_df = trajectory_predictions_df.select(
        ['DFCI_MRN'] + [f'plus_{m}_months_data' for m in months_to_test]
    )


    _stage("writing trajectory scores ...")
    trajectory_predictions_df.write_csv(
        os.path.join(trajectory_path, f'survival_trajectories_w_decay_param_{decay_param}.csv'))

    # The at-risk count now declines for an explicit, auditable reason, so record
    # it next to the scores: any comparison of AUC across landmarks has to be read
    # against the denominator each landmark was computed on.
    pl.DataFrame(
        {
            'months': [m for m, _ in landmark_sizes],
            'n_at_risk': [n for _, n in landmark_sizes],
        }
    ).write_csv(
        os.path.join(trajectory_path, f'landmark_risk_sets_w_decay_param_{decay_param}.csv'))

    n_ok = len(months_to_test) - len(failed_landmarks)
    _resumed, _fit, _failed = ckpt.counts()
    _stage(f"wrote both CSVs to {trajectory_path}")
    _stage(f"[summary] {n_ok}/{len(months_to_test)} landmarks scored plus the month-0 baseline, "
           f"in {(time.time() - run_started) / 3600:.2f} h total.")
    _stage(f"[summary] this session: {_resumed} resumed from checkpoint, {_fit} fit, "
           f"{_failed} failed.")
    _stage(f"[summary] checkpoints + progress log under: "
           f"{os.path.join(trajectory_path, 'checkpoints')}")

    if failed_landmarks:
        failed_months = ", ".join(str(month) for month, _ in failed_landmarks)
        raise RuntimeError(
            f"Trajectory generation failed at month landmarks: {failed_months}. "
            "Completed landmarks are checkpointed — re-running resumes from them and refits "
            "only what is missing (set TRAJECTORY_RETRY_FAILED=1 to also retry the failures)."
        )


if __name__ == "__main__":
    main()
