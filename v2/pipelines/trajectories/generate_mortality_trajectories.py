"""Generate Mortality Trajectories script for model evaluation workflows."""

import io
import os
import time

import numpy as np
import polars as pl
import zstandard as zstd

from config import FEATURE_PATH, NOTES_PATH, SURV_PATH
from schemes import scheme_results_dir
from survival import (LandmarkCheckpoint, dataframe_fingerprint,
                      generate_survival_embedding_df, get_nested_heldout_risk_scores_CoxPH)
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


def _stage(msg: str) -> None:
    """One timestamped progress line, flushed so a piped reader sees it immediately."""
    if SHOW_PROGRESS:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    event = 'death'
    alphas_to_test = np.logspace(-5, 0, 25)
    l1_ratios = [0.5, 1.0]

    # regenerate predication dataframe at time 0 with new decay parameter
    decay_param = 0.1
    note_types = ['Clinician', 'Imaging', 'Pathology']
    _stage(f"pooling baseline embeddings (decay_param={decay_param}, window=0) ...")
    full_prediction_df = (generate_survival_embedding_df(notes_meta, events_data, embeddings_data,
                                                         note_types=note_types, pool_fx={key: 'time_decay_mean' for key in note_types},
                                                         decay_param=decay_param, max_note_window=0)
                              .join(cancer_type_df, on='DFCI_MRN'))
    numeric_cols = [c for c, dtype in full_prediction_df.schema.items() if dtype.is_numeric()]
    full_prediction_df = filter_finite_rows(full_prediction_df, numeric_cols).filter(pl.col(f'tt_{event}') > 0)
    _stage(f"  baseline cohort: {full_prediction_df.height:,} patients")

    # Define model columns
    base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
    events = [col.split('_', 1)[1] for col in full_prediction_df.columns if col.startswith('tt')]
    tt_events = [f"tt_{event}" for event in events]

    # Column groups
    embed_cols = [c for c in full_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
    continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols
    type_cols = [c for c in full_prediction_df.columns if c.startswith('CANCER_TYPE_')]

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

    # Month 0 (baseline) is checkpointed under the same store as the other landmarks.
    if ckpt.landmark_done(0):
        risk_map = ckpt.load_landmark(0)
    else:
        _stage(f"landmark 0/{len(months_to_test)} (month 0, baseline): "
               f"fitting nested CV on {full_prediction_df.height:,} patients ...")
        _t0 = time.time()
        risk_scores = get_nested_heldout_risk_scores_CoxPH(
            full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
            l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
            id_col='DFCI_MRN', max_iter=3000, show_progress=SHOW_PROGRESS,
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
    # No tqdm bar over this loop: it writes a carriage-return redraw to stderr, which interleaves
    # with the stdout stage lines below into unreadable output under a piped subprocess (the 04d
    # notebook driver), and it reports strictly less than the per-landmark lines already do.
    for landmark_i, month_adj in enumerate(months_to_test, start=1):
        landmark_started = time.time()
        landmark_day = month_adj * 30
        # At risk at the landmark: still under follow-up strictly beyond it.
        eligible_mrns = [
            mrn for mrn in cohort_mrns
            if baseline_tt.get(mrn) is not None and baseline_tt[mrn] > landmark_day
        ]
        landmark_sizes.append((month_adj, len(eligible_mrns)))

        # Resume: reload a completed landmark rather than refitting it. Done before the
        # embedding re-pool, which is itself expensive, so a resumed landmark costs a CSV read.
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
            _stage(f"landmark {landmark_i}/{len(months_to_test)} (month {month_adj}): "
                   f"skipping previously failed landmark ({reason}); "
                   "set TRAJECTORY_RETRY_FAILED=1 to refit")
            failed_landmarks.append((month_adj, f"previously failed: {reason}"))
            continue

        _stage(f"landmark {landmark_i}/{len(months_to_test)} (month {month_adj}): "
               f"{len(eligible_mrns):,} patients at risk")
        if not eligible_mrns:
            print(f"Trajectory landmark {month_adj} months: no patients at risk, skipping")
            ckpt.mark_failed(month_adj, "no patients at risk at landmark",
                             meta={'n_at_risk': 0})
            failed_landmarks.append((month_adj, "no patients at risk at landmark"))
            continue

        _stage(f"  pooling embeddings (window={landmark_day}d) ...")
        notes_meta_copy = notes_meta.filter(pl.col('DFCI_MRN').is_in(eligible_mrns))
        events_data_copy = events_data.filter(pl.col('DFCI_MRN').is_in(eligible_mrns))
        monthly_data = (generate_survival_embedding_df(notes_meta_copy, events_data_copy, embeddings_data, note_types=note_types,
                                                      pool_fx={key: 'time_decay_mean' for key in note_types}, decay_param=decay_param,
                                                      max_note_window=landmark_day)
                        .select(['DFCI_MRN', event, f'tt_{event}'] + base_vars + embed_cols)
                        .join(cancer_type_df, on='DFCI_MRN'))
        numeric_cols = [c for c, dtype in monthly_data.schema.items() if dtype.is_numeric()]
        # Every row already satisfies tt > 0 post-shift by construction of
        # `eligible_mrns`; this only drops rows with missing embeddings/covariates.
        monthly_data = filter_finite_rows(monthly_data, numeric_cols)
        _stage(f"  fitting nested CV on {monthly_data.height:,} complete-case patients ...")

        try:
            risk_scores = get_nested_heldout_risk_scores_CoxPH(
                monthly_data, base_vars + type_cols, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
                id_col='DFCI_MRN', max_iter=3000, show_progress=SHOW_PROGRESS,
            )
            ckpt.save_landmark(month_adj, risk_scores,
                               meta={'n_at_risk': len(eligible_mrns),
                                     'n_scored': int(risk_scores.height),
                                     'elapsed_s': round(time.time() - landmark_started, 1)})
            risk_map = dict(zip(risk_scores['DFCI_MRN'].to_list(), risk_scores['risk_score'].to_list()))
            trajectory_predictions_df = trajectory_predictions_df.with_columns(
                pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan).alias(f'plus_{month_adj}_months_data')
            )
            _stage(f"  month {month_adj} done in {(time.time() - landmark_started) / 60:.1f} min "
                   f"({landmark_i}/{len(months_to_test)} landmarks, "
                   f"{(time.time() - run_started) / 3600:.2f} h elapsed)")

        except Exception as exc:
            print(f"Trajectory landmark {month_adj} months failed: {exc}")
            ckpt.mark_failed(month_adj, str(exc),
                             meta={'n_at_risk': len(eligible_mrns),
                                   'elapsed_s': round(time.time() - landmark_started, 1)})
            _stage(f"  month {month_adj} FAILED after "
                   f"{(time.time() - landmark_started) / 60:.1f} min")
            failed_landmarks.append((month_adj, str(exc)))
            continue

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
