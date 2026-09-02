"""Within Treatment Vs Pan Treatment Models script for model evaluation workflows."""

# === Imports ===
import hashlib
import os
import time
import warnings

import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from tqdm import tqdm

from config import FEATURE_PATH, RESULTS_PATH
from schemes import load_embedding_prediction_df
from survival import (RunCheckpoint, dataframe_fingerprint,
                      fit_predict_external_CoxPH, get_nested_heldout_risk_scores_CoxPH,
                      run_grid_CoxPH_parallel)
from shared.polars_utils import filter_finite_rows


def main() -> None:
    # Silence joblib/loky's benign worker-respawn warning; it floods the Jupyter IOPub
    # channel during long runs and triggers "IOStream.flush timed out".
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")

    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    # === Minimum patient counts ===
    # Each within-treatment class is compared against a size-matched pan-treatment model (same
    # train N, that class's own patients excluded) rather than a single full-cohort pan model, so
    # the comparison isn't confounded by the pan arm simply having more training data.
    # MIN_STRATUM_N: a within-treatment model is only FIT for classes with at least this many patients in
    #                the embedding cohort. Smaller classes get no within-model and no per-class matched
    #                pan model, so they don't enter the within-vs-pan comparison.
    # MIN_TRAIN_N:   defensive floor on the actual train-split subset before fitting a within-model.
    # MIN_HELDOUT_N: a treatment class only enters the per-treatment held-out comparison (and the figure)
    #                if it has at least this many held-out patients — small n gives unstable AUC/C-index.
    MIN_STRATUM_N = 500
    MIN_TRAIN_N = 100
    MIN_HELDOUT_N = 30

    # === Load datasets ===
    time_decayed_events_df = load_embedding_prediction_df("icd3_post")

    # Load cancer types
    cancer_type_df = pl.read_csv(
        os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'),
        columns=['DFCI_MRN', 'CANCER_TYPE'])
    _emb_mrns = set(time_decayed_events_df['DFCI_MRN'].unique().to_list())
    cancer_type_sub = cancer_type_df.filter(pl.col('DFCI_MRN').is_in(_emb_mrns))

    # First-line treatment class per patient, derived from the same one-hot source Figure 1 uses
    # (categorical_treatment_data_by_line.csv.gz: DFCI_MRN, treatment_line, PX_on_<class> dummies).
    # Take the first line (treatment_line == 1) and assign each patient the single present class.
    # These dummies are NOT drop_first encoded (a PX_on_OTHER column exists), so a patient with any
    # first-line med has at least one 1; idxmax picks the first class when first line is combination.
    treatment_by_line = pl.read_csv(os.path.join(FEATURE_PATH, 'categorical_treatment_data_by_line.csv.gz'))
    tx_cols = [c for c in treatment_by_line.columns if c.startswith('PX_on_')]
    tx1 = treatment_by_line.filter(pl.col('treatment_line') == 1)
    tx1 = tx1.filter(pl.sum_horizontal(tx_cols) > 0)  # keep patients with a known first-line class
    tx_matrix = tx1.select(tx_cols).to_numpy()
    argmax_idx = np.argmax(tx_matrix, axis=1)
    tx_classes = [tx_cols[i].replace('PX_on_', '', 1) for i in argmax_idx]
    tx1 = tx1.with_columns(pl.Series('TREATMENT_CLASSIFICATION', tx_classes))
    treatment_df = tx1.select(['DFCI_MRN', 'TREATMENT_CLASSIFICATION']).unique(subset='DFCI_MRN')
    treatment_types = treatment_df['TREATMENT_CLASSIFICATION'].unique().to_list()

    # Merge embeddings + cancer types + events
    full_df = (time_decayed_events_df
               .join(treatment_df, on='DFCI_MRN', how='inner')
               .join(cancer_type_sub, on='DFCI_MRN', how='inner'))

    # === Feature definitions ===
    event = 'death'

    # Find all time-to-event columns
    events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
    tt_events = [f"tt_{e}" for e in events]

    # Embedding features
    embed_cols = [c for c in full_df.columns if ('EMBEDDING' in c or '2015' in c)]

    # === Train/held-out split (75% train, 25% held-out evaluation) ===
    held_mrns = set(full_df['DFCI_MRN'].sample(fraction=0.25, seed=1234).to_list())
    train_df = full_df.filter(~pl.col('DFCI_MRN').is_in(held_mrns))
    held_df  = full_df.filter( pl.col('DFCI_MRN').is_in(held_mrns))

    cancer_type_counts = train_df['CANCER_TYPE'].value_counts()
    types_to_keep = set(
        cancer_type_counts.filter(pl.col('count') >= 500)['CANCER_TYPE'].to_list()
    )
    def _encode_cancer_type(frame):
        return frame.with_columns(
            pl.when(pl.col('CANCER_TYPE').is_in(types_to_keep))
            .then(pl.col('CANCER_TYPE')).otherwise(pl.lit('OTHER')).alias('CANCER_TYPE')
        ).to_dummies(columns=['CANCER_TYPE'], drop_first=True)
    train_df = _encode_cancer_type(train_df)
    held_df = _encode_cancer_type(held_df)
    for c in set(train_df.columns) - set(held_df.columns):
        if c.startswith('CANCER_TYPE_'):
            held_df = held_df.with_columns(pl.lit(0).alias(c))
    for c in set(held_df.columns) - set(train_df.columns):
        if c.startswith('CANCER_TYPE_'):
            held_df = held_df.drop(c)

    base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [
        col for col in train_df.columns if col.startswith('CANCER_TYPE_')
    ]
    treatment_counts = train_df['TREATMENT_CLASSIFICATION'].value_counts()
    within_eligible = set(
        treatment_counts.filter(pl.col('count') >= MIN_STRATUM_N)['TREATMENT_CLASSIFICATION'].to_list()
    )

    # Ensure consistent column order
    held_df = held_df.select(train_df.columns)

    # === Identify feature columns ===
    continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

    # === Checkpoint store (writes intermediate results during the run; enables resume) ===
    # Resume by default. Set WITHIN_PAN_RESUME=0 for an intentional full recompute.
    RESUME = os.environ.get('WITHIN_PAN_RESUME', '1') not in {'0', 'false', 'False'}
    train_outdir = os.path.join(RESULTS_PATH, 'pan_vs_within_treatment')
    os.makedirs(train_outdir, exist_ok=True)

    # Fingerprint the deterministic inputs; a mismatch on a later run means the cohort/strata changed,
    # so the stored checkpoints are stale and are ignored (RunCheckpoint starts fresh).
    fingerprint = {
        'script': 'pan_vs_within_treatment',
        'n_train': int(len(train_df)),
        'n_held': int(len(held_df)),
        'n_embed': int(len(embed_cols)),
        'n_base': int(len(base_vars)),
        'strata': sorted(within_eligible),
        'seed': 1234,
        'data_hash': dataframe_fingerprint(
            train_df,
            ['DFCI_MRN', event, f'tt_{event}', 'TREATMENT_CLASSIFICATION']
            + base_vars + embed_cols,
        ),
    }
    ckpt = RunCheckpoint(os.path.join(train_outdir, 'checkpoints'), fingerprint, resume=RESUME)

    alphas_to_test = np.logspace(-5, 0, 25)
    l1_ratios = [0.5, 1.0]

    # No single full-cohort pan model is fit here: every pan comparator is a size-matched
    # per-stratum fit (see _fit_matched_pan below), so the pan arm never simply benefits from
    # more training data than the within arm it's compared against.

    def _provisional_cindex(held_within_df, matched_pan_held_df):
        """Reference-free per-stratum held-out C-index (size-matched pan vs within) for the
        progress manifest."""
        m = (held_within_df
             .join(matched_pan_held_df, on=['DFCI_MRN', 'STRATUM'])
             .join(full_df.select(['DFCI_MRN', f'tt_{event}', event]), on='DFCI_MRN'))
        cols = ['within_treatment_risk_score', 'pan_treatment_risk_score', f'tt_{event}', event]
        m = filter_finite_rows(m, cols)
        if len(m) < MIN_HELDOUT_N or m[event].sum() == 0:
            return None, None, len(m)
        eb = m[event].to_numpy().astype(bool)
        t = m[f'tt_{event}'].to_numpy()
        try:
            cp = concordance_index_censored(eb, t, m['pan_treatment_risk_score'].to_numpy())[0]
            cw = concordance_index_censored(eb, t, m['within_treatment_risk_score'].to_numpy())[0]
        except Exception:
            return None, None, len(m)
        return float(cp), float(cw), len(m)

    def _fit_matched_pan(treatment, n_match, seed):
        """Fit a pan model on a random subsample of train_df, excluding this treatment class's
        own patients and sized to match the within-class train N, so the pan-vs-within
        comparison isn't confounded by the pan arm simply having more training data.
        Checkpointed under a distinct per-class key via RunCheckpoint.save_stratum/
        load_stratum (RunCheckpoint's pan-specific save_pan/load_pan/pan_done slot is unused
        here — there is no single full-cohort pan model to cache)."""
        pan_pool = train_df.filter(pl.col('TREATMENT_CLASSIFICATION') != treatment)
        n_match = min(n_match, len(pan_pool))
        matched_pan_train = pan_pool.sample(n=n_match, seed=seed)

        _, matched_val, matched_model = run_grid_CoxPH_parallel(
            matched_pan_train, base_vars, continuous_vars, embed_cols,
            l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
        )
        if matched_model is None:
            return None, None, None

        _best = filter_finite_rows(matched_val, ['mean_auc(t)']).sort(
            'mean_auc(t)', descending=True
        ).row(0, named=True)
        matched_l1, matched_alpha = _best['l1_ratio'], _best['alpha']
        trained_matched_pan = (
            get_nested_heldout_risk_scores_CoxPH(
                matched_pan_train, base_vars, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
                max_iter=3000, backend="threading")
            .rename({'risk_score': 'pan_treatment_risk_score'})
        )
        trained_matched_pan = trained_matched_pan.with_columns(pl.lit(treatment).alias('STRATUM'))
        matched_pan_held = pl.DataFrame({
            'DFCI_MRN': held_df['DFCI_MRN'],
            'pan_treatment_risk_score': fit_predict_external_CoxPH(
                matched_pan_train, held_df, base_vars, continuous_vars, embed_cols,
                event_col=event, tstop_col=f'tt_{event}', l1_ratio=matched_l1,
                alpha=matched_alpha, max_iter=3000,
            ),
        }).with_columns(pl.lit(treatment).alias('STRATUM'))
        return trained_matched_pan, matched_pan_held, (float(matched_l1), float(matched_alpha))

    # === Train + score within-treatment models (single resumable pass) ===
    # Each stratum's train OOF scores AND held-out predictions are written to disk as it completes,
    # so a crashed/interrupted run reloads finished strata instead of refitting them. A size-matched
    # pan model (same train N as the class, that class's own patients excluded) is fit alongside
    # each within model so the held-out comparison isn't confounded by unequal training-set sizes.
    train_score_frames, held_score_frames = [], []
    matched_pan_train_frames, matched_pan_held_frames = [], []

    for treatment in tqdm(treatment_types, mininterval=30):
        if treatment not in within_eligible:
            continue

        matched_pan_key = f'{treatment}__matched_pan__'

        within_status = ckpt.status(treatment)
        matched_pan_status = ckpt.status(matched_pan_key)
        if within_status == 'skipped' or matched_pan_status == 'skipped':
            continue

        sub_df = train_df.filter(pl.col('TREATMENT_CLASSIFICATION') == treatment)
        if len(sub_df) < MIN_TRAIN_N:
            ckpt.mark_skipped(treatment, 'too_small', meta={'n_train': int(len(sub_df))})
            continue

        _t0 = time.time()
        if ckpt.stratum_done(treatment):
            trained_sub, held_sub = ckpt.load_stratum(treatment)
        else:
            if within_status == 'done':
                ckpt.log(f"incomplete within checkpoint for {treatment}; refitting within model")
            cur_test, cur_val, cur_model = run_grid_CoxPH_parallel(
                sub_df, base_vars, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
            )

            # Skip strata whose final model failed to converge (returned None).
            if cur_model is None:
                ckpt.mark_skipped(treatment, 'no_converge', meta={'n_train': int(len(sub_df))})
                continue

            _best = filter_finite_rows(cur_val, ['mean_auc(t)']).sort(
                'mean_auc(t)', descending=True
            ).row(0, named=True)
            best_l1, best_alpha = _best['l1_ratio'], _best['alpha']
            trained_sub = get_nested_heldout_risk_scores_CoxPH(
                sub_df, base_vars, continuous_vars, embed_cols,
                l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
                max_iter=3000, backend="threading"
            ).rename({'risk_score': 'within_treatment_risk_score'})
            trained_sub = trained_sub.with_columns(pl.lit(treatment).alias('STRATUM'))

            # Held-out predictions for this stratum, taken now while the model is in memory.
            sub_held = held_df.filter(pl.col('TREATMENT_CLASSIFICATION') == treatment)
            held_sub = pl.DataFrame({
                'DFCI_MRN': sub_held['DFCI_MRN'],
                'within_treatment_risk_score': fit_predict_external_CoxPH(
                    sub_df, sub_held, base_vars, continuous_vars, embed_cols,
                    event_col=event, tstop_col=f'tt_{event}', l1_ratio=best_l1,
                    alpha=best_alpha, max_iter=3000,
                ),
            }).with_columns(pl.lit(treatment).alias('STRATUM'))

            # Persist the within half before starting the matched-pan fit. If the process is
            # interrupted below, the next run reloads this work and starts at matched-pan.
            ckpt.save_stratum(treatment, trained_sub, held_sub,
                              meta={'n_train': int(len(sub_df)), 'n_held': int(len(held_sub)),
                                    'l1': float(best_l1), 'alpha': float(best_alpha),
                                    'elapsed_s': round(time.time() - _t0, 1)})

        # Size-matched pan comparator: same train N as this class, class's own patients
        # excluded. Seeded off the base seed + class name so it's reproducible per-class.
        # Tagged with the same STRATUM so downstream merges key on (DFCI_MRN, STRATUM) — a
        # patient can appear in more than one class's matched-pan pool (only their own class
        # is excluded), so DFCI_MRN alone is not unique across matched_pan_*_all.
        matched_seed = 1234 + int(hashlib.md5(treatment.encode('utf-8')).hexdigest()[:8], 16) % 10_000
        if ckpt.stratum_done(matched_pan_key):
            trained_matched_pan, matched_pan_held = ckpt.load_stratum(matched_pan_key)
        else:
            if matched_pan_status == 'done':
                ckpt.log(f"incomplete matched-pan checkpoint for {treatment}; refitting matched-pan model")
            trained_matched_pan, matched_pan_held, matched_hp = _fit_matched_pan(
                treatment, n_match=len(sub_df), seed=matched_seed
            )
            if trained_matched_pan is None:
                ckpt.mark_skipped(matched_pan_key, 'no_converge',
                                  meta={'n_train': int(len(sub_df))})
                continue
            c_pan, c_within, n_held = _provisional_cindex(held_sub, matched_pan_held)
            ckpt.save_stratum(matched_pan_key, trained_matched_pan, matched_pan_held,
                              meta={'n_train': int(len(sub_df)), 'n_held': int(n_held),
                                    'l1': matched_hp[0], 'alpha': matched_hp[1],
                                    'c_pan': c_pan, 'c_within': c_within,
                                    'elapsed_s': round(time.time() - _t0, 1)})
        # Complete the within row's comparison metadata after both independent stages exist.
        # This also repairs the small crash window after matched-pan was saved but before the
        # parent row could be finalized.
        if ckpt.metadata(treatment).get('c_pan') is None:
            c_pan, c_within, n_held = _provisional_cindex(held_sub, matched_pan_held)
            ckpt.update_stratum_meta(treatment, {
                'n_held': int(n_held), 'c_pan': c_pan, 'c_within': c_within,
            })
        train_score_frames.append(trained_sub)
        held_score_frames.append(held_sub)
        matched_pan_train_frames.append(trained_matched_pan)
        matched_pan_held_frames.append(matched_pan_held)

    if not train_score_frames:
        raise RuntimeError("No within-treatment stratum completed successfully.")
    trained_within = pl.concat(train_score_frames, how='diagonal_relaxed')
    within_held_all = pl.concat(held_score_frames, how='diagonal_relaxed')
    # Size-matched pan scores, one row per (stratum, patient) — a patient can appear in more
    # than one class's matched-pan pool (only their own class is excluded from that pool), so
    # these are merged against the within frames on (DFCI_MRN, STRATUM), not DFCI_MRN alone.
    matched_pan_train_all = pl.concat(matched_pan_train_frames, how='diagonal_relaxed')
    matched_pan_held_all = pl.concat(matched_pan_held_frames, how='diagonal_relaxed')

    # === Evaluate on Training Set ===
    complete_train = trained_within.join(
        matched_pan_train_all.select(['DFCI_MRN', 'STRATUM', 'pan_treatment_risk_score']), on=['DFCI_MRN', 'STRATUM']
    ).join(
        full_df.select(['DFCI_MRN', 'TREATMENT_CLASSIFICATION', f'tt_{event}', event]), on='DFCI_MRN'
    )

    # Some within-treatment strata diverge during Cox fitting and emit non-finite OOF
    # risk scores; drop those rows so the pan-vs-within comparison runs on the same set
    # of patients with finite predictions from both models. Also guard the time/event
    # columns: concordance_index_censored / cumulative_dynamic_auc reject any non-finite
    # input, so every metric call below sees only finite values.
    _score_cols = ['pan_treatment_risk_score', 'within_treatment_risk_score']
    _finite_cols = _score_cols + [f'tt_{event}', event]
    _n0 = len(complete_train)
    complete_train = filter_finite_rows(complete_train, _finite_cols)
    if len(complete_train) < _n0:
        print(f"Train: dropped {_n0 - len(complete_train)} rows with non-finite risk scores / outcomes")

    times = complete_train[f'tt_{event}'].to_numpy()
    events_bool = complete_train[event].to_numpy().astype(bool)

    c_pan_train = concordance_index_censored(events_bool, times, complete_train['pan_treatment_risk_score'].to_numpy())[0]
    c_within_train = concordance_index_censored(events_bool, times, complete_train['within_treatment_risk_score'].to_numpy())[0]

    print(f"\nTrain set: Pan-cancer C-index = {c_pan_train:.3f}, Within-cancer C-index = {c_within_train:.3f}")

    # === Held-out Evaluation (assemble per-stratum within scores + size-matched pan scores) ===
    held_scores = within_held_all.join(
        matched_pan_held_all.select(['DFCI_MRN', 'STRATUM', 'pan_treatment_risk_score']), on=['DFCI_MRN', 'STRATUM']
    ).join(
        full_df.select(['DFCI_MRN', 'TREATMENT_CLASSIFICATION', f'tt_{event}', event]),
        on='DFCI_MRN', how='left'
    )

    # Drop non-finite held-out risk scores / outcomes (divergent within-treatment strata).
    _n0 = len(held_scores)
    held_scores = filter_finite_rows(held_scores, _finite_cols)
    if len(held_scores) < _n0:
        print(f"Held-out: dropped {_n0 - len(held_scores)} rows with non-finite risk scores / outcomes")

    # === Compute held-out concordance ===
    times = held_scores[f'tt_{event}'].to_numpy()
    events_bool = held_scores[event].to_numpy().astype(bool)

    c_pan_held = concordance_index_censored(events_bool, times, held_scores['pan_treatment_risk_score'].to_numpy())[0]
    c_within_held = concordance_index_censored(events_bool, times, held_scores['within_treatment_risk_score'].to_numpy())[0]

    print(f"Held-out set: Pan-treatment C-index = {c_pan_held:.3f}, Within-treatment C-index = {c_within_held:.3f}")

    # === Mean time-dependent AUC (project-standard metric) ===
    # IPCW reference + eval-time grid from the TRAINING set, matching evaluate_surv_model
    # (5th–95th percentile, 50 points). Per-stratum eval times are clipped to that stratum's
    # held-out follow-up so cumulative_dynamic_auc never sees out-of-range times.
    y_train_global = Surv.from_arrays(complete_train[event].to_numpy().astype(bool), complete_train[f'tt_{event}'].to_numpy())
    _lo, _hi = np.percentile(complete_train[f'tt_{event}'].to_numpy(), [5, 95])
    base_eval_times = np.linspace(_lo, _hi, 50)

    def _mean_auc(sub_df, risk_col):
        et = base_eval_times[(base_eval_times > sub_df[f'tt_{event}'].min())
                             & (base_eval_times < sub_df[f'tt_{event}'].max())]
        if len(et) == 0:
            return np.nan
        try:
            y_test = Surv.from_arrays(sub_df[event].to_numpy().astype(bool), sub_df[f'tt_{event}'].to_numpy())
            return cumulative_dynamic_auc(y_train_global, y_test, sub_df[risk_col].to_numpy(), et)[1]
        except Exception:
            return np.nan

    auc_pan_held = _mean_auc(held_scores, 'pan_treatment_risk_score')
    auc_within_held = _mean_auc(held_scores, 'within_treatment_risk_score')
    print(f"Held-out set: Pan-treatment mean AUC(t) = {auc_pan_held:.3f}, "
          f"Within-treatment mean AUC(t) = {auc_within_held:.3f}")

    # === Per-Treatment Comparison (Held-out) ===
    cindex_by_treatment = []
    for treatment in tqdm(sorted(held_scores['TREATMENT_CLASSIFICATION'].drop_nulls().unique().to_list()), mininterval=30):
        sub_df = held_scores.filter(pl.col('TREATMENT_CLASSIFICATION') == treatment)
        if sub_df.shape[0] < MIN_HELDOUT_N:
            continue

        times = sub_df[f'tt_{event}'].to_numpy()
        events_bool = sub_df[event].to_numpy().astype(bool)

        c_pan = concordance_index_censored(events_bool, times, sub_df['pan_treatment_risk_score'].to_numpy())[0]
        c_within = concordance_index_censored(events_bool, times, sub_df['within_treatment_risk_score'].to_numpy())[0]
        auc_pan = _mean_auc(sub_df, 'pan_treatment_risk_score')
        auc_within = _mean_auc(sub_df, 'within_treatment_risk_score')

        cindex_by_treatment.append({
            'TREATMENT': treatment,
            'CINDEX_PAN': c_pan,
            'CINDEX_WITHIN': c_within,
            'DELTA_WITHIN_MINUS_PAN': c_within - c_pan,
            'AUC_PAN': auc_pan,
            'AUC_WITHIN': auc_within,
            'DELTA_AUC_WITHIN_MINUS_PAN': auc_within - auc_pan,
            'N_HELDOUT': sub_df.shape[0]
        })

    metrics_df = pl.DataFrame(cindex_by_treatment).sort('DELTA_AUC_WITHIN_MINUS_PAN', descending=True)

    # Prepend an "Overall" row so the figure has a reference line / summary.
    overall_row = pl.DataFrame([{
        'TREATMENT': 'Overall',
        'CINDEX_PAN': c_pan_held,
        'CINDEX_WITHIN': c_within_held,
        'DELTA_WITHIN_MINUS_PAN': c_within_held - c_pan_held,
        'AUC_PAN': auc_pan_held,
        'AUC_WITHIN': auc_within_held,
        'DELTA_AUC_WITHIN_MINUS_PAN': auc_within_held - auc_pan_held,
        'N_HELDOUT': len(held_scores),
    }], schema=metrics_df.schema)
    metrics_df = pl.concat([overall_row, metrics_df])

    # === Save Results === (train_outdir / checkpoints dir were created at the checkpoint-store setup)
    complete_train.write_csv(os.path.join(train_outdir, 'train_risk_scores.csv'))
    held_scores.write_csv(os.path.join(train_outdir, 'held_out_risk_scores.csv'))
    metrics_df.write_csv(os.path.join(train_outdir, 'metrics_by_treatment.csv'))

    print("\n=== Per-Treatment Results (Held-out) ===")
    print(metrics_df)
    print(f"\nSaved per-treatment metrics to: {os.path.join(train_outdir, 'metrics_by_treatment.csv')}")

    n_strata = int((metrics_df['TREATMENT'] != 'Overall').sum())
    _resumed, _fit, _skipped = ckpt.counts()
    print(f"\n[summary] within-treatment models: {_resumed} resumed, {_fit} fit, {_skipped} skipped; "
          f"{n_strata} treatment strata in the comparison across {len(held_scores)} held-out patients "
          f"(strata floored at n>={MIN_HELDOUT_N} held-out patients).")
    print(f"[summary] intermediate checkpoints + progress log under: {os.path.join(train_outdir, 'checkpoints')}")


if __name__ == "__main__":
    main()
