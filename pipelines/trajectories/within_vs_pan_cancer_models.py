"""Within Vs Pan Cancer Models script for model evaluation workflows."""

# === Imports ===
import concurrent.futures
import multiprocessing
import os
import warnings

import numpy as np
import polars as pl
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from tqdm import tqdm

from config import FEATURE_PATH, RESULTS_PATH
from schemes import load_embedding_prediction_df
from survival import RunCheckpoint, dataframe_fingerprint
from pipelines.trajectories.within_vs_pan_worker import fit_stratum
from shared.parallel_utils import resolve_workers, set_single_thread_env
from shared.polars_utils import filter_finite_rows

# Per-worker RAM budget for auto-sizing the stratum pool. A worker holds one stratum's train
# subset plus a matched-pan subsample of the same size, both dense embedding frames.
WORKER_MEM_GB = float(os.environ.get("WITHIN_PAN_WORKER_MEM_GB", "8"))


def main() -> None:
    # Silence joblib/loky's benign worker-respawn warning; it floods the Jupyter IOPub
    # channel during long runs and triggers "IOStream.flush timed out".
    warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")

    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    # === Minimum patient counts ===
    # MIN_STRATUM_N: rare cancer types below this are collapsed into 'OTHER' before modeling.
    # MIN_TRAIN_N:   a within-cancer model is only fit for strata with at least this many train patients.
    # MIN_HELDOUT_N: a stratum only enters the per-type held-out comparison (and the figure) if it
    #                has at least this many held-out patients — small n gives unstable AUC/C-index.
    MIN_STRATUM_N = 500
    MIN_TRAIN_N = 100
    MIN_HELDOUT_N = 30

    # === Load datasets ===
    cancer_type_df = pl.read_csv(
        os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'),
        columns=['DFCI_MRN', 'CANCER_TYPE'],
    )

    time_decayed_events_df = load_embedding_prediction_df("icd3_post")

    # Merge embeddings + cancer types + events
    full_df = (time_decayed_events_df
               .join(cancer_type_df, on='DFCI_MRN', how='inner')
               .drop_nulls(subset=['CANCER_TYPE']))

    # === Feature definitions ===
    base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
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

    # Learn the rarity map on training data only, then apply it unchanged to
    # held-out patients.  Held-out category frequencies cannot shape features.
    cancer_type_counts = train_df['CANCER_TYPE'].value_counts()
    types_to_keep = set(
        cancer_type_counts.filter(pl.col('count') >= MIN_STRATUM_N)['CANCER_TYPE'].to_list()
    )
    def _collapse_types(frame):
        return frame.with_columns(
            pl.when(pl.col('CANCER_TYPE').is_in(types_to_keep))
            .then(pl.col('CANCER_TYPE'))
            .otherwise(pl.lit('OTHER'))
            .alias('CANCER_TYPE')
        ).with_columns(pl.col('CANCER_TYPE').alias('STRATUM_CANCER_TYPE'))
    train_df = _collapse_types(train_df)
    held_df = _collapse_types(held_df)

    # === One-hot encode cancer type ===
    train_df = train_df.to_dummies(columns=['CANCER_TYPE'], drop_first=True)
    held_df  = held_df.to_dummies(columns=['CANCER_TYPE'], drop_first=True)

    # Align dummy columns across splits
    for c in set(train_df.columns) - set(held_df.columns):
        if c.startswith('CANCER_TYPE_'):
            held_df = held_df.with_columns(pl.lit(0).alias(c))
    for c in set(held_df.columns) - set(train_df.columns):
        if c.startswith('CANCER_TYPE_'):
            held_df = held_df.drop(c)

    # Ensure consistent column order
    held_df = held_df.select(train_df.columns)

    # === Dummy consistency checks ===
    train_types = [c for c in train_df.columns if c.startswith('CANCER_TYPE_')]
    held_types = [c for c in held_df.columns if c.startswith('CANCER_TYPE_')]

    missing_in_held = set(train_types) - set(held_types)
    missing_in_train = set(held_types) - set(train_types)

    print(f"\n=== Dummy Variable Consistency Check ===")
    print(f"Train dummy columns: {sorted(train_types)}")
    print(f"Held dummy columns:  {sorted(held_types)}")
    print(f"Missing in held: {missing_in_held}")
    print(f"Missing in train: {missing_in_train}")
    print(f"Column alignment verified: {set(train_types) == set(held_types)}")

    # === Identify feature columns ===
    type_cols = train_types
    continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

    # === Checkpoint store (writes intermediate results during the run; enables resume) ===
    # Resume by default. Set WITHIN_PAN_RESUME=0 for an intentional full recompute.
    RESUME = os.environ.get('WITHIN_PAN_RESUME', '1') not in {'0', 'false', 'False'}
    train_outdir = os.path.join(RESULTS_PATH, 'pan_vs_within_cancer')
    os.makedirs(train_outdir, exist_ok=True)

    # Fingerprint the deterministic inputs; a mismatch on a later run means the cohort/strata changed,
    # so the stored checkpoints are stale and are ignored (RunCheckpoint starts fresh).
    fingerprint = {
        'script': 'pan_vs_within_cancer',
        'n_train': int(len(train_df)),
        'n_held': int(len(held_df)),
        'n_embed': int(len(embed_cols)),
        'n_base': int(len(base_vars)),
        'n_type': int(len(type_cols)),
        'strata': sorted(train_df['STRATUM_CANCER_TYPE'].drop_nulls().unique().to_list()),
        'seed': 1234,
        'data_hash': dataframe_fingerprint(
            train_df,
            ['DFCI_MRN', event, f'tt_{event}', 'STRATUM_CANCER_TYPE']
            + base_vars + type_cols + embed_cols,
        ),
    }
    ckpt = RunCheckpoint(os.path.join(train_outdir, 'checkpoints'), fingerprint, resume=RESUME)

    alphas_to_test = np.logspace(-5, 0, 25)
    l1_ratios = [0.5, 1.0]

    # No single full-cohort pan model is fit here: every pan comparator is a size-matched
    # per-stratum fit (see fit_stratum in within_vs_pan_worker), so the pan arm never simply
    # benefits from more training data than the within arm it's compared against.

    def _provisional_cindex(held_within_df, matched_pan_held_df):
        """Reference-free per-stratum held-out C-index (size-matched pan vs within) for the
        progress manifest."""
        m = (held_within_df
             .join(matched_pan_held_df, on=['DFCI_MRN', 'STRATUM'])
             .join(full_df.select(['DFCI_MRN', f'tt_{event}', event]), on='DFCI_MRN'))
        cols = ['within_cancer_risk_score', 'pan_cancer_risk_score', f'tt_{event}', event]
        m = filter_finite_rows(m, cols)
        if len(m) < MIN_HELDOUT_N or m[event].sum() == 0:
            return None, None, len(m)
        eb = m[event].cast(pl.Boolean).to_numpy()
        t = m[f'tt_{event}'].to_numpy()
        try:
            cp = concordance_index_censored(eb, t, m['pan_cancer_risk_score'].to_numpy())[0]
            cw = concordance_index_censored(eb, t, m['within_cancer_risk_score'].to_numpy())[0]
        except Exception:
            return None, None, len(m)
        return float(cp), float(cw), len(m)

    # === Train + score within-cancer models (single resumable pass) ===
    # Each stratum's train OOF scores AND held-out predictions are written to disk as it completes,
    # so a crashed/interrupted run reloads finished strata instead of refitting them. A size-matched
    # pan model (same train N as the stratum, stratum's own patients excluded) is fit alongside each
    # within model so the held-out comparison isn't confounded by unequal training-set sizes.
    #
    # Strata are fit in a process pool. They are mutually independent -- each within model sees
    # only its own stratum, and each matched-pan model samples from everything except that
    # stratum -- so the only ordering that ever mattered was the checkpoint write, which stays
    # in this process (see below). The previous serial loop left the machine almost idle: the
    # only fan-out was joblib over 5 CV folds, and with l1_ratios of length 2 the grid's
    # parallel_axis="auto" resolves to "fold", so nothing above that inner loop overlapped.
    train_score_frames, held_score_frames = [], []
    matched_pan_train_frames, matched_pan_held_frames = [], []

    spec = {
        'stratum_col': 'STRATUM_CANCER_TYPE',
        'within_score_col': 'within_cancer_risk_score',
        'pan_score_col': 'pan_cancer_risk_score',
        'event': event,
        # The within model is stratified by construction, so cancer-type dummies would be
        # constant within a stratum; only the matched-pan arm gets type_cols. This asymmetry
        # is carried over unchanged from the serial version.
        'base_cols': base_vars,
        'pan_base_cols': base_vars + type_cols,
        'continuous_vars': continuous_vars,
        'embed_cols': embed_cols,
        'l1_ratios': l1_ratios,
        'alphas_to_test': alphas_to_test,
        'max_iter': 3000,
    }

    cancer_strata = sorted(train_df['STRATUM_CANCER_TYPE'].drop_nulls().unique().to_list())

    # Decide what each stratum still needs BEFORE dispatching, so a resumed run ships only the
    # missing halves to the pool and reloads the rest from disk here.
    pending, reloaded = [], []
    for cancer_type in cancer_strata:
        matched_pan_key = f'{cancer_type}__matched_pan__'
        within_status = ckpt.status(cancer_type)
        matched_pan_status = ckpt.status(matched_pan_key)
        if within_status == 'skipped' or matched_pan_status == 'skipped':
            continue

        n_sub = int(train_df.filter(pl.col('STRATUM_CANCER_TYPE') == cancer_type).height)
        if n_sub < MIN_TRAIN_N:
            ckpt.mark_skipped(cancer_type, 'too_small', meta={'n_train': n_sub})
            continue

        need_within = not ckpt.stratum_done(cancer_type)
        need_matched_pan = not ckpt.stratum_done(matched_pan_key)
        if need_within and within_status == 'done':
            ckpt.log(f"incomplete within checkpoint for {cancer_type}; refitting within model")
        if need_matched_pan and matched_pan_status == 'done':
            ckpt.log(f"incomplete matched-pan checkpoint for {cancer_type}; "
                     "refitting matched-pan model")

        if need_within or need_matched_pan:
            pending.append((cancer_type, need_within, need_matched_pan, n_sub))
        else:
            reloaded.append(cancer_type)

    # Fully checkpointed strata never reach the pool.
    for cancer_type in reloaded:
        trained_sub, held_sub = ckpt.load_stratum(cancer_type)
        trained_matched_pan, matched_pan_held = ckpt.load_stratum(f'{cancer_type}__matched_pan__')
        train_score_frames.append(trained_sub)
        held_score_frames.append(held_sub)
        matched_pan_train_frames.append(trained_matched_pan)
        matched_pan_held_frames.append(matched_pan_held)

    # Largest strata first: a stratum's cost scales with its train N, so starting the longest
    # tasks first keeps the pool from ending with one big fit running alone.
    pending.sort(key=lambda task: task[3], reverse=True)

    def _apply_result(result):
        """Checkpoint one worker's stratum and fold it into the accumulators.

        Runs in the parent only. RunCheckpoint holds in-memory manifest state and appends to a
        single CSV, so concurrent writers would interleave rows and diverge; keeping every
        save/skip call here preserves the serial version's on-disk semantics exactly.
        """
        cancer_type = result['stratum']
        matched_pan_key = f'{cancer_type}__matched_pan__'

        if result['error'] is not None:
            ckpt.mark_skipped(cancer_type, 'error')
            ckpt.log(f"stratum {cancer_type} failed: {result['error']}")
            return
        if result['within_skip'] is not None:
            reason, n_train = result['within_skip']
            ckpt.mark_skipped(cancer_type, reason, meta={'n_train': n_train})
            return
        if result['matched_pan_skip'] is not None:
            reason, n_train = result['matched_pan_skip']
            ckpt.mark_skipped(matched_pan_key, reason, meta={'n_train': n_train})
            return

        if result['within'] is not None:
            trained_sub = result['within']['train']
            held_sub = result['within']['held']
            ckpt.save_stratum(cancer_type, trained_sub, held_sub, meta=result['within']['meta'])
        else:
            trained_sub, held_sub = ckpt.load_stratum(cancer_type)

        if result['matched_pan'] is not None:
            trained_matched_pan = result['matched_pan']['train']
            matched_pan_held = result['matched_pan']['held']
            c_pan, c_within, n_held = _provisional_cindex(held_sub, matched_pan_held)
            meta = dict(result['matched_pan']['meta'])
            meta.update({'n_held': int(n_held), 'c_pan': c_pan, 'c_within': c_within})
            ckpt.save_stratum(matched_pan_key, trained_matched_pan, matched_pan_held, meta=meta)
        else:
            trained_matched_pan, matched_pan_held = ckpt.load_stratum(matched_pan_key)

        # Complete the within row's comparison metadata once both halves exist. Also repairs the
        # crash window where matched-pan was saved but the parent row was never finalized.
        if ckpt.metadata(cancer_type).get('c_pan') is None:
            c_pan, c_within, n_held = _provisional_cindex(held_sub, matched_pan_held)
            ckpt.update_stratum_meta(cancer_type, {
                'n_held': int(n_held), 'c_pan': c_pan, 'c_within': c_within,
            })

        train_score_frames.append(trained_sub)
        held_score_frames.append(held_sub)
        matched_pan_train_frames.append(trained_matched_pan)
        matched_pan_held_frames.append(matched_pan_held)

    if pending:
        n_workers = resolve_workers(len(pending), env_var='WITHIN_PAN_WORKERS',
                                    worker_mem_gb=WORKER_MEM_GB)
        print(f"fitting {len(pending)} stratum/strata across {n_workers} worker(s), "
              f"1 thread each ({len(reloaded)} reloaded from checkpoints)", flush=True)

        if n_workers == 1:
            # Explicit serial path, no pool: the fallback if a worker is OOM-killed.
            for cancer_type, need_within, need_matched_pan, _n in tqdm(pending, mininterval=30):
                _apply_result(fit_stratum(cancer_type, spec, train_df, held_df,
                                          need_within, need_matched_pan))
        else:
            # Set BEFORE the pool exists: a spawned child builds its polars/Rayon and BLAS pools
            # while importing them, which happens before any initializer we could pass runs.
            # Without this each of N workers sizes those pools to the whole machine and the run
            # dies on RLIMIT_NPROC rather than merely thrashing.
            set_single_thread_env()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
            ) as pool:
                in_flight = {
                    pool.submit(fit_stratum, cancer_type, spec, train_df, held_df,
                                need_within, need_matched_pan): cancer_type
                    for cancer_type, need_within, need_matched_pan, _n in pending
                }
                n_done = 0
                try:
                    for future in concurrent.futures.as_completed(in_flight):
                        _apply_result(future.result())
                        n_done += 1
                        print(f"  [{n_done}/{len(pending)} strata complete]", flush=True)
                except KeyboardInterrupt:
                    # Without this, pool shutdown blocks on every queued task. Strata already
                    # applied above are checkpointed, so a resume picks up from them.
                    for future in in_flight:
                        future.cancel()
                    print("\nInterrupted — cancelled queued strata, waiting for in-flight ones.")
                    raise

    if not train_score_frames:
        raise RuntimeError("No within-cancer stratum completed successfully.")
    trained_within = pl.concat(train_score_frames, how='diagonal_relaxed')
    within_held_all = pl.concat(held_score_frames, how='diagonal_relaxed')
    # Size-matched pan scores, one row per (stratum, patient) — a patient can appear in more
    # than one stratum's matched-pan pool (only their own stratum is excluded from that pool),
    # so these are merged against the within frames on (DFCI_MRN, STRATUM), not DFCI_MRN alone.
    matched_pan_train_all = pl.concat(matched_pan_train_frames, how='diagonal_relaxed')
    matched_pan_held_all = pl.concat(matched_pan_held_frames, how='diagonal_relaxed')

    # === Evaluate on Training Set ===
    complete_train = trained_within.join(
        matched_pan_train_all.select(['DFCI_MRN', 'STRATUM', 'pan_cancer_risk_score']), on=['DFCI_MRN', 'STRATUM']
    ).join(
        full_df.select(['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]), on='DFCI_MRN'
    )

    # Some within-cancer strata diverge during Cox fitting and emit non-finite OOF risk
    # scores; drop those rows so the pan-vs-within comparison runs on the same set of
    # patients with finite predictions from both models. Also guard the time/event columns:
    # concordance_index_censored / cumulative_dynamic_auc reject any non-finite input, so
    # every metric call below sees only finite values.
    _score_cols = ['pan_cancer_risk_score', 'within_cancer_risk_score']
    _finite_cols = _score_cols + [f'tt_{event}', event]
    _n0 = len(complete_train)
    complete_train = filter_finite_rows(complete_train, _finite_cols)
    if len(complete_train) < _n0:
        print(f"Train: dropped {_n0 - len(complete_train)} rows with non-finite risk scores / outcomes")

    times = complete_train[f'tt_{event}'].to_numpy()
    events_bool = complete_train[event].cast(pl.Boolean).to_numpy()

    c_pan_train = concordance_index_censored(events_bool, times, complete_train['pan_cancer_risk_score'].to_numpy())[0]
    c_within_train = concordance_index_censored(events_bool, times, complete_train['within_cancer_risk_score'].to_numpy())[0]

    print(f"\nTrain set: Pan-cancer C-index = {c_pan_train:.3f}, Within-cancer C-index = {c_within_train:.3f}")

    # === Held-out Evaluation (assemble per-stratum within scores + size-matched pan scores) ===
    held_scores = within_held_all.join(
        matched_pan_held_all.select(['DFCI_MRN', 'STRATUM', 'pan_cancer_risk_score']), on=['DFCI_MRN', 'STRATUM']
    ).join(
        full_df.select(['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]),
        on='DFCI_MRN', how='left'
    )

    # Drop non-finite held-out risk scores / outcomes (divergent within-cancer strata).
    _n0 = len(held_scores)
    held_scores = filter_finite_rows(held_scores, _finite_cols)
    if len(held_scores) < _n0:
        print(f"Held-out: dropped {_n0 - len(held_scores)} rows with non-finite risk scores / outcomes")

    # === Merge consistency checks ===
    print("\n=== Held-out Merge Consistency Check ===")
    print(f"Total predictions: {len(within_held_all)}")
    print(f"Matched to held_df metadata: {held_scores['CANCER_TYPE'].is_not_null().sum()}")
    missing = held_scores.filter(pl.col(f'tt_{event}').is_null())
    if len(missing) > 0:
        print(f"⚠️ {len(missing)} held-out predictions could not be matched to full_df!")
    else:
        print("✅ All held-out predictions successfully matched to metadata.")

    dup_counts = held_scores['DFCI_MRN'].value_counts()
    if (dup_counts['count'] > 1).any():
        print(f"⚠️ {(dup_counts['count'] > 1).sum()} MRNs appear multiple times in held_scores!")

    # === Compute held-out concordance ===
    times = held_scores[f'tt_{event}'].to_numpy()
    events_bool = held_scores[event].cast(pl.Boolean).to_numpy()

    c_pan_held = concordance_index_censored(events_bool, times, held_scores['pan_cancer_risk_score'].to_numpy())[0]
    c_within_held = concordance_index_censored(events_bool, times, held_scores['within_cancer_risk_score'].to_numpy())[0]

    print(f"Held-out set: Pan-cancer C-index = {c_pan_held:.3f}, Within-cancer C-index = {c_within_held:.3f}")

    # === Mean time-dependent AUC (project-standard metric) ===
    # IPCW reference + eval-time grid from the TRAINING set, matching evaluate_surv_model
    # (5th–95th percentile, 50 points). Per-stratum eval times are clipped to that stratum's
    # held-out follow-up so cumulative_dynamic_auc never sees out-of-range times.
    y_train_global = Surv.from_arrays(complete_train[event].cast(pl.Boolean).to_numpy(), complete_train[f'tt_{event}'].to_numpy())
    _lo, _hi = np.percentile(complete_train[f'tt_{event}'].to_numpy(), [5, 95])
    base_eval_times = np.linspace(_lo, _hi, 50)

    def _mean_auc(sub_df, risk_col):
        tt = sub_df[f'tt_{event}'].to_numpy()
        et = base_eval_times[(base_eval_times > tt.min())
                             & (base_eval_times < tt.max())]
        if len(et) == 0:
            return np.nan
        try:
            y_test = Surv.from_arrays(sub_df[event].cast(pl.Boolean).to_numpy(), tt)
            return cumulative_dynamic_auc(y_train_global, y_test, sub_df[risk_col].to_numpy(), et)[1]
        except Exception:
            return np.nan

    auc_pan_held = _mean_auc(held_scores, 'pan_cancer_risk_score')
    auc_within_held = _mean_auc(held_scores, 'within_cancer_risk_score')
    print(f"Held-out set: Pan-cancer mean AUC(t) = {auc_pan_held:.3f}, "
          f"Within-cancer mean AUC(t) = {auc_within_held:.3f}")

    # === Per-Cancer-Type Comparison (Held-out) ===
    cindex_by_type = []
    for cancer_type in tqdm(sorted(held_scores['CANCER_TYPE'].drop_nulls().unique().to_list()), mininterval=30):
        sub_df = held_scores.filter(pl.col('CANCER_TYPE') == cancer_type)
        if sub_df.shape[0] < MIN_HELDOUT_N:
            continue

        times = sub_df[f'tt_{event}'].to_numpy()
        events_bool = sub_df[event].cast(pl.Boolean).to_numpy()

        c_pan = concordance_index_censored(events_bool, times, sub_df['pan_cancer_risk_score'].to_numpy())[0]
        c_within = concordance_index_censored(events_bool, times, sub_df['within_cancer_risk_score'].to_numpy())[0]
        auc_pan = _mean_auc(sub_df, 'pan_cancer_risk_score')
        auc_within = _mean_auc(sub_df, 'within_cancer_risk_score')

        cindex_by_type.append({
            'CANCER_TYPE': cancer_type,
            'CINDEX_PAN': c_pan,
            'CINDEX_WITHIN': c_within,
            'DELTA_WITHIN_MINUS_PAN': c_within - c_pan,
            'AUC_PAN': auc_pan,
            'AUC_WITHIN': auc_within,
            'DELTA_AUC_WITHIN_MINUS_PAN': auc_within - auc_pan,
            'N_HELDOUT': sub_df.shape[0]
        })

    metrics_df = pl.DataFrame(cindex_by_type).sort('DELTA_AUC_WITHIN_MINUS_PAN', descending=True)

    # Prepend an "Overall" row so the figure has a reference line / summary.
    overall_row = pl.DataFrame([{
        'CANCER_TYPE': 'Overall',
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
    metrics_df.write_csv(os.path.join(train_outdir, 'metrics_by_cancer_type.csv'))

    print("\n=== Per-Cancer-Type Results (Held-out) ===")
    print(metrics_df)
    print(f"\nSaved per-cancer-type metrics to: {os.path.join(train_outdir, 'metrics_by_cancer_type.csv')}")

    n_strata = int((metrics_df['CANCER_TYPE'] != 'Overall').sum())
    _resumed, _fit, _skipped = ckpt.counts()
    print(f"\n[summary] within-cancer models: {_resumed} resumed, {_fit} fit, {_skipped} skipped; "
          f"{n_strata} cancer-type strata in the comparison across {len(held_scores)} held-out patients "
          f"(strata floored at n>={MIN_HELDOUT_N} held-out patients).")
    print(f"[summary] intermediate checkpoints + progress log under: {os.path.join(train_outdir, 'checkpoints')}")


if __name__ == "__main__":
    main()
