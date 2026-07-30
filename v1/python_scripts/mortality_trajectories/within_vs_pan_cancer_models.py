"""Within Vs Pan Cancer Models script for model evaluation workflows."""

# === Imports ===
import os
import time
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, RunCheckpoint

# Silence joblib/loky's benign worker-respawn warning; it floods the Jupyter IOPub
# channel during long runs and triggers "IOStream.flush timed out".
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")

# === Paths ===
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')

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
cancer_type_df = pd.read_csv(
    '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
    usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']
).rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'})

time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'level_3_ICD_post_embedding_prediction_df.csv.gz'))

# Merge embeddings + cancer types + events
full_df = (time_decayed_events_df
           .merge(cancer_type_df, on='DFCI_MRN', how='inner')
           .dropna(subset=['CANCER_TYPE']))

# === Feature definitions ===
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
event = 'death'

# Find all time-to-event columns
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{e}" for e in events]

# Embedding features
embed_cols = [c for c in full_df.columns if ('EMBEDDING' in c or '2015' in c)]

# Collapse rare cancer types
cancer_type_counts = full_df['CANCER_TYPE'].value_counts()
types_to_keep = cancer_type_counts[cancer_type_counts >= MIN_STRATUM_N].index.tolist()
full_df['CANCER_TYPE'] = full_df['CANCER_TYPE'].where(full_df['CANCER_TYPE'].isin(types_to_keep), 'OTHER')

# === Train/held-out split (75% train, 25% held-out evaluation) ===
held_mrns = full_df['DFCI_MRN'].sample(frac=0.25, random_state=1234).tolist()
train_df = full_df.loc[~full_df['DFCI_MRN'].isin(held_mrns)].reset_index(drop=True)
held_df  = full_df.loc[ full_df['DFCI_MRN'].isin(held_mrns)].reset_index(drop=True)

# === One-hot encode cancer type ===
train_df = pd.get_dummies(train_df, columns=['CANCER_TYPE'], drop_first=True)
held_df  = pd.get_dummies(held_df,  columns=['CANCER_TYPE'], drop_first=True)

# Align dummy columns across splits
for c in set(train_df.columns) - set(held_df.columns):
    if c.startswith('CANCER_TYPE_'):
        held_df[c] = 0
for c in set(held_df.columns) - set(train_df.columns):
    if c.startswith('CANCER_TYPE_'):
        train_df[c] = 0

# Ensure consistent column order
held_df = held_df[train_df.columns]

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

# === Scale continuous vars ===
scaler = StandardScaler()
train_df[continuous_vars] = scaler.fit_transform(train_df[continuous_vars])
held_df[continuous_vars] = scaler.transform(held_df[continuous_vars])

# Impute any remaining missing features to the TRAIN column mean. Pooled embeddings are
# NaN for patients lacking notes of a given type; run_grid/get_heldout impute internally,
# but the direct model.predict(held_df) below does not — so do it here for every model
# feature (StandardScaler ignores NaN in fit and preserves it in transform). Standardized
# columns have mean ~0; binary covariates get the train proportion. Matches the per-fold
# mean imputation used during fitting and prevents 'Input X contains NaN' at predict time.
_feature_cols = base_vars + type_cols + embed_cols
_train_means = train_df[_feature_cols].mean()
train_df[_feature_cols] = train_df[_feature_cols].fillna(_train_means)
held_df[_feature_cols] = held_df[_feature_cols].fillna(_train_means)

# === Checkpoint store (writes intermediate results during the run; enables resume) ===
RESUME = True
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
    'strata': sorted(c.replace('CANCER_TYPE_', '') for c in type_cols),
    'seed': 1234,
}
ckpt = RunCheckpoint(os.path.join(train_outdir, 'checkpoints'), fingerprint, resume=RESUME)

# === Train Pan-Cancer Model ===
alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]

if ckpt.pan_done():
    trained_pan_cancer, pan_held = ckpt.load_pan()
else:
    _, embed_val_results, pan_cancer_model = run_grid_CoxPH_parallel(
        train_df, base_vars + type_cols, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
    )
    pan_cancer_l1, pan_cancer_alpha = embed_val_results.sort_values(
        by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

    trained_pan_cancer = (
        get_heldout_risk_scores_CoxPH(train_df, base_vars + type_cols, continuous_vars, embed_cols,
                                      event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=3000,
                                      l1_ratio=pan_cancer_l1, alpha=pan_cancer_alpha, backend="threading")
        .rename(columns={'risk_score': 'pan_cancer_risk_score'})
    )
    # Held-out pan predictions, computed once on the full held set (covers every stratum) so the
    # within loop and resume never need the fitted pan model back.
    pan_held = pd.DataFrame({
        'DFCI_MRN': held_df['DFCI_MRN'].values,
        'pan_cancer_risk_score': pan_cancer_model.predict(held_df[base_vars + type_cols + embed_cols]),
    })
    ckpt.save_pan(trained_pan_cancer, pan_held,
                  meta={'n_train': int(len(train_df)), 'n_held': int(len(held_df)),
                        'l1': float(pan_cancer_l1), 'alpha': float(pan_cancer_alpha)})


def _provisional_cindex(held_within_df):
    """Reference-free per-stratum held-out C-index (pan vs within) for the progress manifest."""
    m = (held_within_df
         .merge(pan_held, on='DFCI_MRN')
         .merge(full_df[['DFCI_MRN', f'tt_{event}', event]], on='DFCI_MRN'))
    cols = ['within_cancer_risk_score', 'pan_cancer_risk_score', f'tt_{event}', event]
    m[cols] = m[cols].replace([np.inf, -np.inf], np.nan)
    m = m.dropna(subset=cols)
    if len(m) < MIN_HELDOUT_N or m[event].sum() == 0:
        return None, None, len(m)
    eb = m[event].astype(bool)
    t = m[f'tt_{event}']
    try:
        cp = concordance_index_censored(eb, t, m['pan_cancer_risk_score'])[0]
        cw = concordance_index_censored(eb, t, m['within_cancer_risk_score'])[0]
    except Exception:
        return None, None, len(m)
    return float(cp), float(cw), len(m)


# === Train + score within-cancer models (single resumable pass) ===
# Each stratum's train OOF scores AND held-out predictions are written to disk as it completes,
# so a crashed/interrupted run reloads finished strata instead of refitting them.
train_score_frames, held_score_frames = [], []

for cancer_type in tqdm([c.replace('CANCER_TYPE_', '') for c in type_cols], mininterval=30):
    mask_col = f'CANCER_TYPE_{cancer_type}'

    st = ckpt.status(cancer_type)
    if st == 'done':
        _tr, _hd = ckpt.load_stratum(cancer_type)
        train_score_frames.append(_tr)
        held_score_frames.append(_hd)
        continue
    if st == 'skipped':
        continue

    sub_df = train_df.loc[train_df[mask_col].astype(bool)]
    if len(sub_df) < MIN_TRAIN_N:
        ckpt.mark_skipped(cancer_type, 'too_small', meta={'n_train': int(len(sub_df))})
        continue

    _t0 = time.time()
    cur_test, cur_val, cur_model = run_grid_CoxPH_parallel(
        sub_df, base_vars, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
    )

    # Skip strata whose final model failed to converge (returned None).
    if cur_model is None:
        ckpt.mark_skipped(cancer_type, 'no_converge', meta={'n_train': int(len(sub_df))})
        continue

    best_l1, best_alpha = cur_val.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]
    trained_sub = get_heldout_risk_scores_CoxPH(
        sub_df, base_vars, continuous_vars, embed_cols,
        event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=3000,
        l1_ratio=best_l1, alpha=best_alpha, backend="threading"
    ).rename(columns={'risk_score': 'within_cancer_risk_score'})

    # Held-out predictions for this stratum, taken now while the model is in memory.
    sub_held = held_df.loc[held_df[mask_col].astype(bool)]
    held_sub = pd.DataFrame({
        'DFCI_MRN': sub_held['DFCI_MRN'].values,
        'within_cancer_risk_score': cur_model.predict(sub_held[base_vars + embed_cols]),
    })

    c_pan, c_within, n_held = _provisional_cindex(held_sub)
    ckpt.save_stratum(cancer_type, trained_sub, held_sub,
                      meta={'n_train': int(len(sub_df)), 'n_held': int(n_held),
                            'l1': float(best_l1), 'alpha': float(best_alpha),
                            'c_pan': c_pan, 'c_within': c_within,
                            'elapsed_s': round(time.time() - _t0, 1)})
    train_score_frames.append(trained_sub)
    held_score_frames.append(held_sub)

trained_within = pd.concat(train_score_frames, ignore_index=True)
within_held_all = pd.concat(held_score_frames, ignore_index=True)

# === Evaluate on Training Set ===
complete_train = trained_within.merge(trained_pan_cancer, on='DFCI_MRN').merge(
    full_df[['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]], on='DFCI_MRN'
)

# Some within-cancer strata diverge during Cox fitting and emit non-finite OOF risk
# scores; drop those rows so the pan-vs-within comparison runs on the same set of
# patients with finite predictions from both models. Also guard the time/event columns:
# concordance_index_censored / cumulative_dynamic_auc reject any non-finite input, so
# every metric call below sees only finite values.
_score_cols = ['pan_cancer_risk_score', 'within_cancer_risk_score']
_finite_cols = _score_cols + [f'tt_{event}', event]
complete_train[_finite_cols] = complete_train[_finite_cols].replace([np.inf, -np.inf], np.nan)
_n0 = len(complete_train)
complete_train = complete_train.dropna(subset=_finite_cols)
if len(complete_train) < _n0:
    print(f"Train: dropped {_n0 - len(complete_train)} rows with non-finite risk scores / outcomes")

times = complete_train[f'tt_{event}']
events_bool = complete_train[event].astype(bool)

c_pan_train = concordance_index_censored(events_bool, times, complete_train['pan_cancer_risk_score'])[0]
c_within_train = concordance_index_censored(events_bool, times, complete_train['within_cancer_risk_score'])[0]

print(f"\nTrain set: Pan-cancer C-index = {c_pan_train:.3f}, Within-cancer C-index = {c_within_train:.3f}")

# === Held-out Evaluation (assemble per-stratum within scores + pan held scores) ===
held_scores = within_held_all.merge(pan_held, on='DFCI_MRN').merge(
    full_df[['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]],
    on='DFCI_MRN', how='left'
)

# Drop non-finite held-out risk scores / outcomes (divergent within-cancer strata).
held_scores[_finite_cols] = held_scores[_finite_cols].replace([np.inf, -np.inf], np.nan)
_n0 = len(held_scores)
held_scores = held_scores.dropna(subset=_finite_cols)
if len(held_scores) < _n0:
    print(f"Held-out: dropped {_n0 - len(held_scores)} rows with non-finite risk scores / outcomes")

# === Merge consistency checks ===
print("\n=== Held-out Merge Consistency Check ===")
print(f"Total predictions: {len(within_held_all)}")
print(f"Matched to held_df metadata: {held_scores['CANCER_TYPE'].notna().sum()}")
missing = held_scores[held_scores[f'tt_{event}'].isna()]
if len(missing) > 0:
    print(f"⚠️ {len(missing)} held-out predictions could not be matched to full_df!")
else:
    print("✅ All held-out predictions successfully matched to metadata.")

dup_counts = held_scores['DFCI_MRN'].value_counts()
if (dup_counts > 1).any():
    print(f"⚠️ {sum(dup_counts > 1)} MRNs appear multiple times in held_scores!")

# === Compute held-out concordance ===
times = held_scores[f'tt_{event}']
events_bool = held_scores[event].astype(bool)

c_pan_held = concordance_index_censored(events_bool, times, held_scores['pan_cancer_risk_score'])[0]
c_within_held = concordance_index_censored(events_bool, times, held_scores['within_cancer_risk_score'])[0]

print(f"Held-out set: Pan-cancer C-index = {c_pan_held:.3f}, Within-cancer C-index = {c_within_held:.3f}")

# === Mean time-dependent AUC (project-standard metric) ===
# IPCW reference + eval-time grid from the TRAINING set, matching evaluate_surv_model
# (5th–95th percentile, 50 points). Per-stratum eval times are clipped to that stratum's
# held-out follow-up so cumulative_dynamic_auc never sees out-of-range times.
y_train_global = Surv.from_arrays(complete_train[event].astype(bool), complete_train[f'tt_{event}'])
_lo, _hi = np.percentile(complete_train[f'tt_{event}'], [5, 95])
base_eval_times = np.linspace(_lo, _hi, 50)

def _mean_auc(sub_df, risk_col):
    et = base_eval_times[(base_eval_times > sub_df[f'tt_{event}'].min())
                         & (base_eval_times < sub_df[f'tt_{event}'].max())]
    if len(et) == 0:
        return np.nan
    try:
        y_test = Surv.from_arrays(sub_df[event].astype(bool), sub_df[f'tt_{event}'])
        return cumulative_dynamic_auc(y_train_global, y_test, sub_df[risk_col].values, et)[1]
    except Exception:
        return np.nan

auc_pan_held = _mean_auc(held_scores, 'pan_cancer_risk_score')
auc_within_held = _mean_auc(held_scores, 'within_cancer_risk_score')
print(f"Held-out set: Pan-cancer mean AUC(t) = {auc_pan_held:.3f}, "
      f"Within-cancer mean AUC(t) = {auc_within_held:.3f}")

# === Per-Cancer-Type Comparison (Held-out) ===
cindex_by_type = []
for cancer_type in tqdm(sorted(held_scores['CANCER_TYPE'].dropna().unique()), mininterval=30):
    sub_df = held_scores.loc[held_scores['CANCER_TYPE'] == cancer_type]
    if sub_df.shape[0] < MIN_HELDOUT_N:
        continue

    times = sub_df[f'tt_{event}']
    events_bool = sub_df[event].astype(bool)

    c_pan = concordance_index_censored(events_bool, times, sub_df['pan_cancer_risk_score'])[0]
    c_within = concordance_index_censored(events_bool, times, sub_df['within_cancer_risk_score'])[0]
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

metrics_df = pd.DataFrame(cindex_by_type).sort_values('DELTA_AUC_WITHIN_MINUS_PAN', ascending=False)

# Prepend an "Overall" row so the figure has a reference line / summary.
overall_row = pd.DataFrame([{
    'CANCER_TYPE': 'Overall',
    'CINDEX_PAN': c_pan_held,
    'CINDEX_WITHIN': c_within_held,
    'DELTA_WITHIN_MINUS_PAN': c_within_held - c_pan_held,
    'AUC_PAN': auc_pan_held,
    'AUC_WITHIN': auc_within_held,
    'DELTA_AUC_WITHIN_MINUS_PAN': auc_within_held - auc_pan_held,
    'N_HELDOUT': len(held_scores),
}])
metrics_df = pd.concat([overall_row, metrics_df], ignore_index=True)

# === Save Results === (train_outdir / checkpoints dir were created at the checkpoint-store setup)
complete_train.to_csv(os.path.join(train_outdir, 'train_risk_scores.csv'), index=False)
held_scores.to_csv(os.path.join(train_outdir, 'held_out_risk_scores.csv'), index=False)
metrics_df.to_csv(os.path.join(train_outdir, 'metrics_by_cancer_type.csv'), index=False)

print("\n=== Per-Cancer-Type Results (Held-out) ===")
print(metrics_df)
print(f"\nSaved per-cancer-type metrics to: {os.path.join(train_outdir, 'metrics_by_cancer_type.csv')}")

n_strata = int((metrics_df['CANCER_TYPE'] != 'Overall').sum())
_resumed, _fit, _skipped = ckpt.counts()
print(f"\n[summary] within-cancer models: {_resumed} resumed, {_fit} fit, {_skipped} skipped; "
      f"{n_strata} cancer-type strata in the comparison across {len(held_scores)} held-out patients "
      f"(strata floored at n>={MIN_HELDOUT_N} held-out patients).")
print(f"[summary] intermediate checkpoints + progress log under: {os.path.join(train_outdir, 'checkpoints')}")