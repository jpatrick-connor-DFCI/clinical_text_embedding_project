"""Within Treatment Vs Pan Treatment Models script for model evaluation workflows."""

# === Imports ===
import os
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH

# Silence joblib/loky's benign worker-respawn warning; it floods the Jupyter IOPub
# channel during long runs and triggers "IOStream.flush timed out".
warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor")

# === Paths ===
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# === Load datasets ===
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'level_3_ICD_post_embedding_prediction_df.csv.gz'))

# Load cancer types
cancer_type_df = pd.read_csv(
    '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
    usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'})
cancer_type_sub = cancer_type_df.loc[cancer_type_df['DFCI_MRN'].isin(time_decayed_events_df['DFCI_MRN'].unique())].copy()

cancer_type_counts = cancer_type_sub['CANCER_TYPE'].value_counts()
types_to_keep = cancer_type_counts[cancer_type_counts >= 500].index.tolist()
cancer_type_sub['CANCER_TYPE'] = cancer_type_sub['CANCER_TYPE'].where(cancer_type_sub['CANCER_TYPE'].isin(types_to_keep), 'OTHER')
cancer_type_sub = pd.get_dummies(cancer_type_sub, columns=['CANCER_TYPE'], drop_first=True)

# First-line treatment class per patient, derived from the same one-hot source Figure 1 uses
# (categorical_treatment_data_by_line.csv.gz: DFCI_MRN, treatment_line, PX_on_<class> dummies).
# Take the first line (treatment_line == 1) and assign each patient the single present class.
# These dummies are NOT drop_first encoded (a PX_on_OTHER column exists), so a patient with any
# first-line med has at least one 1; idxmax picks the first class when first line is combination.
treatment_by_line = pd.read_csv(os.path.join(FEATURE_PATH, 'categorical_treatment_data_by_line.csv.gz'))
tx_cols = [c for c in treatment_by_line.columns if c.startswith('PX_on_')]
tx1 = treatment_by_line.loc[treatment_by_line['treatment_line'] == 1].copy()
tx1 = tx1.loc[tx1[tx_cols].sum(axis=1) > 0]  # keep patients with a known first-line class
tx1['TREATMENT_CLASSIFICATION'] = tx1[tx_cols].idxmax(axis=1).str.replace('PX_on_', '', regex=False)
treatment_df = tx1[['DFCI_MRN', 'TREATMENT_CLASSIFICATION']].drop_duplicates(subset='DFCI_MRN')
treatment_types = treatment_df['TREATMENT_CLASSIFICATION'].unique()

# Merge embeddings + cancer types + events
full_df = (time_decayed_events_df
           .merge(treatment_df, on='DFCI_MRN', how='inner')
           .merge(cancer_type_sub, on='DFCI_MRN', how='inner'))

# === Feature definitions ===
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [col for col in full_df if 'CANCER_TYPE' in col]
event = 'death'

# Find all time-to-event columns
events = [col.split('_', 1)[1] for col in time_decayed_events_df.columns if col.startswith('tt')]
tt_events = [f"tt_{e}" for e in events]

# Embedding features
embed_cols = [c for c in full_df.columns if ('EMBEDDING' in c or '2015' in c)]

# === Train/held-out split (75% train, 25% held-out evaluation) ===
held_mrns = full_df['DFCI_MRN'].sample(frac=0.25, random_state=1234).tolist()
train_df = full_df.loc[~full_df['DFCI_MRN'].isin(held_mrns)].reset_index(drop=True)
held_df  = full_df.loc[ full_df['DFCI_MRN'].isin(held_mrns)].reset_index(drop=True)

# Ensure consistent column order
held_df = held_df[train_df.columns]

# === Identify feature columns ===
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

# === Scale continuous vars ===
scaler = StandardScaler()
train_df[continuous_vars] = scaler.fit_transform(train_df[continuous_vars])
held_df[continuous_vars] = scaler.transform(held_df[continuous_vars])

# === Train Pan-Treatment Model ===
alphas_to_test = np.logspace(-5, 0, 25)
l1_ratios = [0.5, 1.0]


_, embed_val_results, pan_treatment_model = run_grid_CoxPH_parallel(
    train_df, base_vars, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
)

pan_treatment_l1, pan_treatment_alpha = embed_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

trained_pan_treatment = (
    get_heldout_risk_scores_CoxPH(train_df, base_vars, continuous_vars, embed_cols,
                                  event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=3000,
                                  l1_ratio=pan_treatment_l1, alpha=pan_treatment_alpha, backend="threading")
    .rename(columns={'risk_score': 'pan_treatment_risk_score'})
)

# === Train Within-Cancer Models ===
within_models = {}
within_scores = []

for treatment in tqdm(treatment_types, mininterval=30):
    sub_df = train_df.loc[train_df['TREATMENT_CLASSIFICATION'] == treatment]
    
    if len(sub_df) < 100:
        continue

    cur_test, cur_val, cur_model = run_grid_CoxPH_parallel(
        sub_df, base_vars, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=3000
    )

    # Skip strata whose final model failed to converge (returned None) — otherwise the
    # held-out loop below would call None.predict(...). Their patients are excluded from
    # both the train and held-out comparison so the two cohorts stay consistent.
    if cur_model is None:
        print(f"  skipping treatment '{treatment}': final model did not converge")
        continue

    best_l1, best_alpha = cur_val.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]
    trained_sub = get_heldout_risk_scores_CoxPH(
        sub_df, base_vars, continuous_vars, embed_cols,
        event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=3000,
        l1_ratio=best_l1, alpha=best_alpha, backend="threading"
    )

    within_models[treatment] = cur_model
    within_scores.append(trained_sub)

trained_within = pd.concat(within_scores).rename(columns={'risk_score': 'within_treatment_risk_score'})

# === Evaluate on Training Set ===
complete_train = trained_within.merge(trained_pan_treatment, on='DFCI_MRN').merge(
    full_df[['DFCI_MRN', 'TREATMENT_CLASSIFICATION', f'tt_{event}', event]], on='DFCI_MRN'
)

# Some within-treatment strata diverge during Cox fitting and emit non-finite OOF
# risk scores; drop those rows so the pan-vs-within comparison runs on the same set
# of patients with finite predictions from both models. Also guard the time/event
# columns: concordance_index_censored / cumulative_dynamic_auc reject any non-finite
# input, so every metric call below sees only finite values.
_score_cols = ['pan_treatment_risk_score', 'within_treatment_risk_score']
_finite_cols = _score_cols + [f'tt_{event}', event]
complete_train[_finite_cols] = complete_train[_finite_cols].replace([np.inf, -np.inf], np.nan)
_n0 = len(complete_train)
complete_train = complete_train.dropna(subset=_finite_cols)
if len(complete_train) < _n0:
    print(f"Train: dropped {_n0 - len(complete_train)} rows with non-finite risk scores / outcomes")

times = complete_train[f'tt_{event}']
events_bool = complete_train[event].astype(bool)

c_pan_train = concordance_index_censored(events_bool, times, complete_train['pan_treatment_risk_score'])[0]
c_within_train = concordance_index_censored(events_bool, times, complete_train['within_treatment_risk_score'])[0]

print(f"\nTrain set: Pan-cancer C-index = {c_pan_train:.3f}, Within-cancer C-index = {c_within_train:.3f}")

# === Held-out Evaluation ===
within_scores, pan_scores, mrns = [], [], []

for treatment in tqdm(treatment_types, mininterval=30):

    sub_df = held_df.loc[held_df['TREATMENT_CLASSIFICATION'] == treatment]
    if len(sub_df) == 0 or treatment not in within_models:
        continue

    within_pred = within_models[treatment].predict(sub_df[base_vars + embed_cols])
    pan_pred = pan_treatment_model.predict(sub_df[base_vars + embed_cols])

    within_scores += within_pred.tolist()
    pan_scores += pan_pred.tolist()
    mrns += sub_df['DFCI_MRN'].tolist()

# === Safe merge with held_df (CANCER_TYPE from full_df) ===
merged = pd.DataFrame({
    'DFCI_MRN': mrns,
    'within_treatment_risk_score': within_scores,
    'pan_treatment_risk_score': pan_scores
})

held_scores = merged.merge(
    full_df[['DFCI_MRN', 'TREATMENT_CLASSIFICATION', f'tt_{event}', event]],
    on='DFCI_MRN', how='left'
)

# Drop non-finite held-out risk scores / outcomes (divergent within-treatment strata).
held_scores[_finite_cols] = held_scores[_finite_cols].replace([np.inf, -np.inf], np.nan)
_n0 = len(held_scores)
held_scores = held_scores.dropna(subset=_finite_cols)
if len(held_scores) < _n0:
    print(f"Held-out: dropped {_n0 - len(held_scores)} rows with non-finite risk scores / outcomes")

# === Compute held-out concordance ===
times = held_scores[f'tt_{event}']
events_bool = held_scores[event].astype(bool)

c_pan_held = concordance_index_censored(events_bool, times, held_scores['pan_treatment_risk_score'])[0]
c_within_held = concordance_index_censored(events_bool, times, held_scores['within_treatment_risk_score'])[0]

print(f"Held-out set: Pan-treatment C-index = {c_pan_held:.3f}, Within-treatment C-index = {c_within_held:.3f}")

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

auc_pan_held = _mean_auc(held_scores, 'pan_treatment_risk_score')
auc_within_held = _mean_auc(held_scores, 'within_treatment_risk_score')
print(f"Held-out set: Pan-treatment mean AUC(t) = {auc_pan_held:.3f}, "
      f"Within-treatment mean AUC(t) = {auc_within_held:.3f}")

# === Per-Treatment Comparison (Held-out) ===
cindex_by_treatment = []
for treatment in tqdm(sorted(held_scores['TREATMENT_CLASSIFICATION'].dropna().unique()), mininterval=30):
    sub_df = held_scores.loc[held_scores['TREATMENT_CLASSIFICATION'] == treatment]

    times = sub_df[f'tt_{event}']
    events_bool = sub_df[event].astype(bool)

    c_pan = concordance_index_censored(events_bool, times, sub_df['pan_treatment_risk_score'])[0]
    c_within = concordance_index_censored(events_bool, times, sub_df['within_treatment_risk_score'])[0]
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

metrics_df = pd.DataFrame(cindex_by_treatment).sort_values('DELTA_AUC_WITHIN_MINUS_PAN', ascending=False)

# Prepend an "Overall" row so the figure has a reference line / summary.
overall_row = pd.DataFrame([{
    'TREATMENT': 'Overall',
    'CINDEX_PAN': c_pan_held,
    'CINDEX_WITHIN': c_within_held,
    'DELTA_WITHIN_MINUS_PAN': c_within_held - c_pan_held,
    'AUC_PAN': auc_pan_held,
    'AUC_WITHIN': auc_within_held,
    'DELTA_AUC_WITHIN_MINUS_PAN': auc_within_held - auc_pan_held,
    'N_HELDOUT': len(held_scores),
}])
metrics_df = pd.concat([overall_row, metrics_df], ignore_index=True)

# === Save Results ===
train_outdir = os.path.join(RESULTS_PATH, 'pan_vs_within_treatment')
os.makedirs(train_outdir, exist_ok=True)

complete_train.to_csv(os.path.join(train_outdir, 'train_risk_scores.csv'), index=False)
held_scores.to_csv(os.path.join(train_outdir, 'held_out_risk_scores.csv'), index=False)
metrics_df.to_csv(os.path.join(train_outdir, 'metrics_by_treatment.csv'), index=False)

print("\n=== Per-Treatment Results (Held-out) ===")
print(metrics_df)
print(f"\nSaved per-treatment metrics to: {os.path.join(train_outdir, 'metrics_by_treatment.csv')}")

n_strata = int((metrics_df['TREATMENT'] != 'Overall').sum())
print(f"\n[summary] {len(within_models)} within-treatment models fit; "
      f"{n_strata} treatment strata in the comparison across {len(held_scores)} held-out patients "
      f"(figure prep applies an additional n>=30 floor).")
