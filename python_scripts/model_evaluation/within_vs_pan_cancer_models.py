# === Imports ===
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH

# === Paths ===
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/icd_results/')

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# === Load datasets ===
cancer_type_df = pd.read_csv(
    '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
    usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']
).rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'})

time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-icd/time_decayed_events_df.csv'))

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
types_to_keep = cancer_type_counts[cancer_type_counts >= 500].index.tolist()
full_df['CANCER_TYPE'] = full_df['CANCER_TYPE'].where(full_df['CANCER_TYPE'].isin(types_to_keep), 'OTHER')

# === Train/held-out split ===
eval_mrns = full_df['DFCI_MRN'].sample(frac=0.75, random_state=1234).tolist()
train_df = full_df.loc[~full_df['DFCI_MRN'].isin(eval_mrns)].reset_index(drop=True)
held_df  = full_df.loc[ full_df['DFCI_MRN'].isin(eval_mrns)].reset_index(drop=True)

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

# === Train Pan-Cancer Model ===
alphas_to_test = np.logspace(-5, 0, 30)
l1_ratios = [0.5, 1.0]

_, embed_val_results, pan_cancer_model = run_grid_CoxPH_parallel(
    train_df, base_vars + type_cols, continuous_vars, embed_cols,
    l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=1000
)

pan_cancer_l1, pan_cancer_alpha = embed_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

trained_pan_cancer = (
    get_heldout_risk_scores_CoxPH(train_df, base_vars + type_cols, continuous_vars, embed_cols,
                                  event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=1000,
                                  l1_ratio=pan_cancer_l1, alpha=pan_cancer_alpha)
    .rename(columns={'risk_score': 'pan_cancer_risk_score'})
)

# === Train Within-Cancer Models ===
within_models = {}
within_scores = []

for cancer_type in tqdm([c.replace('CANCER_TYPE_', '') for c in type_cols]):
    mask_col = f'CANCER_TYPE_{cancer_type}'
    sub_df = train_df.loc[train_df[mask_col].astype(bool)]
    if len(sub_df) < 100:
        continue

    cur_test, cur_val, cur_model = run_grid_CoxPH_parallel(
        sub_df, base_vars, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', max_iter=1000
    )

    best_l1, best_alpha = cur_val.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]
    trained_sub = get_heldout_risk_scores_CoxPH(
        sub_df, base_vars, continuous_vars, embed_cols,
        event_col=event, tstop_col=f'tt_{event}', penalized=True, max_iter=1000,
        l1_ratio=best_l1, alpha=best_alpha
    )

    within_models[cancer_type] = cur_model
    within_scores.append(trained_sub)

trained_within = pd.concat(within_scores).rename(columns={'risk_score': 'within_cancer_risk_score'})

# === Evaluate on Training Set ===
complete_train = trained_within.merge(trained_pan_cancer, on='DFCI_MRN').merge(
    full_df[['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]], on='DFCI_MRN'
)

times = complete_train[f'tt_{event}']
events_bool = complete_train[event].astype(bool)

c_pan_train = concordance_index_censored(events_bool, times, complete_train['pan_cancer_risk_score'])[0]
c_within_train = concordance_index_censored(events_bool, times, complete_train['within_cancer_risk_score'])[0]

print(f"\nTrain set: Pan-cancer C-index = {c_pan_train:.3f}, Within-cancer C-index = {c_within_train:.3f}")

# === Held-out Evaluation ===
within_scores, pan_scores, mrns = [], [], []

for cancer_type in tqdm([c.replace('CANCER_TYPE_', '') for c in type_cols]):
    mask_col = f'CANCER_TYPE_{cancer_type}'
    if mask_col not in held_df.columns or mask_col not in train_df.columns:
        continue

    sub_df = held_df.loc[held_df[mask_col].astype(bool)]
    if len(sub_df) == 0:
        continue

    within_pred = within_models[cancer_type].predict(sub_df[base_vars + embed_cols])
    pan_pred = pan_cancer_model.predict(sub_df[base_vars + type_cols + embed_cols])

    within_scores += within_pred.tolist()
    pan_scores += pan_pred.tolist()
    mrns += sub_df['DFCI_MRN'].tolist()

# === Safe merge with held_df (CANCER_TYPE from full_df) ===
merged = pd.DataFrame({
    'DFCI_MRN': mrns,
    'within_cancer_risk_score': within_scores,
    'pan_cancer_risk_score': pan_scores
})

held_scores = merged.merge(
    full_df[['DFCI_MRN', 'CANCER_TYPE', f'tt_{event}', event]],
    on='DFCI_MRN', how='left'
)

# === Merge consistency checks ===
print("\n=== Held-out Merge Consistency Check ===")
print(f"Total predictions: {len(merged)}")
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

# === Per-Cancer-Type C-index Comparison (Held-out) ===
cindex_by_type = []
for cancer_type in tqdm(sorted(held_scores['CANCER_TYPE'].dropna().unique())):
    sub_df = held_scores.loc[held_scores['CANCER_TYPE'] == cancer_type]
    if sub_df.shape[0] < 30:
        continue

    times = sub_df[f'tt_{event}']
    events_bool = sub_df[event].astype(bool)

    c_pan = concordance_index_censored(events_bool, times, sub_df['pan_cancer_risk_score'])[0]
    c_within = concordance_index_censored(events_bool, times, sub_df['within_cancer_risk_score'])[0]
    delta = c_within - c_pan

    cindex_by_type.append({
        'CANCER_TYPE': cancer_type,
        'CINDEX_PAN': c_pan,
        'CINDEX_WITHIN': c_within,
        'DELTA_WITHIN_MINUS_PAN': delta,
        'N_HELDOUT': sub_df.shape[0]
    })

metrics_df = pd.DataFrame(cindex_by_type).sort_values('DELTA_WITHIN_MINUS_PAN', ascending=False)

# === Save Results ===
train_outdir = os.path.join(RESULTS_PATH, 'pan_vs_within_cancer')
os.makedirs(train_outdir, exist_ok=True)

complete_train.to_csv(os.path.join(train_outdir, 'train_risk_scores.csv'), index=False)
held_scores.to_csv(os.path.join(train_outdir, 'held_out_risk_scores.csv'), index=False)
metrics_df.to_csv(os.path.join(train_outdir, 'cindex_by_cancer_type.csv'), index=False)

print("\n=== Per-Cancer-Type C-Index Results (Held-out) ===")
print(metrics_df)
print(f"\nSaved per-cancer-type metrics to: {os.path.join(train_outdir, 'cindex_by_cancer_type.csv')}")