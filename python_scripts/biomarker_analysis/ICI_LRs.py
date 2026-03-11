"""Unified ICI propensity score generation for line-matched cohorts.

Trains two propensity models per matched cohort:
  1. Embeddings-only: LR on text embeddings
  2. All-covariates: LR on text embeddings + demographics + cancer type + panel version + line_category

Uses 5-fold stratified CV to produce held-out propensity scores.

Usage:
  python ICI_LRs.py
"""

import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from biomarker_common import (
    DATA_PATH, SURV_PATH, load_note_embeddings, load_confounders,
)

MATCHING = '1to1'

# === Paths ===
COHORT_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/matched_cohorts/')
OUTPUT_BASE = os.path.join(DATA_PATH, f'treatment_prediction/matched_{MATCHING}/')
EMBEDDING_DATA_PATH = os.path.join(OUTPUT_BASE, 'prediction_data/')
os.makedirs(EMBEDDING_DATA_PATH, exist_ok=True)

print(f"[ICI_LRs] Matching: {MATCHING}")

# === Load matched cohort ===
cohort_df = pd.read_csv(os.path.join(COHORT_PATH, f'matched_cohort_{MATCHING}.csv'))
cohort_df['treatment_start_date'] = pd.to_datetime(cohort_df['treatment_start_date'])

n_ici = cohort_df['PX_on_ICI'].sum()
n_ctrl = len(cohort_df) - n_ici
print(f"Matched cohort: {int(n_ici)} ICI + {int(n_ctrl)} controls = {len(cohort_df)} total")

# === Load note embeddings ===
notes_meta, embeddings_data = load_note_embeddings()

note_types = ['Clinician', 'Imaging', 'Pathology']
pool_fx = {nt: 'time_decay_mean' for nt in note_types}

# === Load confounders for all-covariates model ===
confounders = load_confounders()

# === Load cancer type dummies and panel version for all-covariates model ===
cancer_type_df = pd.read_csv(
    os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))
somatic_df = pd.read_csv(
    os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))
panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
somatic_panel = somatic_df[['DFCI_MRN'] + panel_cols].drop_duplicates(subset='DFCI_MRN', keep='first')

# === Generate embedding datasets for each buffer ===
buffers = [0, 15, 30, 45]
for buffer in tqdm(buffers, desc="Generating embedding datasets"):
    buffer_path = os.path.join(EMBEDDING_DATA_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_path, exist_ok=True)

    notes_meta_sub = (
        notes_meta[notes_meta['DFCI_MRN'].isin(cohort_df['DFCI_MRN'])]
        .merge(cohort_df[['DFCI_MRN', 'treatment_start_date']], on='DFCI_MRN', how='left')
        .assign(NOTE_TIME_REL_PRED_START_DT=lambda df: (
            pd.to_datetime(df['NOTE_DATETIME']) - pd.to_datetime(df['treatment_start_date'])).dt.days)
    )

    embedding_vals = generate_survival_embedding_df(
        notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data,
        note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT",
        max_note_window=-buffer, pool_fx=pool_fx, decay_param=0.01, continuous_window=False)

    full_dataset = cohort_df.merge(embedding_vals.dropna(), on='DFCI_MRN')
    full_dataset.to_csv(
        os.path.join(buffer_path, f'ICI_prediction_df_w_{buffer}_day_buffer.csv'), index=False)

# Save prediction times for downstream use
cohort_df[['DFCI_MRN', 'treatment_start_date', 'PX_on_ICI', 'line_category']].to_csv(
    os.path.join(EMBEDDING_DATA_PATH, 'prediction_times.csv'), index=False)


# === Train propensity models ===
def train_propensity_cv(pred_df, feature_cols, label):
    """Train elastic-net logistic regression with nested CV for hyperparameter
    tuning and held-out propensity scores.

    Outer loop: 5-fold stratified CV produces held-out predictions.
    Inner loop: LogisticRegressionCV tunes C and l1_ratio via 3-fold CV
                within each outer training fold.
    """
    X = pred_df[['DFCI_MRN'] + feature_cols].copy()
    y = pred_df['PX_on_ICI'].astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    cv_mrns, cv_preds, cv_probs, cv_true = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cv_mrns += X_test['DFCI_MRN'].tolist()
        X_train = X_train.drop(columns=['DFCI_MRN'])
        X_test = X_test.drop(columns=['DFCI_MRN'])

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegressionCV(
            penalty='elasticnet',
            solver='saga',
            Cs=10,
            l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9],
            cv=3,
            scoring='roc_auc',
            max_iter=5000,
            random_state=1234,
            n_jobs=-1,
        )
        clf.fit(X_train_s, y_train.values.ravel())

        n_nonzero = np.sum(clf.coef_ != 0)
        n_total = clf.coef_.size
        print(f"    fold {fold}: C={clf.C_[0]:.4f}, l1_ratio={clf.l1_ratio_[0]:.2f}, "
              f"features={n_nonzero}/{n_total}")

        cv_preds += clf.predict(X_test_s).tolist()
        cv_probs += clf.predict_proba(X_test_s)[:, 1].tolist()
        cv_true += y_test.tolist()

    auc = roc_auc_score(cv_true, cv_probs)
    print(f"  [{label}] AUC = {auc:.4f}")

    return pd.DataFrame({
        'DFCI_MRN': cv_mrns,
        'model_preds': cv_preds,
        'model_probs': cv_probs,
        'ground_truth': cv_true,
    })


PS_MODELS = ['embeddings_only', 'all_covariates']

for buffer in tqdm(buffers, desc="Training propensity models"):
    buffer_input_path = os.path.join(EMBEDDING_DATA_PATH, f'w_{buffer}_day_buffer/')

    pred_df = pd.read_csv(
        os.path.join(buffer_input_path, f'ICI_prediction_df_w_{buffer}_day_buffer.csv'))

    # Embedding feature columns
    embedding_cols = [col for col in pred_df.columns
                      if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]

    # Build all-covariates feature set
    pred_with_covars = (
        pred_df
        .merge(cancer_type_df, on='DFCI_MRN', how='left')
        .merge(somatic_panel, on='DFCI_MRN', how='left')
    )
    # One-hot panel version if not already dummified
    if 'PANEL_VERSION' in pred_with_covars.columns:
        pred_with_covars = pd.get_dummies(
            pred_with_covars, columns=['PANEL_VERSION'], drop_first=True)

    cancer_type_feature_cols = [c for c in pred_with_covars.columns if c.startswith('CANCER_TYPE_')]
    panel_feature_cols = [c for c in pred_with_covars.columns if c.upper().startswith('PANEL_VERSION_')]
    # Line category dummies (drop line 1 as reference)
    pred_with_covars = pd.get_dummies(
        pred_with_covars, columns=['line_category'], prefix='LINE', drop_first=True, dtype=int)
    line_feature_cols = [c for c in pred_with_covars.columns if c.startswith('LINE_')]

    demo_cols = ['GENDER', 'AGE_AT_TREATMENTSTART']
    # Ensure demo cols are present
    if 'GENDER' not in pred_with_covars.columns or 'AGE_AT_TREATMENTSTART' not in pred_with_covars.columns:
        surv_demo = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'),
                                usecols=['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART'])
        pred_with_covars = pred_with_covars.merge(surv_demo, on='DFCI_MRN', how='left')

    all_covar_cols = (embedding_cols + demo_cols + cancer_type_feature_cols +
                      panel_feature_cols + line_feature_cols)
    # Drop any columns that are all NaN
    all_covar_cols = [c for c in all_covar_cols if pred_with_covars[c].notna().any()]

    for ps_model in PS_MODELS:
        output_path = os.path.join(OUTPUT_BASE, f'{ps_model}_propensity/w_{buffer}_day_buffer/')
        os.makedirs(output_path, exist_ok=True)

        if ps_model == 'embeddings_only':
            features = embedding_cols
            source_df = pred_df.dropna(subset=embedding_cols)
        else:
            features = all_covar_cols
            source_df = pred_with_covars.dropna(subset=features)

        label = f"buffer={buffer}, {ps_model}"
        out_df = train_propensity_cv(source_df, features, label)
        out_df.to_csv(os.path.join(output_path, 'predictions.csv'), index=False)

# === Plot ROC curves for the 30-day buffer ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
PLOT_BUFFER = 30

for i, ps_model in enumerate(PS_MODELS):
    pred_path = os.path.join(OUTPUT_BASE, f'{ps_model}_propensity/w_{PLOT_BUFFER}_day_buffer/predictions.csv')
    pred_df = pd.read_csv(pred_path)

    auc = roc_auc_score(pred_df['ground_truth'], pred_df['model_probs'])
    fpr, tpr, _ = roc_curve(pred_df['ground_truth'], pred_df['model_probs'])

    ax = axes[i]
    sns.set_style("whitegrid")
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'AUC = {auc:.3f}')
    sns.lineplot(x=[0, 1], y=[0, 1], ax=ax, linestyle='--', color='gray')
    ax.set_title(f'{ps_model} ({MATCHING}, buffer={PLOT_BUFFER}d)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_BASE, f'propensity_ROC_{MATCHING}.png'), dpi=150)
plt.show()
print(f"[ICI_LRs] Done. Outputs in {OUTPUT_BASE}")
