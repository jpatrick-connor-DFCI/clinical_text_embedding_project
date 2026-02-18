"""ICI propensity score generation: first-line ICI vs never-ICI.

Trains logistic regression propensity models using clinical note embeddings
plus confounders (demographics, cancer type, panel version) to predict
first-line ICI receipt. Patients with ICI only at later lines are excluded.
"""

# %% [code cell 1]
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from biomarker_common import (
    DATA_PATH, load_note_embeddings, load_cohort_treatments,
    add_treatment_line_columns, load_confounders,
)

# Paths
ICI_DATA_PATH = os.path.join(DATA_PATH, 'treatment_prediction/line_ICI_prediction_data/')
ICI_PROP_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/')

manual_ICI_start_df = pd.read_csv('/data/gusev/USERS/mjsaleh/IO_START.csv',index_col=0).rename(columns={'MRN' : 'DFCI_MRN'})

# --- Load cohort ---
cohort_treatment_df = load_cohort_treatments()

# --- One-hot encode treatment subtypes ---
treatments = (
    cohort_treatment_df["Treatment_type"]
    .str.replace(";", "", regex=False)
    .str.split()
    .explode()
)
dummies = pd.get_dummies(treatments, prefix="PX_on").groupby(level=0).max()
cohort_treatment_df = pd.concat([cohort_treatment_df, dummies], axis=1)

# --- Treatment line + time-to-next ---
cohort_treatment_df = add_treatment_line_columns(cohort_treatment_df)
cohort_treatment_df["time_to_next_treatment"] = (
    cohort_treatment_df.groupby("MRN")["LOT_start_date"].diff(-1).dt.days.abs()
)

# --- Columns of interest ---
px_cols = [c for c in cohort_treatment_df if c.startswith("PX_on_")]
cols_to_include = [
    "MRN", "DIAGNOSIS_DT", "STAGE", "LOT_start_date",
    "Treatment_type", "Treatment_subtype", "Medications",
    "time_to_treatment", "treatment_line", "time_to_next_treatment"
] + px_cols
cohort_treatment_df = cohort_treatment_df[cols_to_include]

# Sort for cumulative tracking
cohort_treatment_df = cohort_treatment_df.sort_values(["MRN", "treatment_line"])

# --- Classify patients: first-line ICI vs never-ICI (exclude later-line-only ICI) ---
# Identify whether each patient ever receives ICI at any line
ever_ici = (cohort_treatment_df.groupby("MRN")["PX_on_ICI"]
            .max()
            .rename("ever_ici"))
manual_ever_ici = set(manual_ICI_start_df['DFCI_MRN'].tolist())

# Identify patients with ICI at line 1
line1 = cohort_treatment_df.loc[cohort_treatment_df["treatment_line"] == 1].copy()
line1_ici_mrns = set(line1.loc[line1["PX_on_ICI"] == 1, "MRN"])

# Never-ICI patients: no ICI at any treatment line
never_ici_mrns = set(ever_ici.loc[ever_ici == 0].index) - manual_ever_ici

# Excluded: patients who receive ICI but NOT at line 1 (i.e. only at later lines)
later_line_ici_mrns = (set(ever_ici.loc[ever_ici == 1].index) | manual_ever_ici) - line1_ici_mrns
later_line_ici_mrns = set(ever_ici.loc[ever_ici == 1].index) - line1_ici_mrns

print(f"First-line ICI: {len(line1_ici_mrns)}")
print(f"Never ICI:      {len(never_ici_mrns)}")
print(f"Excluded (ICI at later line only): {len(later_line_ici_mrns)}")

# Build the comparison dataset using line 1 records only
ici_df = line1.loc[line1["MRN"].isin(line1_ici_mrns)].copy()
non_ici_df = line1.loc[line1["MRN"].isin(never_ici_mrns)].copy()

IO_prediction_dataset = (pd.concat([ici_df, non_ici_df])[['MRN', 'LOT_start_date', 'PX_on_ICI']]
                         .rename(columns={'MRN': 'DFCI_MRN', 'LOT_start_date': 'treatment_start_date'}))

# --- Load note embeddings ---
notes_meta, embeddings_data = load_note_embeddings()

note_types = ['Clinician', 'Imaging', 'Pathology']
pool_fx = {nt: 'time_decay_mean' for nt in note_types}

# --- Load confounders (demographics, cancer type, panel version) ---
confounders = load_confounders()
confounder_cols = [c for c in confounders.columns if c != 'DFCI_MRN']

# %% [code cell 2]
# Generate embedding datasets for each buffer
buffers = [0, 15, 30, 45]
for buffer in tqdm(buffers, desc="Generating embedding datasets"):
    buffer_path = os.path.join(ICI_DATA_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_path, exist_ok=True)

    notes_meta_sub = (notes_meta[notes_meta['DFCI_MRN'].isin(IO_prediction_dataset['DFCI_MRN'])]
                      .merge(IO_prediction_dataset[['DFCI_MRN', 'treatment_start_date']], on='DFCI_MRN', how='left')
                      .assign(NOTE_TIME_REL_PRED_START_DT=lambda df: (
                          pd.to_datetime(df['NOTE_DATETIME']) - pd.to_datetime(df['treatment_start_date'])).dt.days))

    IO_prediction_embedding_vals = generate_survival_embedding_df(
        notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data,
        note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT",
        max_note_window=-buffer, pool_fx=pool_fx, decay_param=0.01, continuous_window=False)

    full_IO_prediction_dataset = (IO_prediction_dataset
                                  .merge(IO_prediction_embedding_vals.dropna(), on='DFCI_MRN')
                                  .merge(confounders, on='DFCI_MRN'))

    full_IO_prediction_dataset.to_csv(
        os.path.join(buffer_path, f'line_1_ICI_prediction_df_w_{buffer}_day_buffer.csv'), index=False)

# %% [code cell 3]
# Train propensity models with embeddings + confounders
for buffer in tqdm(buffers, desc="Training propensity models"):
    buffer_input_path = os.path.join(ICI_DATA_PATH, f'w_{buffer}_day_buffer/')
    buffer_output_path = os.path.join(ICI_PROP_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_output_path, exist_ok=True)

    full_ICI_pred_df = pd.read_csv(
        os.path.join(buffer_input_path, f'line_1_ICI_prediction_df_w_{buffer}_day_buffer.csv'))

    # Features: embeddings + confounders
    embed_cols = [col for col in full_ICI_pred_df.columns
                  if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]
    confounder_feature_cols = [c for c in confounder_cols if c in full_ICI_pred_df.columns]
    feature_cols = embed_cols + confounder_feature_cols

    X = full_ICI_pred_df[['DFCI_MRN'] + feature_cols]
    y = full_ICI_pred_df[['PX_on_ICI']].astype(int)

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)

    cv_mrns = []
    cv_preds = []
    cv_probs = []
    cv_true = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

        # Split
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        cv_mrns += X_test['DFCI_MRN'].tolist()

        X_train = X_train.drop(columns=['DFCI_MRN'])
        X_test = X_test.drop(columns=['DFCI_MRN'])

        # Scale
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Fit model
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train.values.ravel())

        # Predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        cv_preds += y_pred.tolist()
        cv_probs += y_prob.tolist()
        cv_true += y_test['PX_on_ICI'].tolist()

    # Save predictions with probabilities
    out_df = pd.DataFrame({
        'DFCI_MRN': cv_mrns,
        'model_preds': cv_preds,
        'model_probs': cv_probs,
        'ground_truth': cv_true})

    out_df.to_csv(os.path.join(buffer_output_path, 'line_1_predictions.csv'), index=False)

# %% [code cell 4]
sns.set_style("whitegrid")

buffers = [0, 15, 30, 45]

fig, axes = plt.subplots(1, len(buffers), figsize=(20, 5))

for y, buffer in enumerate(buffers):
    pred_df = pd.read_csv(os.path.join(ICI_PROP_PATH, f'w_{buffer}_day_buffer/line_1_predictions.csv'))

    auc = roc_auc_score(pred_df['ground_truth'], pred_df['model_probs'])
    fpr, tpr, thresholds = roc_curve(pred_df['ground_truth'], pred_df['model_probs'])

    ax = axes[y]
    sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'AUC = {auc : 0.3f}')
    sns.lineplot(x=[0,1], y=[0,1], ax=ax, linestyle='--', color='gray')

    ax.set_title(f'First-line ICI vs Never ICI, buffer = {buffer} days')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
