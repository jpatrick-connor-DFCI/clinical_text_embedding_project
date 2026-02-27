"""ICI propensity score generation: any-line ICI vs never-ICI.

ICI patients are defined by presence in IO_START.csv (any treatment line).
Never-ICI patients are patients in the survival cohort with confirmed
non-ICI treatment records (profile_rxlines.csv) who are not in IO_START.csv.

Time origin for all patients is first_treatment_date from the survival cohort.

Trains logistic regression propensity models using clinical note embeddings
to predict ICI receipt.
"""

# %% [code cell 1]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from biomarker_common import (
    DATA_PATH, SURV_PATH, load_note_embeddings,
)

# Paths
ICI_DATA_PATH = os.path.join(DATA_PATH, 'treatment_prediction/all_ICI_prediction_data/')
ICI_PROP_PATH = os.path.join(DATA_PATH, 'treatment_prediction/all_ICI_propensity/')
os.makedirs(ICI_DATA_PATH, exist_ok=True)
os.makedirs(ICI_PROP_PATH, exist_ok=True)

# --- Define ICI patients from IO_START.csv ---
manual_ICI_start_df = pd.read_csv('/data/gusev/USERS/mjsaleh/IO_START.csv', index_col=0).rename(columns={'MRN': 'DFCI_MRN'})
ici_mrns = set(manual_ICI_start_df['DFCI_MRN'].unique())

# --- Load survival cohort for first_treatment_date ---
surv_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
surv_df['first_treatment_date'] = pd.to_datetime(surv_df['first_treatment_date'])
cohort_mrns = set(surv_df['DFCI_MRN'].unique())

# --- Load treatment records to identify patients with confirmed non-ICI treatment ---
TREATMENT_FILE = '/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv'
treatment_df = pd.read_csv(TREATMENT_FILE)
treated_mrns = set(treatment_df['MRN'].unique())

# --- Classify patients ---
# ICI: patients in IO_START that are also in the survival cohort
ici_mrns_in_cohort = ici_mrns & cohort_mrns
# Never-ICI: survival cohort patients with treatment records but NOT in IO_START
never_ici_mrns = (cohort_mrns & treated_mrns) - ici_mrns

print(f"ICI (IO_START ∩ cohort):      {len(ici_mrns_in_cohort)}")
print(f"Never ICI (treated, no ICI):  {len(never_ici_mrns)}")
print(f"IO_START not in cohort:       {len(ici_mrns - cohort_mrns)}")

# --- Build prediction dataset using first_treatment_date as time origin ---
surv_dates = surv_df[['DFCI_MRN', 'first_treatment_date']].drop_duplicates(subset='DFCI_MRN', keep='first')

ici_df = surv_dates.loc[surv_dates['DFCI_MRN'].isin(ici_mrns_in_cohort)].copy()
ici_df['PX_on_ICI'] = 1

non_ici_df = surv_dates.loc[surv_dates['DFCI_MRN'].isin(never_ici_mrns)].copy()
non_ici_df['PX_on_ICI'] = 0

IO_prediction_dataset = pd.concat([ici_df, non_ici_df])
IO_prediction_dataset = IO_prediction_dataset.rename(columns={'first_treatment_date': 'treatment_start_date'})

# Save prediction times for downstream use (generate_IPTW_df.py, generate_risk_based_df.py)
IO_prediction_dataset.to_csv(os.path.join(ICI_DATA_PATH, 'prediction_times.csv'), index=False)

# --- Load note embeddings ---
notes_meta, embeddings_data = load_note_embeddings()

note_types = ['Clinician', 'Imaging', 'Pathology']
pool_fx = {nt: 'time_decay_mean' for nt in note_types}

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
                                  .merge(IO_prediction_embedding_vals.dropna(), on='DFCI_MRN'))

    full_IO_prediction_dataset.to_csv(
        os.path.join(buffer_path, f'line_1_ICI_prediction_df_w_{buffer}_day_buffer.csv'), index=False)

# %% [code cell 3]
# Train propensity models with embeddings only
for buffer in tqdm(buffers, desc="Training propensity models"):
    buffer_input_path = os.path.join(ICI_DATA_PATH, f'w_{buffer}_day_buffer/')
    buffer_output_path = os.path.join(ICI_PROP_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_output_path, exist_ok=True)

    full_ICI_pred_df = pd.read_csv(
        os.path.join(buffer_input_path, f'line_1_ICI_prediction_df_w_{buffer}_day_buffer.csv'))

    # Features: embeddings only
    embedding_cols = [col for col in full_ICI_pred_df.columns
                      if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]
    feature_cols = embedding_cols

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

    ax.set_title(f'Any-Line ICI vs Never ICI, buffer = {buffer} days')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

plt.tight_layout()
plt.show()
