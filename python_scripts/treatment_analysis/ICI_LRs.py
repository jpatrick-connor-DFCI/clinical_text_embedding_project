"""Ici Lrs script for treatment analysis workflows."""

# Auto-generated from ICI_LRs.ipynb

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

# Paths
DATA_PATH = "/data/gusev/USERS/jpconnor/clinical_text_project/data/"
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/VTE_data/processed_datasets/")
ICI_DATA_PATH = os.path.join(DATA_PATH, 'treatment_prediction/line_ICI_prediction_data/')
ICI_PROP_PATH = os.path.join(DATA_PATH, 'treatment_prediction/ICI_propensity/')

# --- Load cohort ---
treatment_df = pd.read_csv("/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv")
tt_phecode_df = pd.read_csv(os.path.join(SURV_PATH, "phecode_surv_df.csv"))

cohort_treatment_df = (
    treatment_df.loc[treatment_df["MRN"].isin(tt_phecode_df["DFCI_MRN"].unique())]
    .copy()
)

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
cohort_treatment_df["LOT_start_date"] = pd.to_datetime(cohort_treatment_df["LOT_start_date"])
cohort_treatment_df = cohort_treatment_df.sort_values(["MRN", "LOT_start_date"])
cohort_treatment_df["treatment_line"] = cohort_treatment_df.groupby("MRN").cumcount() + 1
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

# Track whether each patient has had ICI up to the *previous* line
cohort_treatment_df["ever_ici_prior"] = (
    cohort_treatment_df.groupby("MRN")["PX_on_ICI"]
    .cumsum()
    .shift(fill_value=0)
)

# Now stratify by line
ici_sets = {}
for line, df_line in cohort_treatment_df.groupby("treatment_line"):
    # ICI group: first exposure at this line
    ici_df = df_line[(df_line["PX_on_ICI"] == 1) & (df_line["ever_ici_prior"] == 0)].copy()
    
    # non-ICI group: no ICI so far
    non_ici_df = df_line[(df_line["PX_on_ICI"] == 0) & (df_line["ever_ici_prior"] == 0)].copy()
    
    ici_sets[line] = {"ICI": ici_df, "non-ICI": non_ici_df}

notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

note_types = ['Clinician', 'Imaging', 'Pathology']
pool_fx = {nt: 'time_decay_mean' for nt in note_types}

# %% [code cell 2]
buffers = [0, 30, 60, 90]
for buffer in buffers:
    buffer_path = os.path.join(ICI_DATA_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_path, exist_ok=True)
    for line in tqdm(range(1,5)):
        IO_prediction_dataset = (pd.concat([ici_sets[line]['ICI'], ici_sets[line]['non-ICI']])[['MRN', 'LOT_start_date', 'PX_on_ICI']]
                                 .rename(columns={'MRN' : 'DFCI_MRN', 'LOT_start_date' : 'treatment_start_date'}))

        notes_meta_sub = (notes_meta[notes_meta['DFCI_MRN'].isin(IO_prediction_dataset['DFCI_MRN'])]
                          .merge(IO_prediction_dataset[['DFCI_MRN', 'treatment_start_date']], on='DFCI_MRN', how='left')
                          .assign(NOTE_TIME_REL_PRED_START_DT = lambda df: (
                              pd.to_datetime(df['NOTE_DATETIME']) - pd.to_datetime(df['treatment_start_date'])).dt.days))

        IO_prediction_embedding_vals = generate_survival_embedding_df(notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data, 
                                                                      note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT", 
                                                                      max_note_window=-buffer, pool_fx=pool_fx, decay_param=0.01, continuous_window=False)

        full_IO_prediction_dataset = IO_prediction_dataset.merge(IO_prediction_embedding_vals.dropna(), on='DFCI_MRN')

        full_IO_prediction_dataset.to_csv(os.path.join(buffer_path, f'line_{line}_ICI_prediction_df_w_{buffer}_day_buffer.csv'), index=False)

for buffer in [0, 30, 60, 90]:
    buffer_input_path = os.path.join(ICI_DATA_PATH, f'w_{buffer}_day_buffer/')
    buffer_output_path = os.path.join(ICI_PROP_PATH, f'w_{buffer}_day_buffer/')
    os.makedirs(buffer_output_path, exist_ok=True)
    
    for pred_file in tqdm(os.listdir(buffer_input_path)):
        full_ICI_pred_df = pd.read_csv(os.path.join(buffer_input_path, pred_file))

        X = full_ICI_pred_df[['DFCI_MRN'] + [col for col in full_ICI_pred_df.columns 
                                             if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]]
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
            y_prob = clf.predict_proba(X_test)[:, 1]  # probability of class 1

            cv_preds += y_pred.tolist()
            cv_probs += y_prob.tolist()
            cv_true += y_test['PX_on_ICI'].tolist()

        # Save predictions with probabilities
        out_df = pd.DataFrame({
            'DFCI_MRN': cv_mrns,
            'model_preds': cv_preds,
            'model_probs': cv_probs,
            'ground_truth': cv_true})
        
        out_file = os.path.join(buffer_output_path, f'line_{pred_file.split("_")[1]}_predictions.csv')
        out_df.to_csv(out_file, index=False)

sns.set_style("whitegrid")

# %% [code cell 5]
buffers = [0, 30, 60]
lines = [1, 2, 3]

fig, axes = plt.subplots(len(lines), len(buffers), figsize=(16,16))

for y, buffer in enumerate(buffers):
    for x, line in enumerate(lines):
        pred_df = pd.read_csv(os.path.join(ICI_PROP_PATH, f'w_{buffer}_day_buffer/line_{line}_predictions.csv'))
        
        auc = roc_auc_score(pred_df['ground_truth'], pred_df['model_probs'])
        fpr, tpr, thresholds = roc_curve(pred_df['ground_truth'], pred_df['model_probs'])
        
        ax = axes[x, y]
        sns.lineplot(x=fpr, y=tpr, ax=ax, label=f'AUC = {auc : 0.3f}')
        sns.lineplot(x=[0,1], y=[0,1], ax=ax, linestyle='--', color='gray')
        
        ax.set_title(f'line = {line}, buffer = {buffer} days')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc='lower right')
        
plt.tight_layout()
plt.show()
