import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df

# Paths
DATA_PATH = "/data/gusev/USERS/jpconnor/clinical_text_project/data/"
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/VTE_data/processed_datasets/")
ICI_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/line_ICI_prediction_data/')

# --- Load cohort ---
treatment_df = pd.read_csv("/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv")
tt_phecode_df = pd.read_csv(os.path.join(SURV_PATH, "time-to-phecode/tt_vte_plus_phecodes.csv"))

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

for line in tqdm(ici_sets):
    IO_prediction_dataset = (pd.concat([ici_sets[line]['ICI'], ici_sets[line]['non-ICI']])[['MRN', 'LOT_start_date', 'PX_on_IO']]
                             .rename(columns={'MRN' : 'DFCI_MRN', 'LOT_start_date' : 'treatment_start_date'}))
    
    notes_meta_sub = (notes_meta[notes_meta['DFCI_MRN'].isin(IO_prediction_dataset['DFCI_MRN'])]
                      .merge(IO_prediction_dataset[['DFCI_MRN', 'treatment_start_date']], on='DFCI_MRN', how='left')
                      .assign(NOTE_TIME_REL_PRED_START_DT = lambda df: (
                          pd.to_datetime(df['NOTE_DATETIME']) - pd.to_datetime(df['treatment_start_date'])).dt.days))
    
    IO_prediction_embedding_vals = generate_survival_embedding_df(notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data, 
                                                                  note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT", 
                                                                  max_note_window=0, pool_fx=pool_fx, decay_param=0.01, continuous_window=True)
    
    full_IO_prediction_dataset = IO_prediction_dataset.merge(IO_prediction_embedding_vals.dropna(), on='DFCI_MRN')

    full_IO_prediction_dataset.to_csv(os.path.join(ICI_PRED_PATH, f'line_{line}_ICI_prediction_df.csv'), index=False)