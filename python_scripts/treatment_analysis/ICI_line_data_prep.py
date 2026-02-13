"""Ici Line Data Prep script for treatment analysis workflows."""

import os
import pandas as pd
from tqdm import tqdm
from embed_surv_utils import generate_survival_embedding_df
from treatment_analysis_common import (
    DATA_PATH,
    load_note_embeddings,
    load_vte_cohort_treatments,
    add_treatment_line_columns,
)

# Paths
ICI_PRED_PATH = os.path.join(DATA_PATH,'treatment_prediction/line_ICI_prediction_data/')

# --- Load cohort ---
cohort_treatment_df = load_vte_cohort_treatments()

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
cohort_treatment_df = add_treatment_line_columns(
    cohort_treatment_df, mrn_col='MRN', start_col='LOT_start_date'
)
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

notes_meta, embeddings_data = load_note_embeddings()

note_types = ['Clinician', 'Imaging', 'Pathology']
pool_fx = {nt: 'time_decay_mean' for nt in note_types}

for line in tqdm(ici_sets):
    ICI_prediction_dataset = (pd.concat([ici_sets[line]['ICI'], ici_sets[line]['non-ICI']])[['MRN', 'LOT_start_date', 'PX_on_ICI']]
                              .rename(columns={'MRN' : 'DFCI_MRN', 'LOT_start_date' : 'treatment_start_date'}))

    notes_meta_sub = (notes_meta[notes_meta['DFCI_MRN'].isin(ICI_prediction_dataset['DFCI_MRN'])]
                      .merge(ICI_prediction_dataset[['DFCI_MRN', 'treatment_start_date']], on='DFCI_MRN', how='left')
                      .assign(NOTE_TIME_REL_PRED_START_DT = lambda df: (
                          pd.to_datetime(df['NOTE_DATETIME']) - pd.to_datetime(df['treatment_start_date'])).dt.days))

    ICI_prediction_embedding_vals = generate_survival_embedding_df(notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data,
                                                                   note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT",
                                                                   max_note_window=0, pool_fx=pool_fx, decay_param=0.01, continuous_window=True)

    full_ICI_prediction_dataset = ICI_prediction_dataset.merge(ICI_prediction_embedding_vals.dropna(), on='DFCI_MRN')

    full_ICI_prediction_dataset.to_csv(os.path.join(ICI_PRED_PATH, f'line_{line}_ICI_prediction_df.csv'), index=False)
