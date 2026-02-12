import os
import numpy as np
import pandas as pd


# Shared paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')


def load_note_embeddings():
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
    embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
    return notes_meta, embeddings_data


def load_vte_cohort_treatments(
    treatment_file='/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv',
    cohort_file='phecode_surv_df.csv',
):
    treatment_df = pd.read_csv(treatment_file)
    cohort_df = pd.read_csv(os.path.join(SURV_PATH, cohort_file), usecols=['DFCI_MRN'])

    cohort_treatment_df = treatment_df.loc[
        treatment_df['MRN'].isin(cohort_df['DFCI_MRN'].unique())
    ].copy()
    return cohort_treatment_df


def add_treatment_line_columns(
    df,
    mrn_col='MRN',
    start_col='LOT_start_date',
):
    out = df.copy()
    out[start_col] = pd.to_datetime(out[start_col])
    out = out.sort_values([mrn_col, start_col])
    out['treatment_line'] = out.groupby(mrn_col).cumcount() + 1
    return out
