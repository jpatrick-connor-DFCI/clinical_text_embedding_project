"""Shared utilities for biomarker analysis workflows.

Provides standardized data loading functions for note embeddings, survival
cohort, and confounders (demographics, cancer type, panel version).
"""

import os
import numpy as np
import pandas as pd


# Shared paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')


def load_note_embeddings():
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
    embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
    return notes_meta, embeddings_data


def load_survival_cohort(cohort_file='death_met_surv_df.csv'):
    return pd.read_csv(os.path.join(SURV_PATH, cohort_file))


def load_confounders():
    """Load demographic, cancer type, and panel version data for propensity modeling."""
    tt_death_df = pd.read_csv(os.path.join(SURV_PATH, 'death_met_surv_df.csv'))
    demographics = tt_death_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

    cancer_type_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv'))

    somatic_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv'))
    panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
    somatic_df = somatic_df[['DFCI_MRN'] + panel_cols].copy()

    confounders = (demographics
                   .merge(cancer_type_df, on='DFCI_MRN')
                   .merge(somatic_df, on='DFCI_MRN'))
    return confounders
