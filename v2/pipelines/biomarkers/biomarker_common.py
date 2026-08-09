"""Shared utilities for biomarker analysis workflows.

Provides standardized data loading functions for note embeddings, survival
cohort, and confounders (demographics, cancer type, panel version).
"""

import io
import os

import numpy as np
import pandas as pd
import zstandard as zstd

from config import DATA_PATH, NOTES_PATH, SURV_PATH


def load_note_embeddings():
    notes_meta = pd.read_parquet(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.parquet'))
    with open(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy.zst'), 'rb') as f:
        embeddings_data = np.load(io.BytesIO(zstd.decompress(f.read())))
    embeddings_data = embeddings_data.astype(np.float32)
    return notes_meta, embeddings_data


def load_survival_cohort(cohort_file='death_met_surv_df.parquet'):
    return pd.read_parquet(os.path.join(SURV_PATH, cohort_file))


def load_confounders():
    """Load demographic, cancer type, and panel version data for propensity modeling."""
    tt_death_df = pd.read_parquet(os.path.join(SURV_PATH, 'death_met_surv_df.parquet'))
    demographics = tt_death_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

    cancer_type_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv.gz'))

    somatic_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv.gz'))
    panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
    somatic_df = somatic_df[['DFCI_MRN'] + panel_cols].copy()

    confounders = (demographics
                   .merge(cancer_type_df, on='DFCI_MRN')
                   .merge(somatic_df, on='DFCI_MRN'))
    return confounders


MUTATION_TAGS = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')


def get_mutation_type(marker_name):
    """Extract mutation type suffix (e.g., '_SNV', '_AMP') from marker name."""
    marker_upper = marker_name.upper()
    for tag in MUTATION_TAGS:
        if marker_upper.endswith(tag):
            return tag
    return '_OTHER'
