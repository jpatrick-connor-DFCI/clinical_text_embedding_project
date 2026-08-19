"""Shared utilities for biomarker analysis workflows.

Provides standardized data loading functions for note embeddings, survival
cohort, and confounders (demographics, cancer type, panel version).
"""

import io
import os

import numpy as np
import polars as pl
import zstandard as zstd

from config import NOTES_PATH, SURV_PATH
from pipelines.preprocessing.generate_all_non_text_covariates import (
    build_cancer_type_df,
    build_somatic_data_df,
)


def load_note_embeddings():
    notes_meta = pl.read_parquet(os.path.join(NOTES_PATH, 'full_clinical_notes_embeddings_metadata.parquet'))
    with open(os.path.join(NOTES_PATH, 'full_clinical_notes_embeddings_as_array.npy.zst'), 'rb') as f:
        embeddings_data = np.load(io.BytesIO(zstd.decompress(f.read())))
    embeddings_data = embeddings_data.astype(np.float32)
    return notes_meta, embeddings_data


def load_survival_cohort(cohort_file='death_met_surv_df.parquet'):
    return pl.read_parquet(os.path.join(SURV_PATH, cohort_file))


def load_confounders():
    """Load demographic, cancer type, and panel version data for propensity modeling.

    Cancer type and panel version are built from PROFILE_DATA in-process rather
    than read from the feature CSVs. Unlike generate_IPTW_df.py, this function
    has no line landmark in scope, so the somatic build keeps the default
    treatment anchor; only PANEL_VERSION is taken from it, and which panel a
    patient was sequenced on does not move with the anchor.
    """
    tt_death_df = pl.read_parquet(os.path.join(SURV_PATH, 'death_met_surv_df.parquet'))
    demographics = tt_death_df.select(['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART'])

    full_cohort_df = pl.read_parquet(os.path.join(SURV_PATH, 'cohort_df.parquet'))
    cancer_type_df = build_cancer_type_df(full_cohort_df)

    somatic_df = build_somatic_data_df(full_cohort_df)
    panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
    if not panel_cols:
        raise ValueError(
            "No PANEL_VERSION* column survived build_somatic_data_df; the panel-version "
            "confounder would be silently dropped. Check GENOMIC_SPECIMEN.parquet's column names."
        )
    somatic_df = somatic_df.select(['DFCI_MRN'] + panel_cols).unique(
        subset='DFCI_MRN', keep='first')

    confounders = (demographics
                   .join(cancer_type_df, on='DFCI_MRN', how='inner')
                   .join(somatic_df, on='DFCI_MRN', how='inner'))
    return confounders


MUTATION_TAGS = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')


def get_mutation_type(marker_name):
    """Extract mutation type suffix (e.g., '_SNV', '_AMP') from marker name."""
    marker_upper = marker_name.upper()
    for tag in MUTATION_TAGS:
        if marker_upper.endswith(tag):
            return tag
    return '_OTHER'
