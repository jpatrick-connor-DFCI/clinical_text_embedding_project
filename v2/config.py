"""The path module. DATA_PATH and everything derived from it, defined once.

Every path below was previously copy-pasted, verbatim, across 12+ files
(slurm_array_utils.py, biomarker_common.py, _figure_utils.py,
extract_ICD_times.py, generate_all_non_text_covariates.py,
generate_embedding_prediction_datasets.py, build_line_matched_cohort.py, ...).
Import from here instead of redefining.

DATA_PATH is overridable via the CTEP_DATA_PATH env var for local/dev use;
every pipeline script otherwise assumes the cluster layout below.
"""
import os

DATA_PATH = os.environ.get(
    "CTEP_DATA_PATH",
    "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/",
)

# --- Derived from DATA_PATH ---
CODE_PATH = os.path.join(DATA_PATH, "code_data/")
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")
FEATURE_PATH = os.path.join(DATA_PATH, "clinical_and_genomic_features/")
NOTES_PATH = os.path.join(DATA_PATH, "batched_datasets/processed_datasets/")
RESULTS_PATH = os.path.join(SURV_PATH, "results/")
FIGURE_DATA_DIR = os.path.join(RESULTS_PATH, "figure_data/")
BIOMARKER_PATH = os.path.join(DATA_PATH, "biomarker_analysis/")
MATCHED_COHORT_PATH = os.path.join(BIOMARKER_PATH, "matched_cohorts/")
MED_CLASSES_FILE = os.path.join(DATA_PATH, "GPT_generated_med_classes.csv")

# --- Figure rendering output (Python figure-data prep + R plotting share this) ---
FIGURE_OUT_DIR = os.environ.get(
    "CLINICAL_FIGURES_OUT",
    "/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/manuscript_figures/",
)
PNG_OUT_DIR = os.path.join(FIGURE_OUT_DIR, "png/")
PDF_OUT_DIR = os.path.join(FIGURE_OUT_DIR, "pdf/")

# --- PROFILE clinical source data (read-only inputs owned by Gusev lab, not this project) ---
PROFILE_PATH = "/data/gusev/PROFILE/CLINICAL/"
# Still imported by pipelines/biomarkers/build_line_matched_cohort.py, and by
# pipelines/trajectories/within_treatment_vs_pan_treatment_models.py and
# within_vs_pan_cancer_models.py (out of scope for the PROFILE_DATA migration),
# so these two are kept even though the migrated preprocessing pipeline no
# longer reads them.
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, "robust_VTE_pred_project_2025_03_cohort/data/")

# --- PROFILE_DATA: multi-release-deduped parquets produced by PROFILE_data_processing ---
PROFILE_DATA_PATH = os.environ.get(
    "PROFILE_DATA_PATH",
    "/data/gusev/USERS/jpconnor/data/PROFILE_DATA/",
)
CLINICAL_NOTES_PATH = os.path.join(PROFILE_DATA_PATH, "CLINICAL_NOTES/")
CANCER_ANNOTATIONS_PATH = os.path.join(PROFILE_DATA_PATH, "CANCER_ANNOTATIONS/")

# --- Third-party inputs, previously inlined mid-file at the point of use ---
# MED_LINES_FILE: still imported by pipelines/biomarkers/build_line_matched_cohort.py
# (out of scope for the PROFILE_DATA migration), so kept despite being unused
# by the migrated preprocessing pipeline.
MED_LINES_FILE = "/data/gusev/USERS/mjsaleh/profile_lines_of_rx/ALL_MEDICATION_LINES.csv"
IO_START_FILE = "/data/gusev/USERS/mjsaleh/IO_START.csv"
TREATMENT_FILE = "/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv"
PRS_MATRIX_FILE = "/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv"

# --- Recurrent Mets project (external, upstream of embedding-prediction dataset build) ---
METS_PROJECT = "/data/gusev/Recurrent_Mets_Project/"
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, "clinical_to_ag/")
