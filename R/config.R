# Single source of truth for REPO_ROOT, DATA_PATH and paths derived from them, the R
# counterpart to config.py. Every R entry point sources this file by its absolute
# path, and every other repo file is reached through REPO_ROOT, so scripts run from
# any working directory.

REPO_ROOT <- "/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project"

DATA_PATH <- Sys.getenv("CTEP_DATA_PATH", "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/")

CODE_PATH       <- file.path(DATA_PATH, "code_data")
SURV_PATH       <- file.path(DATA_PATH, "time-to-event_analysis")
FEATURE_PATH    <- file.path(DATA_PATH, "clinical_and_genomic_features")
RESULTS_PATH    <- file.path(SURV_PATH, "results")
# Prepared figure-data CSVs, written by figures/prep/* and read by the plot
# scripts here. Must stay in lockstep with FIGURE_DATA_DIR in config.py --
# including the CTEP_FIGURE_DATA_DIR override, or an override set for the
# Python prep tier would leave the R tier reading the old location.
FIGURE_DATA_DIR <- Sys.getenv(
  "CTEP_FIGURE_DATA_DIR",
  unset = file.path(DATA_PATH, "figure_data")
)

# Figure rendering output (Python figure-data prep + R plotting share this).
FIGURE_OUT_DIR <- Sys.getenv(
  "CLINICAL_FIGURES_OUT",
  unset = "/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/manuscript_figures/"
)
PNG_OUT_DIR <- file.path(FIGURE_OUT_DIR, "png")
PDF_OUT_DIR <- file.path(FIGURE_OUT_DIR, "pdf")
