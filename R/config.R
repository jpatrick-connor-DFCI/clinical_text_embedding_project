# Single source of truth for DATA_PATH and paths derived from it, the R
# counterpart to config.py. Scripts are run from the repo root (e.g.
# `Rscript pipelines/preprocessing/generate_icd10_to_phecode_mapping.R` or
# `Rscript R/plot_figure_2.R`), so this relative `source("R/config.R")`
# resolves from that working directory.

DATA_PATH <- Sys.getenv("CTEP_DATA_PATH", "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/")

CODE_PATH       <- file.path(DATA_PATH, "code_data")
SURV_PATH       <- file.path(DATA_PATH, "time-to-event_analysis")
FEATURE_PATH    <- file.path(DATA_PATH, "clinical_and_genomic_features")
RESULTS_PATH    <- file.path(SURV_PATH, "results")
FIGURE_DATA_DIR <- file.path(RESULTS_PATH, "figure_data")

# Figure rendering output (Python figure-data prep + R plotting share this).
FIGURE_OUT_DIR <- Sys.getenv(
  "CLINICAL_FIGURES_OUT",
  unset = "/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/manuscript_figures/"
)
PNG_OUT_DIR <- file.path(FIGURE_OUT_DIR, "png")
PDF_OUT_DIR <- file.path(FIGURE_OUT_DIR, "pdf")
