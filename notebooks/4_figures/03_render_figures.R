# Render manuscript figures (R tier)
#
# Runs the R/plot_figure_*.R scripts in this script's R session. Each script
# reads the CSVs written by the Python prep tier and saves individual panels
# plus compiled Figures 1-4 in PNG and PDF. Compiled figures carry lowercase panel
# labels (a, b, c, ...), with one label per named panel, including panels with internal
# facets or subplots. Missing panels are skipped with their reasons logged. Available
# panels retain their original letters, and the compiled figure lists unavailable
# panels in its caption. A figure with no available panels is skipped without
# stopping the remaining figures.
#
# All performance plots use Harrell's C-index. AUC variants and the former Figure 5a
# propensity ROC panel are no longer rendered. Old AUC/ROC image files in the output
# folders are removed when the rendering utilities are loaded.
#
# The main figures contain exactly these panels, in letter order:
#
#  - Figure 1 (a-d): cancer-type pie, stage barplot, cohort availability flowchart,
#    outcome endpoints.
#  - Figure 2 (a-d): base-versus-text C-index scatter, delta violin, stage KM,
#    text-risk quartile KM.
#  - Figure 3 (a-c): modality rank, significant endpoint barplot, joint Cox violins.
#  - Figure 4 (a-d): risk-score heatmap, KM by risk trajectory, risk-dynamics
#    composition by stage, stage I-II rising-risk versus stage IV falling-risk KM.
#
# Within-versus-pan training comparisons and top-event plots (including the top-event
# supplement) are retired. Superseded standalone panel files are removed when each figure rerenders.
#
# The within-cancer supplements evaluate existing pooled models on matched patients
# within each recorded cancer type, using the independent figures.prep.within_cancer
# outputs. S2 plots text versus base C-indices in cancer-type facets, with points
# colored by endpoint family and the median paired difference at lower right.
# S3 is a cancer-by-modality heatmap of median paired text-minus-comparator
# C-index differences; cells report median absolute C-indices and endpoint counts.
# The median paired difference need not equal the difference of the two medians.
# Unavailable combinations are marked explicitly. These descriptive summaries use
# eligible rows and the shared manuscript endpoint filter, without endpoint-level
# significance tests. Each supplement shows up to 12 cancer types per page.
#
# The new scripts write figS2_within_cancer_cindex and figS3_within_cancer_cindex
# PNG/PDF files to the figure2 and figure3 output groups (later pages add _page2,
# etc.), with matching summary CSVs under tables/ and legends under captions/.
# Each also writes a single-page *_within_cancer_selected_cindex version limited to
# Breast, Leukemia, Lung, Bowel, Brain, Skin, Pancreas, Lymphoma, and CUP. S2 also
# writes figS2_within_cancer_selected_v2_cindex: all cancer types except OTHER on a
# single page, 7 panels per row.
# The S3 script also writes mean within-cancer modality-rank heatmaps (as in
# Figure 3a) for all cancer types (figS3_within_cancer_rank_cindex) and the
# selected ones (figS3_within_cancer_rank_selected_cindex), ranking the shared-cohort
# C-indices in fig3_within_cancer_modality_cindex.csv.
# plot_figure_3_supp_cancer_joint.R renders the joint Cox model refitted within each
# selected cancer type (figures.prep.within_cancer_joint): coefficient violins
# (figS3_within_cancer_joint_betas) and significant-endpoint counts
# (figS3_within_cancer_joint_significant) in the figure3 group.
# plot_figure_3_supp_combined.R renders stacked Cox models on the held-out modality
# risk scores (figures.prep.figure3_combined): overall-survival C-indices for each
# modality with and without text and for text / all but text / all, plus the paired
# differences across endpoints (figS3_combined_models, figure3 group).
# plot_figure_2_supp_cancer_km.R renders Figure 2c/d within each selected cancer
# type (figures.prep.within_cancer_km): KM grids by stage
# (figS2_within_cancer_km_stage) and by within-cancer text risk quartile
# (figS2_within_cancer_km_risk) in the figure2 group.
# plot_figure_os_within_cancer.R writes figOS_within_cancer_cindex to the figure_os
# group: the overall-survival (death_met/death) C-index for those cancer types,
# text versus base and versus each other modality.
# Supplemental caption wording is documented in figures/figure_captions.md.
#
# Compiled Figures 1, 2, and 4 render at 10 x 9 inches; Figure 3 renders at
# 10 x 8.2 inches, with 600-dpi PNGs and vector PDFs. Compiled artwork retains panel
# letters and omits individual plot titles; KM curves have no number-at-risk tables.
# Captions identify panels by letter. Each compiled figure also gets a _captioned PNG/PDF review copy;
# standalone legends are written to $CLINICAL_FIGURES_OUT/captions/. Base wording
# is versioned in figures/manuscript_captions.json; current sample sizes and model
# statistics are added at rendering time. The clean artwork remains available for
# journals requiring captions separately.
#
# Code lookups (R): 01_code_lookups.R - one-time bootstrap.
# Prep tier: 02_figure_data.ipynb (Python kernel) - run first.
# One-time R package bootstrap (any host with R):
#   Rscript R/install_packages.R
#
# Each plot script is sys.source()-ed in its own environment so per-script ggplot objects
# (p2a, p3b, ...) don't leak between them. R scripts always assume the working directory
# is the repo root (they source("R/figure_utils.R") with a relative path), so this script
# sets the R working directory to the repo root before sourcing anything.
#
# Render with:
#   Rscript notebooks/4_figures/03_render_figures.R

find_repo_root <- function() {
  d <- normalizePath(getwd(), mustWork = TRUE)
  while (nchar(d) > 1) {
    if (file.exists(file.path(d, "config.py")) &&
        dir.exists(file.path(d, "R"))) return(d)
    parent <- dirname(d)
    if (parent == d) break
    d <- parent
  }
  stop("Could not find repo root from ", getwd())
}

REPO_ROOT <- find_repo_root()
setwd(REPO_ROOT)

## ---- find-root ----

R_DIR <- file.path(REPO_ROOT, "R")

cat("repo root:   ", REPO_ROOT, "\n", sep = "")
cat("R scripts: ", R_DIR, "\n", sep = "")
cat("R version: ", R.version.string, "\n", sep = "")

## ---- render-plots ----

# Reset both legacy controls in an existing R session before sourcing scripts.
# This also keeps older plot-script copies from inheriting an AUC selection.
Sys.setenv(MANUSCRIPT_METRIC = "cindex", MANUSCRIPT_METRICS = "cindex")
cat("C-index-only renderer: ", file.path(R_DIR, "..", "notebooks", "4_figures",
                                      "03_render_figures.R"), "\n", sep = "")

SCRIPTS <- c(
  "plot_figure_0.R",
  "plot_figure_1.R",
  "plot_figure_2.R",
  "plot_figure_2_supp.R",
  "plot_figure_2_supp_anchor.R",
  "plot_figure_2_supp_cancer.R",
  "plot_figure_2_supp_cancer_km.R",
  "plot_figure_2_supp_family.R",
  "plot_figure_3.R",
  "plot_figure_3_supp_cancer.R",
  "plot_figure_3_supp_cancer_joint.R",
  "plot_figure_3_supp_combined.R",
  "plot_figure_4.R",
  "plot_figure_4_supp.R",
  "plot_figure_5.R",
  "plot_figure_os_within_cancer.R"
)

for (script in SCRIPTS) {
  cat("\n=== ", script, " [cindex] ===\n", sep = "")
  env <- new.env(parent = globalenv())
  sys.source(file.path(R_DIR, script), envir = env)
}

cat("\nDone.\n")
cat("Panel PNGs: $CLINICAL_FIGURES_OUT/png/<figureN>/ (or /data/.../manuscript_figures/png/<figureN>/)\n")
cat("Compiled figures: figureN_cindex.png/.pdf alongside each figure's panels\n")
cat("Panel PDFs: $CLINICAL_FIGURES_OUT/pdf/<figureN>/ (or /data/.../manuscript_figures/pdf/<figureN>/)\n")
