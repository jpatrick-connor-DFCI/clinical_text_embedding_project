# Shared utilities for the manuscript-figure R rendering pipeline.
#
# Mirrors _figure_utils.py: paths, palettes, theme, IO, stats helpers, KM helper.
# Each plot script does:  source(file.path(dirname(sys.frame(1)$ofile), "figure_utils.R"))

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(survival)
  library(ggsurvfit)
})

# ----------------------------------------------------------------------------
# Paths (mirror _figure_utils.py; CLINICAL_FIGURES_OUT env var overrides output)
# ----------------------------------------------------------------------------
DATA_PATH      <- "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
SURV_PATH      <- file.path(DATA_PATH, "time-to-event_analysis")
FEATURE_PATH   <- file.path(DATA_PATH, "clinical_and_genomic_features")
RESULTS_PATH   <- file.path(SURV_PATH, "results")
FIGURE_DATA_DIR <- file.path(RESULTS_PATH, "figure_data")

FIGURE_OUT_DIR <- Sys.getenv(
  "CLINICAL_FIGURES_OUT",
  unset = "/data/gusev/USERS/jpconnor/figures/clinical_text_embedding_project/manuscript_figures/"
)
PANEL_OUT_DIR  <- FIGURE_OUT_DIR
TARGET_OUT_DIR <- file.path(FIGURE_OUT_DIR, "target_figures")

# Create output dirs on demand
dir.create(PANEL_OUT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(TARGET_OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ----------------------------------------------------------------------------
# Palettes (ported verbatim from _figure_utils.py / panel modules)
# ----------------------------------------------------------------------------
SCHEME_COLORS  <- c(death_met = "#E74C3C", icd3_post = "#3498DB",
                    icd4_post = "#2ECC71", phecode_post = "#9B59B6")
SCHEME_LABELS  <- c(death_met = "death_met", icd3_post = "ICD3",
                    icd4_post = "ICD4", phecode_post = "PhecodeX")
SCHEME_SHAPES  <- c(death_met = 18, icd3_post = 16, icd4_post = 17, phecode_post = 15)

MODALITY_ORDER  <- c("stage", "treatment", "somatic", "prs", "text")
MODALITY_COLORS <- c(stage = "#8C8C8C", treatment = "#E8B72E",
                    somatic = "#60BD68", prs = "#B276B2", text = "#D62728")
MODALITY_DISPLAY <- c(stage = "Stage", treatment = "Treatment",
                     somatic = "Somatic", prs = "PRS", text = "Text")

CLUSTER_COLORS <- c("#1F77B4", "#FF7F0E", "#2CA02C", "#D62728",
                    "#9467BD", "#8C564B", "#E377C2", "#7F7F7F")

MODEL_COLORS <- c(text = "#D62728", base = "#4A4A4A")
RISK_COLORS  <- c(low = "#2E86C1", mid = "#F28E2B", high = "#E74C3C")
ORDINAL4     <- c("#2E86C1", "#58A55C", "#F28E2B", "#E74C3C")

BENEFIT_COLOR <- "#2E6F9E"
HARM_COLOR    <- "#E76F51"
TEAL          <- "#2A9D8F"
NS_GRAY       <- "#999999"
LIGHT_GRAY    <- "#EAEAEA"

# Cohort labels / definitions (Fig 5)
COHORT_LABELS <- c(cohort1 = "Cohort 1", cohort2 = "Cohort 2")
COHORT_SHORT  <- c(cohort1 = "first-line, unmatched",
                   cohort2 = "lines 1-3, 1:1 matched")
COHORT_DEFS   <- c(
  cohort1 = "Cohort 1: first-line ICI vs. all never-ICI controls (unmatched, discovery)",
  cohort2 = "Cohort 2: ICI lines 1-3 vs. 1:1 matched controls (matched, validation)"
)


# ----------------------------------------------------------------------------
# Theme (ports apply_style())
# ----------------------------------------------------------------------------
theme_manuscript <- function(base_size = 10) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title       = element_text(size = base_size + 1, face = "bold"),
      axis.title       = element_text(size = base_size),
      axis.text        = element_text(size = base_size - 1),
      legend.title     = element_text(size = base_size - 1),
      legend.text      = element_text(size = base_size - 2),
      legend.background = element_blank(),
      legend.key       = element_blank(),
      strip.background = element_rect(fill = "#F5F5F5", color = NA),
      strip.text       = element_text(size = base_size - 1, face = "bold"),
      plot.margin      = margin(4, 6, 4, 6)
    )
}


# ----------------------------------------------------------------------------
# IO helpers (ggsave produces pdf.fonttype=42 by default via cairo_pdf)
# ----------------------------------------------------------------------------
load_figure_data <- function(name) {
  fp <- file.path(FIGURE_DATA_DIR, name)
  if (!file.exists(fp)) {
    warning(sprintf("Missing figure data: %s", fp))
    return(tibble::tibble())
  }
  suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
}

save_panel <- function(plot, name, width = 6.0, height = 4.8) {
  out_png <- file.path(PANEL_OUT_DIR, paste0(name, ".png"))
  ggsave(out_png, plot, width = width, height = height, dpi = 300,
         bg = "white")
  message(sprintf("[panel] %s", out_png))
  invisible(out_png)
}

save_figure <- function(plot, stem, width = 12.0, height = 10.0) {
  out_png <- file.path(TARGET_OUT_DIR, paste0(stem, ".png"))
  out_pdf <- file.path(TARGET_OUT_DIR, paste0(stem, ".pdf"))
  ggsave(out_png, plot, width = width, height = height, dpi = 300,
         bg = "white")
  ggsave(out_pdf, plot, width = width, height = height,
         device = cairo_pdf, bg = "white")
  message(sprintf("[composed] %s", out_png))
  message(sprintf("[composed] %s", out_pdf))
  invisible(c(out_png, out_pdf))
}

placeholder_panel <- function(msg) {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = msg, color = "#777777",
             hjust = 0.5, vjust = 0.5, size = 3) +
    theme_void() +
    xlim(0, 1) + ylim(0, 1)
}


# ----------------------------------------------------------------------------
# Stats helpers (significance stars + Wilcoxon/Kruskal vs 0 / omnibus)
# ----------------------------------------------------------------------------
p_to_stars <- function(p) {
  if (is.null(p) || length(p) == 0 || is.na(p) || !is.finite(p)) return("n/a")
  if (p < 1e-4) return("****")
  if (p < 1e-3) return("***")
  if (p < 1e-2) return("**")
  if (p < 5e-2) return("*")
  "ns"
}

wilcoxon_vs0 <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) < 2 || length(unique(x)) == 1) return(NA_real_)
  tryCatch(
    suppressWarnings(stats::wilcox.test(x, mu = 0)$p.value),
    error = function(e) NA_real_
  )
}

kruskal_p <- function(groups) {
  groups <- Filter(function(g) length(g) > 0, groups)
  if (length(groups) < 2) return(NA_real_)
  tryCatch(
    suppressWarnings(stats::kruskal.test(groups)$p.value),
    error = function(e) NA_real_
  )
}


# ----------------------------------------------------------------------------
# Survival helper — thin wrapper so all KM panels share one entry point
# ----------------------------------------------------------------------------
build_survfit <- function(df, time_col, event_col, group_col = NULL,
                          start_col = NULL) {
  if (nrow(df) == 0) return(NULL)
  if (!is.null(group_col)) df[[group_col]] <- factor(df[[group_col]])
  if (is.null(start_col)) {
    surv_expr <- substitute(survival::Surv(t, e),
                            list(t = as.name(time_col), e = as.name(event_col)))
  } else {
    surv_expr <- substitute(survival::Surv(s, t, e),
                            list(s = as.name(start_col), t = as.name(time_col),
                                 e = as.name(event_col)))
  }
  rhs <- if (is.null(group_col)) "1" else group_col
  f <- stats::as.formula(paste(deparse(surv_expr), "~", rhs))
  ggsurvfit::survfit2(f, data = df)
}

# ----------------------------------------------------------------------------
# Scheme-formatting (Fig 5 spec parser)
# ----------------------------------------------------------------------------
spec_cohort <- function(spec) sub("\\|.*$", "", spec)

pretty_spec <- function(spec) {
  parts <- strsplit(as.character(spec), "\\|")[[1]]
  if (length(parts) == 3) {
    ps <- if (grepl("embedding", parts[2], ignore.case = TRUE)) "embed PS" else "covar PS"
    return(paste0(parts[3], "\n(", ps, ")"))
  }
  gsub("_", " ", as.character(spec))
}

pretty_model <- function(model) {
  switch(as.character(model),
         covariates_only = "Covariates only",
         covariates_plus_embeddings = "Text + covariates",
         gsub("_", " ", as.character(model)))
}

cohort_label <- function(cohort) {
  unname(ifelse(cohort %in% names(COHORT_LABELS), COHORT_LABELS[cohort], cohort))
}
