# Shared utilities for the manuscript-figure R rendering pipeline.
#
# Mirrors figures/io.py: paths, palettes, theme, IO, stats helpers, KM helper.
# Each plot script does:  source("R/figure_utils.R")   (run from the repo root)

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(survival)
  library(ggsurvfit)
})

source("R/config.R")

# ----------------------------------------------------------------------------
# Metric switch — lets a single plot script render two parallel figure sets,
# one scored by Harrell's C-index, one by mean time-dependent AUC(t). Set the
# MANUSCRIPT_METRIC env var to "cindex" or "auc" before sourcing/Rscript-ing a
# plot_figure_*.R script; defaults to "auc" for interactive/unset use.
# ----------------------------------------------------------------------------
METRIC <- tolower(Sys.getenv("MANUSCRIPT_METRIC", unset = "auc"))
if (!METRIC %in% c("cindex", "auc")) {
  warning(sprintf("Unrecognized MANUSCRIPT_METRIC=%s; falling back to 'auc'", METRIC))
  METRIC <- "auc"
}

# Human-readable label for the active metric ("C-index" / "Mean AUC(t)").
metric_label <- function(metric = METRIC) {
  c(cindex = "C-index", auc = "Mean AUC(t)")[[metric]]
}

# Column-name suffix for the active metric ("cindex" / "auc"), matching the
# `{scheme}_{metric}` naming convention used in the metric-parameterized CSVs.
metric_suffix <- function(metric = METRIC) metric

# File/panel-name suffix for the active metric output, e.g. "figure2_..._cindex.png".
metric_tag <- function(metric = METRIC) paste0("_", metric)


# ----------------------------------------------------------------------------
# Palettes — MODALITY_*, MODEL_COLORS, CLUSTER_COLORS are the single source of
# truth in shared/palette.json, read identically by Python (shared/palette.py)
# and R (here), so the two can no longer silently drift. Everything else below
# (scheme/cohort/risk colors, etc.) has no Python-side consumer and stays local.
# ----------------------------------------------------------------------------
.palette <- jsonlite::fromJSON(file.path("shared", "palette.json"))
MODALITY_ORDER   <- .palette$MODALITY_ORDER
MODALITY_COLORS  <- unlist(.palette$MODALITY_COLORS)
MODALITY_DISPLAY <- unlist(.palette$MODALITY_DISPLAY)
MODEL_COLORS     <- unlist(.palette$MODEL_COLORS)
CLUSTER_COLORS   <- unlist(.palette$CLUSTER_COLORS)

SCHEME_COLORS  <- c(death_met = "#E74C3C", icd3_post = "#3498DB",
                    icd4_post = "#2ECC71", phecode_post = "#9B59B6")
SCHEME_LABELS  <- c(death_met = "Death + Mets", icd3_post = "ICD10 (Level 3)",
                    icd4_post = "ICD10 (Level 4)", phecode_post = "PhecodeX")
SCHEME_SHAPES  <- c(death_met = 18, icd3_post = 16, icd4_post = 17, phecode_post = 15)

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
MANUSCRIPT_BASE_SIZE <- 12
# geom_text()/annotate("text") sizes are in mm, unlike theme text sizes.
MANUSCRIPT_TEXT_SIZE <- 3.6
MANUSCRIPT_SMALL_TEXT_SIZE <- 3.2
MANUSCRIPT_CAPTION_SIZE <- 8

theme_manuscript <- function(base_size = MANUSCRIPT_BASE_SIZE) {
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
# IO helpers
# ----------------------------------------------------------------------------
load_figure_data <- function(name) {
  fp <- file.path(FIGURE_DATA_DIR, name)
  if (!file.exists(fp)) {
    warning(sprintf("Missing figure data: %s", fp))
    return(tibble::tibble())
  }
  suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
}

# Individual panels only (no composed target figures), grouped by figure under
# png/<group>/ and pdf/<group>/; save_panel() writes both formats for every
# panel and creates the group subdirectory on demand. `group` (e.g. "figure1")
# is a required argument at every save_panel() call site — not a script-level
# default — since figure_utils.R is source()'d into globalenv() (source()'s
# default local=FALSE), so a lexical default couldn't see a caller-defined
# constant from the sys.source()-ed plot script's own environment.
save_panel <- function(plot, name, group, width = 6.0, height = 4.8) {
  # A skipped panel (no underlying data) is not written at all -- an absent file
  # is a clearer signal to downstream assembly than a page reading "csv empty".
  if (is_skipped_panel(plot)) {
    message(sprintf("[panel] SKIPPED %s/%s - %s", group, name, skip_reason(plot)))
    return(invisible(character(0)))
  }
  # A NULL here means a panel was built by adding layers to a skipped sentinel
  # (see the dispatch note by placeholder_panel); fail loudly rather than
  # letting ggsave report something unrelated.
  if (is.null(plot)) {
    stop(sprintf("save_panel(%s/%s): plot is NULL - a skipped panel was likely passed to `+`",
                 group, name))
  }
  png_dir <- file.path(PNG_OUT_DIR, group)
  pdf_dir <- file.path(PDF_OUT_DIR, group)
  dir.create(png_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(pdf_dir, showWarnings = FALSE, recursive = TRUE)
  out_png <- file.path(png_dir, paste0(name, ".png"))
  out_pdf <- file.path(pdf_dir, paste0(name, ".pdf"))
  ggsave(out_png, plot, width = width, height = height, dpi = 300, bg = "white")
  ggsave(out_pdf, plot, width = width, height = height, bg = "white")
  message(sprintf("[panel] %s / %s", out_png, out_pdf))
  invisible(c(out_png, out_pdf))
}

# A panel with no data to draw. Returns a sentinel rather than a ggplot: rather
# than emitting a figure whose only content is an error string, save_panel()
# declines to write the file and logs why. Composition helpers below drop these
# from multi-panel grids, so a partially-available figure still renders the
# panels it does have.
placeholder_panel <- function(msg) {
  structure(list(reason = msg), class = "skipped_panel")
}

is_skipped_panel <- function(x) inherits(x, "skipped_panel")

skip_reason <- function(x) if (is_skipped_panel(x)) x$reason else NA_character_

# NOTE: a sentinel must never be fed to `+`. No `+.skipped_panel` method can fix
# this -- the right operand (labs(), theme(), ...) carries ggplot2's "gg" class,
# so both that method and `+.gg` apply, and R resolves the ambiguity by warning
# "incompatible methods" and returning NULL. Every `p <- p + ...` site in the
# plot scripts therefore sits after its function's early return, and any site
# decorating a maybe-skipped panel must call compact_panels() first (see
# build_fig5d). Check with is_skipped_panel() before adding layers.

# Drop skipped panels from a list bound for wrap_plots(); returns NULL when
# nothing is left to draw, so the caller can propagate a single skip upward.
compact_panels <- function(panels) {
  kept <- Filter(function(p) !is_skipped_panel(p), panels)
  if (length(kept) == 0) return(NULL)
  kept
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

# Tidy a survfit2 object into a long, step-ready data frame (adds a clean `stratum` col).
tidy_km <- function(fit) {
  if (is.null(fit)) return(tibble::tibble())
  td <- ggsurvfit::tidy_survfit(fit)
  if ("strata" %in% names(td)) td$stratum <- sub("^[^=]+=", "", as.character(td$strata))
  td
}

# Multivariate log-rank p (survdiff on a Surv(time, event) ~ group formula).
logrank_p <- function(df, time_col, event_col, group_col) {
  if (nrow(df) == 0) return(NA_real_)
  f <- as.formula(sprintf("Surv(%s, %s) ~ %s", time_col, event_col, group_col))
  sd <- tryCatch(survival::survdiff(f, data = df), error = function(e) NULL)
  if (is.null(sd)) return(NA_real_)
  if (!is.null(sd$pvalue)) return(sd$pvalue)
  stats::pchisq(sd$chisq, df = length(sd$n) - 1, lower.tail = FALSE)
}

# Expand a tidy_survfit frame into right-continuous KM "stairs" so a 95% CI band can be
# drawn as a true step (geom_rect spanning each event time to the next), per group.
# Returns an empty frame if CI columns are absent.
step_ci_df <- function(td, group_cols) {
  if (!all(c("conf.low", "conf.high", "time") %in% names(td)) || nrow(td) == 0) {
    return(td[0, , drop = FALSE])
  }
  td %>%
    dplyr::filter(!is.na(conf.low), !is.na(conf.high)) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(group_cols))) %>%
    dplyr::arrange(time, .by_group = TRUE) %>%
    dplyr::mutate(time_next = dplyr::lead(time, default = dplyr::last(time))) %>%
    dplyr::ungroup()
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
