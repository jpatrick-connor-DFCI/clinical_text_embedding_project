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
source("R/figure_captions.R")

# ----------------------------------------------------------------------------
# Manuscript figures use Harrell's C-index exclusively, including direct script
# execution. Legacy metric environment variables cannot enable AUC rendering.
# ----------------------------------------------------------------------------
METRIC <- "cindex"

# Reject unsupported metrics even when a builder is invoked directly.
metric_label <- function(metric = METRIC) {
  metric_suffix(metric)
  "C-index"
}

# Column-name suffix for C-index, matching the
# `{scheme}_{metric}` naming convention used in the metric-parameterized CSVs.
metric_suffix <- function(metric = METRIC) {
  if (!identical(metric, "cindex")) stop("Manuscript figures support only C-index")
  metric
}

# File/panel-name suffix for the active metric output, e.g. "figure2_..._cindex.png".
metric_tag <- function(metric = METRIC) paste0("_", metric_suffix(metric))


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
                   cohort2 = "first-line, 1:1 matched")
COHORT_DEFS   <- c(
  cohort1 = "Cohort 1: first-line ICI vs. all never-ICI controls (unmatched, discovery)",
  cohort2 = "Cohort 2: first-line ICI vs. 1:1 matched first-line controls (matched, validation)"
)


# ----------------------------------------------------------------------------
# Theme (ports apply_style())
# ----------------------------------------------------------------------------
MANUSCRIPT_BASE_SIZE <- 12
# geom_text()/annotate("text") sizes are in mm, unlike theme text sizes.
MANUSCRIPT_TEXT_SIZE <- 3.6
MANUSCRIPT_SMALL_TEXT_SIZE <- 3.2
MANUSCRIPT_CAPTION_SIZE <- 8

# ggplot never wraps a title, so a long interpolated label (event descriptions run
# well past 60 characters in the real data) is drawn as one line and silently
# clipped at the device edge. Wrap every variable-length label before it reaches
# labs(). `width` is in characters and is deliberately conservative: at base_size
# the bold title face fits roughly 2.2 characters per 0.1 inch of panel width.
wrap_title <- function(x, width = 42) {
  if (is.null(x) || length(x) == 0) return(x)
  stringr::str_wrap(as.character(x), width = width)
}

# Wraps the label but keeps a fixed suffix (e.g. a legend hint) on its own line.
wrap_title_suffix <- function(x, suffix, width = 42) {
  paste0(wrap_title(x, width = width), "\n", suffix)
}

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
      # Roomier than the original margin(4, 6, 4, 6): rotated axis text and the
      # last tick label on a wide axis were being trimmed at the device edge.
      plot.margin      = margin(8, 12, 8, 12)
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

# Individual panels and compiled figures, grouped by figure under
# png/<group>/ and pdf/<group>/; save_panel() writes both formats for every
# panel and creates the group subdirectory on demand. `group` (e.g. "figure1")
# is a required argument at every save_panel() call site — not a script-level
# default — since figure_utils.R is source()'d into globalenv() (source()'s
# default local=FALSE), so a lexical default couldn't see a caller-defined
# constant from the sys.source()-ed plot script's own environment.
save_panel <- function(plot, name, group, width = 6.0, height = 4.8, dpi = 300) {
  # A skipped panel (no underlying data) is not written at all -- an absent file
  # is a clearer signal to downstream assembly than a page reading "csv empty".
  if (is_skipped_panel(plot)) {
    # A skipped replacement must not leave an old panel under its reused letter.
    unlink(c(file.path(PNG_OUT_DIR, group, paste0(name, ".png")),
             file.path(PDF_OUT_DIR, group, paste0(name, ".pdf"))))
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
  ggsave(out_png, plot, width = width, height = height, dpi = dpi, bg = "white")
  pdf_device <- if (capabilities("cairo")) grDevices::cairo_pdf else grDevices::pdf
  if (!capabilities("cairo")) warning("Cairo is unavailable; check PDF font/glyph embedding before submission")
  ggsave(out_pdf, plot, width = width, height = height, bg = "white", device = pdf_device)
  message(sprintf("[panel] %s / %s", out_png, out_pdf))
  invisible(c(out_png, out_pdf))
}

# Only obsolete generated manuscript graphics are removed; prepared CSVs and
# unrelated output files are untouched. Run for direct scripts and notebooks too.
remove_retired_figures <- function() {
  for (root in c(PNG_OUT_DIR, PDF_OUT_DIR)) {
    files <- list.files(root, pattern = "\\.(png|pdf)$", recursive = TRUE,
                        full.names = TRUE, ignore.case = TRUE)
    retired <- files[grepl("^fig(ure)?[0-9S].*_auc\\.(png|pdf)$", basename(files),
                           ignore.case = TRUE) |
                       (basename(dirname(files)) == "figure2" &
                          grepl("^figS_scheme_km_.*\\.(png|pdf)$", basename(files))) |
                       (basename(dirname(files)) == "figure4" &
                          grepl("^fig(S1a|S_stage_dynamics_crossed)\\.(png|pdf)$", basename(files))) |
                       (basename(dirname(files)) == "figure5" &
                          grepl("^fig5a\\.(png|pdf)$", basename(files)))]
    if (length(retired)) {
      if (!all(file.remove(retired))) stop("Could not remove retired figures")
      message("Removed ", length(retired), " retired figure files")
    }
  }
}
remove_retired_figures()

# Retire superseded panel letters and untagged filenames after a figure's panel
# selection changes. Current filenames are overwritten (or removed if skipped).
retire_main_panels <- function(number, labels) {
  for (root in c(PNG_OUT_DIR, PDF_OUT_DIR)) {
    files <- list.files(file.path(root, paste0("figure", number)), full.names = TRUE,
                        pattern = sprintf("^fig%s[a-z](_cindex|_auc)?\\.(png|pdf)$", number))
    stems <- tools::file_path_sans_ext(basename(files))
    keep <- paste0("fig", number, labels, "_cindex")
    retired <- files[!stems %in% keep]
    if (length(retired) && !all(file.remove(retired))) {
      stop("Could not remove superseded Figure ", number, " panels")
    }
  }
}

# Wrap each complete panel as one grob so nested patchwork layouts (e.g. 4c)
# receive exactly one letter and retain their own annotations and legends.
# Named panels and explicit designs keep letters tied to standalone filenames.
save_compiled_figure <- function(panels, number, width, height,
                                 design = NULL, ncol = 2, heights = NULL) {
  group <- paste0("figure", number)
  name <- paste0(group, metric_tag())
  if (is.null(names(panels)) || any(!nzchar(names(panels))) || anyDuplicated(names(panels))) {
    stop("Compiled figures require unique panel labels")
  }
  null <- vapply(panels, is.null, logical(1))
  if (any(null)) {
    stop("Unexpected NULL panels: ", paste(names(panels)[null], collapse = ", "))
  }
  missing <- vapply(panels, is_skipped_panel, logical(1))
  missing_labels <- names(panels)[missing]
  if (any(missing)) {
    reasons <- vapply(panels[missing], skip_reason, character(1))
    message(sprintf("[figure%s] Unavailable panels:\n%s", number,
                    paste(sprintf("  %s: %s", missing_labels, reasons), collapse = "\n")))
    panels <- panels[!missing]
    if (!length(panels)) {
      # Do not leave an old figure when none of its current panels can be drawn.
      unlink(c(file.path(PNG_OUT_DIR, group, paste0(name, ".png")),
               file.path(PDF_OUT_DIR, group, paste0(name, ".pdf")),
               file.path(PNG_OUT_DIR, group, paste0(name, "_captioned.png")),
               file.path(PDF_OUT_DIR, group, paste0(name, "_captioned.pdf")),
               file.path(FIGURE_OUT_DIR, "captions", paste0(group, ".txt")),
               file.path(FIGURE_OUT_DIR, "captions", paste0(group, ".md"))))
      message(sprintf("[figure%s] SKIPPED: no available panels", number))
      return(invisible(character(0)))
    }
    # Reserve the unavailable panels' positions in explicit layouts; keeping
    # names avoids relabeling, e.g. panel e as c when c and d are unavailable.
    if (!is.null(design)) {
      for (label in missing_labels) design <- gsub(label, "#", design, fixed = TRUE)
    }
  }
  labeled <- lapply(names(panels), function(label) {
    # Bake each letter into its panel grob. Patchwork tag propagation previously
    # dropped b/c in Figure 3's asymmetric layout and can recurse into KM tables.
    labeled_panel <- cowplot::ggdraw() +
      cowplot::draw_plot(panels[[label]], x = 0.025, y = 0, width = 0.975, height = 0.975) +
      cowplot::draw_label(label, x = 0, y = 1, hjust = 0, vjust = 1,
                          size = 9, fontface = "bold", fontfamily = "sans")
    patchwork::wrap_elements(full = cowplot::as_grob(labeled_panel))
  })
  names(labeled) <- names(panels)
  compiled <- patchwork::wrap_plots(labeled, ncol = ncol, design = design,
                                    heights = heights)
  if (any(missing)) {
    compiled <- compiled + patchwork::plot_annotation(
      caption = paste("Unavailable panels:", paste(missing_labels, collapse = ", ")),
      theme = theme(plot.caption = element_text(size = 10, hjust = 0))
    )
  }
  files <- save_panel(compiled, name, group, width = width, height = height, dpi = 600)
  caption <- figure_caption(number, panels, missing_labels)
  save_captioned_figure(compiled, caption, name, group, width, height)
  invisible(files)
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
# Outlier-event exclusion
#
# Events whose text-vs-base delta is a distributional OUTLIER are dropped from
# every figure, so a reader never sees an event in one panel that was filtered
# out of another. The delta is
#
#     delta = text_cindex - base_cindex        (full cohort, per scheme+event)
#
# and an event is excluded when that delta falls outside
#
#     mean(delta) +/- EVENT_EXCLUSION_SD * sd(delta)
#
# The rule is TWO-SIDED: an implausibly large text win is as much an outlier as
# an implausibly large text loss, and both distort the summaries built on these
# endpoints. The thresholds come from the delta distribution itself, so they
# adapt to the run rather than encoding a fixed effect size.
#
# The exclusion set is always judged on the manuscript's C-index.
# ----------------------------------------------------------------------------
EVENT_EXCLUSION_SD <- 3             # outlier cutoff, in SDs of the delta distribution
EVENT_EXCLUSION_METRIC <- "cindex"  # delta is always judged on the primary metric
FILTER_UNDERPERFORMING_ENDPOINTS <- tolower(Sys.getenv(
  "MANUSCRIPT_FILTER_UNDERPERFORMING_ENDPOINTS", unset = "true"
)) %in% c("1", "true", "yes", "on")
# sd() is undefined below two finite deltas, and a handful of points cannot
# support a +/-3SD rule; below this many, nothing is dropped.
EVENT_EXCLUSION_MIN_N <- 3

# Key an event by scheme + event name; both are needed since the same event
# label can appear under more than one coding scheme.
.event_key <- function(scheme, event) paste(scheme, event, sep = "\u001f")

# Returns the character vector of scheme/event keys to exclude, given the
# full-cohort metrics frame. `metric` selects the delta's metric and defaults to
# the primary C-index (see above) -- it is NOT the active render metric. Empty
# vector when the frame lacks that metric's columns, or has too few finite deltas
# to define a distribution (nothing can be judged, so nothing is dropped).
excluded_event_keys <- function(metrics, metric = EVENT_EXCLUSION_METRIC) {
  if (!FILTER_UNDERPERFORMING_ENDPOINTS) return(character(0))
  if (is.null(metrics) || nrow(metrics) == 0) return(character(0))
  base_col <- paste0("base_", metric_suffix(metric))
  text_col <- paste0("text_", metric_suffix(metric))
  if (!all(c(base_col, text_col, "scheme", "event") %in% names(metrics))) {
    return(character(0))
  }
  delta  <- as.numeric(metrics[[text_col]]) - as.numeric(metrics[[base_col]])
  finite <- is.finite(delta)
  if (sum(finite) < EVENT_EXCLUSION_MIN_N) {
    message(sprintf(
      "[exclude] only %d finite %s delta(s); too few for a +/-%dSD rule -- nothing dropped",
      sum(finite), metric_label(metric), EVENT_EXCLUSION_SD))
    return(character(0))
  }
  mu    <- mean(delta[finite])
  sigma <- stats::sd(delta[finite])
  # A degenerate (zero / non-finite) SD puts every event exactly at the mean:
  # there is no outlier to speak of, and the comparison would drop all or none
  # arbitrarily.
  if (!is.finite(sigma) || sigma <= 0) {
    message("[exclude] delta distribution has zero spread -- nothing dropped")
    return(character(0))
  }
  lo <- mu - EVENT_EXCLUSION_SD * sigma
  hi <- mu + EVENT_EXCLUSION_SD * sigma
  drop <- finite & (delta < lo | delta > hi)
  message(sprintf(
    "[exclude] %s delta: mean %.4f, sd %.4f -> keep [%.4f, %.4f]; %d of %d event(s) outside",
    metric_label(metric), mu, sigma, lo, hi, sum(drop), sum(finite)))
  if (!any(drop)) return(character(0))
  keys <- .event_key(metrics$scheme[drop], metrics$event[drop])
  message(sprintf(
    "[exclude] %d outlier event(s) dropped from all figures: %s",
    length(keys),
    paste(sprintf("%s/%s (delta %+.4f)", metrics$scheme[drop], metrics$event[drop],
                  delta[drop]),
          collapse = "; ")))
  unique(keys)
}

# Compact, non-underflowing manuscript p-value formatter.
format_p_value <- function(p, digits = 2L, floor = 1e-300) {
  if (length(p) == 0 || is.na(p) || !is.finite(p)) return("n/a")
  if (p < floor) return(sprintf("<%s", format(floor, scientific = TRUE, digits = 1)))
  if (p < 0.001) return(format(p, scientific = TRUE, digits = digits))
  sub("^0", "", sprintf(paste0("%.", digits + 1L, "f"), p))
}

# Complete inline p-value expression. Keeping the comparison operator here
# prevents constructions such as `p=<1e-300` at call sites.
format_p_inline <- function(p, digits = 2L, floor = 1e-300) {
  value <- format_p_value(p, digits = digits, floor = floor)
  if (identical(value, "n/a")) return("p=n/a")
  paste0("p", if (startsWith(value, "<")) "" else "=", value)
}

# Censoring can occur at thousands of unique times. Plotting every mark turns a
# KM curve into an opaque band, so retain an evenly spaced display-only sample.
# This never changes the fitted curve, confidence interval, or risk table.
thin_censor_rows <- function(df, groups, max_per_group = 80L) {
  if (is.null(df) || nrow(df) == 0 || !"n.censor" %in% names(df)) return(df[0, , drop = FALSE])
  df %>%
    dplyr::filter(n.censor > 0) %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(groups))) %>%
    dplyr::arrange(time, .by_group = TRUE) %>%
    dplyr::slice(unique(round(seq(1, dplyr::n(), length.out = min(dplyr::n(), max_per_group))))) %>%
    dplyr::ungroup()
}

# Drop excluded events from any scheme+event-keyed frame. A frame without both
# key columns is returned untouched.
drop_excluded_events <- function(df, keys) {
  if (length(keys) == 0 || is.null(df) || nrow(df) == 0) return(df)
  if (!all(c("scheme", "event") %in% names(df))) return(df)
  df[!(.event_key(df$scheme, df$event) %in% keys), , drop = FALSE]
}

# Drop excluded events from a top-k frame, then renumber `rank` densely (1..n)
# within each category so the rank-1/2/3 panels stay filled by the next-best
# surviving events instead of leaving holes where an excluded event ranked.
drop_excluded_events_reranked <- function(topk, keys) {
  if (is.null(topk) || nrow(topk) == 0) return(topk)
  out <- drop_excluded_events(topk, keys)
  if (nrow(out) == 0 || !all(c("category", "rank") %in% names(out))) return(out)
  out %>%
    dplyr::group_by(category) %>%
    dplyr::arrange(rank, .by_group = TRUE) %>%
    dplyr::mutate(rank = dplyr::row_number()) %>%
    dplyr::ungroup()
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
