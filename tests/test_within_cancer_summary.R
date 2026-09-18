# Run from the repository root: Rscript tests/test_within_cancer_summary.R
# The synthetic endpoint summaries never read patient data or render figures.
suppressPackageStartupMessages({ library(dplyr); library(ggplot2) })

# Load the actual endpoint-exclusion helpers without figure_utils.R's package
# dependencies and configured-output cleanup.
for (expr in parse("R/figure_utils.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      as.character(expr[[2]]) %in% c(".event_key", "drop_excluded_events")) {
    eval(expr)
  }
}
source("R/within_cancer_utils.R")

close_to <- function(actual, expected) {
  isTRUE(all.equal(as.numeric(actual), as.numeric(expected), tolerance = 1e-12))
}

# Three endpoints have deliberately unequal patient/event counts. Each endpoint
# contributes once, and two schemes can reuse an event label independently.
metrics <- tibble(
  scheme = c("icd3_post", "icd3_post", "phecode_post"),
  event = c("001", "2", "001"),
  cancer_type = "Lung", comparator = "base", status = "ok",
  text_cindex = c(0.60, 0.70, 0.95),
  comparator_cindex = c(0.40, 0.80, 0.90),
  delta_cindex = text_cindex - comparator_cindex,
  n_patients = c(20L, 40L, 4000L), n_events = c(5L, 10L, 100L)
)
summary <- summarize_within_cancer(metrics, "base")
stopifnot(
  nrow(summary) == 1L, summary$n_endpoints == 3L,
  close_to(summary$median_text_cindex, 0.70),
  close_to(summary$median_comparator_cindex, 0.80),
  close_to(summary$median_delta_cindex, 0.05),
  close_to(summary$q25_delta_cindex, -0.025),
  close_to(summary$q75_delta_cindex, 0.125),
  summary$min_patients == 20L, summary$max_patients == 4000L,
  summary$min_events == 5L, summary$max_events == 100L,
  # A median of paired gains need not equal a difference of median C-indices.
  !close_to(summary$median_delta_cindex,
            summary$median_text_cindex - summary$median_comparator_cindex)
)

# Neither large cohorts nor event-rich endpoints should receive extra weight.
different_counts <- mutate(metrics, n_patients = rev(n_patients),
                           n_events = rev(n_events))
reweighted <- summarize_within_cancer(different_counts, "base")
stopifnot(close_to(reweighted$median_delta_cindex,
                  summary$median_delta_cindex))

# Failed strata and any non-finite metric must not contribute, even if some of
# their metrics are populated. A completely absent cancer/comparator is not 0.
invalid <- bind_rows(
  mutate(metrics[1, ], event = "failed", status = "too_few_events"),
  mutate(metrics[1, ], event = "missing_text", text_cindex = NA_real_),
  mutate(metrics[1, ], event = "infinite_comparator", comparator_cindex = Inf),
  mutate(metrics[1, ], event = "missing_delta", delta_cindex = NA_real_),
  mutate(metrics[1, ], event = "infinite_delta", delta_cindex = Inf),
  mutate(metrics[1, ], cancer_type = "Brain", event = "failed_brain",
         status = "no_comparable_pairs")
)
with_invalid <- summarize_within_cancer(bind_rows(metrics, invalid), "base")
stopifnot(identical(with_invalid, summary))

# Modalities may have different available endpoints. Summarize their own support
# without intersecting it across all modalities or borrowing another's scores.
stage <- mutate(metrics[2, ], comparator = "stage", text_cindex = 0.55,
                comparator_cindex = 0.75, delta_cindex = -0.20)
combined <- summarize_within_cancer(bind_rows(metrics, stage, invalid),
                                   c("base", "stage", "genomic"))
base <- filter(combined, comparator == "base")
stage_summary <- filter(combined, comparator == "stage")
stopifnot(
  nrow(combined) == 2L, base$n_endpoints == 3L,
  stage_summary$n_endpoints == 1L,
  close_to(stage_summary$median_text_cindex, 0.55),
  close_to(stage_summary$median_delta_cindex, -0.20),
  close_to(stage_summary$q25_delta_cindex, -0.20),
  close_to(stage_summary$q75_delta_cindex, -0.20),
  !any(combined$comparator == "genomic"),
  !any(combined$cancer_type == "Brain")
)
only_stage <- summarize_within_cancer(bind_rows(metrics, stage), "stage")
stopifnot(nrow(only_stage) == 1L, only_stage$comparator == "stage")

# Apply the main figure's exclusions before aggregation. Event identity is the
# exact (scheme, event) string pair: numeric-looking labels stay distinct.
numeric_labels <- mutate(metrics[rep(1, 4), ],
                         scheme = c("icd3_post", "icd3_post", "icd3_post", "phecode_post"),
                         event = c("001", "1", "1.00", "001"),
                         text_cindex = c(0.90, 0.60, 0.70, 0.80),
                         comparator_cindex = 0.50,
                         delta_cindex = text_cindex - comparator_cindex)
restricted <- summarize_within_cancer(
  numeric_labels, "base", .event_key("icd3_post", "001")
)
stopifnot(nrow(restricted) == 1L, restricted$n_endpoints == 3L,
          close_to(restricted$median_delta_cindex, 0.20))
all_excluded <- summarize_within_cancer(
  numeric_labels, "base", .event_key(numeric_labels$scheme, numeric_labels$event)
)
stopifnot(nrow(all_excluded) == 0L,
          nrow(summarize_within_cancer(metrics[0, ], "base")) == 0L,
          nrow(summarize_within_cancer(metrics, "genomic")) == 0L)

# The CSV reader must preserve these endpoint strings before exclusions run.
check_csv_identity <- function() {
  fixture_dir <- tempfile("within-cancer-input-")
  dir.create(fixture_dir)
  on.exit(unlink(fixture_dir, recursive = TRUE))
  FIGURE_DATA_DIR <<- fixture_dir
  on.exit(rm("FIGURE_DATA_DIR", envir = .GlobalEnv), add = TRUE)
  readr::write_csv(numeric_labels, file.path(fixture_dir, "metrics.csv"))
  loaded <- read_within_cancer_data("metrics.csv")
  stopifnot(identical(loaded$event, numeric_labels$event))
  result <- summarize_within_cancer(
    loaded, "base", .event_key("icd3_post", "001")
  )
  stopifnot(result$n_endpoints == 3L,
            close_to(result$median_delta_cindex, 0.20))
}
check_csv_identity()

# Duplicate endpoint/comparator rows would weight the same endpoint twice.
duplicate <- try(summarize_within_cancer(bind_rows(metrics, metrics[1, ]), "base"),
                 silent = TRUE)
stopifnot(inherits(duplicate, "try-error"))

# Extract the production plot builder without sourcing the top-level renderer.
for (expr in parse("R/plot_figure_3_supp_cancer.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      identical(expr[[2]], as.name("build_within_cancer_modality_heatmap"))) eval(expr)
}
MANUSCRIPT_SMALL_TEXT_SIZE <- 3.2
MODALITY_DISPLAY <- c(base = "Base", stage = "Stage", genomic = "Genomic")
theme_manuscript <- function(...) theme_classic(...)
heatmap <- build_within_cancer_modality_heatmap(
  combined, c("Lung", "Brain"), c("base", "stage", "genomic"), limit = 0.20
)
cells <- heatmap$data
missing <- filter(cells, is.na(n_endpoints))
stopifnot(nrow(cells) == 6L, nrow(missing) == 4L,
          all(is.na(missing$median_text_cindex)),
          all(is.na(missing$median_comparator_cindex)),
          all(is.na(missing$median_delta_cindex)),
          all(missing$label == "Unavailable"))
built <- ggplot_build(heatmap)
stopifnot(nrow(built$data[[1]]) == 6L,
          sum(built$data[[1]]$fill == "grey92") == 4L,
          sum(built$data[[2]]$label == "Unavailable") == 4L)

message("Within-cancer supplemental-summary tests passed")
