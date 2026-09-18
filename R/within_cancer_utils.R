# Shared reporting for existing models evaluated within cancer types.
# Source R/figure_utils.R first. These helpers do not refit any models.

read_within_cancer_data <- function(name) {
  path <- file.path(FIGURE_DATA_DIR, name)
  if (!file.exists(path)) {
    warning("Missing figure data: ", path)
    return(tibble::tibble())
  }
  # Numeric-looking ICD/phecode identifiers must retain trailing zeroes.
  readr::read_csv(path, show_col_types = FALSE, col_types = readr::cols(
    scheme = readr::col_character(), event = readr::col_character(),
    .default = readr::col_guess()
  ))
}

eligible_within_cancer <- function(metrics, comparators, excluded = character()) {
  required <- c("scheme", "event", "cancer_type", "comparator", "status",
                "text_cindex", "comparator_cindex", "delta_cindex",
                "n_patients", "n_events")
  if (is.null(metrics) || !all(required %in% names(metrics))) return(tibble::tibble())
  d <- drop_excluded_events(metrics, excluded) %>%
    filter(status == "ok", comparator %in% comparators,
           !is.na(cancer_type), nzchar(trimws(cancer_type)),
           is.finite(text_cindex), is.finite(comparator_cindex), is.finite(delta_cindex))
  if (anyDuplicated(d[c("scheme", "event", "cancer_type", "comparator")])) {
    stop("Duplicate within-cancer endpoint/comparator rows; rerun figures.prep.within_cancer")
  }
  d
}

summarize_within_cancer <- function(metrics, comparators, excluded = character()) {
  d <- eligible_within_cancer(metrics, comparators, excluded)
  if (!nrow(d)) return(tibble::tibble())
  d %>%
    group_by(cancer_type, comparator) %>%
    summarise(
      n_endpoints = n(),
      median_text_cindex = median(text_cindex),
      median_comparator_cindex = median(comparator_cindex),
      median_delta_cindex = median(delta_cindex),
      q25_delta_cindex = as.numeric(quantile(delta_cindex, 0.25)),
      q75_delta_cindex = as.numeric(quantile(delta_cindex, 0.75)),
      min_patients = min(n_patients), max_patients = max(n_patients),
      min_events = min(n_events), max_events = max(n_events),
      .groups = "drop"
    ) %>% arrange(cancer_type, match(comparator, comparators))
}

within_cancer_caption <- function(number, metrics, summary, n_excluded) {
  title <- if (number == 2) {
    "Supplemental Figure 2. Text versus base performance within cancer types."
  } else {
    "Supplemental Figure 3. Text versus other modalities within cancer types."
  }
  display <- if (number == 2) {
    paste("Each point represents one endpoint within the indicated cancer type; colors",
          "identify coding families. The diagonal denotes equal performance. Facet labels",
          "report endpoint counts and median paired text-minus-base C-index differences.")
  } else {
    paste("Rows denote cancer types and columns denote comparator modalities. Cell text",
          "reports median Text / Comparator C-indices, the median paired difference",
          "(Text minus Comparator), and the number of endpoints. Color represents the",
          "median paired difference; positive values favor text. The median paired",
          "difference need not equal the difference between the two marginal medians.",
          "Grey cells have no evaluable endpoints; their values are unavailable, not zero.")
  }
  source <- if (number == 2) {
    "Text and base predictions come from the full-cohort held-out score runs."
  } else {
    paste("Text and comparator predictions come from the feature-comparison held-out",
          "score runs; both models include the shared base covariates.")
  }
  methods <- paste(
    "Existing pan-cancer models are evaluated separately within each recorded cancer-type",
    "stratum. Each comparison uses patients with both finite risk scores and valid outcomes.",
    "Harrell's C-index is calculated within joint outer-fold blocks (text fold, comparator",
    "fold) and aggregated by comparable-pair counts. Patients from different fitted-model",
    "blocks are never compared. Counts of patients and observed events in the accompanying",
    "table describe the matched cohort; only blocks with comparable pairs contribute to C-index.",
    "Cancer labels follow the preprocessing categories, including any pooled OTHER category.",
    "Endpoints receive equal weight in medians, are correlated, and may differ between",
    "cancer types or comparators; summaries are descriptive."
  )
  threshold_cols <- c("min_patients_required", "min_events_required")
  thresholds <- if (all(threshold_cols %in% names(metrics)) && nrow(metrics)) {
    rules <- unique(metrics[threshold_cols])
    paste("Eligibility thresholds (matched patients / observed events):",
          paste(paste(rules$min_patients_required, rules$min_events_required, sep = " / "),
                collapse = "; "), "; at least one comparable pair is also required.")
  } else {
    "Eligibility thresholds are recorded by the within-cancer data-preparation run."
  }
  filtering <- if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
    sprintf(paste("The shared manuscript filter excludes endpoints whose pooled text-minus-base",
                  "C-index falls outside the mean +/- %s standard deviations (%d endpoint keys",
                  "excluded across the manuscript). No additional performance-based",
                  "filter is applied within cancer types."), EVENT_EXCLUSION_SD, n_excluded)
  } else "The shared manuscript endpoint outlier filter is disabled."
  counts <- sprintf("Shown: %d cancer types and %d evaluable cancer/comparator cells.",
                    dplyr::n_distinct(summary$cancer_type), nrow(summary))
  paste(title, display, source, methods, thresholds, filtering, counts, sep = "\n\n")
}

save_within_cancer_report <- function(summary, caption, stem) {
  table_dir <- file.path(FIGURE_OUT_DIR, "tables")
  caption_dir <- file.path(FIGURE_OUT_DIR, "captions")
  dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(caption_dir, recursive = TRUE, showWarnings = FALSE)
  readr::write_csv(summary, file.path(table_dir, paste0(stem, ".csv")))
  writeLines(caption, file.path(caption_dir, paste0(stem, ".txt")), useBytes = TRUE)
  writeLines(caption, file.path(caption_dir, paste0(stem, ".md")), useBytes = TRUE)
}

# Remove obsolete pages when a rerun has fewer cancer types or no eligible data.
clear_within_cancer_report <- function(stem, group) {
  for (root in c(PNG_OUT_DIR, PDF_OUT_DIR)) {
    files <- list.files(file.path(root, group), full.names = TRUE,
                        pattern = paste0("^", stem, "(_page[0-9]+)?\\.(png|pdf)$"))
    if (length(files) && !all(file.remove(files))) stop("Could not remove old supplement pages")
  }
  unlink(c(file.path(FIGURE_OUT_DIR, "tables", paste0(stem, ".csv")),
           file.path(FIGURE_OUT_DIR, "captions", paste0(stem, c(".txt", ".md")))))
}

within_cancer_pages <- function(summary, page_size = 12L) {
  cancers <- sort(unique(as.character(summary$cancer_type)))
  if (!length(cancers)) return(list())
  split(cancers, ceiling(seq_along(cancers) / page_size))
}

within_cancer_exclusions <- function() {
  path <- file.path(FIGURE_DATA_DIR, "fig2_full_cohort_metrics.csv")
  # Do not silently produce an unfiltered supplement in a filtered manuscript.
  if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS) && !file.exists(path)) {
    stop("Within-cancer supplements require fig2_full_cohort_metrics.csv for the shared ",
         "endpoint filter. Run figures.prep.figure2, or disable the filter explicitly.")
  }
  if (!isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) return(character())
  metrics <- read_within_cancer_data("fig2_full_cohort_metrics.csv")
  if (!all(c("scheme", "event", "text_cindex", "base_cindex") %in% names(metrics))) {
    stop("fig2_full_cohort_metrics.csv lacks columns required for endpoint filtering")
  }
  excluded_event_keys(metrics)
}
