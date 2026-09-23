# Shared reporting for existing models evaluated within cancer types.
# Source R/figure_utils.R first. These helpers do not refit any models.

# SELECTED_CANCER_TYPES (CANCER_GROUP label -> display name, in display order)
# comes from shared/palette.json via figure_utils.R, shared with the Python prep.
if (!exists("SELECTED_CANCER_TYPES")) {
  SELECTED_CANCER_TYPES <- unlist(jsonlite::fromJSON(file.path("shared", "palette.json"))$SELECTED_CANCER_TYPES)
}
OS_SCHEME <- "death_met"
OS_EVENT <- "death"

# Restrict rows to SELECTED_CANCER_TYPES, normalizing labels to their codes.
# Types with no rows (e.g. pooled into OTHER below the preprocessing size
# threshold, or ineligible for every endpoint) are reported, not invented.
select_cancer_types <- function(d, context) {
  if (!nrow(d)) return(d)
  d <- d %>%
    mutate(cancer_type = toupper(trimws(as.character(cancer_type)))) %>%
    filter(cancer_type %in% names(SELECTED_CANCER_TYPES))
  absent <- setdiff(names(SELECTED_CANCER_TYPES), d$cancer_type)
  if (length(absent)) {
    message(sprintf("[%s] no eligible rows for: %s", context, paste(absent, collapse = ", ")))
  }
  d
}

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

# Per-endpoint modality ranks within each cancer type (1 = best C-index; ties
# share the average rank), mirroring Figure 3a. Input is the shared-cohort table
# (fig3_within_cancer_modality_cindex.csv): every modality is scored on the same
# patients and comparable pairs. Complete-case: an endpoint/cancer is ranked only
# when every modality present in this run has a finite C-index for it.
within_cancer_modality_ranks <- function(modality_cindex, excluded = character()) {
  required <- c("scheme", "event", "cancer_type", "modality", "cindex", "status")
  if (is.null(modality_cindex) || !all(required %in% names(modality_cindex))) return(tibble::tibble())
  d <- drop_excluded_events(modality_cindex, excluded) %>%
    filter(status == "ok", modality %in% MODALITY_ORDER, is.finite(cindex),
           !is.na(cancer_type), nzchar(trimws(cancer_type)))
  if (!nrow(d)) return(tibble::tibble())
  if (anyDuplicated(d[c("scheme", "event", "cancer_type", "modality")])) {
    stop("Duplicate within-cancer endpoint/modality rows; rerun figures.prep.within_cancer")
  }
  present <- intersect(MODALITY_ORDER, unique(d$modality))
  d %>%
    group_by(scheme, event, cancer_type) %>%
    filter(all(present %in% modality)) %>%
    mutate(rank = rank(-cindex, ties.method = "average")) %>%
    ungroup() %>%
    select(scheme, event, cancer_type, modality, cindex, rank)
}

summarize_within_cancer_ranks <- function(ranks) {
  if (!nrow(ranks)) return(tibble::tibble())
  ranks %>%
    group_by(cancer_type, modality) %>%
    summarise(
      n_endpoints = n(),
      mean_rank = mean(rank),
      q25_rank = as.numeric(quantile(rank, 0.25)),
      q75_rank = as.numeric(quantile(rank, 0.75)),
      .groups = "drop"
    ) %>% arrange(cancer_type, match(modality, MODALITY_ORDER))
}

# Per-cancer joint Cox refits (fig3_within_cancer_joint_betas.csv) ready to plot:
# shared endpoint exclusions, one fit variant, BH-FDR within each
# (scheme, event, cancer) fit as in Figure 3b, and complete case within each
# cancer type: an endpoint counts only when every modality fitted for that
# cancer in this run is present (constant scores can drop a modality from a
# small stratum, so the set is per cancer, not run-wide).
prepare_within_cancer_joint <- function(betas, excluded = character(),
                                        variant = "unpenalized", alpha = 0.05) {
  required <- c("cancer_type", "scheme", "event", "fit_variant", "modality", "beta", "p_value")
  if (is.null(betas) || !nrow(betas) || !all(required %in% names(betas))) return(tibble::tibble())
  d <- drop_excluded_events(betas, excluded) %>%
    mutate(cancer_type = toupper(trimws(as.character(cancer_type)))) %>%
    filter(fit_variant == variant, is.finite(beta))
  if (!nrow(d)) return(tibble::tibble())
  if (anyDuplicated(d[c("cancer_type", "scheme", "event", "modality")])) {
    stop("Duplicate within-cancer joint Cox rows; rerun figures.prep.within_cancer_joint")
  }
  d %>%
    group_by(cancer_type) %>%
    mutate(.n_modalities = n_distinct(modality)) %>%
    group_by(cancer_type, scheme, event) %>%
    filter(n_distinct(modality) == .n_modalities) %>%
    mutate(q_value = stats::p.adjust(replace(p_value, is.na(p_value), 1), method = "BH")) %>%
    ungroup() %>%
    mutate(sig = q_value < alpha) %>%
    select(-.n_modalities)
}

summarize_within_cancer_joint <- function(d) {
  if (!nrow(d)) return(tibble::tibble())
  d %>%
    group_by(cancer_type, modality) %>%
    summarise(
      n_endpoints = n(),
      n_significant = sum(sig),
      n_significant_positive = sum(sig & beta > 0),
      prop_significant = n_significant / n_endpoints,
      median_beta = median(beta),
      q25_beta = as.numeric(quantile(beta, 0.25)),
      q75_beta = as.numeric(quantile(beta, 0.75)),
      .groups = "drop"
    ) %>%
    arrange(match(cancer_type, names(SELECTED_CANCER_TYPES)), match(modality, MODALITY_ORDER))
}

within_cancer_rank_caption <- function(summary, n_modalities, n_excluded, selected = FALSE) {
  title <- if (selected) {
    paste("Supplemental Figure 3. Modality rank within selected cancer types. Shown cancer types:",
          paste0(paste(SELECTED_CANCER_TYPES, collapse = ", "), "."),
          "CUP denotes cancer of unknown primary.")
  } else "Supplemental Figure 3. Modality rank within cancer types."
  display <- paste(
    "Rows denote cancer types (with the number of ranked endpoints) and columns denote",
    sprintf("modalities. For each endpoint, the %d modalities are ranked by within-cancer", n_modalities),
    "C-index (1 = best; ties share the average rank). Cells report the mean rank and its",
    "interquartile range across endpoints; blue indicates better-than-middle ranks.",
    "Grey cells have no endpoint with every modality evaluable."
  )
  methods <- paste(
    "Existing pan-cancer models are evaluated within each cancer type without refitting. For",
    "each endpoint, all modalities are scored on the same patients (those with every modality's",
    "held-out prediction, an outcome and a cancer label) and the same comparable pairs: Harrell's",
    "C-index is calculated within blocks sharing every modality's outer fold and aggregated by",
    "comparable-pair counts. An endpoint is ranked within a cancer type only when every modality",
    "is evaluable (complete case), as in Figure 3a. Endpoints receive equal weight, are",
    "correlated, and can differ between cancer types; summaries are descriptive."
  )
  filtering <- if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
    sprintf("The shared manuscript endpoint outlier filter is applied (%d endpoint keys excluded across the manuscript).",
            n_excluded)
  } else "The shared manuscript endpoint outlier filter is disabled."
  counts <- sprintf("Shown: %d cancer types.", dplyr::n_distinct(summary$cancer_type))
  paste(title, display, methods, filtering, counts, sep = "\n\n")
}

within_cancer_caption <- function(number, metrics, summary, n_excluded, selected = FALSE) {
  title <- if (number == 2) {
    "Supplemental Figure 2. Text versus base performance within cancer types."
  } else {
    "Supplemental Figure 3. Text versus other modalities within cancer types."
  }
  if (selected) {
    title <- sub("within cancer types\\.$", "within selected cancer types.", title)
    title <- paste(title, "Shown cancer types:",
                   paste0(paste(SELECTED_CANCER_TYPES, collapse = ", "), "."),
                   "CUP denotes cancer of unknown primary.")
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
