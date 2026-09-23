# Overall-survival C-index within selected cancer types: the pooled models'
# death endpoint from the Figure 2 (text vs base) and Figure 3 (text vs other
# modalities) within-cancer evaluations. No models are refitted.
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

OS_GROUP <- "figure_os"

# One row per cancer/comparator for the OS endpoint; each comparison keeps its
# own matched cohort, so text C-indices can differ between comparators.
load_os_within_cancer <- function(excluded) {
  modalities <- setdiff(MODALITY_ORDER, "text")
  sources <- list(
    list(file = "fig2_within_cancer_cindex.csv", comparators = "base", cohort = "full"),
    list(file = "fig3_within_cancer_cindex.csv", comparators = modalities, cohort = "modality")
  )
  rows <- lapply(sources, function(s) {
    rows <- eligible_within_cancer(read_within_cancer_data(s$file), s$comparators, excluded)
    if (!nrow(rows)) return(NULL)
    rows %>% filter(scheme == OS_SCHEME, event == OS_EVENT) %>%
      mutate(cohort = s$cohort)
  })
  rows <- bind_rows(rows)
  if (!nrow(rows)) return(rows)
  select_cancer_types(rows, "overall survival figure") %>%
    arrange(match(cancer_type, names(SELECTED_CANCER_TYPES)),
            match(comparator, c("base", modalities)))
}

build_os_within_cancer <- function(d) {
  comparators <- intersect(c("base", setdiff(MODALITY_ORDER, "text")), unique(d$comparator))
  model_labels <- c(text = "Text", base = "Base", MODALITY_DISPLAY[setdiff(comparators, "base")])
  model_colors <- c(text = MODEL_COLORS[["text"]], base = MODEL_COLORS[["base"]],
                    MODALITY_COLORS[setdiff(comparators, "base")])
  facet_labels <- setNames(paste("Text vs", model_labels[comparators]), comparators)
  d <- d %>% mutate(
    cancer = factor(SELECTED_CANCER_TYPES[cancer_type], levels = rev(SELECTED_CANCER_TYPES)),
    comparator = factor(comparator, levels = comparators)
  )
  points <- bind_rows(
    transmute(d, cancer, comparator, model = as.character(comparator), cindex = comparator_cindex),
    transmute(d, cancer, comparator, model = "text", cindex = text_cindex)
  ) %>% mutate(model = factor(model, levels = names(model_labels)))
  ggplot(d, aes(y = cancer)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed") +
    geom_segment(aes(x = comparator_cindex, xend = text_cindex, yend = cancer),
                 color = "grey65", linewidth = 0.7) +
    geom_point(data = points, aes(x = cindex, color = model), size = 2.4) +
    geom_text(aes(x = Inf, label = sprintf("%+.3f", delta_cindex)), hjust = 1.05,
              size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "grey25") +
    facet_wrap(~comparator, nrow = 2, labeller = as_labeller(facet_labels)) +
    scale_color_manual(values = model_colors, labels = model_labels, name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0.05, 0.3))) +
    scale_y_discrete(drop = FALSE) +
    labs(x = "Overall survival C-index", y = NULL,
         title = "Overall survival performance within cancer types",
         subtitle = "Existing pan-cancer models; matched held-out patients. Right-hand values: text minus comparator.",
         caption = "C-indices use comparable pairs within joint outer-fold blocks.") +
    theme_manuscript() +
    theme(legend.position = "bottom", panel.grid.major.y = element_line(color = "grey93"),
          plot.caption = element_text(size = 8))
}

os_within_cancer_caption <- function(d, n_excluded) {
  thresholds <- unique(d[c("min_patients_required", "min_events_required")])
  paste(
    "Overall survival C-index within selected cancer types.",
    paste("Each row is one cancer type; each panel pairs text with one comparator model for the",
          "overall-survival (death) endpoint. Grey segments join the comparator (colored) and text",
          "(red) C-indices; right-hand values give the text-minus-comparator difference. Shown",
          "cancer types:", paste0(paste(SELECTED_CANCER_TYPES, collapse = ", "), "."),
          "CUP denotes cancer of unknown primary. Rows without a point were not evaluable."),
    paste("Existing pan-cancer models are evaluated separately within each cancer type. Text",
          "versus base uses the full-cohort held-out scores; text versus other modalities uses",
          "the feature-comparison held-out scores, where both models include the shared base",
          "covariates. Each comparison uses its own matched patients, so text C-indices can",
          "differ between panels. Harrell's C-index is calculated within joint outer-fold blocks",
          "and aggregated by comparable-pair counts. Matched patient and event counts are in the",
          "accompanying table."),
    paste("Eligibility thresholds (matched patients / observed events):",
          paste(paste(thresholds$min_patients_required, thresholds$min_events_required, sep = " / "),
                collapse = "; "), "; at least one comparable pair is also required."),
    if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
      sprintf("The shared manuscript endpoint outlier filter is applied (%d endpoint keys excluded across the manuscript).",
              n_excluded)
    } else "The shared manuscript endpoint outlier filter is disabled.",
    sep = "\n\n"
  )
}

render_os_within_cancer <- function() {
  stem <- "figOS_within_cancer_cindex"
  clear_within_cancer_report(stem, OS_GROUP)
  excluded <- within_cancer_exclusions()
  d <- load_os_within_cancer(excluded)
  if (!nrow(d)) {
    message("[overall survival figure] SKIPPED: no eligible OS rows for the selected cancer types; ",
            "run figures.prep.within_cancer")
    return(invisible(NULL))
  }
  p <- build_os_within_cancer(d)
  n_panels <- n_distinct(d$comparator)
  save_panel(p, stem, OS_GROUP, width = max(6, 4 * ceiling(n_panels / 2)),
             height = 2 + 3.4 * min(n_panels, 2))
  table <- select(d, cancer_type, comparator, cohort, text_cindex, comparator_cindex,
                  delta_cindex, n_patients, n_events, any_of("n_comparable_pairs"))
  save_within_cancer_report(table, os_within_cancer_caption(d, length(excluded)), stem)
  invisible(table)
}

render_os_within_cancer()
