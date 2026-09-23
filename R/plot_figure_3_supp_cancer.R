# Figure 3 supplement: paired within-cancer text/other-modality performance.
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

# `cancer_labels`, if given, maps cancer codes to row labels.
build_within_cancer_modality_heatmap <- function(summary, cancers, comparators, limit,
                                                 cancer_labels = NULL) {
  row_label <- if (is.null(cancer_labels)) identity else function(x) unname(cancer_labels[x])
  # Complete the grid for display only: absent comparisons retain NA metrics.
  grid <- expand.grid(cancer_type = cancers, comparator = comparators, stringsAsFactors = FALSE)
  d <- left_join(grid, summary, by = c("cancer_type", "comparator")) %>%
    mutate(cancer_type = factor(cancer_type, levels = rev(cancers)),
           comparator = factor(comparator, levels = comparators),
           label = ifelse(is.na(n_endpoints), "Unavailable",
                          sprintf("%.3f / %.3f\ndelta %+.3f\nn=%d", median_text_cindex,
                                  median_comparator_cindex, median_delta_cindex, n_endpoints)))
  ggplot(d, aes(comparator, cancer_type, fill = median_delta_cindex)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = label), size = MANUSCRIPT_SMALL_TEXT_SIZE, lineheight = 1.05) +
    scale_fill_gradient2(low = "#E8B0A2", mid = "white", high = "#A5C9E2", midpoint = 0,
                         limits = c(-limit, limit), na.value = "grey92",
                         name = "Median paired\ndelta C-index") +
    scale_x_discrete(labels = MODALITY_DISPLAY[comparators], drop = FALSE, position = "top") +
    scale_y_discrete(labels = function(x) stringr::str_wrap(row_label(x), 28), drop = FALSE) +
    labs(x = NULL, y = NULL) + theme_manuscript() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text.y = element_text(size = 10), axis.text.x = element_text(size = 10),
          legend.position = "right", plot.caption = element_text(size = 8))
}

# Mean within-cancer modality rank (1 = best); row labels carry endpoint counts.
build_within_cancer_rank_heatmap <- function(summary, cancers, modalities, cancer_labels = NULL) {
  row_label <- if (is.null(cancer_labels)) identity else function(x) unname(cancer_labels[x])
  n_by_cancer <- distinct(summary, cancer_type, n_endpoints)
  grid <- expand.grid(cancer_type = cancers, modality = modalities, stringsAsFactors = FALSE)
  d <- left_join(grid, summary, by = c("cancer_type", "modality")) %>%
    mutate(cancer_type = factor(cancer_type, levels = rev(cancers)),
           modality = factor(modality, levels = modalities),
           label = ifelse(is.na(mean_rank), "Unavailable",
                          sprintf("%.2f\n[%.1f-%.1f]", mean_rank, q25_rank, q75_rank)))
  y_labels <- function(x) {
    n <- n_by_cancer$n_endpoints[match(x, n_by_cancer$cancer_type)]
    paste0(stringr::str_wrap(row_label(x), 28), ifelse(is.na(n), "", sprintf("\n(n=%d)", n)))
  }
  ggplot(d, aes(modality, cancer_type, fill = mean_rank)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = label), size = MANUSCRIPT_SMALL_TEXT_SIZE, lineheight = 1.05) +
    scale_fill_gradient2(low = "#A5C9E2", mid = "white", high = "#E8B0A2",
                         midpoint = (length(modalities) + 1) / 2,
                         limits = c(1, length(modalities)), na.value = "grey92",
                         name = "Mean rank\n(1 = best)") +
    scale_x_discrete(labels = MODALITY_DISPLAY[modalities], drop = FALSE, position = "top") +
    scale_y_discrete(labels = y_labels, drop = FALSE) +
    labs(x = NULL, y = NULL) + theme_manuscript() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text.y = element_text(size = 10), axis.text.x = element_text(size = 10),
          legend.position = "right", plot.caption = element_text(size = 8))
}

RANK_PLOT_CAPTION <- paste(
  "Cell: mean rank [IQR] across endpoints with every modality evaluable in that cancer type.",
  "All modalities share patients and comparable pairs within each cancer type and endpoint.", sep = "\n")

# Ranks for every cancer type (paged) and for SELECTED_CANCER_TYPES (one page).
render_figure3_within_cancer_ranks <- function() {
  stems <- c(all = "figS3_within_cancer_rank_cindex", selected = "figS3_within_cancer_rank_selected_cindex")
  for (stem in stems) clear_within_cancer_report(stem, "figure3")
  metrics <- read_within_cancer_data("fig3_within_cancer_modality_cindex.csv")
  if (!nrow(metrics)) {
    message("[figure3 rank supplement] SKIPPED: no within-cancer results; run figures.prep.within_cancer")
    return(invisible(NULL))
  }
  excluded <- within_cancer_exclusions()
  ranks <- within_cancer_modality_ranks(metrics, excluded)
  summary <- summarize_within_cancer_ranks(ranks)
  if (!nrow(summary)) {
    message("[figure3 rank supplement] SKIPPED: no endpoint has every modality eligible within a cancer type")
    return(invisible(NULL))
  }
  modalities <- intersect(MODALITY_ORDER, unique(summary$modality))

  pages <- within_cancer_pages(summary)
  for (i in seq_along(pages)) {
    p <- build_within_cancer_rank_heatmap(summary, pages[[i]], modalities) +
      labs(title = "Modality rank within cancer types",
           subtitle = sprintf("Mean C-index rank (1 = best) | Page %d of %d", i, length(pages)),
           caption = RANK_PLOT_CAPTION)
    name <- if (i == 1L) stems[["all"]] else paste0(stems[["all"]], "_page", i)
    save_panel(p, name, "figure3", width = 11, height = 2.2 + 0.72 * length(pages[[i]]))
  }
  save_within_cancer_report(summary, within_cancer_rank_caption(summary, length(modalities), length(excluded)),
                            stems[["all"]])

  selected <- select_cancer_types(summary, "figure3 selected rank supplement")
  if (!nrow(selected)) {
    message("[figure3 selected rank supplement] SKIPPED: no ranked endpoints for the selected cancer types")
    return(invisible(summary))
  }
  cancers <- names(SELECTED_CANCER_TYPES)
  p <- build_within_cancer_rank_heatmap(selected, cancers, modalities, cancer_labels = SELECTED_CANCER_TYPES) +
    labs(title = "Modality rank within selected cancer types",
         subtitle = "Mean C-index rank (1 = best)", caption = RANK_PLOT_CAPTION)
  save_panel(p, stems[["selected"]], "figure3", width = 11, height = 2.2 + 0.72 * length(cancers))
  save_within_cancer_report(
    selected, within_cancer_rank_caption(selected, length(modalities), length(excluded), selected = TRUE),
    stems[["selected"]])
  invisible(summary)
}

render_figure3_within_cancer <- function() {
  stem <- "figS3_within_cancer_cindex"
  clear_within_cancer_report(stem, "figure3")
  metrics <- read_within_cancer_data("fig3_within_cancer_cindex.csv")
  if (!nrow(metrics)) {
    message("[figure3 supplement] SKIPPED: no within-cancer results; run figures.prep.within_cancer")
    return(invisible(NULL))
  }
  comparators <- setdiff(MODALITY_ORDER, "text")
  excluded <- within_cancer_exclusions()
  summary <- summarize_within_cancer(metrics, comparators, excluded)
  if (!nrow(summary)) {
    message("[figure3 supplement] SKIPPED: no eligible within-cancer modality comparisons")
    return(invisible(NULL))
  }
  pages <- within_cancer_pages(summary)
  # Use the same symmetric color scale across all pages, including all-zero runs.
  limit <- max(0.01, max(abs(summary$median_delta_cindex)))
  for (i in seq_along(pages)) {
    p <- build_within_cancer_modality_heatmap(summary, pages[[i]], comparators, limit) +
      labs(title = "Text versus other modalities within cancer types",
           subtitle = sprintf("Cell: median Text / Comparator C-index; median paired delta; endpoint n | Page %d of %d", i, length(pages)),
           caption = paste("Positive delta favors text. Comparisons use matched held-out patients; endpoint sets can differ.",
                           "C-indices use comparable pairs within joint outer-fold blocks. Summaries are descriptive.", sep = "\n"))
    name <- if (i == 1L) stem else paste0(stem, "_page", i)
    save_panel(p, name, "figure3", width = 12, height = 2.2 + 0.72 * length(pages[[i]]))
  }
  caption <- within_cancer_caption(3, metrics, summary, length(excluded))
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

# The same heatmap restricted to SELECTED_CANCER_TYPES, on a single page. Every
# selected type keeps a row, so an absent type reads as unavailable.
render_figure3_within_cancer_selected <- function() {
  stem <- "figS3_within_cancer_selected_cindex"
  clear_within_cancer_report(stem, "figure3")
  metrics <- select_cancer_types(read_within_cancer_data("fig3_within_cancer_cindex.csv"),
                                 "figure3 selected supplement")
  if (!nrow(metrics)) {
    message("[figure3 selected supplement] SKIPPED: no within-cancer results for the selected cancer types")
    return(invisible(NULL))
  }
  comparators <- setdiff(MODALITY_ORDER, "text")
  excluded <- within_cancer_exclusions()
  summary <- summarize_within_cancer(metrics, comparators, excluded)
  if (!nrow(summary)) {
    message("[figure3 selected supplement] SKIPPED: no eligible modality comparisons")
    return(invisible(NULL))
  }
  cancers <- names(SELECTED_CANCER_TYPES)
  limit <- max(0.01, max(abs(summary$median_delta_cindex)))
  p <- build_within_cancer_modality_heatmap(summary, cancers, comparators, limit,
                                            cancer_labels = SELECTED_CANCER_TYPES) +
    labs(title = "Text versus other modalities within selected cancer types",
         subtitle = "Cell: median Text / Comparator C-index; median paired delta; endpoint n",
         caption = paste("Positive delta favors text. Comparisons use matched held-out patients; endpoint sets can differ.",
                         "C-indices use comparable pairs within joint outer-fold blocks. Summaries are descriptive.", sep = "\n"))
  save_panel(p, stem, "figure3", width = 12, height = 2.2 + 0.72 * length(cancers))
  caption <- within_cancer_caption(3, metrics, summary, length(excluded), selected = TRUE)
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

render_figure3_within_cancer()
render_figure3_within_cancer_selected()
render_figure3_within_cancer_ranks()
