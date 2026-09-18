# Figure 3 supplement: paired within-cancer text/other-modality performance.
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

build_within_cancer_modality_heatmap <- function(summary, cancers, comparators, limit) {
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
    scale_y_discrete(labels = function(x) stringr::str_wrap(x, 28), drop = FALSE) +
    labs(x = NULL, y = NULL) + theme_manuscript() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text.y = element_text(size = 10), axis.text.x = element_text(size = 10),
          legend.position = "right", plot.caption = element_text(size = 8))
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

render_figure3_within_cancer()
