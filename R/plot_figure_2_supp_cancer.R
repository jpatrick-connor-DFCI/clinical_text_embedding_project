# Figure 2 supplement: pooled-model text/base predictions evaluated within cancer types.
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

render_figure2_within_cancer <- function() {
  stem <- "figS2_within_cancer_cindex"
  clear_within_cancer_report(stem, "figure2")
  metrics <- read_within_cancer_data("fig2_within_cancer_cindex.csv")
  if (!nrow(metrics)) {
    message("[figure2 supplement] SKIPPED: no within-cancer results; run figures.prep.within_cancer")
    return(invisible(NULL))
  }
  excluded <- within_cancer_exclusions()
  d <- eligible_within_cancer(metrics, "base", excluded)
  summary <- summarize_within_cancer(metrics, "base", excluded)
  if (!nrow(summary)) {
    message("[figure2 supplement] SKIPPED: no eligible within-cancer text/base comparisons")
    return(invisible(NULL))
  }
  pages <- within_cancer_pages(summary)
  for (i in seq_along(pages)) {
    cancers <- pages[[i]]
    ann <- summary %>% filter(cancer_type %in% cancers) %>%
      mutate(label = sprintf("n=%d endpoints\nmedian delta=%+.3f", n_endpoints, median_delta_cindex))
    p <- ggplot(filter(d, cancer_type %in% cancers), aes(comparator_cindex, text_cindex)) +
      geom_abline(slope = 1, intercept = 0, color = "grey55", linetype = "dashed") +
      geom_point(aes(color = scheme), size = 1.25, alpha = 0.6) +
      geom_text(data = ann, aes(x = 0.03, y = 0.97, label = label), inherit.aes = FALSE,
                hjust = 0, vjust = 1, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
      facet_wrap(~cancer_type, ncol = 3, labeller = label_wrap_gen(30)) +
      scale_color_manual(values = SCHEME_COLORS, labels = SCHEME_LABELS, name = NULL) +
      scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
      scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
      coord_fixed() +
      labs(x = "Base C-index", y = "Text C-index",
           title = "Text versus base performance within cancer types",
           subtitle = sprintf("Existing pan-cancer models; matched held-out patients | Page %d of %d", i, length(pages)),
           caption = "C-indices use comparable pairs within joint outer-fold blocks. Endpoint summaries are descriptive.") +
      theme_manuscript() + theme(legend.position = "bottom", plot.caption = element_text(size = 8))
    name <- if (i == 1L) stem else paste0(stem, "_page", i)
    save_panel(p, name, "figure2", width = 10, height = 1.6 + 3.15 * ceiling(length(cancers) / 3))
  }
  caption <- within_cancer_caption(2, metrics, summary, length(excluded))
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

render_figure2_within_cancer()
