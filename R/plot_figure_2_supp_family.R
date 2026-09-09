# Supplemental Figure 2: paired text-minus-base performance by coding family.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(scales)
})
source("R/figure_utils.R")

metrics <- load_figure_data("fig2_full_cohort_metrics.csv")
metrics <- drop_excluded_events(metrics, excluded_event_keys(metrics))
base_col <- paste0("base_", metric_suffix(METRIC))
text_col <- paste0("text_", metric_suffix(METRIC))

if (nrow(metrics) == 0 || !all(c(base_col, text_col) %in% names(metrics))) {
  p <- placeholder_panel("fig2_full_cohort_metrics.csv empty or missing metric")
} else {
  d <- metrics %>%
    filter(is.finite(.data[[base_col]]), is.finite(.data[[text_col]])) %>%
    mutate(delta = .data[[text_col]] - .data[[base_col]],
           family = factor(scheme, levels = names(SCHEME_LABELS)))
  ann <- d %>% group_by(family) %>%
    summarise(n = n(), median = median(delta), .groups = "drop") %>%
    mutate(label = sprintf("n=%d\nmedian=%+.3f", n, median))
  label_y <- max(d$delta, na.rm = TRUE) + 0.10 * diff(range(d$delta, na.rm = TRUE))
  p <- ggplot(d, aes(family, delta, fill = family)) +
    geom_violin(data = filter(d, ave(delta, family, FUN = length) >= 10),
                scale = "width", alpha = 0.45, color = "grey30") +
    geom_boxplot(width = 0.13, outlier.shape = NA, fill = "white", alpha = 0.85) +
    geom_jitter(width = 0.14, size = 0.65, alpha = 0.28) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_text(data = ann, aes(family, label_y, label = label), inherit.aes = FALSE,
              vjust = 1, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_discrete(labels = SCHEME_LABELS) +
    labs(x = NULL, y = sprintf("Delta %s (Text - Base)", metric_label(METRIC)),
         title = "Paired Improvement by Endpoint Coding Family",
         caption = "Endpoint-level distributions are descriptive because endpoints are correlated; violins are omitted when n<10.") +
    theme_manuscript() + theme(panel.grid.major.y = element_line(color = "grey90"))
}

save_panel(p, paste0("figS2_coding_family_violin", metric_tag(METRIC)),
           group = "figure2", width = 8.2, height = 6.0)
