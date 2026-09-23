# Figure 2 supplement: pooled-model text/base predictions evaluated within cancer types.
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

# `d` holds endpoint rows and `summary` the per-cancer rows for the facets shown;
# facets follow the factor levels of `d$cancer_type` when it is a factor.
build_within_cancer_scatter <- function(d, summary, subtitle, ncol = 3) {
  ann <- summary %>%
    mutate(label = sprintf("median delta=%+.3f", median_delta_cindex))
  ggplot(d, aes(comparator_cindex, text_cindex)) +
    geom_abline(slope = 1, intercept = 0, color = "grey55", linetype = "dashed") +
    geom_point(aes(color = scheme), size = 1.25, alpha = 0.6) +
    geom_text(data = ann, aes(x = 0.97, y = 0.03, label = label), inherit.aes = FALSE,
              hjust = 1, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    facet_wrap(~cancer_type, ncol = ncol, labeller = label_wrap_gen(30)) +
    scale_color_manual(values = SCHEME_COLORS, labels = SCHEME_LABELS, name = NULL) +
    scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
    scale_y_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1)) +
    coord_fixed() +
    labs(x = "Base C-index", y = "Text C-index",
         title = "Text versus base performance within cancer types",
         subtitle = subtitle,
         caption = "C-indices use comparable pairs within joint outer-fold blocks. Endpoint summaries are descriptive.") +
    theme_manuscript() + theme(legend.position = "bottom", plot.caption = element_text(size = 8))
}

scatter_height <- function(n_cancers) 1.6 + 3.15 * ceiling(n_cancers / 3)

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
    p <- build_within_cancer_scatter(
      filter(d, cancer_type %in% cancers), filter(summary, cancer_type %in% cancers),
      sprintf("Existing pan-cancer models; matched held-out patients | Page %d of %d", i, length(pages)))
    name <- if (i == 1L) stem else paste0(stem, "_page", i)
    save_panel(p, name, "figure2", width = 10, height = scatter_height(length(cancers)))
  }
  caption <- within_cancer_caption(2, metrics, summary, length(excluded))
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

# The same comparison restricted to SELECTED_CANCER_TYPES, on a single page.
render_figure2_within_cancer_selected <- function() {
  stem <- "figS2_within_cancer_selected_cindex"
  clear_within_cancer_report(stem, "figure2")
  metrics <- select_cancer_types(read_within_cancer_data("fig2_within_cancer_cindex.csv"),
                                 "figure2 selected supplement")
  if (!nrow(metrics)) {
    message("[figure2 selected supplement] SKIPPED: no within-cancer results for the selected cancer types")
    return(invisible(NULL))
  }
  excluded <- within_cancer_exclusions()
  summary <- summarize_within_cancer(metrics, "base", excluded)
  if (!nrow(summary)) {
    message("[figure2 selected supplement] SKIPPED: no eligible text/base comparisons")
    return(invisible(NULL))
  }
  as_display <- function(x) factor(SELECTED_CANCER_TYPES[x], levels = SELECTED_CANCER_TYPES)
  d <- eligible_within_cancer(metrics, "base", excluded) %>% mutate(cancer_type = as_display(cancer_type))
  shown <- summary %>% mutate(cancer_type = as_display(cancer_type))
  p <- build_within_cancer_scatter(d, shown, "Existing pan-cancer models; matched held-out patients")
  save_panel(p, stem, "figure2", width = 10, height = scatter_height(n_distinct(shown$cancer_type)))
  caption <- within_cancer_caption(2, metrics, summary, length(excluded), selected = TRUE)
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

# Selected v2: every recorded cancer type except the pooled OTHER category, on a
# single page of 7 columns (3 rows for 21 types).
V2_NCOL <- 7L

cancer_display_name <- function(code) {
  ifelse(code %in% names(SELECTED_CANCER_TYPES), SELECTED_CANCER_TYPES[code],
         ifelse(nchar(code) <= 3, code, tools::toTitleCase(tolower(gsub("_", " ", code)))))
}

render_figure2_within_cancer_selected_v2 <- function() {
  stem <- "figS2_within_cancer_selected_v2_cindex"
  clear_within_cancer_report(stem, "figure2")
  metrics <- read_within_cancer_data("fig2_within_cancer_cindex.csv")
  if (nrow(metrics)) {
    metrics <- metrics %>%
      mutate(cancer_type = toupper(trimws(as.character(cancer_type)))) %>%
      filter(cancer_type != "OTHER")
  }
  excluded <- within_cancer_exclusions()
  summary <- summarize_within_cancer(metrics, "base", excluded)
  if (!nrow(summary)) {
    message("[figure2 selected v2 supplement] SKIPPED: no eligible text/base comparisons outside OTHER")
    return(invisible(NULL))
  }
  levels <- sort(unique(cancer_display_name(summary$cancer_type)))
  as_display <- function(x) factor(cancer_display_name(x), levels = levels)
  d <- eligible_within_cancer(metrics, "base", excluded) %>% mutate(cancer_type = as_display(cancer_type))
  shown <- summary %>% mutate(cancer_type = as_display(cancer_type))
  p <- build_within_cancer_scatter(d, shown, "Existing pan-cancer models; matched held-out patients",
                                   ncol = V2_NCOL)
  n_rows <- ceiling(length(levels) / V2_NCOL)
  save_panel(p, stem, "figure2", width = 0.6 + 2.3 * V2_NCOL, height = 1.9 + 2.4 * n_rows)
  caption <- within_cancer_caption(2, metrics, summary, length(excluded))
  caption <- paste(caption, "The pooled OTHER cancer-type category is omitted.", sep = "\n\n")
  save_within_cancer_report(summary, caption, stem)
  invisible(summary)
}

render_figure2_within_cancer()
render_figure2_within_cancer_selected()
render_figure2_within_cancer_selected_v2()
