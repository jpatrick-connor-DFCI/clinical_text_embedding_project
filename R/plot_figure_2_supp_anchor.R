# Render Figure 2 supplement: anchor sensitivity (overall survival from sequencing date).
#
# Shows only the outputs of notebooks/2_models/05_sequencing_os_comparison.ipynb
# (death_met:death, t=0 at sequencing), in two panels on two different populations:
#   A  Full sequencing-anchored cohort: text + baseline covariates vs the baseline model.
#   B  Common full-data cohort (stage, treatment, somatic and germline all available):
#      text vs each other modality.
# The cohorts differ, so models are compared within a panel, never across.
#
# Reads fig2_anchor_sensitivity.csv (figures/prep/figure2_anchor.py).

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr)
})

source("/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project/R/figure_utils.R")


# One horizontal dot per model with its C-index printed alongside; `models` fixes
# the row order (top to bottom) and `labels`/`colors` are keyed by model.
build_cindex_dots <- function(d, models, labels, colors, title, metric = METRIC) {
  if (nrow(d) == 0) return(placeholder_panel(sprintf("no %s rows", tolower(title))))
  metric_col <- metric_suffix(metric)
  d <- d %>%
    filter(model %in% models, !is.na(.data[[metric_col]])) %>%
    mutate(model = factor(model, levels = rev(models)))
  if (nrow(d) == 0) return(placeholder_panel(sprintf("no finite %s for %s", metric_col, tolower(title))))
  n_label <- d %>% filter(!is.na(n)) %>% distinct(n, n_events)
  subtitle <- if (nrow(n_label) == 1) {
    sprintf("n = %s patients, %s deaths", format(n_label$n, big.mark = ","),
            format(n_label$n_events, big.mark = ","))
  } else if (nrow(n_label) > 1) {
    sprintf("n = %s-%s patients per model", format(min(n_label$n), big.mark = ","),
            format(max(n_label$n), big.mark = ","))
  }

  ggplot(d, aes(x = .data[[metric_col]], y = model, color = model)) +
    geom_vline(xintercept = 0.5, linetype = "dashed", color = NS_GRAY) +
    geom_segment(aes(x = 0.5, xend = .data[[metric_col]], yend = model), linewidth = 0.7) +
    geom_point(size = 3) +
    geom_text(aes(label = sprintf("%.3f", .data[[metric_col]])), hjust = -0.35,
              size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "grey25") +
    scale_color_manual(values = colors, guide = "none") +
    scale_y_discrete(labels = labels) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.15))) +
    labs(x = sprintf("Overall survival %s (sequencing anchor)", metric_label(metric)),
         y = NULL, title = title, subtitle = subtitle) +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey93"))
}


sens <- load_figure_data("fig2_anchor_sensitivity.csv")
if (nrow(sens) == 0) {
  sens <- tibble::tibble(cohort = character(), model = character(), n = numeric(),
                         n_events = numeric(), cindex = numeric())
}

pS_full <- build_cindex_dots(
  filter(sens, cohort == "full"),
  models = c("text", "base"),
  labels = c(text = "Text + base", base = "Base"),
  colors = MODEL_COLORS[c("text", "base")],
  title = "Full cohort"
)
pS_modalities <- build_cindex_dots(
  filter(sens, cohort == "common"),
  models = rev(MODALITY_ORDER),  # text on top, as in the palette's reverse order
  labels = MODALITY_DISPLAY,
  colors = MODALITY_COLORS,
  title = "Common full-data cohort"
)

.tag <- metric_tag(METRIC)
save_panel(pS_full,       paste0("figS_anchor_full", .tag),       group = "figure2", width = 6.4, height = 3.2)
save_panel(pS_modalities, paste0("figS_anchor_modalities", .tag), group = "figure2", width = 6.4, height = 4.6)
