# Render Figure 2 supplement: anchor sensitivity (treatment-date vs sequencing-date time-zero).
#
# Two panels, each built on the both-anchors-eligible intersection cohort (so a shift is
# attributable to the timescale, not to cohort composition):
#   A  Paired scatter of the text model's per-event metric (C-index or mean AUC(t), per
#      figure_utils.R::METRIC) under the sequencing anchor vs the treatment anchor,
#      one point per (scheme, event); diagonal = no sensitivity to anchor choice.
#   B  Delta (sequencing - treatment) per event, ordered by scheme, with a zero
#      reference line — shows the direction/magnitude of any anchor-driven shift.
#
# Reuses fig2_anchor_sensitivity.csv (prep_figure_2_anchor.py / figure2_anchor.py),
# restricted to model == "text" and cohort == "intersection" rows.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
})

source("R/figure_utils.R")


# ============================================================================
# Panel A: paired scatter, sequencing vs treatment
# ============================================================================
build_scatter_panel <- function(wide, metric = METRIC) {
  if (nrow(wide) == 0) return(placeholder_panel("no both-anchor events"))
  metric_col <- if (metric == "cindex") "cindex" else "mean_auc"
  lims <- range(c(wide[[paste0(metric_col, "_treatment")]],
                  wide[[paste0(metric_col, "_sequencing")]]), na.rm = TRUE)

  ggplot(wide, aes(x = .data[[paste0(metric_col, "_treatment")]],
                    y = .data[[paste0(metric_col, "_sequencing")]],
                    color = scheme)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#999999") +
    geom_point(size = 2, alpha = 0.75) +
    scale_color_manual(values = SCHEME_COLORS, labels = SCHEME_LABELS, name = NULL) +
    coord_equal(xlim = lims, ylim = lims) +
    labs(x = sprintf("%s (treatment anchor)", metric_label(metric)),
         y = sprintf("%s (sequencing anchor)", metric_label(metric)),
         title = "Anchor sensitivity: paired event metrics") +
    theme_manuscript() +
    theme(legend.position = "bottom")
}


# ============================================================================
# Panel B: per-event delta (sequencing - treatment), ordered by scheme
# ============================================================================
build_delta_panel <- function(wide, metric = METRIC) {
  if (nrow(wide) == 0) return(placeholder_panel("no both-anchor events"))
  metric_col <- if (metric == "cindex") "cindex" else "mean_auc"
  d <- wide %>%
    mutate(delta = .data[[paste0(metric_col, "_sequencing")]] - .data[[paste0(metric_col, "_treatment")]]) %>%
    filter(!is.na(delta)) %>%
    arrange(scheme, delta) %>%
    mutate(row_id = row_number())

  ggplot(d, aes(x = row_id, y = delta, color = scheme)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "#999999") +
    geom_point(size = 1.6, alpha = 0.8) +
    scale_color_manual(values = SCHEME_COLORS, labels = SCHEME_LABELS, name = NULL) +
    labs(x = "Event (ordered by scheme, then delta)",
         y = sprintf("Delta %s (sequencing - treatment)", metric_label(metric)),
         title = "Anchor sensitivity: per-event delta") +
    theme_manuscript() +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          legend.position = "bottom")
}


# ============================================================================
# Compose supplementary figure
# ============================================================================
sens <- load_figure_data("fig2_anchor_sensitivity.csv")

# Same full-cohort exclusion as figure 2, recomputed here (separate process).
sens <- drop_excluded_events(
  sens, excluded_event_keys(load_figure_data("fig2_full_cohort_metrics.csv")))

wide <- tibble::tibble()
if (nrow(sens) > 0) {
  text_intersection <- sens %>% filter(model == "text", cohort == "intersection")
  wide <- text_intersection %>%
    select(anchor, scheme, event, cindex, mean_auc) %>%
    pivot_wider(names_from = anchor, values_from = c(cindex, mean_auc)) %>%
    filter(!is.na(cindex_treatment), !is.na(cindex_sequencing))
}

pS_scatter <- build_scatter_panel(wide)
pS_delta   <- build_delta_panel(wide)

# Both panels plot the active metric (cindex vs mean_auc) and are trimmed on it,
# so each render needs its own filename -- untagged, the two runs collided.
.tag <- metric_tag(METRIC)
save_panel(pS_scatter, paste0("figS_anchor_scatter", .tag), group = "figure2", width = 6.4, height = 6.4)
save_panel(pS_delta,   paste0("figS_anchor_delta", .tag),   group = "figure2", width = 7.8, height = 5.2)
