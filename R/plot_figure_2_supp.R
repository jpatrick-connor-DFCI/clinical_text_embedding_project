# Render Figure 2 supplement: within-stage KM curves stratified by overall risk score.
#
# Shows that the overall (text) risk score separates survival within stage-defined
# strata. Two panels, each stratified by the patient's
# overall risk-score quartile (the same quartiles used in Figure 2G, defined across
# the known-stage cohort — i.e. NOT re-binned within stage):
#   A  Stage IV patients, by overall risk-score quartile
#   B  Stages I-II patients pooled, by overall risk-score quartile
#
# Reuses fig2_km_stage_vs_risk.csv and fig2_stage_vs_risk_{cindex_by_stage,auc}.csv
# (all written by prep_figure_2.py); the latter two carry the per-stage annotation
# for the active metric (see figure_utils.R::METRIC — MANUSCRIPT_METRIC=cindex|auc).

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(scales); library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")  # provides tidy_km, logrank_p, step_ci_df


# ============================================================================
# One KM panel: a stage-defined stratum, by overall risk-score quartile (+95% CI)
# ============================================================================
RISK_QUARTILE_COLORS <- setNames(ORDINAL4, c("Q1", "Q2", "Q3", "Q4"))

build_stage_panel <- function(df, perf_df, stage_values, stage_label, perf_group,
                              title_text, metric = METRIC) {
  # Test the input frame before filtering: an empty tibble from a missing CSV has
  # no columns at all, so filter(stage_group %in% ...) would error on the absent
  # column rather than falling through to the skip below.
  if (nrow(df) == 0) return(placeholder_panel(paste0("no ", stage_label, " patients")))
  sub <- df %>% filter(stage_group %in% stage_values)
  if (nrow(sub) == 0) return(placeholder_panel(paste0("no ", stage_label, " patients")))
  sub <- sub %>% mutate(risk_quartile = factor(risk_quartile,
                                               levels = names(RISK_QUARTILE_COLORS)))

  fit <- survfit2(Surv(months, death) ~ risk_quartile, data = sub)
  td  <- tidy_km(fit) %>%
    mutate(stratum = factor(stratum, levels = names(RISK_QUARTILE_COLORS)))
  td_ci <- step_ci_df(td, "stratum")

  lr  <- logrank_p(sub, "months", "death", "risk_quartile")
  # Within-stage performance of the text risk score for OS (fig2_stage_vs_risk_auc.csv
  # or fig2_stage_vs_risk_cindex_by_stage.csv, both precomputed in prep_figure_2.py),
  # matching whichever metric is active (MANUSCRIPT_METRIC=cindex|auc).
  perf_col <- if (metric == "cindex") "cindex" else "mean_auc"
  perf <- if (nrow(perf_df) > 0) perf_df[[perf_col]][perf_df$stage_group == perf_group][1] else NA_real_
  ann <- sprintf("n=%s\n%s=%.3f\nlogrank p=%.1e",
                 scales::comma(nrow(sub)), metric_label(metric), perf, lr)

  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.15, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = RISK_QUARTILE_COLORS, name = "Overall risk") +
    scale_fill_manual(values = RISK_QUARTILE_COLORS, guide = "none") +
    coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
    annotate("text", x = 1, y = 0.06, label = ann,
             hjust = 0, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE,
             fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment", y = "Event-free survival", title = title_text) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.97),
          legend.justification = c(1, 1),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# Compose supplementary figure
# ============================================================================
d <- load_figure_data("fig2_km_stage_vs_risk.csv")
if (nrow(d) > 0) {
  d <- d %>% mutate(months = tt_death / 30.44, death = as.integer(death))
}
perf_csv <- if (METRIC == "cindex") "fig2_stage_vs_risk_cindex_by_stage.csv" else "fig2_stage_vs_risk_auc.csv"
perf_df <- load_figure_data(perf_csv)

pS_iv <- build_stage_panel(
  d, perf_df, "IV", "Stage IV", "IV",
  "Stage IV: survival by overall risk-score quartile")
pS_i_ii <- build_stage_panel(
  d, perf_df, c("I", "II"), "Stages I-II", "I-II",
  "Stages I-II: survival by overall risk-score quartile")

.tag <- metric_tag(METRIC)
save_panel(pS_iv, paste0("figS2_stage4_by_risk", .tag), group = "figure2", width = 7.2, height = 6.0)
save_panel(pS_i_ii, paste0("figS2_stage1_2_by_risk", .tag), group = "figure2", width = 7.2, height = 6.0)
