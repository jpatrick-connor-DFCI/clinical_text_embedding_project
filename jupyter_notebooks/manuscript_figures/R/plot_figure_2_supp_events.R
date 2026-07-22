# Render Figure 2 supplement: remaining per-event held-out-risk KM panels
# (ranks 2-3 of each category) for the Δ C-index top-3 events shown in Figure 2 H-M.
#
# Figure 2 itself shows only the rank-1 (largest positive Δ C-index) event's KM
# curve per category (mets / ICD10 / phecodes); this supplement shows the
# rank-2 and rank-3 events for each category as a 3x2 grid.
#
# C-index only — this script does not honor MANUSCRIPT_METRIC; it always reads
# fig2_scheme_delta_topk.csv / fig2_scheme_event_km.csv (ranked by delta
# C-index, largest positive delta only) and produces a single un-suffixed
# output.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(survival); library(ggsurvfit)
})

script_dir <- local({
  if (exists("R_DIR", envir = globalenv(), inherits = FALSE)) {
    return(get("R_DIR", envir = globalenv()))
  }
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
  if (length(fa) && nzchar(fa[1])) return(dirname(normalizePath(fa[1])))
  for (n in seq_len(sys.nframe())) {
    ofile <- sys.frame(n)$ofile
    if (!is.null(ofile) && nzchar(ofile)) return(dirname(normalizePath(ofile)))
  }
  stop("Could not determine script directory (set R_DIR in globalenv, run via Rscript, or source() directly)")
})
source(file.path(script_dir, "figure_utils.R"))  # provides tidy_km, logrank_p, step_ci_df
FIGURE_GROUP <- "figure2"


# ============================================================================
# Shared KM-by-risk-tertile renderer (text solid, base dashed) — local copy of
# plot_figure_2.R::km_tertile_panel so this script has no cross-script
# dependency (mirrors the plot_figure_2_supp.R / plot_figure_4_supp.R pattern
# of keeping their own local helpers).
# ============================================================================
km_tertile_panel <- function(km, time_col, event_col, title) {
  if (nrow(km) == 0) return(placeholder_panel("no data for KM panel"))
  km <- km %>% mutate(months = .data[[time_col]] / 30.44,
                      .event = as.integer(.data[[event_col]]))

  fit_t <- survfit2(Surv(months, .event) ~ text_tertile, data = km)
  fit_b <- survfit2(Surv(months, .event) ~ base_tertile, data = km)
  td <- bind_rows(
    tidy_km(fit_t) %>% mutate(model = "text"),
    tidy_km(fit_b) %>% mutate(model = "base")
  ) %>%
    mutate(stratum = factor(stratum, levels = c("low", "mid", "high")),
           model   = factor(model,   levels = c("text", "base")))

  lr_t <- logrank_p(km, "months", ".event", "text_tertile")
  lr_b <- logrank_p(km, "months", ".event", "base_tertile")

  td_ci <- step_ci_df(td, c("stratum", "model"))

  ggplot(td, aes(x = time, y = estimate, color = stratum, linetype = model)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.12, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = RISK_COLORS, name = NULL) +
    scale_fill_manual(values = RISK_COLORS, guide = "none") +
    scale_linetype_manual(values = c(text = "solid", base = "dashed"), guide = "none") +
    coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
    annotate("text", x = 1, y = 0.06,
             label = sprintf("text logrank p=%.1e\nbase logrank p=%.1e", lr_t, lr_b),
             hjust = 0, vjust = 0, size = 2.6, fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment", y = "Overall survival", title = title) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.98),
          legend.justification = c(1, 1),
          legend.background = element_rect(fill = "white", color = NA),
          legend.spacing.y = unit(0.05, "in"))
}

build_event_km_panel <- function(km_data, topk, category, rank_n) {
  ev <- topk %>% filter(category == !!category, rank == rank_n)
  if (nrow(ev) == 0) return(placeholder_panel(sprintf("no rank-%d %s event", rank_n, category)))
  km <- km_data %>% filter(category == !!category, scheme == ev$scheme[1], event == ev$event[1])
  if (nrow(km) == 0) return(placeholder_panel(sprintf("%s: no held-out risk scores yet", ev$event_lbl[1])))
  km_tertile_panel(km, "tt", "event_flag",
                   sprintf("%s\n(text solid, base dashed)", ev$event_lbl[1]))
}


# ============================================================================
# Compose supplementary figure: 3x2 grid, rows = category, cols = rank2/rank3
# ============================================================================
topk    <- load_figure_data("fig2_scheme_delta_topk.csv")
km_data <- load_figure_data("fig2_scheme_event_km.csv")

pS_mets2     <- build_event_km_panel(km_data, topk, "mets", 2)
pS_mets3     <- build_event_km_panel(km_data, topk, "mets", 3)
pS_icd2      <- build_event_km_panel(km_data, topk, "ICD10", 2)
pS_icd3      <- build_event_km_panel(km_data, topk, "ICD10", 3)
pS_phecodes2 <- build_event_km_panel(km_data, topk, "phecodes", 2)
pS_phecodes3 <- build_event_km_panel(km_data, topk, "phecodes", 3)

save_panel(pS_mets2,     "figS_scheme_km_mets2", width = 5.6, height = 4.6)
save_panel(pS_mets3,     "figS_scheme_km_mets3", width = 5.6, height = 4.6)
save_panel(pS_icd2,      "figS_scheme_km_icd2", width = 5.6, height = 4.6)
save_panel(pS_icd3,      "figS_scheme_km_icd3", width = 5.6, height = 4.6)
save_panel(pS_phecodes2, "figS_scheme_km_phecodes2", width = 5.6, height = 4.6)
save_panel(pS_phecodes3, "figS_scheme_km_phecodes3", width = 5.6, height = 4.6)
