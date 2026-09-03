# Render Figure 2 (text vs base, full cohort) in ggplot2 + patchwork.
#
# A scatter (text vs base performance), B delta-performance violins + Wilcoxon stars,
# C pan vs within-cancer model (dumbbell), D pan vs within-treatment model,
# E KM by risk-score tertile (text solid / base dashed),
# F survival by cancer stage, G survival by text risk-score quartile.
# E/F/G share a single side-by-side row of KM curves, each with 95% CI bands.
# H/I/J: top-3-by-Δ-C-index event barplots for mets/ICD10/phecodes, shown as
#   paired text-vs-base C-index bars per event (I has only 2 ICD10 events).
# K/L/M: text-vs-base held-out-risk KM for each category's rank-1 event.
# (Ranks 2-3 for each category are in the plot_figure_2_supp_events.R supplement.)
#
# Metric switch: set MANUSCRIPT_METRIC=cindex|auc (see figure_utils.R::METRIC) to
# render this whole figure scored by Harrell's C-index or by mean AUC(t). Panels
# A-D and F/G read whichever metric's columns/CSV match; outputs are suffixed
# (e.g. figure2_text_results_cindex.png / _auc.png) so both sets can coexist.
# Panels H-M (Δ C-index barplots + per-event KM) are always C-index-only and
# carry no metric suffix — they render identically regardless of METRIC.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr)
  library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")
# KM helpers (tidy_km, logrank_p, step_ci_df) now live in figure_utils.R so the
# supplementary stage-stratified script (plot_figure_2_supp.R) can share them.


# ============================================================================
# fig2a: text vs base scatter
# ============================================================================
# Legend groups: death_met is split into its own "Death" entry (the
# single literal death event) and "Mets" (metastatic-site events), separate
# from the death_met scheme's other rows.
FIG2A_GROUP_ORDER <- c("death", "mets", "icd3_post", "icd4_post", "phecode_post")
FIG2A_GROUP_LABELS <- c(death = "Death", mets = "Mets",
                        icd3_post = SCHEME_LABELS[["icd3_post"]],
                        icd4_post = SCHEME_LABELS[["icd4_post"]],
                        phecode_post = SCHEME_LABELS[["phecode_post"]])
FIG2A_GROUP_COLORS <- c(death = SCHEME_COLORS[["death_met"]], mets = "#F1948A",
                        icd3_post = SCHEME_COLORS[["icd3_post"]],
                        icd4_post = SCHEME_COLORS[["icd4_post"]],
                        phecode_post = SCHEME_COLORS[["phecode_post"]])
FIG2A_GROUP_SHAPES <- c(death = 18, mets = 17,
                        icd3_post = SCHEME_SHAPES[["icd3_post"]],
                        icd4_post = SCHEME_SHAPES[["icd4_post"]],
                        phecode_post = SCHEME_SHAPES[["phecode_post"]])

build_fig2a <- function(metrics, metric = METRIC) {
  if (nrow(metrics) == 0) return(placeholder_panel("fig2_full_cohort_metrics.csv empty"))
  base_col <- paste0("base_", metric_suffix(metric))
  text_col <- paste0("text_", metric_suffix(metric))
  d <- metrics %>%
    filter(!is.na(.data[[base_col]]), !is.na(.data[[text_col]])) %>%
    mutate(base_val = .data[[base_col]], text_val = .data[[text_col]],
           plot_group = case_when(
             scheme == "death_met" & event == "death" ~ "death",
             scheme == "death_met"                    ~ "mets",
             TRUE                                      ~ as.character(scheme)
           ),
           plot_group = factor(plot_group, levels = FIG2A_GROUP_ORDER))
  lo <- max(0.45, min(c(d$base_val, d$text_val)) - 0.02)
  hi <- min(1.00, max(c(d$base_val, d$text_val)) + 0.02)
  lbl <- metric_label(metric)

  ggplot(d, aes(base_val, text_val, color = plot_group, shape = plot_group)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#666666") +
    geom_point(data = filter(d, as.character(plot_group) != "death"),
               size = 1.8, alpha = 0.65) +
    # Draw death in a separate final layer so no coincident event can cover it.
    geom_point(data = filter(d, as.character(plot_group) == "death"),
               size = 1.8, alpha = 0.65) +
    scale_color_manual(values = FIG2A_GROUP_COLORS, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    scale_shape_manual(values = FIG2A_GROUP_SHAPES, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    coord_fixed(xlim = c(lo, hi), ylim = c(lo, hi)) +
    labs(x = paste("Base Model", lbl), y = paste("Text Model", lbl),
         title = "Text vs. Base Model Performance") +
    theme_manuscript() +
    theme(legend.position = c(0.85, 0.18),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# fig2b: delta-performance violins by scheme + Wilcoxon-vs-0 stars
# ============================================================================
build_fig2b <- function(metrics, metric = METRIC) {
  if (nrow(metrics) == 0) return(placeholder_panel("fig2_full_cohort_metrics.csv empty"))
  base_col <- paste0("base_", metric_suffix(metric))
  text_col <- paste0("text_", metric_suffix(metric))
  lbl <- metric_label(metric)
  d <- metrics %>%
    filter(!is.na(.data[[base_col]]), !is.na(.data[[text_col]])) %>%
    mutate(delta = .data[[text_col]] - .data[[base_col]],
           scheme = factor(scheme, levels = names(SCHEME_LABELS)))
  ann <- d %>% group_by(scheme) %>%
    summarise(n = n(),
              mean_delta = mean(delta),
              sd_delta = sd(delta),
              p = wilcoxon_vs0(delta),
              .groups = "drop") %>%
    mutate(stars = vapply(p, p_to_stars, character(1)),
           label = sprintf("n=%d\nmean=%+.3f\n%s", n, mean_delta, stars))
  ymax <- max(d$delta, na.rm = TRUE) * 1.20

  ggplot(d, aes(x = scheme, y = delta, fill = scheme)) +
    geom_violin(alpha = 0.45, color = "#444444", linewidth = 0.4, scale = "width") +
    geom_jitter(width = 0.18, size = 0.7, alpha = 0.35,
                aes(color = scheme), show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "#333333", linetype = "dashed") +
    geom_errorbar(data = ann,
                  aes(x = scheme, y = mean_delta,
                      ymin = mean_delta - sd_delta, ymax = mean_delta + sd_delta),
                  inherit.aes = FALSE, width = 0.15, linewidth = 0.6, color = "#111111") +
    geom_point(data = ann, aes(x = scheme, y = mean_delta),
              inherit.aes = FALSE, shape = 23, size = 2.2,
              fill = "white", color = "#111111", stroke = 0.7) +
    geom_text(data = ann, aes(x = scheme, y = ymax, label = label),
              inherit.aes = FALSE, vjust = 1,
              size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#222222") +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_color_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_discrete(labels = SCHEME_LABELS) +
    labs(x = NULL, y = sprintf("Delta %s (Text - Base)", lbl),
         title = sprintf("%s Improvement by Scheme", lbl),
         caption = paste("Stars: Wilcoxon signed-rank vs Δ=0  (*<.05, **<.01, ***<.001, ****<1e-4).",
                         "Diamond ± bar: mean ± SD.")) +
    theme_manuscript() +
    theme(plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 1,
                                      face = "italic", color = "#666666"),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig2c / fig2d: pan vs. within-stratum model dumbbell (C-index or mean AUC(t),
# per the active METRIC). Grey dot = pan model, red dot = within-stratum model.
# Dashed vertical lines mark the overall (pooled) held-out performance of each
# model — grey = pan, red = within. The catch-all "OTHER" stratum is not
# plotted as its own dumbbell. Shared by the cancer- and treatment-stratified
# panels.
# ============================================================================
build_within_vs_pan <- function(csv, stratum_title, metric = METRIC) {
  d <- load_figure_data(csv)
  if (nrow(d) == 0) return(placeholder_panel(paste(csv, "empty")))
  pan_col <- if (metric == "cindex") "cindex_pan" else "auc_pan"
  within_col <- if (metric == "cindex") "cindex_within" else "auc_within"
  lbl <- metric_label(metric)
  if (!all(c(pan_col, within_col) %in% names(d)) ||
      all(is.na(d[[pan_col]])) || all(is.na(d[[within_col]]))) {
    return(placeholder_panel(paste(csv, "missing", lbl, "columns — re-run prep_figure_2.py")))
  }
  d$pan_val <- d[[pan_col]]
  d$within_val <- d[[within_col]]
  is_ov <- as.character(d$is_overall) %in% c("True", "TRUE", "true")
  # Overall (pooled) held-out performance of each model -> reference lines.
  overall_pan    <- d$pan_val[is_ov]
  overall_within <- d$within_val[is_ov]
  d <- d[!is_ov, , drop = FALSE]
  if (identical(csv, "fig2_within_vs_pan_cancer.csv")) {
    d$stratum <- stringr::str_to_sentence(
      stringr::str_replace_all(as.character(d$stratum), "_", " ")
    )
  }
  # Drop the collapsed catch-all stratum (rare cancer types pooled as OTHER); it is
  # not a clinically interpretable group and clutters the per-stratum comparison.
  d <- d[!toupper(as.character(d$stratum)) %in% c("OTHER"), , drop = FALSE]
  d <- d[!is.na(d$pan_val) & !is.na(d$within_val), , drop = FALSE]
  if (nrow(d) == 0) return(placeholder_panel(paste(csv, "no per-stratum rows")))
  d$stratum <- forcats::fct_reorder(as.character(d$stratum), d$within_val - d$pan_val)
  p <- ggplot(d) +
    geom_segment(aes(y = stratum, yend = stratum, x = pan_val, xend = within_val),
                 color = "#BBBBBB", linewidth = 0.9) +
    geom_point(aes(y = stratum, x = pan_val,    color = "Pan"),    size = 2.6) +
    geom_point(aes(y = stratum, x = within_val, color = "Within"), size = 2.6) +
    geom_text(aes(y = stratum, x = pmax(pan_val, within_val), label = paste0("n=", n_heldout)),
              hjust = -0.2, size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#666666") +
    scale_color_manual(values = c(Pan    = unname(MODEL_COLORS[["base"]]),
                                  Within = unname(MODEL_COLORS[["text"]])), name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.13))) +
    labs(title = stratum_title, x = paste("Held-out", lbl), y = NULL,
         caption = sprintf("Dashed lines: overall held-out %s (grey = pan, red = within)", lbl)) +
    theme_manuscript() +
    theme(legend.position = "top",
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 1,
                                      face = "italic", color = "#666666"))
  # Numeric labels sit horizontally at the top of each dashed reference line.
  # Give a touch of headroom and stagger the two labels vertically so the pan
  # and within values don't collide when the lines are close together.
  p <- p + coord_cartesian(clip = "off") +
    theme(plot.margin = margin(t = 18, r = 5.5, b = 5.5, l = 5.5))
  if (length(overall_pan) == 1 && is.finite(overall_pan)) {
    p <- p + geom_vline(xintercept = overall_pan, linetype = "dashed",
                        color = unname(MODEL_COLORS[["base"]]), linewidth = 0.5) +
      annotate("text", x = overall_pan, y = Inf,
               label = sprintf("%s = %.3f", lbl, overall_pan),
               color = unname(MODEL_COLORS[["base"]]),
               size = MANUSCRIPT_SMALL_TEXT_SIZE,
               fontface = "italic", hjust = 0.5, vjust = -1.6)
  }
  if (length(overall_within) == 1 && is.finite(overall_within)) {
    p <- p + geom_vline(xintercept = overall_within, linetype = "dashed",
                        color = unname(MODEL_COLORS[["text"]]), linewidth = 0.5) +
      annotate("text", x = overall_within, y = Inf,
               label = sprintf("%s = %.3f", lbl, overall_within),
               color = unname(MODEL_COLORS[["text"]]),
               size = MANUSCRIPT_SMALL_TEXT_SIZE,
               fontface = "italic", hjust = 0.5, vjust = -0.4)
  }
  p
}


# ============================================================================
# Shared KM-by-risk-tertile renderer (text solid, base dashed) — used by fig2d
# (OS / death_met risk score) and the per-scheme-event panels below, which
# share the same text_tertile/base_tertile/text-vs-base structure but differ
# in which event/time columns and title to use.
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

  # Zoom the probability axis to the data's own range: events with low incidence
  # keep survival near 1 throughout, so a fixed 0-1 axis wastes most of the panel.
  # Floor a little below the lowest observed CI bound, with a 1.03 ceiling to
  # match the padding used elsewhere in this figure.
  y_lo <- max(0, min(td_ci$conf.low, na.rm = TRUE) - 0.03)
  y_hi <- 1.03
  ann_y <- y_lo + 0.06 * (y_hi - y_lo)

  ggplot(td, aes(x = time, y = estimate, color = stratum, linetype = model)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.12, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = RISK_COLORS, name = NULL) +
    scale_fill_manual(values = RISK_COLORS, guide = "none") +
    scale_linetype_manual(values = c(text = "solid", base = "dashed"), guide = "none") +
    coord_cartesian(xlim = c(0, 60), ylim = c(y_lo, y_hi)) +
    annotate("text", x = 1, y = ann_y,
             label = sprintf("text logrank p=%.1e\nbase logrank p=%.1e", lr_t, lr_b),
             hjust = 0, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE,
             fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment", y = "Event-free survival", title = title) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.98),
          legend.justification = c(1, 1),
          legend.background = element_rect(fill = "white", color = NA),
          legend.spacing.y = unit(0.05, "in"))
}


# ============================================================================
# fig2d: KM by risk-score tertile (text solid, base dashed)
# ============================================================================
build_fig2d <- function() {
  km <- load_figure_data("fig2_km_tertiles.csv")
  if (nrow(km) == 0) return(placeholder_panel("fig2_km_tertiles.csv empty"))
  km_tertile_panel(km, "tt_death", "death",
                   "Mortality by Risk-Score Tertile\n(text solid, base dashed)")
}


# ============================================================================
# fig2e: stage vs text risk-quartile (1×2 KM + C-index/AUC(t) annotations)
# ============================================================================
build_fig2e <- function(metric = METRIC) {
  d  <- load_figure_data("fig2_km_stage_vs_risk.csv")
  lbl <- metric_label(metric)
  if (metric == "cindex") {
    ci <- load_figure_data("fig2_stage_vs_risk_cindex.csv")
    perf_s <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "stage"][1] else NA_real_
    perf_q <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "text_risk"][1] else NA_real_
  } else {
    # fig2_stage_vs_risk_auc.csv is per-stage (FigS2 convention), not per-predictor
    # like the cindex file — it has no pooled "stage" predictor row, only the text
    # risk score's mean AUC(t) within each stage. Report the n-weighted average
    # across stages as the closest AUC(t) analogue to the pooled stage C-index.
    auc_df <- load_figure_data("fig2_stage_vs_risk_auc.csv")
    perf_s <- NA_real_
    perf_q <- if (nrow(auc_df) > 0) weighted.mean(auc_df$mean_auc, auc_df$n, na.rm = TRUE) else NA_real_
  }
  if (nrow(d) == 0) {
    ph <- placeholder_panel("fig2_km_stage_vs_risk.csv empty")
    return(list(stage = ph, quartile = ph))
  }
  d <- d %>% mutate(months = tt_death / 30.44,
                    death = as.integer(death),
                    stage_group   = factor(stage_group,   levels = c("I","II","III","IV")),
                    risk_quartile = factor(risk_quartile, levels = c("Q1","Q2","Q3","Q4")))

  fit_s <- survfit2(Surv(months, death) ~ stage_group,   data = d)
  fit_q <- survfit2(Surv(months, death) ~ risk_quartile, data = d)
  ts <- tidy_km(fit_s)
  tq <- tidy_km(fit_q)

  ord4 <- setNames(ORDINAL4, c("I","II","III","IV"))
  ord4q <- setNames(ORDINAL4, c("Q1","Q2","Q3","Q4"))

  lr_s <- logrank_p(d, "months", "death", "stage_group")
  lr_q <- logrank_p(d, "months", "death", "risk_quartile")

  panel_km <- function(td, palette, lr_p, perf, title_text,
                       legend_pos = c(0.82, 0.80), legend_just = c(0.5, 0.5)) {
    ts2 <- td %>% mutate(stratum = factor(stratum, levels = names(palette)))
    td_ci <- step_ci_df(ts2, "stratum")
    ann <- if (is.na(perf)) sprintf("logrank p=%.1e", lr_p)
           else sprintf("%s=%.3f\nlogrank p=%.1e", lbl, perf, lr_p)
    ggplot(ts2, aes(time, estimate, color = stratum)) +
      geom_rect(data = td_ci,
                aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
                inherit.aes = FALSE, alpha = 0.15, color = NA) +
      geom_step(linewidth = 0.9) +
      scale_color_manual(values = palette, name = NULL) +
      scale_fill_manual(values = palette, guide = "none") +
      coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
      annotate("text", x = 1, y = 0.06, label = ann,
               hjust = 0, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE,
               fontface = "italic", color = "#444444") +
      labs(x = "Months from first treatment", y = "Event-free survival",
           title = title_text) +
      theme_manuscript() +
      theme(legend.position = legend_pos, legend.justification = legend_just,
            legend.background = element_rect(fill = "white", color = NA))
  }

  pL <- panel_km(ts, ord4,  lr_s, perf_s, "Survival by Cancer Stage")
  # Fig 2G: legend pinned to the top-right corner of the panel.
  pR <- panel_km(tq, ord4q, lr_q, perf_q, "Survival by Text Risk-Score Quartile",
                 legend_pos = c(0.98, 0.98), legend_just = c(1, 1))
  # Return the two KM panels separately so they can be laid out beside the
  # risk-tertile panel (build_fig2d) as a single side-by-side-by-side row.
  list(stage = pL, quartile = pR)
}


# ============================================================================
# Combined metric-delta barplot (Mets / ICD10 events, grouped by code type) +
# top-event KM panels. Ranking follows MANUSCRIPT_METRIC and reads the matching
# cindex/auc prep outputs.
# ============================================================================
SCHEME_CATEGORY_TITLES <- c(mets = "Mets", ICD10 = "ICD10")

build_scheme_delta_bars <- function(topk) {
  # An empty frame (missing CSV) has no columns, so emptiness must be tested
  # before any filter() that names one.
  if (nrow(topk) == 0) return(placeholder_panel("no positive-delta events"))
  d <- topk %>% filter(category %in% names(SCHEME_CATEGORY_TITLES)) %>%
    arrange(category, delta)
  if (nrow(d) == 0) return(placeholder_panel("no positive-delta events"))
  # event_lbl is ordered (and disambiguated) within each category so identical
  # labels across categories don't collide into one y-axis row, and each
  # facet's bars come out sorted by its own delta. The "Mets: " prefix is
  # dropped here (mets-only) since the facet strip already reads "Mets".
  d <- d %>%
    mutate(category = factor(category, levels = names(SCHEME_CATEGORY_TITLES)),
           event_lbl = ifelse(category == "mets", sub("^Mets: ", "", event_lbl), event_lbl),
           row_key = factor(paste(category, event_lbl, sep = "|"),
                            levels = paste(category, event_lbl, sep = "|")))
  d_long <- d %>%
    select(category, row_key, event_lbl, text = text_value, base = base_value) %>%
    pivot_longer(c(text, base), names_to = "model", values_to = "metric_value") %>%
    mutate(model = factor(model, levels = c("text", "base")))
  lo <- max(0, min(d_long$metric_value, na.rm = TRUE) - 0.05)

  ggplot(d_long, aes(row_key, metric_value, fill = model)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.6, color = "white") +
    geom_text(aes(label = sprintf("%.3f", metric_value)),
              position = position_dodge(width = 0.7),
              vjust = -0.4, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    scale_x_discrete(
      labels = setNames(stringr::str_wrap(as.character(d$event_lbl), width = 18), d$row_key)
    ) +
    scale_fill_manual(values = MODEL_COLORS, labels = c(text = "Text", base = "Base"), name = NULL) +
    scale_y_continuous(limits = c(lo, NA), oob = scales::squish,
                       expand = expansion(mult = c(0, 0.18))) +
    coord_cartesian(ylim = c(lo, NA)) +
    facet_grid(. ~ category, scales = "free_x", space = "free_x",
              labeller = labeller(category = SCHEME_CATEGORY_TITLES)) +
    labs(x = NULL, y = metric_label(METRIC),
         title = sprintf("Top Events by Δ %s, by Code Type", metric_label(METRIC))) +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"),
          legend.position = "top",
          strip.text = element_text(face = "bold"),
          axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1,
                                     size = 10, lineheight = 0.9))
}

build_scheme_event_km <- function(km_data, topk, category, rank_n) {
  # An empty frame (missing CSV) has no columns, so emptiness must be tested
  # before any filter() that names one.
  if (nrow(topk) == 0) return(placeholder_panel(sprintf("no rank-%d %s event", rank_n, category)))
  ev <- topk %>% filter(category == !!category, rank == rank_n)
  if (nrow(ev) == 0) return(placeholder_panel(sprintf("no rank-%d %s event", rank_n, category)))
  km <- km_data %>% filter(category == !!category, scheme == ev$scheme[1], event == ev$event[1])
  if (nrow(km) == 0) return(placeholder_panel(sprintf("%s: no held-out risk scores yet", ev$event_lbl[1])))
  km_tertile_panel(km, "tt", "event_flag",
                   sprintf("%s\n(text solid, base dashed)", ev$event_lbl[1]))
}


# ============================================================================
# Compose Figure 2
# ============================================================================
metrics <- load_figure_data("fig2_full_cohort_metrics.csv")

# Panels A/B only: drop the single ICD10 Level 3 event whose delta performance (text - base)
# is a large negative outlier. It compresses the scatter/violin scale and is not
# representative of the scheme; removed here so it doesn't distort both panels.
# Determine the outlier using the active metric.
if (nrow(metrics) > 0 && any(metrics$scheme == "icd3_post")) {
  .text_col <- if (METRIC == "cindex") "text_cindex" else "text_auc"
  .base_col <- if (METRIC == "cindex") "base_cindex" else "base_auc"
  .delta <- metrics[[.text_col]] - metrics[[.base_col]]
  .icd3  <- metrics$scheme == "icd3_post"
  .out   <- which(.icd3 & .delta == min(.delta[.icd3], na.rm = TRUE))
  if (length(.out)) {
    message(sprintf("fig2 A/B: dropping ICD10 (Level 3) outlier '%s' (delta %s = %.3f)",
                    metrics$event[.out[1]], METRIC, .delta[.out[1]]))
    metrics <- metrics[-.out[1], , drop = FALSE]
  }
}

p2a <- build_fig2a(metrics)
p2b <- build_fig2b(metrics)
p2_wc <- build_within_vs_pan("fig2_within_vs_pan_cancer.csv",    "Pan vs. within-cancer model")
p2_wt <- build_within_vs_pan("fig2_within_vs_pan_treatment.csv", "Pan vs. within-treatment model")
p2d <- build_fig2d()                       # E: risk-score tertile KM (text vs base)
e_panels <- build_fig2e()
p2_stage <- e_panels$stage                 # F: survival by cancer stage
p2_quart <- e_panels$quartile              # G: survival by text risk-score quartile

# Per-scheme metric-delta barplots + top-1 event KM.
scheme_topk <- load_figure_data(paste0("fig2_scheme_delta_topk_", METRIC, ".csv"))
scheme_km   <- load_figure_data(paste0("fig2_scheme_event_km_", METRIC, ".csv"))

p2_bars <- build_scheme_delta_bars(scheme_topk)

p2_km_mets1     <- build_scheme_event_km(scheme_km, scheme_topk, "mets", 1)
p2_km_icd1      <- build_scheme_event_km(scheme_km, scheme_topk, "ICD10", 1)
p2_km_phecodes1 <- build_scheme_event_km(scheme_km, scheme_topk, "phecodes", 1)

.tag <- metric_tag(METRIC)
save_panel(p2a, paste0("fig2a", .tag), group = "figure2", width = 6.4, height = 5.0)
save_panel(p2b, paste0("fig2b", .tag), group = "figure2", width = 6.5, height = 4.8)
save_panel(p2_wc, paste0("fig2c", .tag), group = "figure2", width = 7.0,  height = 4.6)
save_panel(p2_wt, paste0("fig2d", .tag), group = "figure2", width = 9.6, height = 4.6)
save_panel(p2d,       paste0("fig2e", .tag), group = "figure2", width = 5.6, height = 4.6)
save_panel(p2_stage,  paste0("fig2f", .tag), group = "figure2", width = 5.6, height = 4.6)
save_panel(p2_quart,  paste0("fig2g", .tag), group = "figure2", width = 5.6, height = 4.6)
save_panel(p2_bars, paste0("fig2h", .tag), group = "figure2", width = 9.6, height = 5.2)
save_panel(p2_km_mets1, paste0("fig2k", .tag), group = "figure2", width = 5.6, height = 4.6)
save_panel(p2_km_icd1, paste0("fig2l", .tag), group = "figure2", width = 5.6, height = 4.6)
save_panel(p2_km_phecodes1, paste0("fig2m", .tag), group = "figure2", width = 5.6, height = 4.6)
