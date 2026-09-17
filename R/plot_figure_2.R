# Render Figure 2 (text vs base, full cohort) in ggplot2 + patchwork.
#
# A scatter (text vs base performance), B delta-performance violins + paired Wilcoxon p values,
# C pan vs within-cancer model (dumbbell), D pan vs within-treatment model,
# E KM by risk-score tertile (text solid / base dashed),
# F survival by cancer stage, G survival by text risk-score quartile.
# H: top-3-by-Δ-C-index event bars for mets/ICD10/phecodes.
# I: 24-month calibration; J: decision-curve analysis.
# K/L/M: text-vs-base held-out-risk KM for each category's rank-1 event.
# (Ranks 2-3 for each category are in the plot_figure_2_supp_events.R supplement.)
#
# All performance panels and annotations use Harrell's C-index.

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
    # Draw death in a separate final layer so no coincident event can cover it,
    # and draw it larger and fully opaque: it is the single literal death
    # endpoint among thousands of coded events, so at the shared size/alpha it
    # was indistinguishable from the surrounding cloud.
    geom_point(data = filter(d, as.character(plot_group) == "death"),
               size = 3.6, alpha = 1) +
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
# fig2b: delta-performance violins by scheme + paired Wilcoxon summaries
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
              median_delta = median(delta),
              q25 = quantile(delta, .25),
              q75 = quantile(delta, .75),
              .groups = "drop") %>%
    mutate(label = sprintf("n=%d\nmedian=%+.3f\nmean=%+.3f",
                           n, median_delta, mean_delta))
  # Order violins left-to-right by DESCENDING median delta, so the scheme the
  # text model improves most sits furthest left. Pin this order on the x scale
  # below as well as in both data frames: this avoids a future layer with a
  # differently ordered factor resetting the displayed order. Scheme name
  # supplies a deterministic tie-break for exactly equal medians.
  scheme_order <- ann %>%
    arrange(desc(median_delta), as.character(scheme)) %>%
    pull(scheme) %>%
    as.character()
  d   <- d   %>% mutate(scheme = factor(as.character(scheme), levels = scheme_order))
  ann <- ann %>% mutate(scheme = factor(as.character(scheme), levels = scheme_order))
  message(sprintf("[fig2b] schemes left-to-right by descending median delta: %s",
                  paste(scheme_order, collapse = ", ")))
  ymax <- max(d$delta, na.rm = TRUE) * 1.20

  ggplot(d, aes(x = scheme, y = delta, fill = scheme)) +
    geom_violin(data = filter(d, ave(delta, scheme, FUN = length) >= 10),
                alpha = 0.45, color = "#444444", linewidth = 0.4, scale = "width") +
    geom_jitter(width = 0.18, size = 0.7, alpha = 0.35,
                aes(color = scheme), show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "#333333", linetype = "dashed") +
    geom_errorbar(data = ann,
                  aes(x = scheme, y = median_delta, ymin = q25, ymax = q75),
                  inherit.aes = FALSE, width = 0.15, linewidth = 0.6, color = "#111111") +
    geom_point(data = ann, aes(x = scheme, y = median_delta),
              inherit.aes = FALSE, shape = 23, size = 2.2,
              fill = "white", color = "#111111", stroke = 0.7) +
    geom_text(data = ann, aes(x = scheme, y = ymax, label = label),
              inherit.aes = FALSE, vjust = 1,
              size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#222222") +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_color_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_discrete(limits = scheme_order,
                     labels = unname(SCHEME_LABELS[scheme_order])) +
    labs(x = NULL, y = sprintf("Delta %s (Text - Base)", lbl),
         title = sprintf("%s Improvement by Scheme", lbl),
         caption = paste("Each point is an endpoint; endpoints are correlated, so summaries are descriptive.",
                         "Diamond and bar: median and IQR; violins are omitted when n<10.")) +
    theme_manuscript() +
    theme(plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 1,
                                      face = "italic", color = "#666666"),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig2c / fig2d: pan vs. within-stratum C-index dumbbell.
# Grey dot = pan model, red dot = within-stratum model.
# Dashed vertical lines mark the overall (pooled) held-out performance of each
# model — grey = pan, red = within. The catch-all "OTHER" stratum is not
# plotted as its own dumbbell. Shared by the cancer- and treatment-stratified
# panels.
# ============================================================================
build_within_vs_pan <- function(csv, stratum_title, metric = METRIC) {
  d <- load_figure_data(csv)
  if (nrow(d) == 0) return(placeholder_panel(paste(csv, "empty")))
  pan_col <- "cindex_pan"
  within_col <- "cindex_within"
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

  high_low_hr <- function(group_col) {
    dd <- km %>% filter(.data[[group_col]] %in% c("low", "high")) %>%
      mutate(.grp = relevel(factor(.data[[group_col]]), ref = "low"))
    fit <- tryCatch(coxph(Surv(months, .event) ~ .grp, data = dd), error = function(e) NULL)
    if (is.null(fit)) return("HR n/a")
    sm <- summary(fit)$conf.int[1, ]
    sprintf("HR high vs low %.2f (95%% CI %.2f–%.2f)", sm["exp(coef)"], sm["lower .95"], sm["upper .95"])
  }
  hr_t <- high_low_hr("text_tertile")
  hr_b <- high_low_hr("base_tertile")

  n_t <- km %>% count(text_tertile) %>% mutate(model = "text") %>% rename(stratum = text_tertile)
  n_b <- km %>% count(base_tertile) %>% mutate(model = "base") %>% rename(stratum = base_tertile)
  n_labels <- bind_rows(n_t, n_b) %>%
    mutate(key = paste(model, stratum), display = sprintf("%s %s (n=%s)", model, stratum, comma(n)))
  td <- td %>% mutate(key = paste(model, as.character(stratum)),
                      display = factor(n_labels$display[match(key, n_labels$key)], levels = n_labels$display))

  td_ci <- step_ci_df(td, c("stratum", "model"))

  # Zoom the probability axis to the data's own range: events with low incidence
  # keep survival near 1 throughout, so a fixed 0-1 axis wastes most of the panel.
  # Floor a little below the lowest observed CI bound, with a 1.03 ceiling to
  # match the padding used elsewhere in this figure.
  y_lo <- max(0, min(td_ci$conf.low, na.rm = TRUE) - 0.03)
  y_hi <- 1.03
  stats_caption <- sprintf("Text: %s; %s.  Base: %s; %s.",
                           format_p_inline(lr_t), hr_t,
                           format_p_inline(lr_b), hr_b)

  main <- ggplot(td, aes(x = time, y = estimate, color = stratum, linetype = model)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.12, color = NA) +
    geom_step(linewidth = 0.9) +
    geom_point(data = thin_censor_rows(td, c("stratum", "model"), 50),
               shape = 3, size = 1.0, stroke = 0.45) +
    scale_color_manual(values = RISK_COLORS, name = NULL) +
    scale_fill_manual(values = RISK_COLORS, guide = "none") +
    scale_linetype_manual(values = c(text = "solid", base = "dashed"), guide = "none") +
    coord_cartesian(xlim = c(0, 60), ylim = c(y_lo, y_hi)) +
    labs(x = "Months from first treatment", y = "Event-free survival", title = title,
         caption = stringr::str_wrap(stats_caption, width = 105)) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.98),
          legend.justification = c(1, 1),
          legend.background = element_rect(fill = "white", color = NA),
          legend.spacing.y = unit(0.05, "in"),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 0,
                                      face = "italic", color = "#444444"))

  risk_times <- seq(0, 60, 12)
  risk_rows <- function(fit, model) {
    s <- summary(fit, times = risk_times, extend = TRUE)
    tibble(time = s$time, n.risk = s$n.risk,
           stratum = sub("^[^=]+=", "", s$strata), model = model)
  }
  rt <- bind_rows(risk_rows(fit_t, "text"), risk_rows(fit_b, "base")) %>%
    mutate(key = paste(model, stratum),
           row = factor(n_labels$display[match(key, n_labels$key)],
                        levels = rev(n_labels$display)))
  table <- ggplot(rt, aes(time, row, label = comma(n.risk), color = stratum)) +
    geom_text(size = 3) +
    scale_color_manual(values = RISK_COLORS, guide = "none") +
    scale_x_continuous(limits = c(0, 60), breaks = risk_times) +
    labs(x = NULL, y = "Number at risk") +
    theme_void(base_size = 9) +
    theme(axis.text.y = element_text(), axis.title.y = element_text(angle = 90),
          plot.margin = margin(0, 5.5, 5.5, 5.5))
  main / table + plot_layout(heights = c(4.2, 1.2))
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
# fig2e: stage vs text risk-quartile (1×2 KM + C-index annotations)
# ============================================================================
build_fig2e <- function(metric = METRIC) {
  d  <- load_figure_data("fig2_km_stage_vs_risk.csv")
  lbl <- metric_label(metric)
  ci <- load_figure_data("fig2_stage_vs_risk_cindex.csv")
  perf_s <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "stage"][1] else NA_real_
  perf_q <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "text_risk"][1] else NA_real_
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
    ann <- if (is.na(perf)) sprintf("log-rank %s", format_p_inline(lr_p))
           else sprintf("%s=%.3f\nlog-rank %s", lbl, perf, format_p_inline(lr_p))
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
# Combined metric-delta barplot (Mets / ICD10 / phecode events, grouped by code
# type) + top-event KM panels. Ranking uses the prepared C-index outputs.
# ============================================================================
# Must cover every category the prep tier emits (CATEGORY_ORDER in
# figures/prep/figure2.py): build_scheme_delta_bars filters to these names, so a
# category missing here is dropped from the panel without warning. "phecodes" was
# absent, which silently discarded its top-3 events even though the rest of the
# figure (fig2m) and the supplement both report that category.
SCHEME_CATEGORY_TITLES <- c(mets = "Mets", ICD10 = "ICD10", phecodes = "PhecodeX")

build_scheme_delta_bars <- function(topk) {
  # An empty frame (missing CSV) has no columns, so emptiness must be tested
  # before any filter() that names one.
  if (nrow(topk) == 0) return(placeholder_panel("no positive-delta events"))
  d <- topk %>% filter(category %in% names(SCHEME_CATEGORY_TITLES)) %>%
    arrange(category, delta)
  if (nrow(d) == 0) return(placeholder_panel("no positive-delta events"))
  # One bar group per (scheme, event), ordered by delta within each facet. The
  # key is built from scheme+event rather than the label because event_lbl is not
  # unique: ICD10 pools icd3_post and icd4_post, and a level-3 code and its
  # level-4 child can share a description, which made the label-keyed factor die
  # on a duplicated level. Distinct events must stay distinct bars even when they
  # print the same text. The "Mets: " prefix is dropped (mets-only) since the
  # facet strip already reads "Mets".
  d <- d %>%
    mutate(category = factor(category, levels = names(SCHEME_CATEGORY_TITLES)),
           event_lbl = ifelse(category == "mets", sub("^Mets: ", "", event_lbl), event_lbl),
           row_key = factor(paste(category, scheme, event, sep = "|"),
                            levels = paste(category, scheme, event, sep = "|")))
  # Two bars carrying the identical description would be unreadable, so append the
  # event code to distinguish them. Only labels that actually repeat are touched,
  # which keeps the common case uncluttered.
  d <- d %>%
    group_by(category, event_lbl) %>%
    mutate(event_lbl = if (n() > 1) paste0(event_lbl, " (", event, ")") else event_lbl) %>%
    ungroup()
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
    # Wrap hard. facet_grid(space = "free_x") splits width by bar count, not by
    # label length, so every facet gets the same room however long its labels
    # are -- widening the device scales the facets and leaves the text the same
    # size, and the longest labels (mets runs to ~70 characters) keep colliding
    # with their neighbours. The wrap width, not the canvas, is what separates
    # adjacent tick labels here.
    scale_x_discrete(
      labels = setNames(stringr::str_wrap(as.character(d$event_lbl), width = 22), d$row_key)
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
                   wrap_title_suffix(ev$event_lbl[1], "(text solid, base dashed)"))
}


# ============================================================================
# fig2i/j: cross-fitted 24-month calibration and decision-curve analysis
# ============================================================================
build_fig2i <- function() {
  d <- load_figure_data("fig2_calibration.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig2_calibration.csv empty — outer folds required"))
  d <- d %>% mutate(model = factor(model, levels = c("base", "text")))
  ggplot(d, aes(mean_predicted, observed, color = model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey55") +
    geom_errorbar(aes(ymin = pmax(0, observed - 1.96 * se),
                      ymax = pmin(1, observed + 1.96 * se)), width = 0.008) +
    geom_line(linewidth = 0.8) + geom_point(size = 3, alpha = 0.85) +
    scale_color_manual(values = c(base = MODEL_COLORS[["base"]],
                                  text = MODEL_COLORS[["text"]]),
                       labels = c(base = "Base", text = "Text"), name = NULL) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(x = "Cross-fitted predicted 24-month mortality",
         y = "Observed 24-month mortality",
         title = "Recalibrated Predictions at 24 Months",
         subtitle = "Cross-fitted Platt recalibration",
         caption = "Equal-frequency bins; approximate 95% CIs. IPCW accounts for censoring. This panel assesses recalibrated, not raw, predictions.") +
    theme_manuscript() + theme(legend.position = "bottom",
                               plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE))
}

build_fig2j <- function() {
  d <- load_figure_data("fig2_decision_curve.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig2_decision_curve.csv empty — outer folds required"))
  d <- d %>% distinct(model, threshold, .keep_all = TRUE) %>%
    arrange(model, threshold) %>% mutate(model = factor(model,
      levels = c("text", "base", "treat_all", "treat_none")))
  cols <- c(text = MODEL_COLORS[["text"]], base = MODEL_COLORS[["base"]],
            treat_all = "grey45", treat_none = "grey75")
  types <- c(text = "solid", base = "solid", treat_all = "dashed", treat_none = "dotted")
  ggplot(d, aes(threshold, net_benefit, color = model, linetype = model)) +
    geom_line(linewidth = 0.9) +
    scale_color_manual(values = cols,
      labels = c(text = "Text", base = "Base", treat_all = "Treat all", treat_none = "Treat none"),
      name = NULL) +
    scale_linetype_manual(values = types, guide = "none") +
    scale_x_continuous(labels = percent, limits = c(0.02, 0.30)) +
    labs(x = "24-month mortality-risk threshold", y = "Net benefit",
         title = "Decision-Curve Analysis at 24 Months",
         caption = "Net benefit for a hypothetical intervention above the predicted-risk threshold; cross-fitted predictions with IPCW for censoring.") +
    theme_manuscript() + theme(legend.position = "bottom")
}


# ============================================================================
# Compose Figure 2
# ============================================================================
metrics <- load_figure_data("fig2_full_cohort_metrics.csv")

# Events where the text model is substantially worse than base in the full cohort
# are dropped from EVERY figure (see figure_utils.R::excluded_event_keys). This
# replaces the former hand-picked "drop the worst ICD10 Level 3 outlier from
# panels A/B" rule: the exclusion is now threshold-based, applies to all panels
# rather than just A/B, and is computed once here so the supplementary scripts
# can reuse the identical set.
EXCLUDED_EVENTS <- excluded_event_keys(metrics)
metrics <- drop_excluded_events(metrics, EXCLUDED_EVENTS)

p2a <- build_fig2a(metrics)
p2b <- build_fig2b(metrics)
p2_wc <- build_within_vs_pan("fig2_within_vs_pan_cancer.csv",    "Pan vs. within-cancer model")
p2_wt <- build_within_vs_pan("fig2_within_vs_pan_treatment.csv", "Pan vs. within-treatment model")
p2d <- build_fig2d()                       # E: risk-score tertile KM (text vs base)
e_panels <- build_fig2e()
p2_stage <- e_panels$stage                 # F: survival by cancer stage
p2_quart <- e_panels$quartile              # G: survival by text risk-score quartile
p2_cal <- build_fig2i()
p2_dca <- build_fig2j()

# Per-scheme metric-delta barplots + top-1 event KM.
scheme_topk <- load_figure_data(paste0("fig2_scheme_delta_topk_", METRIC, ".csv"))
scheme_km   <- load_figure_data(paste0("fig2_scheme_event_km_", METRIC, ".csv"))

# Apply the same exclusion to the per-event panels. topk is re-ranked so the
# rank-1 KM panels (K/L/M) fall through to the next-best surviving event rather
# than disappearing when the original rank-1 event was excluded.
scheme_topk <- drop_excluded_events_reranked(scheme_topk, EXCLUDED_EVENTS)
scheme_km   <- drop_excluded_events(scheme_km, EXCLUDED_EVENTS)

p2_bars <- build_scheme_delta_bars(scheme_topk)

p2_km_mets1     <- build_scheme_event_km(scheme_km, scheme_topk, "mets", 1)
p2_km_icd1      <- build_scheme_event_km(scheme_km, scheme_topk, "ICD10", 1)
p2_km_phecodes1 <- build_scheme_event_km(scheme_km, scheme_topk, "phecodes", 1)

.tag <- metric_tag(METRIC)
save_panel(p2a, paste0("fig2a", .tag), group = "figure2", width = 7.8, height = 6.0)
save_panel(p2b, paste0("fig2b", .tag), group = "figure2", width = 7.8, height = 5.8)
save_panel(p2_wc, paste0("fig2c", .tag), group = "figure2", width = 8.6,  height = 5.8)
save_panel(p2_wt, paste0("fig2d", .tag), group = "figure2", width = 11.5, height = 5.8)
save_panel(p2d,       paste0("fig2e", .tag), group = "figure2", width = 7.6, height = 6.0)
save_panel(p2_stage,  paste0("fig2f", .tag), group = "figure2", width = 7.6, height = 6.0)
save_panel(p2_quart,  paste0("fig2g", .tag), group = "figure2", width = 7.6, height = 6.0)
# Widest panel in the set: up to 3 events x 3 code-type facets (mets / ICD10 /
# phecodes), each x-label a wrapped multi-line event description. Width is driven
# by label crowding, not by the bars; the extra height is headroom for labels that
# wrap to four lines.
save_panel(p2_bars, paste0("fig2h", .tag), group = "figure2", width = 18.0, height = 7.6)
save_panel(p2_cal, paste0("fig2i", .tag), group = "figure2", width = 7.2, height = 6.0)
save_panel(p2_dca, paste0("fig2j", .tag), group = "figure2", width = 7.2, height = 6.0)
save_panel(p2_km_mets1, paste0("fig2k", .tag), group = "figure2", width = 7.6, height = 6.0)
save_panel(p2_km_icd1, paste0("fig2l", .tag), group = "figure2", width = 7.6, height = 6.0)
save_panel(p2_km_phecodes1, paste0("fig2m", .tag), group = "figure2", width = 7.6, height = 6.0)

# Complete manuscript figure, with one lowercase label per named panel.
save_compiled_figure(
  list(a = p2a, b = p2b, c = p2_wc, d = p2_wt, e = p2d,
       f = p2_stage, g = p2_quart, h = p2_bars, i = p2_cal, j = p2_dca,
       k = p2_km_mets1, l = p2_km_icd1, m = p2_km_phecodes1),
  number = 2, width = 24, height = 36, design = "abc\ndde\nfgg\nhhh\nijk\nlmm",
  heights = c(6, 6, 6, 7, 6, 6)
)
