# Render Figure 2 (text vs base, full cohort) in ggplot2 + patchwork.
#
# A scatter (text vs base performance), B delta-performance violins + Wilcoxon stars,
# C pan vs within-cancer model (dumbbell), D pan vs within-treatment model,
# E KM by risk-score tertile (text solid / base dashed),
# F survival by cancer stage, G survival by text risk-score quartile.
# E/F/G share a single side-by-side row of KM curves, each with 95% CI bands.
#
# Metric switch: set MANUSCRIPT_METRIC=cindex|auc (see figure_utils.R::METRIC) to
# render this whole figure scored by Harrell's C-index or by mean AUC(t). Panels
# A-D and F/G read whichever metric's columns/CSV match; outputs are suffixed
# (e.g. figure2_text_results_cindex.png / _auc.png) so both sets can coexist.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggrepel); library(stringr)
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
source(file.path(script_dir, "figure_utils.R"))
# KM helpers (tidy_km, logrank_p, step_ci_df) now live in figure_utils.R so the
# supplementary stage-stratified script (plot_figure_2_supp.R) can share them.


# Human-readable label for a single event code. death_met events are either the
# literal "death" or a metastatic-site name (see MET_SITES in
# generate_embedding_prediction_datasets.py); everything else (ICD-10 codes,
# phecodes) just gets underscore-cleanup + title-casing.
event_description <- function(scheme, event) {
  ifelse(scheme == "death_met" & event == "death", "Death",
  ifelse(scheme == "death_met", paste("Mets:", stringr::str_to_title(event)),
         stringr::str_trunc(stringr::str_to_title(gsub("_", " ", event)), 22)))
}

# ============================================================================
# fig2a: text vs base scatter
# ============================================================================
build_fig2a <- function(metrics, metric = METRIC) {
  if (nrow(metrics) == 0) return(placeholder_panel("fig2_full_cohort_metrics.csv empty"))
  base_col <- paste0("base_", metric_suffix(metric))
  text_col <- paste0("text_", metric_suffix(metric))
  d <- metrics %>%
    filter(!is.na(.data[[base_col]]), !is.na(.data[[text_col]])) %>%
    mutate(base_val = .data[[base_col]], text_val = .data[[text_col]],
           delta = text_val - base_val,
           scheme_lbl = SCHEME_LABELS[as.character(scheme)])
  lo <- max(0.45, min(c(d$base_val, d$text_val)) - 0.02)
  hi <- min(1.00, max(c(d$base_val, d$text_val)) + 0.02)
  top <- d %>% slice_max(delta, n = 5) %>%
    mutate(event_lbl = event_description(scheme, event))
  lbl <- metric_label(metric)

  ggplot(d, aes(base_val, text_val, color = scheme, shape = scheme)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#666666") +
    geom_point(size = 1.8, alpha = 0.65) +
    ggrepel::geom_text_repel(data = top, aes(label = event_lbl),
                             size = 2.2, color = "#222222", min.segment.length = 0.1,
                             max.overlaps = 12) +
    scale_color_manual(values = SCHEME_COLORS,
                       labels = SCHEME_LABELS[names(SCHEME_COLORS)],
                       name = NULL) +
    scale_shape_manual(values = SCHEME_SHAPES,
                       labels = SCHEME_LABELS[names(SCHEME_SHAPES)],
                       name = NULL) +
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
              inherit.aes = FALSE, vjust = 1, size = 2.5, color = "#222222") +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_color_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_discrete(labels = SCHEME_LABELS) +
    labs(x = NULL, y = sprintf("Delta %s (Text - Base)", lbl),
         title = sprintf("%s Improvement by Scheme", lbl),
         caption = paste("Stars: Wilcoxon signed-rank vs Δ=0  (*<.05, **<.01, ***<.001, ****<1e-4).",
                         "Diamond ± bar: mean ± SD.")) +
    theme_manuscript() +
    theme(plot.caption = element_text(size = 6.5, hjust = 1,
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
              hjust = -0.2, size = 2.5, color = "#666666") +
    scale_color_manual(values = c(Pan    = unname(MODEL_COLORS[["base"]]),
                                  Within = unname(MODEL_COLORS[["text"]])), name = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0.02, 0.13))) +
    labs(title = stratum_title, x = paste("Held-out", lbl), y = NULL,
         caption = sprintf("Dashed lines: overall held-out %s (grey = pan, red = within)", lbl)) +
    theme_manuscript() +
    theme(legend.position = "top",
          plot.caption = element_text(size = 6.5, hjust = 1, face = "italic", color = "#666666"))
  if (length(overall_pan) == 1 && is.finite(overall_pan)) {
    p <- p + geom_vline(xintercept = overall_pan, linetype = "dashed",
                        color = unname(MODEL_COLORS[["base"]]), linewidth = 0.5) +
      annotate("text", x = overall_pan, y = Inf,
               label = sprintf("Pan avg %s = %.3f", lbl, overall_pan),
               color = unname(MODEL_COLORS[["base"]]), size = 2.5,
               fontface = "italic", angle = 90, hjust = 1.05, vjust = -0.4)
  }
  if (length(overall_within) == 1 && is.finite(overall_within)) {
    p <- p + geom_vline(xintercept = overall_within, linetype = "dashed",
                        color = unname(MODEL_COLORS[["text"]]), linewidth = 0.5) +
      annotate("text", x = overall_within, y = Inf,
               label = sprintf("Within avg %s = %.3f", lbl, overall_within),
               color = unname(MODEL_COLORS[["text"]]), size = 2.5,
               fontface = "italic", angle = 90, hjust = 1.05, vjust = 1.4)
  }
  p
}


# ============================================================================
# fig2d: KM by risk-score tertile (text solid, base dashed)
# ============================================================================
build_fig2d <- function() {
  km <- load_figure_data("fig2_km_tertiles.csv")
  if (nrow(km) == 0) return(placeholder_panel("fig2_km_tertiles.csv empty"))
  km <- km %>% mutate(months = tt_death / 30.44,
                      death = as.integer(death))

  fit_t <- survfit2(Surv(months, death) ~ text_tertile, data = km)
  fit_b <- survfit2(Surv(months, death) ~ base_tertile, data = km)
  td <- bind_rows(
    tidy_km(fit_t) %>% mutate(model = "text"),
    tidy_km(fit_b) %>% mutate(model = "base")
  ) %>%
    mutate(stratum = factor(stratum, levels = c("low", "mid", "high")),
           model   = factor(model,   levels = c("text", "base")))

  n_by <- km %>% group_by(text_tertile) %>% summarise(n = n(), .groups = "drop") %>%
    mutate(label = sprintf("text %s (n=%s)", text_tertile, scales::comma(n)))
  n_by_b <- km %>% group_by(base_tertile) %>% summarise(n = n(), .groups = "drop") %>%
    mutate(label = sprintf("base %s (n=%s)", base_tertile, scales::comma(n)))

  lr_t <- logrank_p(km, "months", "death", "text_tertile")
  lr_b <- logrank_p(km, "months", "death", "base_tertile")

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
    labs(x = "Months from first treatment", y = "Overall survival",
         title = "Mortality by Risk-Score Tertile\n(text solid, base dashed)") +
    theme_manuscript() +
    theme(legend.position = c(0.82, 0.78),
          legend.background = element_rect(fill = "white", color = NA),
          legend.spacing.y = unit(0.05, "in"))
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

  panel_km <- function(td, palette, lr_p, perf, title_text) {
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
               hjust = 0, vjust = 0, size = 2.6,
               fontface = "italic", color = "#444444") +
      labs(x = "Months from first treatment", y = "Overall survival",
           title = title_text) +
      theme_manuscript() +
      theme(legend.position = c(0.82, 0.80),
            legend.background = element_rect(fill = "white", color = NA))
  }

  pL <- panel_km(ts, ord4,  lr_s, perf_s, "Survival by Cancer Stage")
  pR <- panel_km(tq, ord4q, lr_q, perf_q, "Survival by Text Risk-Score Quartile")
  # Return the two KM panels separately so they can be laid out beside the
  # risk-tertile panel (build_fig2d) as a single side-by-side-by-side row.
  list(stage = pL, quartile = pR)
}


# ============================================================================
# Compose Figure 2
# ============================================================================
metrics <- load_figure_data("fig2_full_cohort_metrics.csv")

# Panels A/B only: drop the single ICD-3 event whose delta performance (text - base)
# is a large negative outlier. It compresses the scatter/violin scale and is not
# representative of the scheme; removed here so it doesn't distort both panels.
# Always determined from C-index (regardless of the active METRIC) so the same
# event is excluded from both the C-index and AUC(t) renderings — trimming
# identically instead of letting AUC(t) pick its own (possibly different) outlier.
if (nrow(metrics) > 0 && any(metrics$scheme == "icd3_post")) {
  .delta <- metrics[["text_cindex"]] - metrics[["base_cindex"]]
  .icd3  <- metrics$scheme == "icd3_post"
  .out   <- which(.icd3 & .delta == min(.delta[.icd3], na.rm = TRUE))
  if (length(.out)) {
    message(sprintf("fig2 A/B: dropping ICD-3 outlier '%s' (delta cindex = %.3f), applied to both metrics",
                    metrics$event[.out[1]], .delta[.out[1]]))
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

# E/F/G: three KM curves side-by-side-by-side.
risk_row <- p2d | p2_stage | p2_quart

.tag <- metric_tag(METRIC)
save_panel(p2a, paste0("fig2a", .tag), width = 6.4, height = 5.0)
save_panel(p2b, paste0("fig2b", .tag), width = 6.0, height = 4.8)
save_panel(p2_wc, paste0("fig2c", .tag), width = 7.0,  height = 4.6)
save_panel(p2_wt, paste0("fig2d", .tag), width = 11.2, height = 4.6)
save_panel(p2d,       paste0("fig2e", .tag), width = 5.6, height = 4.6)
save_panel(p2_stage,  paste0("fig2f", .tag), width = 5.6, height = 4.6)
save_panel(p2_quart,  paste0("fig2g", .tag), width = 5.6, height = 4.6)

fig2 <- (p2a + p2b) /
        (p2_wc + p2_wt + plot_layout(widths = c(1.25, 2))) /
        risk_row +
        plot_annotation(tag_levels = "A") &
        theme(plot.tag = element_text(size = 14, face = "bold"))

save_figure(fig2, paste0("figure2_text_results", .tag), width = 16.5, height = 14.2)
