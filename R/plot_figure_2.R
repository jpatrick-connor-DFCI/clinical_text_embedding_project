# Figure 2: a base-vs-text C-index scatter, b delta-C-index violin,
# c survival by stage, d survival by text-risk quartile.
# Individual panels and compiled PNG/PDF with panel letters and no plot titles.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr)
  library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")
source("R/publication_style.R")
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
               size = 0.65, alpha = 0.5) +
    # Draw death in a separate final layer so no coincident event can cover it,
    # and draw it larger and fully opaque: it is the single literal death
    # endpoint among thousands of coded events, so at the shared size/alpha it
    # was indistinguishable from the surrounding cloud.
    geom_point(data = filter(d, as.character(plot_group) == "death"),
               size = 1.5, alpha = 1) +
    scale_color_manual(values = FIG2A_GROUP_COLORS, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    scale_shape_manual(values = FIG2A_GROUP_SHAPES, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    coord_fixed(xlim = c(lo, hi), ylim = c(lo, hi), expand = FALSE) +
    labs(x = paste("Base model", lbl), y = paste("Text model", lbl),
         title = "Base versus text model") +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.04), legend.justification = c(1, 0),
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
    mutate(label = sprintf("n = %d", n))
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
    geom_point(position = position_jitter(width = 0.15, seed = 2026), size = 0.45, alpha = 0.3,
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
                     labels = stringr::str_wrap(unname(SCHEME_LABELS[scheme_order]), 10)) +
    labs(x = NULL, y = "C-index difference (text - base)",
         title = "Change in discrimination") +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig2c/d: stage vs text risk-quartile (1×2 KM + C-index annotations)
# ============================================================================
build_stage_and_risk_km <- function(metric = METRIC) {
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

  panel_km <- function(td, palette, lr_p, perf, title_text) {
    ts2 <- td %>% mutate(stratum = factor(stratum, levels = names(palette)))
    td_ci <- step_ci_df(ts2, "stratum")
    ann <- if (is.na(perf)) sprintf("log-rank %s", format_p_inline(lr_p))
           else sprintf("%s = %.3f; log-rank %s", lbl, perf, format_p_inline(lr_p))
    ggplot(ts2, aes(time, estimate, color = stratum)) +
      scale_x_continuous(breaks = seq(0, 60, 12), expand = expansion(mult = SURVIVAL_X_EXPANSION)) +
      geom_rect(data = td_ci,
                aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
                inherit.aes = FALSE, alpha = 0.15, color = NA) +
      geom_step(linewidth = 0.5) +
      scale_color_manual(values = palette, name = NULL) +
      scale_fill_manual(values = palette, guide = "none") +
      coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
      labs(x = "Months from first treatment", y = "Overall survival",
           title = title_text, subtitle = ann) +
      guides(color = guide_legend(nrow = 1, byrow = TRUE)) +
      theme_manuscript() +
      theme(legend.position = "bottom",
            legend.background = element_rect(fill = "white", color = NA))
  }

  pL <- panel_km(ts, ord4,  lr_s, perf_s, "Overall survival by stage")
  pR <- panel_km(tq, ord4q, lr_q, perf_q, "Overall survival by text risk")
  # Return the stage and text-risk quartile curves as separate panels.
  attr(pL, "caption_detail") <- sprintf("Panels c and d include %s patients with known stage.",
                                        scales::comma(nrow(d)))
  list(stage = pL, quartile = pR)
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

retire_main_panels(2, letters[1:4])

p2a <- build_fig2a(metrics)
p2b <- build_fig2b(metrics)
km_panels <- build_stage_and_risk_km()
p2c <- km_panels$stage
p2d <- km_panels$quartile

.tag <- metric_tag()
save_panel(p2a, paste0("fig2a", .tag), group = "figure2", width = 3.5, height = 3.2, dpi = 600)
save_panel(p2b, paste0("fig2b", .tag), group = "figure2", width = 3.5, height = 3.2, dpi = 600)
save_panel(p2c, paste0("fig2c", .tag), group = "figure2", width = 3.5, height = 3.2, dpi = 600)
save_panel(p2d, paste0("fig2d", .tag), group = "figure2", width = 3.5, height = 3.2, dpi = 600)
save_compiled_figure(
  list(a = p2a, b = p2b, c = p2c, d = p2d),
  number = 2, width = COMPILED_FIGURE_WIDTH, height = COMPILED_FIGURE_HEIGHT
)
