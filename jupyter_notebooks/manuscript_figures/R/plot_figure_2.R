# Render Figure 2 (text vs base, full cohort) in ggplot2 + patchwork.
#
# A scatter (text vs base c-index), B Δc-index violins + Wilcoxon stars,
# C cancer×endpoint heatmap, D KM by risk-score tertile (text solid / base dashed),
# E stage vs text risk-quartile KM (1×2) + c-index annotation.

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

# ----------------------------------------------------------------------------
# KM helper: tidy survfit2 output into a long step-ready data frame
# ----------------------------------------------------------------------------
tidy_km <- function(fit) {
  if (is.null(fit)) return(tibble::tibble())
  td <- ggsurvfit::tidy_survfit(fit)
  if ("strata" %in% names(td)) td$stratum <- sub("^[^=]+=", "", as.character(td$strata))
  td
}

# Multivariate log-rank p (survdiff on a formula)
logrank_p <- function(df, time_col, event_col, group_col) {
  if (nrow(df) == 0) return(NA_real_)
  f <- as.formula(sprintf("Surv(%s, %s) ~ %s", time_col, event_col, group_col))
  sd <- tryCatch(survival::survdiff(f, data = df), error = function(e) NULL)
  if (is.null(sd)) return(NA_real_)
  if (!is.null(sd$pvalue)) return(sd$pvalue)
  stats::pchisq(sd$chisq, df = length(sd$n) - 1, lower.tail = FALSE)
}


# ============================================================================
# fig2a: text vs base scatter
# ============================================================================
build_fig2a <- function(metrics) {
  if (nrow(metrics) == 0) return(placeholder_panel("fig2_full_cohort_metrics.csv empty"))
  d <- metrics %>%
    filter(!is.na(base_cindex), !is.na(text_cindex)) %>%
    mutate(delta = text_cindex - base_cindex,
           scheme_lbl = SCHEME_LABELS[as.character(scheme)])
  lo <- max(0.45, min(c(d$base_cindex, d$text_cindex)) - 0.02)
  hi <- min(1.00, max(c(d$base_cindex, d$text_cindex)) + 0.02)
  top <- d %>% slice_max(delta, n = 5)

  ggplot(d, aes(base_cindex, text_cindex, color = scheme, shape = scheme)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#666") +
    geom_point(size = 1.8, alpha = 0.65) +
    ggrepel::geom_text_repel(data = top, aes(label = stringr::str_trunc(event, 22)),
                             size = 2.2, color = "#222", min.segment.length = 0.1,
                             max.overlaps = 12) +
    scale_color_manual(values = SCHEME_COLORS,
                       labels = SCHEME_LABELS[names(SCHEME_COLORS)],
                       name = NULL) +
    scale_shape_manual(values = SCHEME_SHAPES,
                       labels = SCHEME_LABELS[names(SCHEME_SHAPES)],
                       name = NULL) +
    coord_fixed(xlim = c(lo, hi), ylim = c(lo, hi)) +
    labs(x = "Base Model C-index", y = "Text Model C-index",
         title = "Text vs. Base Model Performance") +
    theme_manuscript() +
    theme(legend.position = c(0.85, 0.18),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# fig2b: Δc-index violins by scheme + Wilcoxon-vs-0 stars
# ============================================================================
build_fig2b <- function(metrics) {
  if (nrow(metrics) == 0) return(placeholder_panel("fig2_full_cohort_metrics.csv empty"))
  d <- metrics %>%
    filter(!is.na(base_cindex), !is.na(text_cindex)) %>%
    mutate(delta = text_cindex - base_cindex,
           scheme = factor(scheme, levels = names(SCHEME_LABELS)))
  ann <- d %>% group_by(scheme) %>%
    summarise(n = n(),
              med = median(delta),
              p = wilcoxon_vs0(delta),
              .groups = "drop") %>%
    mutate(stars = vapply(p, p_to_stars, character(1)),
           label = sprintf("n=%d\nmed=%+.3f\n%s", n, med, stars))
  ymax <- max(d$delta, na.rm = TRUE) * 1.20

  ggplot(d, aes(x = scheme, y = delta, fill = scheme)) +
    geom_violin(alpha = 0.45, color = "#444", linewidth = 0.4, scale = "width") +
    geom_jitter(width = 0.18, size = 0.7, alpha = 0.35,
                aes(color = scheme), show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "#333", linetype = "dashed") +
    geom_text(data = ann, aes(x = scheme, y = ymax, label = label),
              inherit.aes = FALSE, vjust = 1, size = 2.5, color = "#222") +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_color_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_discrete(labels = SCHEME_LABELS) +
    labs(x = NULL, y = "Delta C-index (Text - Base)",
         title = "C-index Improvement by Scheme",
         caption = "Stars: Wilcoxon signed-rank vs Δ=0  (*<.05, **<.01, ***<.001, ****<1e-4)") +
    theme_manuscript() +
    theme(plot.caption = element_text(size = 6.5, hjust = 1,
                                      face = "italic", color = "#666"),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig2c: heatmap of text c-index by cancer type × endpoint
# ============================================================================
build_fig2c <- function() {
  d <- load_figure_data("fig2_cancer_endpoint_heatmap.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig2_cancer_endpoint_heatmap.csv empty"))
  d <- d %>%
    filter(!is.na(text_cindex)) %>%
    mutate(event = gsub("_", " ", as.character(event)),
           cancer_type = as.character(cancer_type),
           star = ifelse(text_cindex >= 0.60, "*", ""))
  event_order <- d %>% group_by(event) %>%
    summarise(m = mean(text_cindex)) %>% arrange(desc(m)) %>% pull(event)
  cancer_order <- d %>% group_by(cancer_type) %>%
    summarise(s = sum(n)) %>% arrange(desc(s)) %>% pull(cancer_type)
  d <- d %>%
    mutate(event = factor(event, levels = event_order),
           cancer_type = factor(cancer_type, levels = cancer_order))
  ggplot(d, aes(cancer_type, event, fill = text_cindex)) +
    geom_tile(color = "white") +
    geom_text(aes(label = star), color = "white", size = 4, fontface = "bold",
              vjust = 0.55) +
    scale_fill_distiller(palette = "Blues", direction = 1,
                         limits = c(0.5, 1.0), name = "C-index",
                         na.value = "grey90") +
    labs(x = NULL, y = NULL,
         title = "Text Model C-index by Cancer Type and Endpoint") +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          axis.text.y = element_text(size = 7),
          panel.grid = element_blank(),
          axis.line = element_blank(), axis.ticks = element_blank())
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

  ggplot(td, aes(x = time, y = estimate, color = stratum, linetype = model)) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = RISK_COLORS, name = NULL) +
    scale_linetype_manual(values = c(text = "solid", base = "dashed"), name = NULL) +
    coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
    annotate("text", x = 1, y = 0.06,
             label = sprintf("text logrank p=%.1e\nbase logrank p=%.1e", lr_t, lr_b),
             hjust = 0, vjust = 0, size = 2.6, fontface = "italic", color = "#444") +
    labs(x = "Months from first treatment", y = "Overall survival",
         title = "Mortality by Risk-Score Tertile\n(text solid, base dashed)") +
    theme_manuscript() +
    theme(legend.position = c(0.82, 0.78),
          legend.background = element_rect(fill = "white", color = NA),
          legend.spacing.y = unit(0.05, "in"))
}


# ============================================================================
# fig2e: stage vs text risk-quartile (1×2 KM + c-index annotations)
# ============================================================================
build_fig2e <- function() {
  d  <- load_figure_data("fig2_km_stage_vs_risk.csv")
  ci <- load_figure_data("fig2_stage_vs_risk_cindex.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig2_km_stage_vs_risk.csv empty"))
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
  cidx_s <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "stage"][1] else NA_real_
  cidx_q <- if (nrow(ci) > 0) ci$cindex[ci$predictor == "text_risk"][1] else NA_real_

  panel_km <- function(td, palette, lr_p, cidx, title_text) {
    ts2 <- td %>% mutate(stratum = factor(stratum, levels = names(palette)))
    ann <- sprintf("c-index=%.3f\nlogrank p=%.1e",
                   ifelse(is.na(cidx), NA, cidx), lr_p)
    ggplot(ts2, aes(time, estimate, color = stratum)) +
      geom_step(linewidth = 0.9) +
      scale_color_manual(values = palette, name = NULL) +
      coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
      annotate("text", x = 1, y = 0.06, label = ann,
               hjust = 0, vjust = 0, size = 2.6,
               fontface = "italic", color = "#444") +
      labs(x = "Months from first treatment", y = "Overall survival",
           title = title_text) +
      theme_manuscript() +
      theme(legend.position = c(0.82, 0.80),
            legend.background = element_rect(fill = "white", color = NA))
  }

  pL <- panel_km(ts, ord4,  lr_s, cidx_s, "Survival by Cancer Stage")
  pR <- panel_km(tq, ord4q, lr_q, cidx_q, "Survival by Text Risk-Score Quartile") +
        labs(y = NULL)
  pL | pR
}


# ============================================================================
# Compose Figure 2
# ============================================================================
metrics <- load_figure_data("fig2_full_cohort_metrics.csv")

p2a <- build_fig2a(metrics)
p2b <- build_fig2b(metrics)
p2c <- build_fig2c()
p2d <- build_fig2d()
p2e <- build_fig2e()

save_panel(p2a, "fig2a", width = 6.4, height = 5.0)
save_panel(p2b, "fig2b", width = 6.0, height = 4.8)
save_panel(p2c, "fig2c", width = 10.8, height = 4.2)
save_panel(p2d, "fig2d", width = 6.4, height = 5.0)
save_panel(p2e, "fig2e", width = 11.0, height = 4.8)

fig2 <- (p2a + p2b) /
        p2c /
        p2d /
        p2e +
        plot_annotation(tag_levels = "A") &
        theme(plot.tag = element_text(size = 14, face = "bold"))

save_figure(fig2, "figure2_text_results", width = 15.5, height = 16.8)
