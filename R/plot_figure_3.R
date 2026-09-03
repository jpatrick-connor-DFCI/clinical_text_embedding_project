# Render Figure 3 (modality comparison) in ggplot2 + patchwork.
#
# A complete-case significant endpoints per modality (joint Cox BH-FDR<.05),
# B modality risk-score correlation heatmap (death endpoint),
# C average modality rank across endpoints (1 = best),
# D Wald-z violins (β/SE) + Wilcoxon-vs-0 stars + Tukey-IQR trim.
#
# Metric switch: panel C is ranked by whichever metric MANUSCRIPT_METRIC selects
# (see figure_utils.R::METRIC) — "cindex" (Harrell's C-index) or "auc" (mean AUC(t)).
# A/B/D don't depend on the survival ranking metric and are unaffected.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggcorrplot); library(stringr)
})

source("R/figure_utils.R")

FDR_ALPHA  <- 0.05
IQR_WHISKER <- 1.5


# BH-FDR per (scheme, event) cell — matches the Python pipeline
bh_within_cell <- function(betas) {
  if (nrow(betas) == 0) return(betas)
  betas %>%
    group_by(scheme, event) %>%
    mutate(q_value = stats::p.adjust(replace(p_value, is.na(p_value), 1),
                                     method = "BH")) %>%
    ungroup() %>%
    mutate(sig = q_value < FDR_ALPHA & !is.na(beta))
}


# ============================================================================
# fig3a: significant endpoints per modality (complete-case only)
# ============================================================================
build_fig3a <- function(betas) {
  if (nrow(betas) == 0 || !"sig" %in% names(betas))
    return(placeholder_panel("fig3_joint_betas.csv missing p-values"))

  # Complete-case endpoints = (scheme, event) groups with all modalities fit.
  # "All" means every modality present somewhere in this run, not every modality
  # in MODALITY_ORDER: one that never fit anywhere (its feature-comp tasks did
  # not finish) would otherwise disqualify every endpoint and blank the panel.
  fitted_mods <- intersect(MODALITY_ORDER, unique(betas$modality[!is.na(betas$beta)]))
  present <- betas %>% filter(!is.na(beta)) %>%
    group_by(scheme, event) %>%
    summarise(mods = list(unique(modality)), .groups = "drop") %>%
    filter(vapply(mods, function(s) all(fitted_mods %in% s), logical(1))) %>%
    select(scheme, event)
  cc <- betas %>% inner_join(present, by = c("scheme", "event"))
  counts <- cc %>%
    group_by(modality) %>%
    summarise(n = sum(sig, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(n)) %>%
    mutate(modality = factor(modality, levels = modality))

  ggplot(counts, aes(modality, n, fill = modality)) +
    geom_col(width = 0.6, color = "white") +
    geom_text(aes(label = n), vjust = -0.3, size = MANUSCRIPT_TEXT_SIZE) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_x_discrete(labels = MODALITY_DISPLAY) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL,
         y = sprintf("# endpoints (joint Cox BH-FDR < %.2f)", FDR_ALPHA),
         title = "Significant Endpoints per Modality") +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig3b: modality risk-score correlation heatmap (death endpoint)
# ============================================================================
build_fig3b <- function() {
  d <- load_figure_data("fig3_risk_score_corr.csv")
  if (nrow(d) == 0 || !"modality" %in% names(d))
    return(placeholder_panel("fig3_risk_score_corr.csv empty"))
  mat <- d %>% select(any_of(c("modality", MODALITY_ORDER))) %>%
    tibble::column_to_rownames("modality") %>% as.matrix()
  mods <- intersect(MODALITY_ORDER, rownames(mat))
  mat <- mat[mods, mods, drop = FALSE]
  mat[!is.finite(mat)] <- NA_real_

  ggcorrplot::ggcorrplot(mat, lab = TRUE, lab_size = MANUSCRIPT_TEXT_SIZE,
                         colors = c("#2E86C1", "#FFFFFF", "#E74C3C"),
                         outline.color = "white") +
    scale_x_discrete(labels = MODALITY_DISPLAY) +
    scale_y_discrete(labels = MODALITY_DISPLAY) +
    labs(title = "Modality Risk-Score Correlation",
         x = NULL, y = NULL, fill = "Pearson r") +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          panel.grid = element_blank())
}


# Friedman omnibus test across modalities (repeated-measures ranks, one block
# per endpoint) — answers whether modality rank differs at all before any
# pairwise comparison would be considered.
friedman_p <- function(ranks_long) {
  if (nrow(ranks_long) == 0) return(NA_real_)
  mat <- ranks_long %>%
    tidyr::pivot_wider(id_cols = c(scheme, event), names_from = modality, values_from = rank) %>%
    select(-scheme, -event) %>%
    as.matrix()
  if (nrow(mat) < 2 || ncol(mat) < 2) return(NA_real_)
  tryCatch(stats::friedman.test(mat)$p.value, error = function(e) NA_real_)
}

# Re-aggregate per-modality mean/SEM rank from the long per-endpoint ranks.
# Mirrors figures/prep/figure3.py::_modality_avg_rank so a filtered 3C matches
# what the Python tier would have produced from the surviving endpoints:
# mean over endpoints, SEM = sd(ddof=1)/sqrt(n_events), 0 when n_events <= 1.
avg_rank_from_long <- function(ranks_long) {
  if (is.null(ranks_long) || nrow(ranks_long) == 0) return(tibble::tibble())
  if (!all(c("modality", "rank") %in% names(ranks_long))) return(tibble::tibble())
  n_events <- dplyr::n_distinct(ranks_long[, c("scheme", "event")])
  ranks_long %>%
    group_by(modality) %>%
    summarise(mean_rank = mean(rank, na.rm = TRUE),
              sem_rank = if (dplyr::n() > 1) stats::sd(rank, na.rm = TRUE) / sqrt(dplyr::n()) else 0,
              .groups = "drop") %>%
    mutate(n_events = n_events)
}


# ============================================================================
# fig3c: average modality rank across endpoints (1 = best)
# ============================================================================
build_fig3c <- function(metric = METRIC, excluded = EXCLUDED_EVENTS) {
  ranks_long <- drop_excluded_events(
    load_figure_data(sprintf("fig3_modality_ranks_long_%s.csv", metric_suffix(metric))),
    excluded)
  # fig3_modality_avg_rank_*.csv is pre-aggregated per modality in the Python tier
  # and carries no event column, so it cannot be filtered directly. When events are
  # disqualified, re-derive the mean/SEM from the (filtered) long companion, which
  # is the same complete-case rank matrix the aggregate was built from -- otherwise
  # 3C would still be averaging over events no other panel reports.
  d <- avg_rank_from_long(ranks_long)
  if (nrow(d) == 0) {
    d <- load_figure_data(sprintf("fig3_modality_avg_rank_%s.csv", metric_suffix(metric)))
  }
  if (nrow(d) == 0) return(placeholder_panel("fig3_modality_avg_rank_*.csv empty"))
  # Drop MODALITY_ORDER levels this run has no rank for, so a modality that
  # never ran leaves no empty slot on the axis; ranked_mods also sizes the
  # x-range below, which is 1..n over what was actually ranked.
  d <- d %>% filter(!is.na(mean_rank))
  ranked_mods <- intersect(MODALITY_ORDER, unique(d$modality))
  d <- d %>%
    mutate(modality = factor(modality, levels = ranked_mods)) %>%
    arrange(mean_rank) %>%
    mutate(modality = fct_reorder(modality, mean_rank, .desc = TRUE))

  fp <- friedman_p(ranks_long)
  stars <- p_to_stars(fp)
  lbl <- metric_label(metric)

  ggplot(d, aes(mean_rank, modality, fill = as.character(modality))) +
    geom_col(width = 0.62, color = "white") +
    geom_errorbarh(aes(xmin = mean_rank - sem_rank,
                       xmax = mean_rank + sem_rank),
                   height = 0.25, color = "#222222", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", mean_rank),
                  x = mean_rank + sem_rank + 0.05),
              hjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_y_discrete(labels = MODALITY_DISPLAY) +
    coord_cartesian(xlim = c(0.5, length(ranked_mods) + 0.5)) +
    labs(x = sprintf("Average rank across endpoints (1 = best, ranked by %s)", lbl), y = NULL,
         title = "Average Modality Rank",
         caption = sprintf("Friedman test across modalities: p=%s  %s",
                           ifelse(is.na(fp), "n/a", sprintf("%.1e", fp)), stars)) +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 1,
                                      face = "italic", color = "#666666"))
}


# ============================================================================
# fig3d: Wald-z (β/SE) violins by modality + Tukey-trim + Wilcoxon-vs-0 stars
# ============================================================================
tukey_trim <- function(x, k = IQR_WHISKER) {
  qs <- stats::quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iqr <- qs[2] - qs[1]
  lo  <- qs[1] - k * iqr
  hi  <- qs[2] + k * iqr
  list(kept = x[x >= lo & x <= hi & is.finite(x)],
       n_trim = sum(x < lo | x > hi, na.rm = TRUE))
}

build_fig3d <- function(betas) {
  if (nrow(betas) == 0)
    return(placeholder_panel("fig3_joint_betas.csv empty"))
  d <- betas %>%
    mutate(z = beta / se) %>%
    filter(is.finite(z))

  trimmed <- d %>% group_by(modality) %>%
    summarise(t = list(tukey_trim(z)), .groups = "drop") %>%
    mutate(kept = purrr::map(t, "kept"), n_trim = purrr::map_int(t, "n_trim"))
  # purrr may not be loaded; fall back to base
  if (!requireNamespace("purrr", quietly = TRUE)) {
    trimmed <- d %>% group_by(modality) %>%
      do(tibble(t = list(tukey_trim(.$z)))) %>%
      mutate(kept = lapply(t, function(x) x$kept),
             n_trim = vapply(t, function(x) x$n_trim, integer(1))) %>%
      ungroup() %>% select(modality, kept, n_trim)
  }

  plot_df <- trimmed %>%
    select(modality, kept) %>%
    tidyr::unnest(kept) %>%
    rename(z = kept) %>%
    mutate(modality = factor(modality, levels = MODALITY_ORDER))

  ann <- trimmed %>%
    rowwise() %>%
    mutate(p = wilcoxon_vs0(unlist(kept)),
           stars = p_to_stars(p),
           mean_z = mean(unlist(kept), na.rm = TRUE),
           sd_z = sd(unlist(kept), na.rm = TRUE)) %>%
    ungroup() %>%
    select(modality, stars, mean_z, sd_z)

  # Order the x-axis by mean standardized coefficient, descending left-to-right.
  mod_order <- ann %>% arrange(desc(mean_z)) %>% pull(modality) %>% as.character()
  plot_df <- plot_df %>% mutate(modality = factor(as.character(modality), levels = mod_order))
  ann      <- ann      %>% mutate(modality = factor(as.character(modality), levels = mod_order))
  means_str <- paste(sprintf("%s: %.2f", MODALITY_DISPLAY[as.character(ann$modality)], ann$mean_z),
                     collapse = "   ")

  ymax <- max(plot_df$z, na.rm = TRUE)
  ggplot(plot_df, aes(modality, z, fill = modality)) +
    geom_violin(scale = "width", alpha = 0.45, color = "#444444", linewidth = 0.4) +
    geom_jitter(aes(color = modality), width = 0.18, size = 0.7, alpha = 0.30,
                show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "#333333", linetype = "dashed") +
    geom_hline(yintercept = c(-1.96, 1.96), color = "#999999",
               linetype = "dotted") +
    geom_errorbar(data = ann,
                  aes(x = modality, y = mean_z, ymin = mean_z - sd_z, ymax = mean_z + sd_z),
                  inherit.aes = FALSE, width = 0.15, linewidth = 0.6, color = "#111111") +
    geom_point(data = ann, aes(x = modality, y = mean_z),
              inherit.aes = FALSE, shape = 23, size = 2.2,
              fill = "white", color = "#111111", stroke = 0.7) +
    geom_text(data = ann, aes(modality, ymax * 1.04, label = stars),
              inherit.aes = FALSE,
              size = MANUSCRIPT_TEXT_SIZE, fontface = "bold", color = "#222222") +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_color_manual(values = MODALITY_COLORS, guide = "none") +
    scale_x_discrete(labels = MODALITY_DISPLAY) +
    labs(x = NULL, y = "Joint Cox standardized coefficient (z = β/SE)",
         title = "Joint Cox Model: Standardized Coefficient by Modality",
         caption = paste0("Mean z by modality: ", means_str, "\n",
                          "z from L2-penalized fit; ±1.96 lines are descriptive, not an exact test.",
                          "  Stars: Wilcoxon vs z=0  (*<.05, **<.01, ***<.001, ****<1e-4).",
                          "  Diamond ± bar: mean ± SD.")) +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 0,
                                      face = "italic", color = "#777777"),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# Compose
# ============================================================================
# Disqualified events (text much worse than base in the full cohort) are removed
# from every figure, so drop them here before anything else reads `betas`.
# Filtering precedes bh_within_cell() so no disqualified endpoint is corrected or
# counted; BH runs within each (scheme, event) cell, so the surviving cells'
# q-values are the same either way.
EXCLUDED_EVENTS <- excluded_event_keys(load_figure_data("fig2_full_cohort_metrics.csv"))

betas <- drop_excluded_events(load_figure_data("fig3_joint_betas.csv"), EXCLUDED_EVENTS)
if (nrow(betas) > 0 && "p_value" %in% names(betas)) betas <- bh_within_cell(betas)

p3a <- build_fig3a(betas)
p3b <- build_fig3b()
p3c <- build_fig3c()
p3d <- build_fig3d(betas)

.tag <- metric_tag(METRIC)
save_panel(p3a, "fig3a", group = "figure3", width = 6.0, height = 4.8)
save_panel(p3b, "fig3b", group = "figure3", width = 6.0, height = 5.0)
save_panel(p3c, paste0("fig3c", .tag), group = "figure3", width = 6.0, height = 5.0)
save_panel(p3d, "fig3d", group = "figure3", width = 6.0, height = 5.0)
