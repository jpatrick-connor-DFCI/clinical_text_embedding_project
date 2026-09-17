# Render Figure 3 (six-modality comparison, including somatic) in ggplot2 + patchwork.
#
# A complete-case significant endpoints per modality (joint Cox BH-FDR<.05),
# B modality risk-score correlation heatmap (death endpoint),
# C average modality rank across endpoints (1 = best),
# D unpenalized standardized-beta violins + paired endpoint summaries.
#
# Panel C ranks modalities using Harrell's C-index.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggcorrplot); library(stringr)
})

source("R/figure_utils.R")

FDR_ALPHA  <- 0.05
IQR_WHISKER <- 1.5

FIG3_MODALITIES <- MODALITY_ORDER


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
# Complete-case endpoints = (scheme, event) groups with all modalities fit.
# "All" means every modality present somewhere in this run, not every modality
# in MODALITY_ORDER: one that never fit anywhere (its feature-comp tasks did
# not finish) would otherwise disqualify every endpoint and blank the panel.
#
# 3A and 3D share this so the joint-Cox panels report one endpoint set. An
# endpoint missing a modality is not comparable across modalities: counting it
# in 3D's per-modality violins while 3A excludes it made the two panels describe
# different endpoints from the same file.
complete_case_events <- function(betas) {
  fitted_mods <- intersect(FIG3_MODALITIES, unique(betas$modality[!is.na(betas$beta)]))
  betas %>% filter(!is.na(beta)) %>%
    group_by(scheme, event) %>%
    summarise(mods = list(unique(modality)), .groups = "drop") %>%
    filter(vapply(mods, function(s) all(fitted_mods %in% s), logical(1))) %>%
    select(scheme, event)
}

build_fig3a <- function(betas) {
  if (nrow(betas) == 0 || !"sig" %in% names(betas))
    return(placeholder_panel("fig3_joint_betas.csv missing p-values"))

  present <- complete_case_events(betas)
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
         title = "Significant Endpoints per Modality",
         subtitle = sprintf("%d complete-case endpoints", nrow(present))) +
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
  mat <- d %>% select(any_of(c("modality", FIG3_MODALITIES))) %>%
    tibble::column_to_rownames("modality") %>% as.matrix()
  mods <- intersect(FIG3_MODALITIES, rownames(mat))
  mat <- mat[mods, mods, drop = FALSE]
  mat[!is.finite(mat)] <- NA_real_

  ggcorrplot::ggcorrplot(mat, type = "lower", lab = TRUE, lab_size = MANUSCRIPT_TEXT_SIZE,
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
              q25_rank = quantile(rank, .25, na.rm = TRUE),
              q75_rank = quantile(rank, .75, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(n_events = n_events)
}


# ============================================================================
# fig3c: average modality rank across endpoints (1 = best)
# ============================================================================
build_fig3c <- function(betas, metric = METRIC, excluded = EXCLUDED_EVENTS) {
  ranks_long <- drop_excluded_events(
    load_figure_data(sprintf("fig3_modality_ranks_long_%s.csv", metric_suffix(metric))),
    excluded)
  # Use the exact joint-Cox complete-case endpoint set reported in panels A/D.
  if (!is.null(betas) && nrow(betas) > 0 && nrow(ranks_long) > 0) {
    ranks_long <- ranks_long %>%
      inner_join(complete_case_events(betas), by = c("scheme", "event"))
  }
  # fig3_modality_avg_rank_*.csv is pre-aggregated per modality in the Python tier
  # and carries no event column, so it cannot be filtered directly. When events are
  # disqualified, re-derive the mean/SEM from the (filtered) long companion, which
  # is the same complete-case rank matrix the aggregate was built from -- otherwise
  # 3C would still be averaging over events no other panel reports.
  d <- avg_rank_from_long(ranks_long)
  if (nrow(d) == 0) {
    # Fallback only: this file is pre-aggregated and carries no event column, so
    # its mean ranks still reflect the held-out modality and are not re-ranked.
    d <- load_figure_data(sprintf("fig3_modality_avg_rank_%s.csv", metric_suffix(metric)))
  }
  if (nrow(d) == 0) return(placeholder_panel("fig3_modality_avg_rank_*.csv empty"))
  if (!"q25_rank" %in% names(d)) d$q25_rank <- d$mean_rank - d$sem_rank
  if (!"q75_rank" %in% names(d)) d$q75_rank <- d$mean_rank + d$sem_rank
  # Drop MODALITY_ORDER levels this run has no rank for, so a modality that
  # never ran leaves no empty slot on the axis; ranked_mods also sizes the
  # x-range below, which is 1..n over what was actually ranked.
  d <- d %>% filter(!is.na(mean_rank))
  ranked_mods <- intersect(FIG3_MODALITIES, unique(d$modality))
  d <- d %>%
    mutate(modality = factor(modality, levels = ranked_mods)) %>%
    arrange(mean_rank) %>%
    mutate(modality = fct_reorder(modality, mean_rank, .desc = TRUE))

  lbl <- metric_label(metric)

  ggplot(d, aes(mean_rank, modality, fill = as.character(modality))) +
    geom_col(width = 0.62, color = "white") +
    geom_errorbarh(aes(xmin = q25_rank, xmax = q75_rank),
                   height = 0.25, color = "#222222", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", mean_rank),
                  x = q75_rank + 0.05),
              hjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_y_discrete(labels = MODALITY_DISPLAY) +
    coord_cartesian(xlim = c(0.5, length(ranked_mods) + 0.5)) +
    labs(x = sprintf("Average rank across endpoints (1 = best, ranked by %s)", lbl), y = NULL,
         title = "Average Modality Rank",
         caption = sprintf("%s joint-Cox complete-case endpoints; bars show mean rank and IQR across correlated endpoints.",
                           if ("n_events" %in% names(d)) d$n_events[1] else "?")) +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 1,
                                      face = "italic", color = "#666666"))
}


# ============================================================================
# fig3d: unpenalized β violins by modality + Tukey trim + Wilcoxon-vs-0
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
  # Same complete-case endpoint set as 3A, so both joint-Cox panels summarise
  # the same endpoints (see complete_case_events).
  cc_events <- complete_case_events(betas)
  d <- betas %>%
    inner_join(cc_events, by = c("scheme", "event")) %>%
    filter(is.finite(beta))
  if (nrow(d) == 0)
    return(placeholder_panel("no complete-case endpoints with finite beta"))

  trimmed <- d %>% group_by(modality) %>%
    summarise(t = list(tukey_trim(beta)), .groups = "drop") %>%
    mutate(kept = purrr::map(t, "kept"), n_trim = purrr::map_int(t, "n_trim"))
  # purrr may not be loaded; fall back to base
  if (!requireNamespace("purrr", quietly = TRUE)) {
    trimmed <- d %>% group_by(modality) %>%
      do(tibble(t = list(tukey_trim(.$beta)))) %>%
      mutate(kept = lapply(t, function(x) x$kept),
             n_trim = vapply(t, function(x) x$n_trim, integer(1))) %>%
      ungroup() %>% select(modality, kept, n_trim)
  }

  plot_df <- trimmed %>%
    select(modality, kept) %>%
    tidyr::unnest(kept) %>%
    rename(beta = kept) %>%
    mutate(modality = factor(modality, levels = FIG3_MODALITIES))

  ann <- trimmed %>%
    rowwise() %>%
    mutate(mean_beta = mean(unlist(kept), na.rm = TRUE),
           median_beta = median(unlist(kept), na.rm = TRUE),
           q25 = quantile(unlist(kept), .25, na.rm = TRUE),
           q75 = quantile(unlist(kept), .75, na.rm = TRUE)) %>%
    ungroup() %>%
    select(modality, mean_beta, median_beta, q25, q75)

  # Order the x-axis by mean standardized coefficient, descending left-to-right.
  mod_order <- ann %>% arrange(desc(mean_beta)) %>% pull(modality) %>% as.character()
  plot_df <- plot_df %>% mutate(modality = factor(as.character(modality), levels = mod_order))
  ann      <- ann      %>% mutate(modality = factor(as.character(modality), levels = mod_order))
  means_str <- paste(sprintf("%s: %.2f", MODALITY_DISPLAY[as.character(ann$modality)], ann$mean_beta),
                     collapse = "   ")

  ymax <- max(plot_df$beta, na.rm = TRUE)
  ggplot(plot_df, aes(modality, beta, fill = modality)) +
    geom_violin(scale = "width", alpha = 0.45, color = "#444444", linewidth = 0.4) +
    geom_jitter(aes(color = modality), width = 0.18, size = 0.7, alpha = 0.30,
                show.legend = FALSE) +
    geom_hline(yintercept = 0, color = "#333333", linetype = "dashed") +
    geom_errorbar(data = ann,
                  aes(x = modality, y = median_beta, ymin = q25, ymax = q75),
                  inherit.aes = FALSE, width = 0.15, linewidth = 0.6, color = "#111111") +
    geom_point(data = ann, aes(x = modality, y = median_beta),
              inherit.aes = FALSE, shape = 23, size = 2.2,
              fill = "white", color = "#111111", stroke = 0.7) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_color_manual(values = MODALITY_COLORS, guide = "none") +
    scale_x_discrete(labels = MODALITY_DISPLAY) +
    labs(x = NULL, y = "Joint Cox coefficient β per 1-SD risk score",
         title = "Unpenalized Joint Cox Model: Coefficients by Modality",
         caption = paste0(sprintf("%d complete-case endpoints.  ", nrow(cc_events)),
                          "Mean β by modality: ", means_str, "\n",
                          "Unpenalized coefficients; ridge estimates are retained as sensitivity output.",
                          "  Diamond and bar: median and IQR across correlated endpoints.")) +
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
if (nrow(betas) > 0) {
  if (!"fit_variant" %in% names(betas)) betas$fit_variant <- "legacy_ridge_0.01"
  requested_variant <- Sys.getenv("MANUSCRIPT_JOINT_COX_VARIANT", unset = "unpenalized")
  betas <- betas %>% filter(fit_variant == requested_variant)
}
if (nrow(betas) > 0 && "p_value" %in% names(betas)) betas <- bh_within_cell(betas)

p3a <- build_fig3a(betas)
p3b <- build_fig3b()
p3c <- build_fig3c(betas)
p3d <- build_fig3d(betas)

.tag <- metric_tag(METRIC)
# Preserve the existing C-index panel filenames.
save_panel(p3a, paste0("fig3a", .tag), group = "figure3", width = 7.2, height = 5.8)
save_panel(p3b, "fig3b", group = "figure3", width = 7.2, height = 6.0)
save_panel(p3c, paste0("fig3c", .tag), group = "figure3", width = 7.2, height = 6.0)
save_panel(p3d, paste0("fig3d", .tag), group = "figure3", width = 9.2, height = 6.4)

# Complete manuscript figure, with one lowercase label per named panel.
save_compiled_figure(
  list(a = p3a, b = p3b, c = p3c, d = p3d),
  number = 3, width = 20, height = 14
)
