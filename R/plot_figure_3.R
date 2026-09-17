# Figure 3: a C-index modality rank, b significant endpoint counts,
# c joint Cox coefficient violins. Individual panels and compiled PNG/PDF,
# with panel letters and no plot titles.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr)
})

source("R/figure_utils.R")
source("R/publication_style.R")

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
# fig3b: significant endpoints per modality (complete-case only)
# ============================================================================
# Complete-case endpoints = (scheme, event) groups with all modalities fit.
# "All" means every modality present somewhere in this run, not every modality
# in MODALITY_ORDER: one that never fit anywhere (its feature-comp tasks did
# not finish) would otherwise disqualify every endpoint and blank the panel.
#
# Panels b and c share this so the joint-Cox panels report one endpoint set. An
# endpoint missing a modality is not comparable across modalities: counting it
# in the per-modality violins while the count panel excludes it would describe
# different endpoints from the same file.
complete_case_events <- function(betas) {
  fitted_mods <- intersect(FIG3_MODALITIES, unique(betas$modality[!is.na(betas$beta)]))
  betas %>% filter(!is.na(beta)) %>%
    group_by(scheme, event) %>%
    summarise(mods = list(unique(modality)), .groups = "drop") %>%
    filter(vapply(mods, function(s) all(fitted_mods %in% s), logical(1))) %>%
    select(scheme, event)
}

build_significant_endpoints <- function(betas) {
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
    scale_x_discrete(labels = function(x) stringr::str_wrap(unname(MODALITY_DISPLAY[x]), 9)) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = NULL,
         y = "Significant endpoints",
         title = "Significant endpoints",
         subtitle = sprintf("BH-FDR < %.2f; n = %d endpoints", FDR_ALPHA, nrow(present))) +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# fig3a: average modality rank across endpoints (1 = best)
# ============================================================================
# Summarize the surviving endpoint ranks after the shared exclusions and
# joint-Cox complete-case restriction. The rank panel displays means and IQRs.
avg_rank_from_long <- function(ranks_long) {
  if (is.null(ranks_long) || nrow(ranks_long) == 0) return(tibble::tibble())
  if (!all(c("scheme", "event", "modality", "rank") %in% names(ranks_long))) {
    return(tibble::tibble())
  }
  n_events <- dplyr::n_distinct(ranks_long[, c("scheme", "event")])
  ranks_long %>%
    group_by(modality) %>%
    summarise(mean_rank = mean(rank, na.rm = TRUE),
              q25_rank = quantile(rank, .25, na.rm = TRUE),
              q75_rank = quantile(rank, .75, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(n_events = n_events)
}

build_modality_rank <- function(betas, metric = METRIC, excluded = EXCLUDED_EVENTS) {
  ranks_long <- drop_excluded_events(
    load_figure_data(sprintf("fig3_modality_ranks_long_%s.csv", metric_suffix(metric))),
    excluded)
  # Use the exact joint-Cox complete-case endpoint set reported in panels b/c.
  if (!is.null(betas) && nrow(betas) > 0 && nrow(ranks_long) > 0) {
    ranks_long <- ranks_long %>%
      inner_join(complete_case_events(betas), by = c("scheme", "event"))
  }
  # fig3_modality_avg_rank_*.csv is pre-aggregated per modality in the Python tier
  # and carries no event column, so it cannot be filtered directly. When events are
  # disqualified, re-derive the mean/IQR from the (filtered) long companion, which
  # is the same complete-case rank matrix the aggregate was built from -- otherwise
  # 3a would still be averaging over events no other panel reports.
  d <- avg_rank_from_long(ranks_long)
  if (nrow(d) == 0) return(placeholder_panel("No complete modality ranks after endpoint filtering"))
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
    geom_point(shape = 21, size = 2, color = "#333333") +
    geom_errorbarh(aes(xmin = q25_rank, xmax = q75_rank),
                   height = 0.25, color = "#222222", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", mean_rank),
                  x = q75_rank + 0.12),
              hjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_y_discrete(labels = MODALITY_DISPLAY) +
    coord_cartesian(xlim = c(0.5, length(ranked_mods) + 0.85)) +
    labs(x = "Mean C-index rank (1 = best)", y = NULL,
         title = "Modality rank",
         subtitle = sprintf("n = %s endpoints with complete ranks", d$n_events[1])) +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"))
}


# ============================================================================
# fig3c: unpenalized Cox coefficient violins by modality with display trimming
# ============================================================================
tukey_trim <- function(x, k = IQR_WHISKER) {
  qs <- stats::quantile(x, c(0.25, 0.75), na.rm = TRUE)
  iqr <- qs[2] - qs[1]
  lo  <- qs[1] - k * iqr
  hi  <- qs[2] + k * iqr
  list(kept = x[x >= lo & x <= hi & is.finite(x)],
       n_trim = sum(x < lo | x > hi, na.rm = TRUE))
}

build_joint_cox_violins <- function(betas) {
  if (nrow(betas) == 0)
    return(placeholder_panel("fig3_joint_betas.csv empty"))
  # Same complete-case endpoint set as panel b, so both joint-Cox panels summarise
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
  p <- ggplot(plot_df, aes(modality, beta, fill = modality)) +
    geom_violin(scale = "width", alpha = 0.45, color = "#444444", linewidth = 0.4) +
    geom_point(aes(color = modality), position = position_jitter(width = 0.16, seed = 2026),
               size = 0.45, alpha = 0.30, show.legend = FALSE) +
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
    labs(x = NULL, y = "Joint Cox coefficient (per SD)",
         title = "Joint Cox coefficients",
         subtitle = sprintf("n = %d joint-model endpoints", nrow(cc_events))) +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5),
          panel.grid.major.y = element_line(color = "grey90"))
  attr(p, "caption_detail") <- sprintf("The joint-model analyses include %d endpoints before display trimming.", nrow(cc_events))
  p
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

retire_main_panels(3, letters[1:3])

p3a <- build_modality_rank(betas)
p3b <- build_significant_endpoints(betas)
p3c <- build_joint_cox_violins(betas)

.tag <- metric_tag()
save_panel(p3a, paste0("fig3a", .tag), group = "figure3", width = 3.5, height = 3.2, dpi = 600)
save_panel(p3b, paste0("fig3b", .tag), group = "figure3", width = 3.5, height = 3.2, dpi = 600)
save_panel(p3c, paste0("fig3c", .tag), group = "figure3", width = MANUSCRIPT_WIDTH, height = 2.8, dpi = 600)
save_compiled_figure(
  list(a = p3a, b = p3b, c = p3c),
  number = 3, width = COMPILED_FIGURE_WIDTH, height = 8.2, design = "ab\ncc"
)
