# Figure 3 supplement: stacked Cox models combining the held-out modality risk
# scores (figures.prep.figure3_combined).
#   a  Overall survival: each non-text modality alone vs. with the text score
#   b  Overall survival: text only, all modalities except text, all modalities
#   c  Overall survival: paired differences between the panel-b models
#   d  The panel-c differences across every endpoint (shared endpoint filter applied)
# and a second page (figS3_combined_scatter) of per-endpoint C-index scatters, as in
# Figure 2a: each modality vs. modality + text, all but text vs. all, all but text vs. text.
suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

COMBINED_GROUP <- "figure3"
COMBINED_NON_TEXT <- setdiff(MODALITY_ORDER, "text")
COMBINED_SET_MODELS <- c("text", "all_minus_text", "all")
COMBINED_SET_COLORS <- c(text = MODALITY_COLORS[["text"]], all_minus_text = "#4A4A4A",
                         all = BENEFIT_COLOR)
# Mirrors figures.prep.figure3_combined.CONTRASTS: model minus reference.
COMBINED_CONTRASTS <- tibble::tibble(
  model = c(paste0(COMBINED_NON_TEXT, "+text"), "all", "all", "text"),
  reference = c(COMBINED_NON_TEXT, "all_minus_text", "text", "all_minus_text")
)
COMBINED_SET_CONTRASTS <- filter(COMBINED_CONTRASTS, model %in% COMBINED_SET_MODELS,
                                 reference %in% COMBINED_SET_MODELS)

combined_model_label <- function(model) {
  base <- sub("\\+text$", "", model)
  unname(case_when(
    model == "all" ~ "All modalities",
    model == "all_minus_text" ~ "All except text",
    model == "text" ~ "Text",
    grepl("\\+text$", model) ~ paste(MODALITY_DISPLAY[base], "+ text"),
    TRUE ~ MODALITY_DISPLAY[model]
  ))
}

combined_contrast_label <- function(model, reference) {
  paste(combined_model_label(model), "vs.", combined_model_label(reference))
}

format_estimate <- function(estimate, lower, upper, signed = FALSE) {
  sprintf(if (signed) "%+.3f (%.3f, %.3f)" else "%.3f (%.3f, %.3f)", estimate, lower, upper)
}

# Rows listed top to bottom, with right-hand labels on a secondary axis.
labelled_rows <- function(left, right, right_name = NULL) {
  at <- rev(seq_along(left))
  scale_y_continuous(breaks = at, labels = left, expand = expansion(add = 0.6),
                     sec.axis = sec_axis(~., breaks = at, labels = right, name = right_name))
}

theme_combined <- function() {
  theme_manuscript() +
    theme(axis.line.y.right = element_blank(), axis.ticks.y.right = element_blank(),
          axis.text.y.right = element_text(size = MANUSCRIPT_BASE_SIZE - 3, color = "grey25"),
          axis.title.y.right = element_text(size = MANUSCRIPT_BASE_SIZE - 3, color = "grey25"),
          panel.grid.major.y = element_line(color = "grey93"),
          plot.title = element_text(size = MANUSCRIPT_BASE_SIZE, face = "bold"),
          plot.subtitle = element_text(size = MANUSCRIPT_BASE_SIZE - 3, color = "grey30"))
}

build_combined_os_modalities <- function(os, delta) {
  if (!nrow(os) || !nrow(delta)) return(placeholder_panel("no overall-survival results"))
  pairs <- delta %>%
    filter(reference %in% COMBINED_NON_TEXT, model == paste0(reference, "+text")) %>%
    mutate(modality = reference) %>%
    arrange(match(modality, COMBINED_NON_TEXT))
  if (!nrow(pairs)) return(placeholder_panel("no overall-survival modality/text pairs"))
  mods <- pairs$modality
  at <- setNames(rev(seq_along(mods)), mods)
  points <- os %>%
    filter(model %in% c(mods, paste0(mods, "+text"))) %>%
    mutate(modality = sub("\\+text$", "", model),
           variant = ifelse(grepl("\\+text$", model), "with_text", "alone"),
           y = at[modality] + ifelse(variant == "alone", 0.16, -0.16))
  text_cindex <- os$cindex[os$model == "text"]
  p <- ggplot(points, aes(y = y)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed")
  if (length(text_cindex) == 1) {
    p <- p +
      geom_vline(xintercept = text_cindex, color = MODALITY_COLORS[["text"]], linetype = "dotted") +
      annotate("text", x = text_cindex, y = length(mods) + 0.55, label = "Text alone",
               hjust = -0.08, size = MANUSCRIPT_SMALL_TEXT_SIZE, color = MODALITY_COLORS[["text"]])
  }
  p +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper, color = modality), linewidth = 0.6) +
    geom_point(aes(x = cindex, fill = modality, shape = variant), size = 2.4,
               color = "grey15", stroke = 0.4) +
    scale_shape_manual(values = c(alone = 21, with_text = 24),
                       labels = c(alone = "Modality alone", with_text = "Modality + text"),
                       name = NULL) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_color_manual(values = MODALITY_COLORS, guide = "none") +
    labelled_rows(unname(MODALITY_DISPLAY[mods]),
                  format_estimate(pairs$delta_cindex, pairs$ci_lower, pairs$ci_upper, signed = TRUE),
                  "Gain from adding text (95% CI)") +
    guides(shape = guide_legend(override.aes = list(fill = "grey60"))) +
    labs(x = "Overall survival C-index", y = NULL, title = "Each modality with and without text") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_combined_os_models <- function(os) {
  if (!nrow(os)) return(placeholder_panel("no overall-survival results"))
  d <- os %>% filter(model %in% COMBINED_SET_MODELS) %>%
    arrange(match(model, COMBINED_SET_MODELS))
  if (!nrow(d)) return(placeholder_panel("no overall-survival combined models"))
  d <- mutate(d, y = rev(seq_len(n())))
  ggplot(d, aes(x = cindex, y = y, color = model)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), linewidth = 0.7) +
    geom_point(size = 2.6) +
    scale_color_manual(values = COMBINED_SET_COLORS, guide = "none") +
    labelled_rows(combined_model_label(d$model),
                  format_estimate(d$cindex, d$ci_lower, d$ci_upper), "C-index (95% CI)") +
    labs(x = "Overall survival C-index", y = NULL, title = "Text, other modalities, and both") +
    theme_combined()
}

build_combined_os_contrasts <- function(delta) {
  if (!nrow(delta)) return(placeholder_panel("no overall-survival results"))
  d <- inner_join(COMBINED_SET_CONTRASTS, delta, by = c("model", "reference"))
  if (!nrow(d)) return(placeholder_panel("no overall-survival contrasts"))
  d <- mutate(d, y = rev(seq_len(n())))
  ggplot(d, aes(x = delta_cindex, y = y)) +
    geom_vline(xintercept = 0, color = "grey45", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), linewidth = 0.7, color = "grey20") +
    geom_point(size = 2.6, shape = 23, fill = "white", color = "grey20", stroke = 0.7) +
    labelled_rows(combined_contrast_label(d$model, d$reference),
                  format_estimate(d$delta_cindex, d$ci_lower, d$ci_upper, signed = TRUE),
                  "Difference (95% CI)") +
    labs(x = "Overall survival C-index difference", y = NULL, title = "Paired differences") +
    theme_combined()
}

# Per-endpoint paired differences for each contrast, over endpoints where both
# models were evaluated.
combined_endpoint_deltas <- function(cindex, excluded) {
  ok <- drop_excluded_events(cindex, excluded) %>%
    filter(status == "ok", is.finite(cindex)) %>%
    select(scheme, event, model, cindex)
  bind_rows(lapply(seq_len(nrow(COMBINED_CONTRASTS)), function(i) {
    m <- COMBINED_CONTRASTS$model[i]
    r <- COMBINED_CONTRASTS$reference[i]
    inner_join(filter(ok, model == m) %>% select(scheme, event, model_cindex = cindex),
               filter(ok, model == r) %>% select(scheme, event, reference_cindex = cindex),
               by = c("scheme", "event")) %>%
      mutate(model = m, reference = r, delta_cindex = model_cindex - reference_cindex)
  }))
}

summarize_endpoint_deltas <- function(deltas) {
  deltas %>%
    group_by(model, reference) %>%
    summarise(n_endpoints = n(), median_delta = median(delta_cindex),
              q25_delta = unname(quantile(delta_cindex, 0.25)),
              q75_delta = unname(quantile(delta_cindex, 0.75)),
              share_positive = mean(delta_cindex > 0), .groups = "drop")
}

build_combined_endpoint_contrasts <- function(deltas, summary) {
  if (!nrow(deltas)) return(placeholder_panel("no endpoints with every combined model"))
  s <- inner_join(COMBINED_SET_CONTRASTS, summary, by = c("model", "reference"))
  if (!nrow(s)) return(placeholder_panel("no endpoints with every combined model"))
  s <- mutate(s, y = rev(seq_len(n())))
  d <- inner_join(deltas, select(s, model, reference, y), by = c("model", "reference"))
  schemes <- intersect(names(SCHEME_LABELS), unique(d$scheme))
  ggplot(d, aes(x = delta_cindex, y = y)) +
    geom_vline(xintercept = 0, color = "grey45", linetype = "dashed") +
    geom_violin(aes(group = y), orientation = "y", scale = "width", width = 0.8,
                fill = "grey88", color = "grey55", linewidth = 0.3) +
    geom_point(aes(color = scheme), position = position_jitter(height = 0.18, width = 0, seed = 2026),
               size = 0.5, alpha = 0.45) +
    geom_boxplot(aes(group = y), orientation = "y", width = 0.22, outlier.shape = NA,
                 fill = NA, color = "grey15", linewidth = 0.4) +
    scale_color_manual(values = SCHEME_COLORS[schemes], labels = SCHEME_LABELS[schemes], name = NULL) +
    labelled_rows(combined_contrast_label(s$model, s$reference),
                  sprintf("median %+.3f; %.0f%% > 0; n = %d",
                          s$median_delta, 100 * s$share_positive, s$n_endpoints),
                  NULL) +
    guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
    labs(x = "C-index difference", y = NULL, title = "Paired differences across endpoints") +
    theme_combined() +
    theme(legend.position = "bottom")
}

# Per-endpoint C-index scatter of `contrasts` (reference on x, model on y), styled
# as Figure 2a with a dotted y = x line; one facet per contrast when there are several.
build_combined_scatter <- function(deltas, contrasts, title) {
  if (!nrow(deltas)) return(placeholder_panel("no endpoints with every combined model"))
  d <- inner_join(deltas, contrasts, by = c("model", "reference"))
  if (!nrow(d)) return(placeholder_panel(paste("no endpoints for", title)))
  labels <- combined_contrast_label(contrasts$model, contrasts$reference)
  d <- d %>% mutate(
    contrast = factor(combined_contrast_label(model, reference), levels = labels),
    plot_group = fig2a_plot_group(scheme, event)
  )
  # Unlike Figure 2a, no 0.45 floor: single-modality C-indices (e.g. PRS) sit near 0.5.
  lo <- min(c(d$reference_cindex, d$model_cindex)) - 0.02
  hi <- min(1.00, max(c(d$reference_cindex, d$model_cindex)) + 0.02)
  counts <- d %>% group_by(contrast) %>%
    summarise(label = sprintf("%.0f%% above; n = %d", 100 * mean(delta_cindex > 0), n()),
              .groups = "drop")
  one <- nrow(contrasts) == 1
  p <- ggplot(d, aes(reference_cindex, model_cindex, color = plot_group, shape = plot_group)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "#666666") +
    geom_point(data = filter(d, plot_group != "death"), size = 1.0, alpha = 0.5) +
    # Overall survival last, larger and opaque, as in Figure 2a.
    geom_point(data = filter(d, plot_group == "death"), size = 2.2, alpha = 1) +
    geom_text(data = counts, aes(x = lo + 0.01, y = hi - 0.01, label = label),
              inherit.aes = FALSE, hjust = 0, vjust = 1, size = MANUSCRIPT_SMALL_TEXT_SIZE,
              color = "grey25") +
    scale_color_manual(values = FIG2A_GROUP_COLORS, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    scale_shape_manual(values = FIG2A_GROUP_SHAPES, labels = FIG2A_GROUP_LABELS,
                       name = NULL, drop = FALSE) +
    scale_x_continuous(breaks = scales::breaks_pretty(4)) +
    scale_y_continuous(breaks = scales::breaks_pretty(4)) +
    coord_fixed(xlim = c(lo, hi), ylim = c(lo, hi), expand = FALSE) +
    labs(x = if (one) paste(combined_model_label(contrasts$reference), "C-index") else "Modality alone C-index",
         y = if (one) paste(combined_model_label(contrasts$model), "C-index") else "Modality + text C-index",
         title = title) +
    theme_manuscript() +
    theme(plot.title = element_text(size = MANUSCRIPT_BASE_SIZE, face = "bold"),
          panel.spacing.x = unit(1.2, "lines"))
  if (one) {
    p + theme(legend.position = "none")
  } else {
    p + facet_wrap(~contrast, nrow = 1,
                   labeller = as_labeller(setNames(unname(MODALITY_DISPLAY[contrasts$reference]), labels))) +
      theme(legend.position = "bottom")
  }
}

combined_scatter_caption <- function(n_excluded) {
  paste(
    "Supplemental Figure 3. Per-endpoint C-indices of the combined-modality models.",
    paste(
      "Each point is one endpoint, plotted as in Figure 2a; the dotted line is y = x, so points",
      "above it favor the y-axis model. (a) Each non-text modality alone (x) versus combined with",
      "the text risk score (y), one panel per modality. (b) All modalities except text (x) versus",
      "all modalities (y). (c) All modalities except text (x) versus text alone (y). Corner",
      "values give the share of endpoints above the line and the number of endpoints; overall",
      "survival (Death) is drawn larger."
    ),
    paste(
      "Models, cohorts and C-indices are as in the combined-model supplement: stacked Cox models",
      "cross-fitted on the out-of-fold modality risk scores, with every model for an endpoint",
      "scored on the same patients and comparable pairs. Endpoints without every model evaluated",
      "are omitted."
    ),
    if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
      sprintf(paste("The shared manuscript endpoint outlier filter is applied (%d endpoint",
                    "keys excluded across the manuscript)."), n_excluded)
    } else "The shared manuscript endpoint outlier filter is disabled.",
    sep = "\n\n"
  )
}

combined_caption <- function(os, n_excluded) {
  n_boot <- if (nrow(os)) max(os$n_boot) else NA
  paste(
    "Supplemental Figure 3. Combining held-out modality risk scores with the text risk score.",
    paste(
      "(a) Overall survival C-index of each non-text modality's risk score alone (circles) and",
      "combined with the text risk score (triangles); the dotted line marks text alone and",
      "right-hand values give the paired gain from adding text. (b) Overall survival C-index",
      "of text alone, all non-text modalities combined, and all modalities combined. (c) Paired",
      "differences between the panel-b models for overall survival. (d) The panel-c differences",
      "for every endpoint where all models were evaluated; violins and points show endpoints",
      "(colored by endpoint scheme), boxes their median and interquartile range, and right-hand",
      "values the median difference, the share of endpoints above zero and the number of",
      "endpoints."
    ),
    paste(
      "Inputs are the out-of-fold risk scores behind Figure 3 (each modality model includes the",
      "shared base covariates). Each score is standardized within its own outer fold, and the",
      "scores in a combination enter an unpenalized Cox model (Breslow ties) that is",
      "cross-fitted over the text model's outer folds, so every prediction is out of sample.",
      "Single-score models are fitted the same way. For each endpoint, all models use the",
      "patients with every modality's held-out score (at least 20 patients, 5 events and 5",
      "non-events), and Harrell's C-index is computed within blocks of patients sharing every",
      "outer fold and pooled by comparable-pair counts, so differences between models are paired.",
      if (is.finite(n_boot)) {
        sprintf(paste("Overall survival intervals are 95%% percentile intervals from %d patient",
                      "bootstrap resamples with the fitted models held fixed."), n_boot)
      } else "",
      "Endpoints receive equal weight in panel d and are correlated; summaries are descriptive."
    ),
    if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
      sprintf(paste("Panel d applies the shared manuscript endpoint outlier filter (%d endpoint",
                    "keys excluded across the manuscript)."), n_excluded)
    } else "The shared manuscript endpoint outlier filter is disabled.",
    sep = "\n\n"
  )
}

render_figure3_combined <- function() {
  stems <- c(os_modalities = "figS3_combined_os_modality_text",
             os_models = "figS3_combined_os_models",
             os_contrasts = "figS3_combined_os_contrasts",
             endpoints = "figS3_combined_endpoint_contrasts",
             compiled = "figS3_combined_models",
             scatter_modalities = "figS3_combined_scatter_modality_text",
             scatter_all = "figS3_combined_scatter_all",
             scatter_text = "figS3_combined_scatter_text",
             scatter_compiled = "figS3_combined_scatter")
  for (stem in stems) clear_within_cancer_report(stem, COMBINED_GROUP)
  cindex <- read_within_cancer_data("fig3_combined_cindex.csv")
  os <- read_within_cancer_data("fig3_combined_os_cindex.csv")
  delta <- read_within_cancer_data("fig3_combined_os_delta.csv")
  if (!nrow(cindex) && !nrow(os)) {
    message("[figure3 combined models] SKIPPED: no results; run figures.prep.figure3_combined")
    return(invisible(NULL))
  }
  excluded <- within_cancer_exclusions()
  deltas <- if (nrow(cindex)) combined_endpoint_deltas(cindex, excluded) else tibble::tibble()
  summary <- if (nrow(deltas)) summarize_endpoint_deltas(deltas) else tibble::tibble(
    model = character(), reference = character())

  panels <- list(
    os_modalities = build_combined_os_modalities(os, delta),
    os_models = build_combined_os_models(os),
    os_contrasts = build_combined_os_contrasts(delta),
    endpoints = build_combined_endpoint_contrasts(deltas, summary)
  )
  sizes <- list(os_modalities = c(7, 4.2), os_models = c(6, 2.4), os_contrasts = c(7, 2.4),
                endpoints = c(8, 3.6))
  for (name in names(panels)) {
    save_panel(panels[[name]], stems[[name]], COMBINED_GROUP,
               width = sizes[[name]][1], height = sizes[[name]][2])
  }
  kept <- compact_panels(panels)
  if (is.null(kept)) {
    message("[figure3 combined models] SKIPPED: no panel had data")
    return(invisible(NULL))
  }
  compiled <- if (length(kept) == length(panels)) {
    wrap_plots(kept, design = "AB\nAC\nDD", widths = c(1.1, 1), heights = c(1, 1, 1.5))
  } else {
    wrap_plots(kept, ncol = 1)
  }
  save_panel(compiled + plot_annotation(tag_levels = "a"), stems[["compiled"]], COMBINED_GROUP,
             width = 14, height = 10)

  table <- COMBINED_CONTRASTS %>%
    mutate(contrast = combined_contrast_label(model, reference), .before = 1)
  if (nrow(delta)) {
    table <- left_join(table, delta %>% select(model, reference, os_delta_cindex = delta_cindex,
                                               os_ci_lower = ci_lower, os_ci_upper = ci_upper),
                       by = c("model", "reference"))
  }
  table <- left_join(table, summary, by = c("model", "reference"))
  save_within_cancer_report(table, combined_caption(os, length(excluded)), stems[["compiled"]])
  render_combined_scatters(deltas, stems, length(excluded))
  invisible(table)
}

render_combined_scatters <- function(deltas, stems, n_excluded) {
  pair <- function(model, reference) tibble::tibble(model = model, reference = reference)
  scatters <- list(
    scatter_modalities = build_combined_scatter(
      deltas, pair(paste0(COMBINED_NON_TEXT, "+text"), COMBINED_NON_TEXT),
      "Each modality with and without text"),
    scatter_all = build_combined_scatter(deltas, pair("all", "all_minus_text"),
                                         "All modalities with and without text"),
    scatter_text = build_combined_scatter(deltas, pair("text", "all_minus_text"),
                                          "Text alone versus all other modalities")
  )
  sizes <- list(scatter_modalities = c(14, 3.9), scatter_all = c(3.5, 3.2),
                scatter_text = c(3.5, 3.2))
  for (name in names(scatters)) {
    save_panel(scatters[[name]], stems[[name]], COMBINED_GROUP,
               width = sizes[[name]][1], height = sizes[[name]][2], dpi = 600)
  }
  kept <- compact_panels(scatters)
  if (is.null(kept)) return(invisible(NULL))
  compiled <- if (length(kept) == length(scatters)) {
    wrap_plots(kept, design = "AAAA\n#BC#", heights = c(1, 1.15))
  } else {
    wrap_plots(kept, ncol = 1)
  }
  save_panel(compiled + plot_annotation(tag_levels = "a"), stems[["scatter_compiled"]],
             COMBINED_GROUP, width = 14, height = 8.5)
  if (!nrow(deltas)) return(invisible(NULL))
  scatter_table <- deltas %>%
    filter(model %in% c(paste0(COMBINED_NON_TEXT, "+text"), "all", "text"),
           reference %in% c(COMBINED_NON_TEXT, "all_minus_text")) %>%
    select(scheme, event, model, reference, reference_cindex, model_cindex, delta_cindex)
  save_within_cancer_report(scatter_table, combined_scatter_caption(n_excluded),
                            stems[["scatter_compiled"]])
}

render_figure3_combined()
