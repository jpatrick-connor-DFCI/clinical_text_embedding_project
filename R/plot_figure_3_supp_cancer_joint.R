# Figure 3 supplement: the joint Cox model (Figure 3b/3c) refitted within each
# selected cancer type (figures.prep.within_cancer_joint).
suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

JOINT_FDR_ALPHA <- 0.05
JOINT_IQR_WHISKER <- 1.5

# Display-only Tukey trimming, as in Figure 3c; summaries use every endpoint.
tukey_keep <- function(x, k = JOINT_IQR_WHISKER) {
  if (length(x) < 4) return(rep(TRUE, length(x)))
  qs <- stats::quantile(x, c(0.25, 0.75))
  x >= qs[1] - k * diff(qs) & x <= qs[2] + k * diff(qs)
}

joint_cancer_labels <- function(d) {
  n <- d %>% distinct(cancer_type, scheme, event) %>% count(cancer_type)
  n <- setNames(n$n, n$cancer_type)
  codes <- names(SELECTED_CANCER_TYPES)
  setNames(sprintf("%s (n=%s)", SELECTED_CANCER_TYPES, ifelse(is.na(n[codes]), "0", n[codes])), codes)
}

build_within_cancer_joint_violins <- function(d, summary, variant) {
  modalities <- intersect(MODALITY_ORDER, unique(d$modality))
  labels <- joint_cancer_labels(d)
  plot_df <- d %>%
    group_by(cancer_type, modality) %>% filter(tukey_keep(beta)) %>% ungroup() %>%
    mutate(cancer_type = factor(cancer_type, levels = names(SELECTED_CANCER_TYPES)),
           modality = factor(modality, levels = modalities))
  ann <- summary %>%
    mutate(cancer_type = factor(cancer_type, levels = names(SELECTED_CANCER_TYPES)),
           modality = factor(modality, levels = modalities))
  ggplot(plot_df, aes(modality, beta, fill = modality)) +
    geom_hline(yintercept = 0, color = "#333333", linetype = "dashed") +
    geom_violin(scale = "width", alpha = 0.45, color = "#444444", linewidth = 0.3) +
    geom_point(aes(color = modality), position = position_jitter(width = 0.16, seed = 2026),
               size = 0.4, alpha = 0.3, show.legend = FALSE) +
    geom_errorbar(data = ann, aes(x = modality, ymin = q25_beta, ymax = q75_beta),
                  inherit.aes = FALSE, width = 0.15, linewidth = 0.5, color = "#111111") +
    geom_point(data = ann, aes(x = modality, y = median_beta), inherit.aes = FALSE,
               shape = 23, size = 1.8, fill = "white", color = "#111111", stroke = 0.6) +
    facet_wrap(~cancer_type, ncol = 3, drop = FALSE, labeller = as_labeller(labels)) +
    scale_fill_manual(values = MODALITY_COLORS, guide = "none") +
    scale_color_manual(values = MODALITY_COLORS, guide = "none") +
    scale_x_discrete(labels = MODALITY_DISPLAY, drop = FALSE) +
    labs(x = NULL, y = "Joint Cox coefficient (per SD within cancer type)",
         title = "Joint Cox coefficients within selected cancer types",
         subtitle = sprintf("Refitted per cancer type (%s); n = complete-case endpoints", variant),
         caption = "Diamond and bar: median and IQR over all endpoints; violins and points omit Tukey outliers.") +
    theme_manuscript() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.major.y = element_line(color = "grey92"),
          strip.background = element_blank(), plot.caption = element_text(size = 8))
}

build_within_cancer_joint_significance <- function(summary, d) {
  modalities <- intersect(MODALITY_ORDER, unique(summary$modality))
  cancers <- names(SELECTED_CANCER_TYPES)
  labels <- joint_cancer_labels(d)
  grid <- expand.grid(cancer_type = cancers, modality = modalities, stringsAsFactors = FALSE)
  cells <- left_join(grid, summary, by = c("cancer_type", "modality")) %>%
    mutate(cancer_type = factor(cancer_type, levels = rev(cancers)),
           modality = factor(modality, levels = modalities),
           label = ifelse(is.na(n_endpoints), "Unavailable",
                          sprintf("%d/%d\n(%d+)", n_significant, n_endpoints, n_significant_positive)))
  ggplot(cells, aes(modality, cancer_type, fill = prop_significant)) +
    geom_tile(color = "white", linewidth = 0.8) +
    geom_text(aes(label = label), size = MANUSCRIPT_SMALL_TEXT_SIZE, lineheight = 1.05) +
    scale_fill_gradient(low = "white", high = "#A5C9E2", limits = c(0, 1), na.value = "grey92",
                        labels = scales::percent, name = "Significant\nendpoints") +
    scale_x_discrete(labels = MODALITY_DISPLAY[modalities], position = "top") +
    scale_y_discrete(labels = function(x) unname(labels[x]), drop = FALSE) +
    labs(x = NULL, y = NULL,
         title = "Significant joint Cox coefficients within selected cancer types",
         subtitle = sprintf("Significant / complete-case endpoints (BH-FDR < %.2f; + = positive coefficient)",
                            JOINT_FDR_ALPHA)) +
    theme_manuscript() +
    theme(axis.line = element_blank(), axis.ticks = element_blank(),
          axis.text = element_text(size = 10), legend.position = "right")
}

within_cancer_joint_caption <- function(summary, variant, n_excluded, panel) {
  display <- if (panel == "violins") {
    paste("Each panel is one cancer type (n = complete-case endpoints). Violins and points show",
          "per-endpoint standardized coefficients by modality; diamonds and bars give the median",
          "and interquartile range over all endpoints, and display alone omits values beyond",
          "1.5 IQR (Tukey), as in Figure 3c.")
  } else {
    paste("Rows are cancer types (n = complete-case endpoints) and columns are modalities. Cells",
          sprintf("give endpoints with a BH-FDR < %.2f coefficient over complete-case endpoints,", JOINT_FDR_ALPHA),
          "with the number of those coefficients that are positive in parentheses; shading is",
          "the significant fraction. Grey cells had no fitted endpoint.")
  }
  paste(
    paste("Supplemental Figure 3. Joint Cox model refitted within selected cancer types:",
          paste0(paste(SELECTED_CANCER_TYPES, collapse = ", "), "."),
          "CUP denotes cancer of unknown primary."),
    display,
    paste("For each endpoint and cancer type, the held-out modality risk scores of Figure 3 are",
          "standardized within the cancer type and entered jointly into a Cox model",
          sprintf("(%s fit; lifelines).", variant),
          "Fits require at least 20 patients, 5 events and 5 non-events and at least two",
          "non-constant modalities; fits with any |coefficient| > 5 are discarded. BH-FDR is",
          "applied within each endpoint's fit. An endpoint is included for a cancer type only",
          "when every modality fitted for that cancer type is present (complete case); a",
          "modality whose scores are constant within a cancer type is absent from it. Endpoints",
          "receive equal weight and are correlated; summaries are descriptive."),
    if (isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
      sprintf("The shared manuscript endpoint outlier filter is applied (%d endpoint keys excluded across the manuscript).",
              n_excluded)
    } else "The shared manuscript endpoint outlier filter is disabled.",
    sep = "\n\n"
  )
}

render_figure3_within_cancer_joint <- function() {
  stems <- c(violins = "figS3_within_cancer_joint_betas",
             significance = "figS3_within_cancer_joint_significant")
  for (stem in stems) clear_within_cancer_report(stem, "figure3")
  betas <- read_within_cancer_data("fig3_within_cancer_joint_betas.csv")
  if (!nrow(betas)) {
    message("[figure3 within-cancer joint Cox] SKIPPED: no fits; run figures.prep.within_cancer_joint")
    return(invisible(NULL))
  }
  variant <- Sys.getenv("MANUSCRIPT_JOINT_COX_VARIANT", unset = "unpenalized")
  excluded <- within_cancer_exclusions()
  d <- prepare_within_cancer_joint(betas, excluded, variant, JOINT_FDR_ALPHA)
  d <- select_cancer_types(d, "figure3 within-cancer joint Cox")
  if (!nrow(d)) {
    message("[figure3 within-cancer joint Cox] SKIPPED: no complete-case ", variant,
            " fits for the selected cancer types")
    return(invisible(NULL))
  }
  summary <- summarize_within_cancer_joint(d)

  save_panel(build_within_cancer_joint_violins(d, summary, variant), stems[["violins"]], "figure3",
             width = 10, height = 9)
  save_within_cancer_report(summary, within_cancer_joint_caption(summary, variant, length(excluded), "violins"),
                            stems[["violins"]])
  save_panel(build_within_cancer_joint_significance(summary, d), stems[["significance"]], "figure3",
             width = 9, height = 2.2 + 0.6 * length(SELECTED_CANCER_TYPES))
  save_within_cancer_report(summary, within_cancer_joint_caption(summary, variant, length(excluded), "significance"),
                            stems[["significance"]])
  invisible(summary)
}

render_figure3_within_cancer_joint()
