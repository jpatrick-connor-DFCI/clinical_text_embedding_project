# Render Figure 5 (ICI biomarker discovery) in ggplot2 + patchwork.
#
# A propensity ROC curves (covariates-only vs covariates+embeddings),
# B covariate-balance love plot (SMD before vs after IPTW),
# C cohort-grouped biomarker robustness dot-matrix + definitions caption,
# D 3-panel marker × ICI KM strip with carried-through interaction HR,
# E forest of interaction estimates for the stability-selected markers.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr); library(cowplot)
  library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")


# ============================================================================
# Manual ROC + trapezoidal AUC (matches the Python helper)
# ============================================================================
compute_roc <- function(y, score) {
  ok <- is.finite(y) & is.finite(score)
  y <- as.integer(y[ok]); score <- score[ok]
  if (length(unique(y)) < 2) return(list(fpr = c(0, 1), tpr = c(0, 1), auc = NA_real_))
  ord <- order(-score)
  y <- y[ord]
  n_pos <- max(sum(y == 1), 1L); n_neg <- max(sum(y == 0), 1L)
  cum_tp <- cumsum(y == 1) / n_pos
  cum_fp <- cumsum(y == 0) / n_neg
  fpr <- c(0, cum_fp, 1); tpr <- c(0, cum_tp, 1)
  auc <- sum(diff(fpr) * (head(tpr, -1) + tpr[-1]) / 2)
  list(fpr = fpr, tpr = tpr, auc = auc)
}


# ============================================================================
# fig5a: propensity-score ROC curves
# ============================================================================
build_fig5a <- function() {
  ps <- load_figure_data("fig5_ps_predictions.csv")
  if (nrow(ps) == 0) return(placeholder_panel("fig5_ps_predictions.csv empty"))
  ps <- ps %>% filter(!is.na(model_probs), !is.na(ground_truth))

  model_order <- c("covariates_only", "covariates_plus_embeddings")
  model_colors <- c("covariates_only" = "#F28E2B",
                    "covariates_plus_embeddings" = TEAL)

  roc_df <- bind_rows(lapply(model_order, function(m) {
    sub <- ps[ps$ps_model == m, ]
    if (nrow(sub) == 0) return(NULL)
    r <- compute_roc(sub$ground_truth, sub$model_probs)
    tibble::tibble(model = m, fpr = r$fpr, tpr = r$tpr,
                   label = sprintf("%s (AUC=%.2f)", pretty_model(m), r$auc))
  }))
  legend_labels <- roc_df %>% distinct(model, label) %>%
    arrange(match(model, model_order))

  # Cohort subtitle
  cohorts <- if ("cohort" %in% names(ps)) unique(ps$cohort) else character(0)
  cohort_sub <- paste(vapply(cohorts, function(c) {
    sprintf("%s (%s)", cohort_label(c), COHORT_SHORT[c])
  }, character(1)), collapse = " · ")
  ttl <- "Propensity Score Model: Predicting ICI Receipt"
  if (nzchar(cohort_sub)) ttl <- paste0(ttl, "\n", cohort_sub)

  main <- ggplot(roc_df, aes(fpr, tpr, color = model)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#777777") +
    geom_step(linewidth = 1.1) +
    scale_color_manual(values = model_colors,
                       breaks = legend_labels$model,
                       labels = legend_labels$label, name = NULL) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1.02)) +
    labs(x = "False Positive Rate", y = "True Positive Rate", title = ttl) +
    theme_manuscript() +
    theme(legend.position = c(0.32, 0.90),
          legend.justification = c(0, 1),
          legend.background = element_rect(fill = "white", color = NA))

  main
}


# ============================================================================
# fig5b: covariate balance love plot
# ============================================================================
build_fig5b <- function() {
  d <- load_figure_data("fig5_love_smd.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig5_love_smd.csv empty"))
  d <- d %>%
    mutate(smd_unweighted = abs(smd_unweighted),
           smd_weighted = abs(smd_weighted),
           absu = smd_unweighted) %>%
    arrange(absu) %>%
    mutate(covariate_lbl = gsub("_", " ", as.character(covariate)),
           covariate_lbl = factor(covariate_lbl, levels = covariate_lbl))

  ggplot(d) +
    geom_segment(aes(x = smd_unweighted, xend = smd_weighted,
                     y = covariate_lbl, yend = covariate_lbl),
                 color = "#CCCCCC", linewidth = 0.6) +
    geom_point(aes(x = smd_unweighted, y = covariate_lbl, color = "Unweighted"),
               size = 2.2) +
    geom_point(aes(x = smd_weighted, y = covariate_lbl, color = "IPTW-weighted"),
               size = 2.2) +
    geom_vline(xintercept = 0.1, linetype = "dotted", color = "#999999") +
    scale_color_manual(values = c("Unweighted" = HARM_COLOR,
                                   "IPTW-weighted" = TEAL),
                       breaks = c("Unweighted", "IPTW-weighted"),
                       name = NULL) +
    labs(x = "Absolute standardized mean difference", y = NULL,
         title = "Covariate Balance: Before vs After IPTW",
         subtitle = sprintf("All structured + 10 worst-balanced embedding dimensions · max shown weighted |SMD| = %.3f%s",
                            max(d$smd_weighted, na.rm = TRUE),
                            ifelse(max(d$smd_weighted, na.rm = TRUE) > 0.10,
                                   " (exceeds 0.10)", ""))) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.05),
          legend.justification = c(1, 0),
          legend.background = element_rect(fill = "white", color = NA),
          panel.grid.major.x = element_line(color = "grey92"),
          axis.text.y = element_text(size = 9))
}


# ============================================================================
# fig5c: cohort-grouped robustness dot-matrix
# ============================================================================
build_fig5c <- function() {
  robust <- load_figure_data("fig5_robust_hits.csv")
  if (nrow(robust) == 0) return(placeholder_panel("fig5_robust_hits.csv empty"))
  r <- robust %>% filter(!is.na(marker), !is.na(spec), !is.na(HR_markerxICI)) %>%
    mutate(support = -log10(pmax(p_markerxICI, 1e-300)),
           marker_key = paste0(marker, " (", cancer_type, ")"))

  marker_rank <- r %>% group_by(marker_key) %>%
    summarise(n_specs = n_distinct(spec),
              support = max(support, na.rm = TRUE),
              mean_HR = median(mean_HR, na.rm = TRUE),
              .groups = "drop") %>%
    arrange(desc(n_specs), desc(support)) %>%
    head(15)
  markers <- marker_rank$marker_key

  # Order specs grouped by cohort, balanced
  all_specs <- unique(r$spec)
  spec_cohorts_all <- vapply(all_specs, spec_cohort, character(1))
  cohorts_in_order <- intersect(c("cohort1", "cohort2"), unique(spec_cohorts_all))
  per_cohort <- max(1, floor(6 / max(length(cohorts_in_order), 1L)))
  specs <- unlist(lapply(cohorts_in_order, function(c) {
    head(all_specs[spec_cohorts_all == c], per_cohort)
  }))
  specs <- head(specs, 6)
  spec_cohorts <- vapply(specs, spec_cohort, character(1))

  sub <- r %>% filter(marker_key %in% markers, spec %in% specs) %>%
    mutate(marker_key = factor(marker_key, levels = rev(markers)),
           spec = factor(spec, levels = specs),
           cohort_lbl = factor(cohort_label(spec_cohorts[match(spec, specs)]),
                               levels = cohort_label(cohorts_in_order)),
           dir = ifelse(HR_markerxICI < 1, "Sig., ICI benefit", "Sig., ICI harm"),
           ci_excludes = is.finite(CI95_marker_ICI_low) & is.finite(CI95_marker_ICI_high) &
                         (CI95_marker_ICI_low > 1 | CI95_marker_ICI_high < 1))

  # Grid base: empty cells in grey ns
  grid <- expand.grid(marker_key = factor(rev(markers), levels = rev(markers)),
                      spec = factor(specs, levels = specs),
                      stringsAsFactors = FALSE) %>%
    mutate(cohort_lbl = factor(cohort_label(spec_cohorts[match(spec, specs)]),
                               levels = cohort_label(cohorts_in_order)))

  defn <- paste(COHORT_DEFS[cohorts_in_order], collapse = "\n")
  n_sig <- if ("n_significant_markers" %in% names(r) &&
               any(!is.na(r$n_significant_markers))) {
    as.integer(r$n_significant_markers[!is.na(r$n_significant_markers)][1])
  } else NA_integer_
  denom_txt <- if (!is.na(n_sig))
                  sprintf(" of %d significant in ≥1 spec", n_sig) else ""
  subtitle_txt <- sprintf(
    "top %d shown · %d robust (≥2 specs, consistent direction)%s",
    length(markers),
    length(unique(r$marker_key)),
    denom_txt
  )

  pretty_x <- setNames(vapply(specs, pretty_spec, character(1)), as.character(specs))

  ggplot() +
    geom_point(data = grid, aes(spec, marker_key), shape = 21,
               fill = "white", color = NS_GRAY, size = 2.4) +
    geom_point(data = sub, aes(spec, marker_key, color = dir,
                              size = abs(log(HR_markerxICI)),
                              alpha = ci_excludes)) +
    scale_color_manual(values = c("Sig., ICI benefit" = BENEFIT_COLOR,
                                   "Sig., ICI harm" = HARM_COLOR),
                       name = NULL) +
    scale_size_continuous(range = c(2, 7), name = "|log interaction HR|") +
    scale_alpha_manual(values = c(`TRUE` = 1, `FALSE` = 0.35),
                       labels = c(`TRUE` = "95% CI excludes 1", `FALSE` = "95% CI includes 1"),
                       name = NULL) +
    scale_x_discrete(labels = pretty_x) +
    facet_grid(. ~ cohort_lbl, scales = "free_x", space = "free_x") +
    labs(x = NULL, y = NULL,
         title = "Biomarker Specification Stability",
         subtitle = subtitle_txt,
         caption = paste(defn, "Specification stability is not external replication.", sep = "\n")) +
    theme_manuscript() +
    theme(plot.subtitle = element_text(size = 9, color = "#666666"),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 0.5,
                                      color = "#555555"),
          axis.text.x = element_text(size = 9),
          axis.text.y = element_text(face = "bold", size = 10),
          strip.text = element_text(face = "bold", size = 11),
          panel.grid.major = element_line(color = "#EAEAEA"),
          legend.position = "bottom")
}


# ============================================================================
# fig5d: 3-panel marker × ICI KM strip (interaction HR carried through)
# ============================================================================
build_fig5d <- function() {
  km_examples <- load_figure_data("fig5_km_examples.csv")
  km_top      <- load_figure_data("fig5_km_top_hit.csv")
  meta        <- load_figure_data("fig5_top_hit_meta.csv")

  strata_def <- list(
    list(label = "Marker+ / ICI+", color = "#1B4F72",
         pred = function(d) d$marker_value == 1 & d$PX_on_ICI == 1),
    list(label = "Marker- / ICI+", color = "#5DADE2",
         pred = function(d) d$marker_value == 0 & d$PX_on_ICI == 1),
    list(label = "Marker- / ICI-", color = "#F28E2B",
         pred = function(d) d$marker_value == 0 & d$PX_on_ICI == 0),
    list(label = "Marker+ / ICI-", color = "#C0392B",
         pred = function(d) d$marker_value == 1 & d$PX_on_ICI == 0)
  )
  strat_pal <- setNames(vapply(strata_def, function(x) x$color, character(1)),
                        vapply(strata_def, function(x) x$label, character(1)))

  hr_suffix <- function(d) {
    if (!"hr" %in% names(d) || is.na(d$hr[1])) return("")
    # Track 2 rows carry hr_label = "HR(marker×ICI)"; Track 1 fallback carries
    # "HR(marker | ICI)". Use the per-row label so the title is honest about
    # which estimate is being shown (interaction vs prognostic-within-ICI).
    lbl <- if ("hr_label" %in% names(d) && !is.na(d$hr_label[1]) && nzchar(d$hr_label[1]))
             d$hr_label[1] else "HR"
    sprintf("\n%s=%.2f", lbl, d$hr[1])
  }

  km_panel <- function(data, title) {
    if (nrow(data) == 0) return(placeholder_panel("no KM data"))
    data <- data %>%
      mutate(months = tt_death / 30.44, death = as.integer(death)) %>%
      filter(months > 0)
    rows <- bind_rows(lapply(strata_def, function(s) {
      sub <- data[s$pred(data), ]
      if (nrow(sub) < 5) return(NULL)
      sub$stratum <- s$label
      sub
    }))
    if (nrow(rows) == 0) return(placeholder_panel("insufficient KM strata"))
    n_by <- rows %>% group_by(stratum) %>% summarise(n = n(), .groups = "drop") %>%
      mutate(label = sprintf("%s (n=%s)", stratum, scales::comma(n)))
    rows <- rows %>% mutate(stratum = factor(stratum, levels = n_by$stratum))
    fit <- survfit2(Surv(months, death) ~ stratum, data = rows)
    td  <- ggsurvfit::tidy_survfit(fit) %>%
      mutate(stratum = sub("^[^=]+=", "", as.character(strata)),
             stratum = factor(stratum, levels = n_by$stratum,
                              labels = n_by$label))
    pal <- setNames(strat_pal[n_by$stratum], n_by$label)
    ci <- step_ci_df(td, "stratum")
    ggplot(td, aes(time, estimate, color = stratum)) +
      geom_rect(data = ci, aes(xmin = time, xmax = time_next,
                               ymin = conf.low, ymax = conf.high, fill = stratum),
                inherit.aes = FALSE, alpha = 0.12, color = NA) +
      geom_step(linewidth = 0.9) +
      geom_point(data = td %>% filter(n.censor > 0), shape = 3, size = 1.1) +
      scale_color_manual(values = pal, name = NULL) +
      scale_fill_manual(values = pal, guide = "none") +
      coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.02)) +
      labs(x = "Months", y = NULL, title = title) +
      theme_manuscript() +
      theme(plot.title = element_text(size = 11, face = "bold"),
            legend.position = c(0.98, 0.95),
            legend.justification = c(1, 1),
            legend.background = element_rect(fill = "white", color = NA),
            legend.text = element_text(size = 8))
  }

  panels <- list()
  if (nrow(km_examples) > 0) {
    ex_ids <- unique(km_examples$example_id)
    ex_ids <- head(ex_ids[!is.na(ex_ids)], 3)
    for (ex in ex_ids) {
      sub <- km_examples %>% filter(example_id == ex)
      ttl <- paste0(sub$title[1], "\n", sub$cancer[1], hr_suffix(sub))
      panels[[length(panels) + 1]] <- km_panel(sub, ttl)
    }
  } else if (nrow(km_top) > 0) {
    ttl <- if (nrow(meta) > 0) sprintf("%s\n%s", meta$marker[1], meta$cancer[1]) else "Top hit"
    panels[[1]] <- km_panel(km_top, ttl)
  }
  # Render however many examples actually have data -- the grid narrows rather
  # than padding out to three with empty panels.
  panels <- compact_panels(panels)
  if (is.null(panels)) return(placeholder_panel("no KM examples with data"))
  panels[[1]] <- panels[[1]] + labs(y = "Survival Probability")

  pgrid <- wrap_plots(panels, nrow = 1)
  spec_txt <- ""
  if (nrow(meta) > 0) {
    m0 <- meta[1, ]
    spec_txt <- sprintf("%s · %s PS · %s",
                        cohort_label(m0$cohort), pretty_model(m0$ps_model),
                        toupper(as.character(m0$weight_type)))
    pgrid <- pgrid +
      plot_annotation(title = sprintf("Exploratory marker-stratified survival   (%s)",
                                      spec_txt)) &
      theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5))
  }
  pgrid
}


# ============================================================================
# fig5e: forest of marker x ICI interaction estimates for the markers meeting
# the pre-registered stability criterion (>=2 specifications, direction-
# consistent), ranked by primary-spec interaction p-value.
#
# Brackets are genuine Wald 95% CIs from the primary spec (se_mx =
# sqrt(V[mx, mx]) on the interaction term, computed in
# run_IPTW_analysis._fit_track2_marker and propagated as
# CI95_markerxICI_low/high), not an inter-spec range.
#
# There is no literature-validation annotation: markers are selected by a
# statistical rule, so the claim this panel makes is about stability, not about
# agreement with prior literature.
# ============================================================================
DIRECTION_LABEL  <- c(benefit = "HR < 1 · greater ICI benefit",
                      harm    = "HR > 1 · reduced ICI benefit")
DIRECTION_COLORS <- c(`HR < 1 · greater ICI benefit` = BENEFIT_COLOR,
                      `HR > 1 · reduced ICI benefit` = HARM_COLOR)

build_fig5e <- function() {
  d <- load_figure_data("fig5_forest_headline.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig5_forest_headline.csv empty"))
  d <- d %>%
    mutate(dir_lbl = DIRECTION_LABEL[ifelse(HR < 1, "benefit", "harm")],
           dir_lbl = factor(dir_lbl, levels = unname(DIRECTION_LABEL)),
           # Row label = gene with cohort tag (cohort 2 = validation cohort).
           row_lbl = sprintf("%s · %s · %s", label, cancer,
                             ifelse(cohort == "cohort2", "C2", "C1"))) %>%
    arrange(desc(row_number()))                      # patchwork plots bottom-up
  row_order <- unique(d$row_lbl)
  d$row_lbl <- factor(d$row_lbl, levels = row_order)

  xmin <- min(c(d$CI95_low, d$HR), na.rm = TRUE) * 0.85
  xmax <- max(c(d$CI95_high, d$HR), na.rm = TRUE) * 1.15

  ggplot(d, aes(x = HR, y = row_lbl, color = dir_lbl)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "#777777") +
    geom_errorbarh(aes(xmin = CI95_low, xmax = CI95_high),
                   height = 0.25, linewidth = 0.55) +
    geom_point(size = 2.6, shape = 17) +
    scale_color_manual(values = DIRECTION_COLORS, name = NULL,
                       breaks = unname(DIRECTION_LABEL), drop = FALSE) +
    scale_x_log10(limits = c(xmin, xmax),
                  breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 10),
                  oob = scales::squish) +
    labs(x = "Marker × ICI interaction HR (log scale, 95% CI)",
         y = NULL,
         title = "Stability-Selected Markers · Interaction with ICI Exposure") +
    theme_manuscript() +
    theme(legend.position = "top",
          plot.title = element_text(size = 12, face = "bold"),
          panel.grid.major.y = element_line(color = "grey92"),
          axis.title.x = element_text(size = 10),
          axis.text.y = element_text(size = 10))
}


# Supplemental diagnostic: propensity overlap, weight tails, and effective N.
build_figS5a <- function() {
  d <- load_figure_data("fig5_weight_diagnostics.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig5_weight_diagnostics.csv empty"))
  d <- d %>% mutate(arm = factor(treatment, levels = c(0, 1),
                                 labels = c("Control", "ICI")))
  ess <- d %>% group_by(arm) %>%
    summarise(n = n(), ess = sum(weight)^2 / sum(weight^2), .groups = "drop")
  ess_txt <- paste(sprintf("%s: n=%s, ESS=%.0f", ess$arm, comma(ess$n), ess$ess), collapse = " · ")
  p_overlap <- ggplot(d, aes(propensity, color = arm, fill = arm)) +
    geom_density(alpha = 0.18, linewidth = 0.9) +
    scale_color_manual(values = c(Control = HARM_COLOR, ICI = TEAL), name = NULL) +
    scale_fill_manual(values = c(Control = HARM_COLOR, ICI = TEAL), guide = "none") +
    labs(x = "Held-out propensity score", y = "Density", title = "Propensity Overlap") +
    theme_manuscript() + theme(legend.position = "top")
  p_weight <- ggplot(d, aes(weight, fill = arm)) +
    geom_histogram(bins = 40, position = "identity", alpha = 0.45) +
    scale_fill_manual(values = c(Control = HARM_COLOR, ICI = TEAL), name = NULL) +
    labs(x = "Trimmed stabilized ATE weight", y = "Patients", title = "Weight Distribution") +
    theme_manuscript() + theme(legend.position = "top")
  (p_overlap | p_weight) + plot_annotation(subtitle = ess_txt)
}


# ============================================================================
# Compose Figure 5
# ============================================================================
p5a <- build_fig5a()
p5b <- build_fig5b()
p5c <- build_fig5c()
p5d <- build_fig5d()
p5e <- build_fig5e()
pS5a <- build_figS5a()

save_panel(p5a, "fig5a", group = "figure5", width = 9.6, height = 7.2)
save_panel(p5b, "fig5b", group = "figure5", width = 7.8, height = 7.2)
save_panel(p5c, "fig5c", group = "figure5", width = 10.8, height = 7.8)
save_panel(p5d, "fig5d", group = "figure5", width = 17.0, height = 6.2)
save_panel(p5e, "fig5e", group = "figure5", width = 13.0, height = 7.0)
save_panel(pS5a, "figS5a_weight_diagnostics", group = "figure5", width = 13.0, height = 5.8)
