# Render Figure 5 (ICI biomarker discovery) in ggplot2 + patchwork.
#
# A propensity ROC curves (covariates-only vs covariates+embeddings),
# B covariate-balance love plot (SMD before vs after IPTW),
# C cohort-grouped biomarker robustness dot-matrix + definitions caption,
# D 3-panel marker × ICI KM strip with carried-through interaction HR.

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
    mutate(absu = abs(smd_unweighted)) %>%
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
    geom_vline(xintercept = 0, color = "#333333", linewidth = 0.4) +
    geom_vline(xintercept = c(-0.1, 0.1), linetype = "dotted", color = "#999999") +
    scale_color_manual(values = c("Unweighted" = HARM_COLOR,
                                   "IPTW-weighted" = TEAL),
                       breaks = c("Unweighted", "IPTW-weighted"),
                       name = NULL) +
    labs(x = "Standardized mean difference", y = NULL,
         title = "Covariate Balance (SMD): before vs after IPTW") +
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
           dir = ifelse(HR_markerxICI < 1, "Sig., ICI benefit", "Sig., ICI harm"))

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
    geom_point(data = sub, aes(spec, marker_key, color = dir, size = pmin(support, 4) + 1),
               alpha = 0.95) +
    scale_color_manual(values = c("Sig., ICI benefit" = BENEFIT_COLOR,
                                   "Sig., ICI harm" = HARM_COLOR),
                       name = NULL) +
    scale_size_continuous(range = c(2, 7), guide = "none") +
    scale_x_discrete(labels = pretty_x) +
    facet_grid(. ~ cohort_lbl, scales = "free_x", space = "free_x") +
    labs(x = NULL, y = NULL,
         title = "Biomarker Association Robustness",
         subtitle = subtitle_txt,
         caption = defn) +
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
    ggplot(td, aes(time, estimate, color = stratum)) +
      geom_step(linewidth = 0.9) +
      scale_color_manual(values = pal, name = NULL) +
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
  while (length(panels) < 3) panels[[length(panels) + 1]] <- placeholder_panel("no additional example")
  panels[[1]] <- panels[[1]] + labs(y = "Survival Probability")

  pgrid <- wrap_plots(panels, nrow = 1)
  spec_txt <- ""
  if (nrow(meta) > 0) {
    m0 <- meta[1, ]
    spec_txt <- sprintf("%s · %s PS · %s",
                        cohort_label(m0$cohort), pretty_model(m0$ps_model),
                        toupper(as.character(m0$weight_type)))
    pgrid <- pgrid +
      plot_annotation(title = sprintf("Representative marker × ICI interactions   (%s)",
                                      spec_txt)) &
      theme(plot.title = element_text(size = 12, face = "bold", hjust = 0.5))
  }
  pgrid
}


# ============================================================================
# fig5e: synthesis forest — T1 (prognostic-in-ICI) vs T2 (predictive-of-ICI)
# for the headline gene set, with literature-validation strip on the right.
#
# T1 and T2 brackets are both genuine Wald 95% CIs from the primary spec
# (T2: se_mx = sqrt(V[mx, mx]) on the marker x ICI interaction term, computed
# in run_IPTW_analysis._fit_track2_marker and propagated as
# CI95_markerxICI_low/high). No longer an inter-spec range.
# ============================================================================
TRACK_LABEL  <- c(T1_prognostic_in_ICI = "T1 · Prognostic within ICI",
                  T2_predictive_of_ICI = "T2 · Predictive of ICI benefit")
TRACK_COLORS <- c(`T1 · Prognostic within ICI` = BENEFIT_COLOR,
                  `T2 · Predictive of ICI benefit` = HARM_COLOR)
TRACK_SHAPES <- c(`T1 · Prognostic within ICI` = 16,
                  `T2 · Predictive of ICI benefit` = 17)
# 7-tier validation ramp (Very Strong → No Evidence). Use a sequential blue so
# strong-evidence rows pop and absent-evidence rows are faint.
VAL_LEVELS <- c("Very Strong","Strong","Moderate","Partial","Weak","Indirect","No Evidence")
VAL_COLORS <- c("#0B3D91","#1F5FBF","#3A82DC","#79A8E7","#B0CBED","#D4E2F4","#EFEFEF")

build_fig5e <- function() {
  d <- load_figure_data("fig5_forest_headline.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig5_forest_headline.csv empty"))
  d <- d %>%
    mutate(track_lbl = TRACK_LABEL[as.character(track)],
           track_lbl = factor(track_lbl, levels = unname(TRACK_LABEL)),
           # Row label = gene with cohort tag (cohort 2 = validation).
           # Compose order matches HEADLINE_FOREST in prep_figure_5.py — display
           # MET first as the protagonist, then T1-replication genes, then T2-only.
           row_lbl = sprintf("%s · %s · %s", label, cancer,
                             ifelse(cohort == "cohort2", "C2", "C1"))) %>%
    arrange(desc(row_number()))                      # patchwork plots bottom-up
  row_order <- unique(d$row_lbl)
  d$row_lbl <- factor(d$row_lbl, levels = row_order)

  # Validation strip: one tile per row at fixed x position, colored by level
  val_df <- d %>% distinct(row_lbl, validation_level) %>%
    mutate(validation_level = ifelse(is.na(validation_level) | validation_level == "",
                                     "No Evidence", validation_level),
           validation_level = factor(validation_level, levels = VAL_LEVELS))

  # Dodge T1 / T2 within each row so they don't sit on top of each other
  dodge <- position_dodge(width = 0.55)
  xmin <- min(c(d$CI95_low, d$HR), na.rm = TRUE) * 0.85
  xmax <- max(c(d$CI95_high, d$HR), na.rm = TRUE) * 1.15

  forest <- ggplot(d, aes(x = HR, y = row_lbl,
                          color = track_lbl, shape = track_lbl)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "#777777") +
    geom_errorbarh(aes(xmin = CI95_low, xmax = CI95_high),
                   height = 0.25, linewidth = 0.55, position = dodge) +
    geom_point(size = 2.6, position = dodge) +
    scale_color_manual(values = TRACK_COLORS, name = NULL,
                       breaks = unname(TRACK_LABEL)) +
    scale_shape_manual(values = TRACK_SHAPES, name = NULL,
                       breaks = unname(TRACK_LABEL)) +
    scale_x_log10(limits = c(xmin, xmax),
                  breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 10),
                  oob = scales::squish) +
    labs(x = "Hazard ratio (log scale, 95% CI)\nT1: Cox HR · T2: interaction HR",
         y = NULL,
         title = "Synthesis · Prognostic vs Predictive Signal for Headline Markers") +
    theme_manuscript() +
    theme(legend.position = "top",
          plot.title = element_text(size = 12, face = "bold"),
          panel.grid.major.y = element_line(color = "grey92"),
          axis.title.x = element_text(size = 10),
          axis.text.y = element_text(size = 10))

  strip <- ggplot(val_df, aes(x = 1, y = row_lbl, fill = validation_level)) +
    geom_tile(color = "white", linewidth = 0.6) +
    geom_text(aes(label = validation_level),
              size = 3.0, color = ifelse(val_df$validation_level %in% c("Very Strong","Strong"),
                                          "white", "#222222")) +
    scale_fill_manual(values = VAL_COLORS, name = NULL, drop = FALSE) +
    scale_x_continuous(expand = c(0, 0)) +
    labs(x = NULL, y = NULL, title = "Literature\nvalidation") +
    theme_void() +
    theme(plot.title = element_text(size = 10, face = "bold", hjust = 0.5,
                                    margin = margin(b = 4)),
          axis.text.y = element_blank(),
          legend.position = "none",
          plot.margin = margin(0, 4, 0, 0))

  forest + strip + plot_layout(widths = c(5, 1))
}


# ============================================================================
# Compose Figure 5
# ============================================================================
p5a <- build_fig5a()
p5b <- build_fig5b()
p5c <- build_fig5c()
p5d <- build_fig5d()
p5e <- build_fig5e()

save_panel(p5a, "fig5a", group = "figure5", width = 8.0, height = 6.0)
save_panel(p5b, "fig5b", group = "figure5", width = 6.5, height = 6.0)
save_panel(p5c, "fig5c", group = "figure5", width = 9.0, height = 6.5)
save_panel(p5d, "fig5d", group = "figure5", width = 15.0, height = 5.0)
save_panel(p5e, "fig5e", group = "figure5", width = 11.0, height = 5.8)
