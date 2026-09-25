# Published within-cancer-type prognostic scores (MDCalc-style) versus the
# held-out text risk score for overall survival (figures.prep.published_scores).
# No models are refitted here; C-indices, Cox HRs and KM groups come from the
# Python prep tier. Only the primary run (treatment anchor, 30-day lab window)
# is plotted; the sequencing-anchor and 90-day-window runs are sensitivity
# results left in the report tables only.
suppressPackageStartupMessages({ library(ggplot2); library(patchwork); library(dplyr) })
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

PS_GROUP <- "published_scores"
PS_ANCHOR <- "treatment"
PS_LAB_WINDOW <- 30L
PS_MODELS <- c("published", "text", "published+text")
PS_MODEL_LABELS <- c(published = "Published score", text = "Text",
                     "published+text" = "Published + text")
PS_MODEL_COLORS <- c(published = MODEL_COLORS[["published"]], text = MODEL_COLORS[["text"]],
                     "published+text" = MODEL_COLORS[["published+text"]])
PS_CONTRAST_LABELS <- c(
  "published+text vs. published" = "published+text vs. published",
  "published+text vs. text" = "published+text vs. text",
  "text vs. published" = "text vs. published"
)

ps_contrast_label <- function(model, reference) paste(model, "vs.", reference)

# Restrict to the primary anchor/window and drop scores with no ok row anywhere.
ps_primary <- function(d) {
  if (!nrow(d)) return(d)
  filter(d, anchor == PS_ANCHOR, lab_window_days == PS_LAB_WINDOW)
}

# Score display order/labels: whatever scores actually have data, in catalog order.
ps_score_order <- function(...) {
  ids <- unique(unlist(lapply(list(...), function(d) if ("score" %in% names(d)) d$score else character())))
  catalog <- c("mgps", "rmh", "lipi", "albi", "meld", "capra_mod",
              "ipi_noecog", "imdc_noecog", "mskcc_noecog")
  intersect(catalog, ids)
}

build_published_cindex <- function(cindex) {
  d <- ps_primary(cindex) %>% filter(status == "ok", is.finite(cindex))
  if (!nrow(d)) return(placeholder_panel("no published-score C-index results"))
  scores <- ps_score_order(d)
  if (!length(scores)) return(placeholder_panel("no published-score C-index results"))
  d <- d %>% mutate(
    score = factor(score, levels = rev(scores)),
    model = factor(model, levels = PS_MODELS)
  )
  ggplot(d, aes(x = cindex, y = score, color = model)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), position = position_dodge(width = 0.6),
                   linewidth = 0.6) +
    geom_point(position = position_dodge(width = 0.6), size = 2.2) +
    scale_color_manual(values = PS_MODEL_COLORS, labels = PS_MODEL_LABELS, name = NULL) +
    scale_y_discrete(labels = rev(scores)) +
    labs(x = "Overall survival C-index", y = NULL,
         title = "Published score, text, and both combined") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_published_delta <- function(delta) {
  d <- ps_primary(delta) %>% filter(is.finite(delta_cindex))
  if (!nrow(d)) return(placeholder_panel("no published-score contrast results"))
  scores <- ps_score_order(d)
  if (!length(scores)) return(placeholder_panel("no published-score contrast results"))
  d <- d %>% mutate(
    score = factor(score, levels = rev(scores)),
    contrast = factor(ps_contrast_label(model, reference), levels = unname(PS_CONTRAST_LABELS))
  )
  ggplot(d, aes(x = delta_cindex, y = score, color = contrast)) +
    geom_vline(xintercept = 0, color = "grey45", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), position = position_dodge(width = 0.6),
                   linewidth = 0.6) +
    geom_point(position = position_dodge(width = 0.6), size = 2.2, shape = 23, fill = "white",
               stroke = 0.7) +
    scale_color_manual(values = c("grey20", "grey45", "grey70"), name = NULL) +
    scale_y_discrete(labels = rev(scores)) +
    labs(x = "C-index difference", y = NULL, title = "Paired differences") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_published_cox <- function(cox) {
  d <- ps_primary(cox) %>% filter(status == "ok", term == "text_z", is.finite(hr))
  if (!nrow(d)) return(placeholder_panel("no published-score Cox results"))
  scores <- ps_score_order(d)
  if (!length(scores)) return(placeholder_panel("no published-score Cox results"))
  d <- d %>% mutate(
    score = factor(score, levels = rev(scores)),
    p_label = ifelse(is.na(lrt_p), "", sprintf("LRT p = %.2g", lrt_p))
  )
  ggplot(d, aes(x = hr, y = score)) +
    geom_vline(xintercept = 1, color = "grey45", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), linewidth = 0.7, color = "grey20") +
    geom_point(size = 2.4, shape = 21, fill = MODEL_COLORS[["text"]], color = "grey15") +
    geom_text(aes(x = Inf, label = p_label), hjust = 1.05, size = MANUSCRIPT_SMALL_TEXT_SIZE,
              color = "grey25") +
    scale_x_log10(expand = expansion(mult = c(0.05, 0.35))) +
    scale_y_discrete(labels = rev(scores)) +
    labs(x = "Text HR per SD (adjusted for published score)", y = NULL,
         title = "Added text signal, adjusted for the published score") +
    theme_combined()
}

# One facet per score, KM by published risk group (evaluate_km in the prep
# tier already cut both the published groups and the size-matched text groups
# on the same patients; this panel shows the published grouping).
build_published_km <- function(km) {
  d <- ps_primary(km) %>% filter(is.finite(time))
  if (!nrow(d)) return(placeholder_panel("no published-score KM data"))
  scores <- ps_score_order(d)
  if (!length(scores)) return(placeholder_panel("no published-score KM data"))
  d <- d %>%
    filter(!is.na(published_group)) %>%
    group_by(score) %>% filter(n_distinct(published_group) >= 2) %>% ungroup()
  if (!nrow(d)) return(placeholder_panel("no score has >=2 published risk groups"))
  scores <- ps_score_order(d)
  td <- d %>%
    group_by(score) %>%
    group_modify(function(sub, key) tidy_km(build_survfit(sub, "time", "event_flag", "published_group"))) %>%
    ungroup() %>%
    mutate(score = factor(score, levels = scores))
  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_step(linewidth = 0.5) +
    facet_wrap(~score, ncol = min(3, length(scores)), scales = "free_x") +
    labs(x = "Time from anchor", y = "Overall survival",
         title = "Kaplan-Meier by published risk group", color = "Risk group") +
    theme_manuscript() +
    theme(legend.position = "bottom", strip.background = element_blank())
}

published_scores_caption <- function(cindex, cohort) {
  d <- ps_primary(cindex)
  n_boot <- if (nrow(d)) max(d$n_boot, na.rm = TRUE) else NA
  scores <- ps_score_order(d)
  paste(
    "Published within-cancer-type prognostic scores versus the held-out text risk score",
    "(overall survival).",
    paste(
      "(a) Overall survival C-index of each published score alone, the text risk score alone,",
      "and the two combined, within that score's eligible, lab-observable, complete-case",
      "population, with 95% bootstrap CIs.",
      "(b) Paired differences between panel-a models.",
      "(c) Hazard ratio per SD of the text risk score, adjusted for the published score, from a",
      "joint Cox model; annotated with the likelihood-ratio-test p-value for adding text.",
      "(d) Kaplan-Meier curves by published risk group, one panel per score."
    ),
    paste(
      "Scores shown:", if (length(scores)) paste(scores, collapse = ", ") else "none evaluable.",
      "IPI, IMDC and MSKCC are ECOG-free modified variants (points are a lower bound, original",
      "cutpoints applied), since no ECOG/KPS/performance-status source exists in this cohort.",
      "The text model already includes age, sex and cancer type."
    ),
    paste(
      "Each score's population is that score's own eligible, lab-observable, complete-case",
      "cohort (see the accompanying cohort table), not the manuscript's full cohort. The published",
      "score's model is the unfitted clinical score (raw points, or the continuous ALBI/MELD",
      "value); the combined model cross-fits an unpenalized Cox model (Breslow ties) on the",
      "standardized published score and text score. Harrell's C-index is computed within blocks",
      "of patients sharing the text model's outer fold, on one shared set of comparable pairs",
      "across all three models, so differences are paired.",
      if (is.finite(n_boot)) {
        sprintf("Intervals are 95%% percentile intervals from %d patient bootstrap resamples with the fitted models held fixed.", n_boot)
      } else ""
    ),
    "Results for the sequencing-anchor and 90-day lab-window sensitivity runs are in the accompanying tables only.",
    sep = "\n\n"
  )
}

render_published_scores <- function() {
  stems <- c(cindex = "figPS_published_cindex", delta = "figPS_published_delta",
             cox = "figPS_published_cox", km = "figPS_published_km",
             compiled = "figPS_published_scores")
  for (stem in stems) clear_within_cancer_report(stem, PS_GROUP)

  cindex <- read_within_cancer_data("pubscore_cindex.csv")
  delta <- read_within_cancer_data("pubscore_delta.csv")
  cox <- read_within_cancer_data("pubscore_cox.csv")
  km <- read_within_cancer_data("pubscore_km.csv")
  cohort <- read_within_cancer_data("pubscore_cohort.csv")
  if (!nrow(cindex)) {
    message("[published scores] SKIPPED: no results; run figures.prep.published_scores")
    return(invisible(NULL))
  }

  panels <- list(
    cindex = build_published_cindex(cindex),
    delta = build_published_delta(delta),
    cox = build_published_cox(cox),
    km = build_published_km(km)
  )
  n_scores <- length(ps_score_order(ps_primary(cindex)))
  sizes <- list(cindex = c(7, 1.6 + 0.5 * n_scores), delta = c(7, 1.6 + 0.5 * n_scores),
                cox = c(7, 1.6 + 0.5 * n_scores), km = c(4 * min(3, max(n_scores, 1)), 3.6 * ceiling(n_scores / 3)))
  for (name in names(panels)) {
    save_panel(panels[[name]], stems[[name]], PS_GROUP, width = sizes[[name]][1], height = sizes[[name]][2])
  }
  kept <- compact_panels(panels[c("cindex", "delta", "cox")])
  if (!is.null(kept)) {
    compiled <- wrap_plots(kept, ncol = 1) + plot_annotation(tag_levels = "a")
    save_panel(compiled, stems[["compiled"]], PS_GROUP, width = 8, height = 3 * length(kept) + 1)
  }

  table <- ps_primary(cindex) %>%
    filter(status == "ok") %>%
    select(score, variant, model, cindex, ci_lower, ci_upper, n_patients, n_events, n_boot)
  if (nrow(cohort)) {
    table <- left_join(table, ps_primary(cohort) %>%
                         select(score, n_eligible, n_complete, frac_tied_published,
                                spearman_published_text, unblocked_published_cindex, underpowered),
                       by = "score")
  }
  save_within_cancer_report(table, published_scores_caption(cindex, cohort), stems[["compiled"]])
  invisible(table)
}

render_published_scores()
