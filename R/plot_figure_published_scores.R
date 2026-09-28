# Published within-cancer-type prognostic scores (MDCalc-style) versus the
# held-out text risk score for overall survival (figures.prep.published_scores).
# No models are refitted here; C-indices, Cox HRs and KM groups come from the
# Python prep tier. Each score appears twice: `calculated` from structured
# data and `regex` as documented in clinical notes; every row is labelled
# with its evaluated patients and deaths. Only the primary run (treatment
# anchor, 30-day lab window) is plotted; the sequencing-anchor and 90-day-
# window runs are sensitivity results left in the report tables only.
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
PS_SCORES <- c(mgps = "mGPS", rmh = "RMH", lipi = "LIPI", albi = "ALBI", meld = "MELD",
               capra_mod = "CAPRA", ipi = "IPI", ipi_noecog = "IPI (ECOG-free)",
               imdc = "IMDC", imdc_noecog = "IMDC (ECOG-free)",
               mskcc = "MSKCC", mskcc_noecog = "MSKCC (ECOG-free)")
PS_SOURCES <- c(calculated = "calculated", regex = "notes (regex)")
PS_COUNT_LEVELS <- c(n_eligible = "Eligible", n_complete = "Scored",
                     n_patients = "Evaluated", n_events = "Deaths")

ps_contrast_label <- function(model, reference) paste(model, "vs.", reference)

# Restrict to the primary anchor/window and the whole-population stratum (the
# only one the prep emits); tables from before the regex source existed have
# no `source` column and are all calculated.
ps_primary <- function(d) {
  if (!nrow(d)) return(d)
  if (!"source" %in% names(d)) d$source <- "calculated"
  d <- filter(d, anchor == PS_ANCHOR, lab_window_days == PS_LAB_WINDOW)
  if ("stratum" %in% names(d)) d <- filter(d, stratum == "all")
  d
}

# One row per (score, source) with patients evaluated, in catalog then source
# order; `row` is the y-axis label shared by every panel.
ps_rows <- function(cohort) {
  d <- ps_primary(cohort)
  if (!nrow(d)) return(tibble::tibble(score = character(), source = character(), row = character()))
  d %>%
    filter(score %in% names(PS_SCORES), source %in% names(PS_SOURCES)) %>%
    mutate(
      n_label = ifelse(is.na(n_patients), "n = NA",
                       sprintf("n = %s, %s deaths", scales::comma(n_patients),
                               scales::comma(n_events))),
      row = sprintf("%s, %s\n%s", PS_SCORES[score], PS_SOURCES[source], n_label),
      order = match(score, names(PS_SCORES)) * 10 + match(source, names(PS_SOURCES))
    ) %>%
    arrange(order) %>%
    select(score, source, row, n_patients, n_events, n_eligible, n_complete)
}

# Join the shared row labels; top-to-bottom in catalog order.
ps_attach_rows <- function(d, rows) {
  d <- inner_join(d, select(rows, score, source, row), by = c("score", "source"))
  mutate(d, row = factor(row, levels = rev(rows$row)))
}

build_published_n <- function(rows) {
  d <- rows %>% filter(!is.na(n_eligible))
  if (!nrow(d)) return(placeholder_panel("no published-score cohort counts"))
  long <- d %>%
    tidyr::pivot_longer(names(PS_COUNT_LEVELS), names_to = "count", values_to = "n") %>%
    filter(is.finite(n), n > 0) %>%
    mutate(count = factor(PS_COUNT_LEVELS[count], levels = unname(PS_COUNT_LEVELS)),
           row = factor(row, levels = rev(rows$row)))
  if (!nrow(long)) return(placeholder_panel("no published-score cohort counts"))
  span <- long %>% group_by(row) %>% summarise(lo = min(n), hi = max(n), .groups = "drop")
  ggplot(long, aes(x = n, y = row)) +
    geom_segment(data = span, aes(x = lo, xend = hi, y = row, yend = row), inherit.aes = FALSE,
                 color = "grey80", linewidth = 0.6) +
    geom_point(aes(shape = count, fill = count), size = 2.3, color = "grey15", stroke = 0.5) +
    scale_shape_manual(values = c(Eligible = 21, Scored = 22, Evaluated = 21, Deaths = 4), name = NULL,
                       drop = FALSE) +
    scale_fill_manual(values = c(Eligible = "white", Scored = "grey70", Evaluated = "grey15", Deaths = "grey15"),
                      name = NULL, drop = FALSE) +
    scale_x_log10(labels = scales::label_comma()) +
    labs(x = "Patients (log scale)", y = NULL, title = "Patients evaluated with each score") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_published_cindex <- function(cindex, rows) {
  d <- ps_primary(cindex) %>% filter(status == "ok", is.finite(cindex))
  if (!nrow(d)) return(placeholder_panel("no published-score C-index results"))
  d <- ps_attach_rows(d, rows) %>% mutate(model = factor(model, levels = PS_MODELS))
  if (!nrow(d)) return(placeholder_panel("no published-score C-index results"))
  ggplot(d, aes(x = cindex, y = row, color = model)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), position = position_dodge(width = 0.6),
                   linewidth = 0.6) +
    geom_point(position = position_dodge(width = 0.6), size = 2.2) +
    scale_color_manual(values = PS_MODEL_COLORS, labels = PS_MODEL_LABELS, name = NULL) +
    labs(x = "Overall survival C-index", y = NULL,
         title = "Published score, text, and both combined") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_published_delta <- function(delta, rows) {
  d <- ps_primary(delta) %>% filter(is.finite(delta_cindex))
  if (!nrow(d)) return(placeholder_panel("no published-score contrast results"))
  d <- ps_attach_rows(d, rows) %>% mutate(
    contrast = factor(ps_contrast_label(model, reference), levels = unname(PS_CONTRAST_LABELS))
  )
  if (!nrow(d)) return(placeholder_panel("no published-score contrast results"))
  ggplot(d, aes(x = delta_cindex, y = row, color = contrast)) +
    geom_vline(xintercept = 0, color = "grey45", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), position = position_dodge(width = 0.6),
                   linewidth = 0.6) +
    geom_point(position = position_dodge(width = 0.6), size = 2.2, shape = 23, fill = "white",
               stroke = 0.7) +
    scale_color_manual(values = c("grey20", "grey45", "grey70"), name = NULL) +
    labs(x = "C-index difference", y = NULL, title = "Paired differences") +
    theme_combined() +
    theme(legend.position = "bottom")
}

build_published_cox <- function(cox, rows) {
  d <- ps_primary(cox) %>% filter(status == "ok", term == "text_z", is.finite(hr))
  if (!nrow(d)) return(placeholder_panel("no published-score Cox results"))
  d <- ps_attach_rows(d, rows) %>%
    mutate(p_label = ifelse(is.na(lrt_p), "", sprintf("LRT p = %.2g", lrt_p)))
  if (!nrow(d)) return(placeholder_panel("no published-score Cox results"))
  ggplot(d, aes(x = hr, y = row)) +
    geom_vline(xintercept = 1, color = "grey45", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), linewidth = 0.7, color = "grey20") +
    geom_point(size = 2.4, shape = 21, fill = MODEL_COLORS[["text"]], color = "grey15") +
    geom_text(aes(x = Inf, label = p_label), hjust = 1.05, size = MANUSCRIPT_SMALL_TEXT_SIZE,
              color = "grey25") +
    scale_x_log10(expand = expansion(mult = c(0.05, 0.35))) +
    labs(x = "Text HR per SD (adjusted for published score)", y = NULL,
         title = "Added text signal, adjusted for the published score") +
    theme_combined()
}

# One facet per (score, source), KM by published risk group (evaluate_km in
# the prep tier already cut both the published groups and the size-matched
# text groups on the same patients; this panel shows the published grouping).
build_published_km <- function(km, rows) {
  d <- ps_primary(km) %>% filter(is.finite(time), !is.na(published_group))
  if (!nrow(d)) return(placeholder_panel("no published-score KM data"))
  d <- d %>%
    group_by(score, source) %>% filter(n_distinct(published_group) >= 2) %>% ungroup()
  if (!nrow(d)) return(placeholder_panel("no score has >=2 published risk groups"))
  panels <- rows %>% semi_join(d, by = c("score", "source"))
  td <- d %>%
    group_by(score, source) %>%
    group_modify(function(sub, key) tidy_km(build_survfit(sub, "time", "event_flag", "published_group"))) %>%
    ungroup() %>%
    inner_join(select(panels, score, source, row), by = c("score", "source")) %>%
    mutate(row = factor(row, levels = panels$row))
  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_step(linewidth = 0.5) +
    facet_wrap(~row, ncol = min(4, nrow(panels)), scales = "free_x") +
    labs(x = "Time from anchor", y = "Overall survival",
         title = "Kaplan-Meier by published risk group", color = "Risk group") +
    theme_manuscript() +
    theme(legend.position = "bottom", strip.background = element_blank())
}

published_scores_caption <- function(cindex, cohort) {
  d <- ps_primary(cindex)
  n_boot <- if (nrow(d)) max(d$n_boot, na.rm = TRUE) else NA
  rows <- ps_rows(cohort)
  lookback <- if (nrow(cohort) && "note_lookback_days" %in% names(cohort)) {
    suppressWarnings(max(cohort$note_lookback_days, na.rm = TRUE))
  } else NA
  lookback_text <- if (is.finite(lookback)) sprintf("%d", as.integer(lookback)) else "the configured"
  paste(
    "Published within-cancer-type prognostic scores versus the held-out text risk score",
    "(overall survival).",
    paste(
      "(a) Patients per score and source: eligible, with a score (complete structured inputs, or a",
      "documented score in the notes), and evaluated (score, text risk score and outcome all",
      "available), with deaths among those evaluated.",
      "(b) Overall survival C-index of each published score alone, the text risk score alone,",
      "and the two combined, with 95% bootstrap CIs.",
      "(c) Paired differences between panel-b models.",
      "(d) Hazard ratio per SD of the text risk score, adjusted for the published score, from a",
      "joint Cox model; annotated with the likelihood-ratio-test p-value for adding text.",
      "A separate panel shows Kaplan-Meier curves by published risk group, one per score and source.",
      "Every row is labelled with its evaluated patients and deaths."
    ),
    paste(
      "Each score has two sources, evaluated on separate cohorts drawn from the same eligible",
      "population. Calculated scores are computed from structured data (labs, registry, ICD codes)",
      "within the eligible, lab-observable, complete-case population. The calculated IPI, IMDC and",
      "MSKCC take performance status from the latest ECOG or Karnofsky score regex-extracted from",
      "progress notes over the same window as the notes scores (KPS converted to ECOG; ECOG >= 2",
      "scores the item, matching KPS < 80% for IMDC and MSKCC), so they are limited to patients",
      "with documented performance status. Their ECOG-free variants drop that item (points are a",
      "lower bound, original cutpoints applied), so they cover more patients; they have no",
      "notes source.",
      sprintf(paste(
        "Notes (regex) scores are the scores as documented by clinicians, regex-extracted from",
        "progress notes dated 0-%s days before the anchor (latest note used). These are the full",
        "published scores including performance status; IMDC and MSKCC are used as their",
        "documented risk group, ALBI as its grade."), lookback_text),
      "The text model already includes age, sex and cancer type."
    ),
    paste(
      "The published score's model is the unfitted clinical score (raw points, or the continuous",
      "ALBI/MELD value for calculated scores); the combined model cross-fits an unpenalized Cox",
      "model (Breslow ties) on the standardized published score and text score. Harrell's C-index",
      "is computed within blocks of patients sharing the text model's outer fold, on one shared set",
      "of comparable pairs across all three models, so differences are paired.",
      if (is.finite(n_boot)) {
        sprintf("Intervals are 95%% percentile intervals from %d patient bootstrap resamples with the fitted models held fixed.", n_boot)
      } else ""
    ),
    sprintf("Score rows shown: %d.", nrow(rows)),
    "Results for the sequencing-anchor and 90-day lab-window sensitivity runs are in the accompanying tables only.",
    sep = "\n\n"
  )
}

render_published_scores <- function() {
  stems <- c(n = "figPS_published_n", cindex = "figPS_published_cindex",
             delta = "figPS_published_delta", cox = "figPS_published_cox",
             km = "figPS_published_km", compiled = "figPS_published_scores")
  for (stem in stems) clear_within_cancer_report(stem, PS_GROUP)

  cindex <- read_within_cancer_data("pubscore_cindex.csv")
  delta <- read_within_cancer_data("pubscore_delta.csv")
  cox <- read_within_cancer_data("pubscore_cox.csv")
  km <- read_within_cancer_data("pubscore_km.csv")
  cohort <- read_within_cancer_data("pubscore_cohort.csv")
  if (!nrow(cindex) || !nrow(cohort)) {
    message("[published scores] SKIPPED: no results; run figures.prep.published_scores")
    return(invisible(NULL))
  }

  rows <- ps_rows(cohort)
  panels <- list(
    n = build_published_n(rows),
    cindex = build_published_cindex(cindex, rows),
    delta = build_published_delta(delta, rows),
    cox = build_published_cox(cox, rows),
    km = build_published_km(km, rows)
  )
  row_height <- function(d) 1.6 + 0.55 * max(if (nrow(d)) n_distinct(d$score, d$source) else 0, 1)
  ok_rows <- ps_primary(cindex) %>% filter(status == "ok")
  km_primary <- ps_primary(km)
  n_km <- if (nrow(km_primary)) max(nrow(distinct(km_primary, score, source)), 1) else 1
  sizes <- list(n = c(7.5, row_height(rows)), cindex = c(7.5, row_height(ok_rows)),
                delta = c(7.5, row_height(ok_rows)), cox = c(7.5, row_height(ok_rows)),
                km = c(3.4 * min(4, n_km), 3.4 * ceiling(n_km / 4)))
  for (name in names(panels)) {
    save_panel(panels[[name]], stems[[name]], PS_GROUP, width = sizes[[name]][1], height = sizes[[name]][2])
  }
  kept <- compact_panels(panels[c("n", "cindex", "delta", "cox")])
  if (!is.null(kept)) {
    heights <- vapply(names(kept), function(name) sizes[[name]][2], numeric(1))
    compiled <- wrap_plots(kept, ncol = 1, heights = heights) + plot_annotation(tag_levels = "a")
    save_panel(compiled, stems[["compiled"]], PS_GROUP, width = 8, height = sum(heights) + 0.5)
  }

  table <- ps_primary(cindex) %>%
    filter(status == "ok") %>%
    select(score, source, variant, model, cindex, ci_lower, ci_upper, n_patients, n_events, n_boot)
  cohort_cols <- intersect(c("score", "source", "n_eligible", "n_complete", "n_both_sources",
                             "frac_tied_published", "spearman_published_text",
                             "unblocked_published_cindex", "underpowered"), names(ps_primary(cohort)))
  table <- left_join(table, ps_primary(cohort) %>% select(all_of(cohort_cols)), by = c("score", "source"))
  save_within_cancer_report(table, published_scores_caption(cindex, cohort), stems[["compiled"]])
  invisible(table)
}

render_published_scores()
