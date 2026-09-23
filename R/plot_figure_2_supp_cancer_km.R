# Figure 2 supplement: Figure 2c/d (overall survival by stage and by text risk
# quartile) within each selected cancer type (figures.prep.within_cancer_km).
suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(survival); library(ggsurvfit)
})
source("R/figure_utils.R")
source("R/within_cancer_utils.R")

KM_STRATIFICATIONS <- list(
  stage = list(column = "stage_group", predictor = "stage", legend = "Stage",
               palette = setNames(ORDINAL4, c("I", "II", "III", "IV")),
               title = "Overall survival by stage within selected cancer types",
               stem = "figS2_within_cancer_km_stage"),
  risk = list(column = "risk_quartile", predictor = "text_risk", legend = "Text risk",
              palette = setNames(ORDINAL4, c("Q1", "Q2", "Q3", "Q4")),
              title = "Overall survival by text risk quartile within selected cancer types",
              stem = "figS2_within_cancer_km_risk")
)

# KM curves for one cancer type; a single-level stratum has no `strata` column.
cancer_km <- function(sub, column, levels) {
  sub <- sub %>% mutate(group = factor(.data[[column]], levels = levels)) %>% filter(!is.na(group))
  if (!nrow(sub)) return(tibble::tibble())
  present <- levels[levels %in% sub$group]
  td <- if (length(present) > 1) {
    tidy_km(survfit2(Surv(months, death) ~ group, data = sub))
  } else {
    tidy_km(survfit2(Surv(months, death) ~ 1, data = sub)) %>% mutate(stratum = present)
  }
  td %>% mutate(stratum = factor(stratum, levels = levels))
}

km_summary <- function(d, cindex, spec) {
  stats <- d %>%
    group_by(cancer_type) %>%
    group_modify(function(sub, key) tibble::tibble(
      n = nrow(sub), n_events = sum(sub$death),
      n_groups = dplyr::n_distinct(sub[[spec$column]]),
      logrank_p = logrank_p(sub, "months", "death", spec$column)
    )) %>% ungroup()
  perf <- cindex %>% filter(predictor == spec$predictor) %>%
    select(cancer_type, cindex, status, n_cohort = n)
  full_join(perf, stats, by = "cancer_type") %>%
    mutate(stratification = spec$legend,
           status = ifelse(is.na(status), "ok", status),
           order = match(cancer_type, names(SELECTED_CANCER_TYPES))) %>%
    arrange(order) %>%
    select(cancer_type, stratification, status, n = n_cohort, n_events, n_groups, cindex, logrank_p)
}

build_within_cancer_km <- function(d, summary, spec) {
  cancers <- names(SELECTED_CANCER_TYPES)
  shown <- summary$status == "ok"
  labels <- setNames(
    ifelse(shown, sprintf("%s (n=%s)", SELECTED_CANCER_TYPES[summary$cancer_type],
                          scales::comma(summary$n)),
           sprintf("%s (not shown)", SELECTED_CANCER_TYPES[summary$cancer_type])),
    summary$cancer_type)
  td <- d %>%
    group_by(cancer_type) %>%
    group_modify(function(sub, key) cancer_km(sub, spec$column, names(spec$palette))) %>%
    ungroup() %>%
    mutate(cancer_type = factor(cancer_type, levels = cancers))
  td_ci <- step_ci_df(td, c("cancer_type", "stratum"))
  ann <- summary %>% filter(shown) %>%
    mutate(cancer_type = factor(cancer_type, levels = cancers),
           label = sprintf("%s=%.3f\nlog-rank %s", metric_label(), cindex,
                           vapply(logrank_p, format_p_inline, character(1))))
  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.15, color = NA) +
    geom_step(linewidth = 0.5) +
    geom_text(data = ann, aes(x = 1, y = 0.03, label = label), inherit.aes = FALSE,
              hjust = 0, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#444444") +
    facet_wrap(~cancer_type, ncol = 3, drop = FALSE, labeller = as_labeller(labels)) +
    scale_color_manual(values = spec$palette, name = spec$legend, drop = FALSE) +
    scale_fill_manual(values = spec$palette, guide = "none", drop = FALSE) +
    scale_x_continuous(breaks = seq(0, 60, 12), expand = expansion(mult = SURVIVAL_X_EXPANSION)) +
    coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
    labs(x = "Months from first treatment", y = "Overall survival", title = spec$title,
         subtitle = "Patients with known stage (I-IV); n = patients per cancer type") +
    guides(color = guide_legend(nrow = 1)) +
    theme_manuscript() +
    theme(legend.position = "bottom", strip.background = element_blank())
}

within_cancer_km_caption <- function(summary, spec) {
  display <- if (spec$predictor == "stage") {
    "Each panel shows Kaplan-Meier overall survival by major stage (I-IV) with 95% confidence bands."
  } else {
    paste("Each panel shows Kaplan-Meier overall survival by text risk quartile with 95% confidence",
          "bands. Quartiles are the pan-cancer quartiles of Figure 2d, defined across the whole",
          "known-stage cohort; each panel plots only that cancer type's patients, so group sizes",
          "differ and a quartile with no patients in a cancer type is absent.")
  }
  shown <- summary$cancer_type[summary$status == "ok"]
  hidden <- setdiff(names(SELECTED_CANCER_TYPES), shown)
  paste(
    paste("Supplemental Figure 2. Figure 2c/d within selected cancer types:",
          paste0(paste(SELECTED_CANCER_TYPES, collapse = ", "), "."),
          "CUP denotes cancer of unknown primary."),
    display,
    paste("The cohort matches Figure 2c/d: patients with a known major stage and a held-out",
          "overall-survival text risk score from the full-cohort model, split by cancer type.",
          sprintf("Each panel reports the %s of %s for overall survival within the cancer type",
                  metric_label(), if (spec$predictor == "stage") "stage" else "the text risk score"),
          "and the log-rank p across the plotted groups. Cancer types need at least 20 patients",
          "and 5 deaths to be shown."),
    if (length(hidden)) {
      sprintf("Not shown (too few patients or deaths with known stage): %s.",
              paste(SELECTED_CANCER_TYPES[hidden], collapse = ", "))
    } else "All selected cancer types are shown.",
    sep = "\n\n"
  )
}

render_figure2_within_cancer_km <- function() {
  for (spec in KM_STRATIFICATIONS) clear_within_cancer_report(spec$stem, "figure2")
  d <- load_figure_data("fig2_km_stage_vs_risk_by_cancer.csv")
  cindex <- load_figure_data("fig2_stage_vs_risk_cindex_by_cancer.csv")
  if (!nrow(d) || !nrow(cindex)) {
    message("[figure2 within-cancer KM] SKIPPED: no data; run figures.prep.within_cancer_km")
    return(invisible(NULL))
  }
  d <- select_cancer_types(d, "figure2 within-cancer KM") %>%
    mutate(months = tt_death / 30.44, death = as.integer(death))
  cindex <- cindex %>% mutate(cancer_type = toupper(trimws(cancer_type)))
  for (spec in KM_STRATIFICATIONS) {
    summary <- km_summary(d, cindex, spec)
    save_panel(build_within_cancer_km(d, summary, spec), spec$stem, "figure2", width = 10, height = 10)
    save_within_cancer_report(summary, within_cancer_km_caption(summary, spec), spec$stem)
  }
  invisible(NULL)
}

render_figure2_within_cancer_km()
