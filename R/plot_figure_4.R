# Render Figure 4 (mortality-risk dynamics) + Figure S1 (silhouette appendix).
#
# A per-patient mortality-risk trajectory heatmap (raster) grouped by risk-slope group,
# B conditional KM survival from the slope-window landmark (left-truncated entry),
# D mean trajectory per slope group vs. a cohort-average reference band,
# E stage-matched slope-group composition (dynamics vs. baseline stage),
# C disease-severity small multiples (mean lines of therapy, pre-index met burden,
#   conditional RMST, mean maximum stage).
# S1: silhouette vs k (slope-group-count justification).

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr); library(viridisLite)
  library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")

N_SLOPE_GROUPS <- 3
# prep_figure_4 relabels slope groups 0..N-1 by ASCENDING mean OLS slope of
# months 0..L model risk (group 0 = most falling risk, group N-1 = most rising).
# Names therefore describe risk DYNAMICS (the quantity the groups are actually
# ordered on), not a risk level. Length must equal N_SLOPE_GROUPS.
GROUP_NAMES <- c("Falling Risk", "Stable Risk", "Rising Risk")
stopifnot(length(GROUP_NAMES) == N_SLOPE_GROUPS)

# Diverging palette keyed by group id 0..2: falling=benefit(blue), stable=grey,
# rising=harm(red). Use GROUP_COLORS[group_id + 1] anywhere a group needs a color.
GROUP_COLORS <- c(BENEFIT_COLOR, NS_GRAY, HARM_COLOR)

cluster_label <- function(k, n = NA_integer_) {
  # Coerce via character so factors return their underlying integer (not level codes).
  k_int <- suppressWarnings(as.integer(as.character(k)))
  nm <- GROUP_NAMES[pmin(k_int + 1L, length(GROUP_NAMES))]
  if (any(!is.na(n))) sprintf("%s (n=%s)", nm, scales::comma(n)) else nm
}

month_to_num <- function(x) {
  as.numeric(stringr::str_extract(as.character(x), "\\d+"))
}

logrank_p <- function(df, time_col, event_col, group_col, start_col = NULL) {
  if (nrow(df) == 0) return(NA_real_)
  # survdiff() does NOT support left-truncated Surv(start, stop, event); fall back
  # to the coxph score test, which is asymptotically equivalent to log-rank.
  if (!is.null(start_col)) {
    f <- as.formula(sprintf("Surv(%s, %s, %s) ~ %s",
                            start_col, time_col, event_col, group_col))
    cx <- tryCatch(survival::coxph(f, data = df), error = function(e) NULL)
    if (is.null(cx)) return(NA_real_)
    return(unname(summary(cx)$sctest["pvalue"]))
  }
  f <- as.formula(sprintf("Surv(%s, %s) ~ %s", time_col, event_col, group_col))
  sd <- tryCatch(survival::survdiff(f, data = df), error = function(e) NULL)
  if (is.null(sd)) return(NA_real_)
  if (!is.null(sd$pvalue)) return(sd$pvalue)
  stats::pchisq(sd$chisq, df = length(sd$n) - 1, lower.tail = FALSE)
}


# ============================================================================
# fig4a: per-patient mortality-risk trajectory heatmap
# ============================================================================
build_fig4a <- function() {
  d <- load_figure_data("fig4_trajectories_heatmap.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig4_trajectories_heatmap.csv empty"))
  month_cols <- setdiff(names(d), c("DFCI_MRN", "cluster"))
  d <- d %>% arrange(cluster) %>% mutate(row_idx = row_number())
  long <- d %>%
    select(row_idx, cluster, all_of(month_cols)) %>%
    pivot_longer(all_of(month_cols), names_to = "month_col", values_to = "risk") %>%
    mutate(month = month_to_num(month_col)) %>%
    filter(!is.na(month))

  # Boundaries between cluster blocks
  bounds <- d %>% count(cluster) %>% arrange(cluster) %>%
    mutate(top = cumsum(n), mid = top - n / 2 + 0.5)
  full_n <- load_figure_data("fig4_cluster_severity.csv")
  if (nrow(full_n) > 0 && all(c("cluster", "n_patients") %in% names(full_n))) {
    bounds <- bounds %>% left_join(full_n %>% select(cluster, full_n = n_patients), by = "cluster")
  } else bounds$full_n <- NA_integer_

  yticks <- bounds$mid
  ylabs <- sprintf("%s\nshown %s / full %s", cluster_label(bounds$cluster),
                   comma(bounds$n), ifelse(is.na(bounds$full_n), "NA", comma(bounds$full_n)))

  # The prep z-scores each month column across the cohort (StandardScaler), so
  # `risk` is a signed Z-score in roughly [-3, 3]. A diverging palette centered
  # at 0 makes "above-cohort-average" vs "below-cohort-average" pop within each
  # month — under a sequential scale on raw values the cluster structure was
  # washed out by the cohort-wide mean trend toward event time.
  Z_LIMIT <- 2.5
  ggplot(long, aes(month, row_idx, fill = risk)) +
    geom_raster() +
    scale_fill_distiller(
      palette  = "RdBu", direction = -1,
      limits   = c(-Z_LIMIT, Z_LIMIT), oob = scales::squish,
      breaks   = c(-Z_LIMIT, 0, Z_LIMIT),
      labels   = c(sprintf("≤ -%.1f", Z_LIMIT), "0", sprintf("≥ +%.1f", Z_LIMIT)),
      name     = "Mortality risk\n(z vs cohort,\nper month)",
      na.value = "grey90"
    ) +
    scale_x_continuous(expand = c(0, 0), breaks = pretty(long$month, n = 6)) +
    scale_y_reverse(expand = c(0, 0), breaks = yticks, labels = ylabs) +
    geom_hline(data = bounds[-nrow(bounds), ],
               aes(yintercept = top + 0.5),
               color = "white", linewidth = 0.6) +
    labs(x = "Months post-treatment", y = NULL,
         title = "Mortality-Risk Trajectories by Dynamics Group",
         subtitle = stringr::str_wrap(
           "Display-stratified random sample (up to 500 patients/group); group labels report full analytic-cohort N",
           width = 78)) +
    theme_manuscript() +
    theme(panel.grid = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 9))
}


# ============================================================================
# fig4b: conditional KM from the slope-window landmark (left-truncated entry)
# ============================================================================
build_fig4b <- function() {
  km <- load_figure_data("fig4_km_data.csv")
  if (nrow(km) == 0) return(placeholder_panel("fig4_km_data.csv empty"))
  landmark_values <- if ("landmark_month" %in% names(km)) {
    unique(km$landmark_month[!is.na(km$landmark_month)])
  } else numeric(0)
  LANDMARK <- if (length(landmark_values) == 1) as.numeric(landmark_values) else 12
  km <- km %>%
    mutate(months     = tt_death / 30.44,
           death      = as.integer(death),
           cluster_id = suppressWarnings(as.integer(as.character(cluster)))) %>%
    filter(months > LANDMARK, !is.na(cluster_id))
  if (nrow(km) == 0) return(placeholder_panel(
    sprintf("no patients survive to month %s", LANDMARK)))
  km$entry <- LANDMARK

  # Stable cluster-id → (label, color) map; we stratify on the SHORT id to avoid
  # any quirks of strata-name handling on long labels with parens/equals.
  cluster_ids  <- sort(unique(km$cluster_id))
  n_by_id      <- as.integer(table(km$cluster_id)[as.character(cluster_ids)])
  labels_by_id <- setNames(cluster_label(cluster_ids, n_by_id),
                           as.character(cluster_ids))
  colors_by_id <- setNames(GROUP_COLORS[cluster_ids + 1L],
                           as.character(cluster_ids))

  km <- km %>% mutate(strat = as.character(cluster_id))
  fit <- survfit2(Surv(entry, months, death) ~ strat, data = km)
  td <- ggsurvfit::tidy_survfit(fit) %>%
    mutate(strat_id = sub("^[^=]+=", "", as.character(strata)),
           label    = factor(labels_by_id[strat_id],
                             levels = unname(labels_by_id))) %>%
    # survfit2 emits a synthetic time=0, estimate=1 row per stratum (curve
    # start) that predates the landmark left-truncation point; drawing it
    # produces an unstratified flat segment before entry once geom_step
    # connects it to the first real event. Drop anything before entry.
    filter(time >= LANDMARK)

  pal <- setNames(unname(colors_by_id), unname(labels_by_id))
  lp  <- logrank_p(km, "months", "death", "strat", start_col = "entry")
  ci  <- step_ci_df(td, "label")
  ref_id <- if ("1" %in% as.character(cluster_ids)) "1" else as.character(cluster_ids[1])
  km$strat <- relevel(factor(km$strat), ref = ref_id)
  known_stage <- km %>% filter(stage %in% c("I", "II", "III", "IV")) %>%
    mutate(stage = factor(stage, levels = c("I", "II", "III", "IV")))
  stage_adjusted <- nrow(known_stage) >= 20 && n_distinct(known_stage$stage) >= 2
  cx_data <- if (stage_adjusted) known_stage else km
  cx_formula <- if (stage_adjusted) Surv(entry, months, death) ~ strat + stage
                else Surv(entry, months, death) ~ strat
  cx <- tryCatch(coxph(cx_formula, data = cx_data), error = function(e) NULL)
  hr_text <- "HR unavailable"
  if (!is.null(cx)) {
    cs <- summary(cx)$conf.int
    cs <- cs[grepl("^strat", rownames(cs)), , drop = FALSE]
    hr_text <- paste(sprintf("%s vs %s: HR %.2f (95%% CI %.2f–%.2f)",
                             labels_by_id[sub("^strat", "", rownames(cs))],
                             labels_by_id[ref_id], cs[, "exp(coef)"],
                             cs[, "lower .95"], cs[, "upper .95"]), collapse = "\n")
  }

  main <- ggplot(td, aes(time, estimate, color = label)) +
    { if (nrow(ci) > 0) geom_rect(data = ci,
                                  aes(xmin = time, xmax = time_next,
                                      ymin = conf.low, ymax = conf.high,
                                      fill = label),
                                  color = NA, alpha = 0.15, inherit.aes = FALSE) } +
    geom_step(linewidth = 0.9) +
    geom_point(data = thin_censor_rows(td, "label", 60), shape = 3, size = 1.0) +
    scale_color_manual(values = pal, name = NULL, drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none", drop = FALSE) +
    coord_cartesian(xlim = c(LANDMARK, 120)) +
    labs(x = "Months from first treatment",
         y = sprintf(paste0("Overall Survival Probability\n",
                           "(conditional on survival to month %s)"), LANDMARK),
         title = "KM Overall Survival by Risk-Dynamics Group",
         subtitle = sprintf("Landmark-eligible cohort: N=%s; follow-up conditional on survival to month %s",
                            comma(nrow(km)), LANDMARK)) +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.18), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))

  risk_times <- seq(LANDMARK, 120, 12)
  s <- summary(fit, times = risk_times, extend = TRUE)
  rt <- tibble(time = s$time, n.risk = s$n.risk,
               strat_id = sub("^[^=]+=", "", s$strata)) %>%
    mutate(row = factor(labels_by_id[strat_id], levels = rev(unname(labels_by_id))))
  risk_table <- ggplot(rt, aes(time, row, label = comma(n.risk), color = row)) +
    geom_text(size = 3) + scale_color_manual(values = pal, guide = "none") +
    scale_x_continuous(limits = c(LANDMARK, 120), breaks = risk_times) +
    labs(x = NULL, y = "Number at risk") + theme_void(base_size = 9) +
    theme(axis.text.y = element_text(), axis.title.y = element_text(angle = 90))
  stats_text <- sprintf("Landmark score test: %s. %s%s",
                        format_p_inline(lp),
                        ifelse(stage_adjusted, "Stage-adjusted ", ""),
                        gsub("\n", "; ", hr_text))
  stats_panel <- ggplot() +
    annotate("text", x = 0, y = 1, label = stringr::str_wrap(stats_text, width = 105),
             hjust = 0, vjust = 1, size = MANUSCRIPT_SMALL_TEXT_SIZE,
             fontface = "italic", color = "#444444") +
    xlim(0, 1) + ylim(0, 1) + theme_void()
  main / stats_panel / risk_table + plot_layout(heights = c(4.0, 0.55, 1.1))
}


# ============================================================================
# fig4d: mean trajectory per dynamics group vs. a cohort-average reference band
# ============================================================================
build_fig4d <- function() {
  d <- load_figure_data("fig4_group_trajectories.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig4_group_trajectories.csv empty"))

  # `group` mixes integers and the literal "cohort" pseudo-group, so readr loads
  # it as character; split the cohort-wide reference band from the slope groups.
  band <- d %>% filter(group == "cohort")
  grp  <- d %>% filter(group != "cohort") %>% mutate(group_id = as.integer(group))
  if (nrow(grp) == 0) return(placeholder_panel("no slope groups in fig4_group_trajectories.csv"))

  pal <- setNames(GROUP_COLORS[seq_len(N_SLOPE_GROUPS)], as.character(seq_len(N_SLOPE_GROUPS) - 1L))
  grp <- grp %>% mutate(group_lab = factor(cluster_label(group_id), levels = GROUP_NAMES))
  lab_by_id <- setNames(GROUP_NAMES, as.character(seq_len(N_SLOPE_GROUPS) - 1L))

  ggplot() +
    # Cohort-average band drawn first/underneath as a neutral grey reference.
    geom_ribbon(data = band, aes(month, ymin = q25, ymax = q75),
                fill = "grey50", alpha = 0.25, inherit.aes = FALSE) +
    geom_line(data = band, aes(month, mean_risk),
              color = "grey40", linetype = "dashed", linewidth = 0.8) +
    geom_ribbon(data = grp, aes(month, ymin = q25, ymax = q75, fill = group_lab),
                alpha = 0.15) +
    geom_line(data = grp, aes(month, mean_risk, color = group_lab), linewidth = 0.9) +
    scale_color_manual(values = setNames(unname(pal), lab_by_id), name = NULL, drop = FALSE) +
    scale_fill_manual(values = setNames(unname(pal), lab_by_id), guide = "none", drop = FALSE) +
    labs(x = "Months post-treatment", y = "Cox linear predictor (log relative hazard)",
         title = "Mean Risk Trajectory by Dynamics Group",
         subtitle = "Descriptive trajectories; ribbons are within-group IQRs") +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# fig4e: stage-matched dynamics-group composition (dynamics vs. baseline stage)
# ============================================================================
build_fig4e <- function() {
  d <- load_figure_data("fig4_slope_by_stage.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig4_slope_by_stage.csv empty"))

  d <- d %>%
    mutate(stage    = factor(stage, levels = c("I", "II", "III", "IV")),
           group_lab = factor(cluster_label(cluster), levels = GROUP_NAMES)) %>%
    filter(!is.na(stage))
  if (nrow(d) == 0) return(placeholder_panel("no recognized stages in fig4_slope_by_stage.csv"))

  pal <- setNames(GROUP_COLORS[seq_len(N_SLOPE_GROUPS)], GROUP_NAMES)
  tab <- xtabs(n_patients ~ stage + group_lab, d)
  chi <- suppressWarnings(chisq.test(tab, correct = FALSE))
  total_n <- sum(tab)
  cramer_v <- sqrt(unname(chi$statistic) /
                   (total_n * min(nrow(tab) - 1, ncol(tab) - 1)))

  ggplot(d, aes(stage, n_patients, fill = group_lab)) +
    geom_col(position = "fill", width = 0.7, color = "white") +
    scale_fill_manual(values = pal, name = NULL, drop = FALSE) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = "Stage", y = "Proportion of stage",
         title = "Risk-Dynamics Composition by Stage",
         subtitle = sprintf("N=%s; Cramér's V=%.3f (descriptive association)",
                            comma(total_n), cramer_v)) +
    theme_manuscript()
}


# ============================================================================
# fig4c: disease-severity small multiples
# ============================================================================
# The four displayed metrics come from fig4_cluster_severity.csv and all describe
# disease severity rather than the risk dynamics themselves: mean lines of
# therapy, mean pre-index met burden (the N_MET_SITES covariate, not the
# post-index mean_met_sites also present in that file), conditional RMST from the
# landmark, and mean maximum stage on a I=1..IV=4 ordinal scale. Each is NULL-
# guarded by by_id(), so a metric whose source data was unavailable at prep time
# drops out of the panel instead of rendering an empty facet.
build_fig4c <- function() {
  severity <- load_figure_data("fig4_cluster_severity.csv")
  if (nrow(severity) == 0) return(placeholder_panel("fig4_cluster_severity.csv empty"))
  clusters <- sort(unique(severity$cluster))
  if (length(clusters) == 0) return(placeholder_panel("severity CSV has no clusters"))

  by_id <- function(col) {
    if (!col %in% names(severity)) return(NULL)
    v <- severity[[col]]
    if (all(is.na(v))) return(NULL)
    setNames(v, as.character(severity$cluster))
  }

  characteristics <- list(
    list(title = "Mean lines of therapy", units = "Lines", vals = by_id("mean_n_lines"),
         low = by_id("mean_n_lines_low"), high = by_id("mean_n_lines_high"), is_pct = FALSE),
    list(title = "Mean met burden (pre-index)", units = "Met sites", vals = by_id("mean_met_burden"),
         low = by_id("mean_met_burden_low"), high = by_id("mean_met_burden_high"), is_pct = FALSE),
    list(title = "Conditional RMST: month 12–120", units = "Months after landmark", vals = by_id("rmst_months"),
         low = by_id("rmst_months_low"), high = by_id("rmst_months_high"), is_pct = FALSE),
    list(title = "Mean maximum stage", units = "Stage (1=I … 4=IV)", vals = by_id("mean_max_stage"),
         low = by_id("mean_max_stage_low"), high = by_id("mean_max_stage_high"), is_pct = FALSE)
  )

  panel_for <- function(spec) {
    if (is.null(spec$vals)) return(placeholder_panel(paste("no data:", spec$title)))
    df <- tibble::tibble(cluster = clusters,
                         value = unname(spec$vals[as.character(clusters)]),
                         low = if (is.null(spec$low)) NA_real_ else unname(spec$low[as.character(clusters)]),
                         high = if (is.null(spec$high)) NA_real_ else unname(spec$high[as.character(clusters)]))
    p <- ggplot(df, aes(factor(cluster), value, fill = factor(cluster))) +
      geom_col(width = 0.65, color = "white") +
      geom_errorbar(data = filter(df, is.finite(low), is.finite(high)),
                    aes(ymin = low, ymax = high), width = 0.15, linewidth = 0.6) +
      scale_fill_manual(values = setNames(GROUP_COLORS[clusters + 1],
                                          as.character(clusters)),
                        guide = "none") +
      scale_x_discrete(labels = setNames(GROUP_NAMES[clusters + 1],
                                         as.character(clusters))) +
      labs(x = NULL, y = spec$units, title = spec$title) +
      theme_manuscript() +
      theme(panel.grid.major.y = element_line(color = "grey90"))
    # is_pct panels are bounded percentages; the other metrics (incl. mean
    # slope, which can be negative for Falling) are left with a free y-range.
    if (isTRUE(spec$is_pct)) p <- p + coord_cartesian(ylim = c(0, 100))
    p
  }
  # Only characteristics with data get a cell; if none do, skip the figure.
  ps <- compact_panels(lapply(characteristics, panel_for))
  if (is.null(ps)) return(placeholder_panel("no severity characteristics with data"))
  # Shape the grid to how many panels survived, so a partial set fills the space
  # instead of leaving blank cells in a fixed 2x2.
  n_col <- if (length(ps) <= 3) length(ps) else 2
  wrap_plots(ps, ncol = n_col) +
    plot_annotation(title = "Disease-Severity Characteristics by Risk-Dynamics Group") &
    theme(plot.title = element_text(size = 13, face = "bold", hjust = 0.5))
}


# ============================================================================
# figS1a: silhouette vs k (appendix)
# ============================================================================
build_figS1a <- function() {
  d <- load_figure_data("fig4_silhouette.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig4_silhouette.csv empty"))
  d <- d %>% arrange(k)
  best <- d$k[which.max(d$silhouette)]
  best_val <- max(d$silhouette)

  ggplot(d, aes(k, silhouette)) +
    geom_line(color = "#2E86C1", linewidth = 1) +
    geom_point(size = 2, color = "#2E86C1") +
    geom_vline(xintercept = N_SLOPE_GROUPS, color = "#E74C3C",
               linetype = "dashed", linewidth = 1) +
    annotate("point", x = best, y = best_val, color = "#E74C3C", size = 3) +
    annotate("text", x = N_SLOPE_GROUPS, y = max(d$silhouette) * 1.04,
             label = sprintf("chosen k=%d", N_SLOPE_GROUPS),
             hjust = -0.05, size = MANUSCRIPT_TEXT_SIZE,
             fontface = "italic", color = "#E74C3C") +
    annotate("text", x = best, y = best_val + 0.005,
             label = sprintf("best silhouette (k=%d)", best),
             hjust = -0.05, size = MANUSCRIPT_TEXT_SIZE,
             fontface = "italic", color = "#E74C3C") +
    labs(x = "Number of slope groups (k)", y = "Mean silhouette score",
         title = "Risk-Slope Group-Count Selection") +
    theme_manuscript() +
    theme(panel.grid.major = element_line(color = "grey90"))
}


# ============================================================================
# Compose Figure 4 + Figure S1
# ============================================================================
p4a <- build_fig4a()
p4b <- build_fig4b()
p4d <- build_fig4d()
p4e <- build_fig4e()
p4c <- build_fig4c()
pS1 <- build_figS1a()

save_panel(p4a, "fig4a",  group = "figure4", width = 8.8, height = 5.8)
save_panel(p4b, "fig4b",  group = "figure4", width = 9.8, height = 7.0)
save_panel(p4d, "fig4d",  group = "figure4", width = 7.8, height = 5.8)
save_panel(p4e, "fig4e",  group = "figure4", width = 7.2, height = 5.4)
save_panel(p4c, "fig4c",  group = "figure4", width = 9.6, height = 8.4)
save_panel(pS1, "figS1a", group = "figure4", width = 7.2, height = 5.4)

# Complete manuscript figure, with one lowercase label per named panel.
save_compiled_figure(
  list(a = p4a, b = p4b, c = p4c, d = p4d, e = p4e),
  number = 4, width = 20, height = 23, design = "aabb\ncccc\nddee", heights = c(7, 9, 6)
)
