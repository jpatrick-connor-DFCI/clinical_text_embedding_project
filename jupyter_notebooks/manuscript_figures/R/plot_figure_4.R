# Render Figure 4 (mortality-risk dynamics) + Figure S1 (silhouette appendix).
#
# A per-patient mortality-risk trajectory heatmap (raster) grouped by risk-slope group,
# B conditional KM survival from the slope-window landmark (left-truncated entry),
# D mean trajectory per slope group vs. a cohort-average reference band,
# E stage-matched slope-group composition (dynamics vs. baseline stage),
# C disease-severity small multiples (% Stage IV, mean # met, 10-yr RMST, mean slope).
# S1: silhouette vs k (slope-group-count justification).

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr); library(viridisLite)
  library(survival); library(ggsurvfit)
})

script_dir <- local({
  if (exists("R_DIR", envir = globalenv(), inherits = FALSE)) {
    return(get("R_DIR", envir = globalenv()))
  }
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
  if (length(fa) && nzchar(fa[1])) return(dirname(normalizePath(fa[1])))
  for (n in seq_len(sys.nframe())) {
    ofile <- sys.frame(n)$ofile
    if (!is.null(ofile) && nzchar(ofile)) return(dirname(normalizePath(ofile)))
  }
  stop("Could not determine script directory (set R_DIR in globalenv, run via Rscript, or source() directly)")
})
source(file.path(script_dir, "figure_utils.R"))

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

  yticks <- bounds$mid
  ylabs  <- cluster_label(bounds$cluster, bounds$n)

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
         title = "Mortality-Risk Trajectories by Dynamics Group") +
    theme_manuscript() +
    theme(panel.grid = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 7))
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

  ggplot(td, aes(time, estimate, color = label)) +
    { if (nrow(ci) > 0) geom_rect(data = ci,
                                  aes(xmin = time, xmax = time_next,
                                      ymin = conf.low, ymax = conf.high,
                                      fill = label),
                                  color = NA, alpha = 0.15, inherit.aes = FALSE) } +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = pal, name = NULL, drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none", drop = FALSE) +
    coord_cartesian(xlim = c(LANDMARK, 120)) +
    annotate("text", x = LANDMARK + 2, y = 0.05,
             label = sprintf("Log-rank p=%.1e", lp),
             hjust = 0, size = 2.7, fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment",
         y = sprintf(paste0("Overall Survival Probability\n",
                           "(conditional on survival to month %s)"), LANDMARK),
         title = "KM Overall Survival by Risk-Dynamics Group") +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.18), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))
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
    labs(x = "Months post-treatment", y = "Model mortality risk (raw)",
         title = "Mean Risk Trajectory by Dynamics Group") +
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

  ggplot(d, aes(stage, n_patients, fill = group_lab)) +
    geom_col(position = "fill", width = 0.7, color = "white") +
    scale_fill_manual(values = pal, name = NULL, drop = FALSE) +
    scale_y_continuous(labels = scales::percent) +
    labs(x = "Stage", y = "Proportion of stage",
         title = "Risk-Dynamics Composition by Stage") +
    theme_manuscript()
}


# ============================================================================
# fig4c: disease-severity + slope small multiples
# ============================================================================
# The four displayed metrics come from fig4_cluster_severity.csv. Mean slope is
# the quantity the dynamics groups are defined on and can be negative for the
# Falling group (no 0..100 clamp applied).
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
    list(title = "% Stage IV",       units = "Percentage (%)", vals = by_id("pct_stage_iv"),   is_pct = TRUE),
    list(title = "Mean # met sites", units = "Sites (0-7)",    vals = by_id("mean_met_sites"), is_pct = FALSE),
    list(title = "10-yr RMST",       units = "Months",         vals = by_id("rmst_months"),    is_pct = FALSE),
    list(title = "Mean risk slope",  units = "Risk / month",   vals = by_id("mean_slope"),     is_pct = FALSE)
  )

  panel_for <- function(spec) {
    if (is.null(spec$vals)) return(placeholder_panel(paste("no data:", spec$title)))
    df <- tibble::tibble(cluster = clusters,
                         value = unname(spec$vals[as.character(clusters)]))
    p <- ggplot(df, aes(factor(cluster), value, fill = factor(cluster))) +
      geom_col(width = 0.65, color = "white") +
      scale_fill_manual(values = setNames(GROUP_COLORS[clusters + 1],
                                          as.character(clusters)),
                        guide = "none") +
      scale_x_discrete(labels = setNames(GROUP_NAMES[clusters + 1],
                                         as.character(clusters))) +
      labs(x = "Risk dynamics", y = spec$units, title = spec$title) +
      theme_manuscript() +
      theme(panel.grid.major.y = element_line(color = "grey90"))
    # is_pct panels are bounded percentages; the other metrics (incl. mean
    # slope, which can be negative for Falling) are left with a free y-range.
    if (isTRUE(spec$is_pct)) p <- p + coord_cartesian(ylim = c(0, 100))
    p
  }
  ps <- lapply(characteristics, panel_for)
  wrap_plots(ps, nrow = 2, ncol = 2) +
    plot_annotation(title = "Disease-Severity Characteristics by Risk-Dynamics Group") &
    theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5))
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
             hjust = -0.05, size = 3, fontface = "italic", color = "#E74C3C") +
    annotate("text", x = best, y = best_val + 0.005,
             label = sprintf("best silhouette (k=%d)", best),
             hjust = -0.05, size = 3, fontface = "italic", color = "#E74C3C") +
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

save_panel(p4a, "fig4a",  width = 6.4, height = 4.8)
save_panel(p4b, "fig4b",  width = 6.4, height = 4.8)
save_panel(p4d, "fig4d",  width = 6.4, height = 4.8)
save_panel(p4e, "fig4e",  width = 6.0, height = 4.4)
save_panel(p4c, "fig4c",  width = 8.0, height = 7.0)
save_panel(pS1, "figS1a", width = 6.0, height = 4.4)

fig4 <- (p4a + p4b) / (p4d + p4e) / p4c +
        plot_layout(heights = c(1, 1, 1.4)) +
        plot_annotation(tag_levels = "A") &
        theme(plot.tag = element_text(size = 14, face = "bold"))

save_figure(fig4, "figure4_trajectories", width = 14.0, height = 18.0)
save_figure(pS1, "figureS1_cluster_silhouette", width = 7.0, height = 5.5)
