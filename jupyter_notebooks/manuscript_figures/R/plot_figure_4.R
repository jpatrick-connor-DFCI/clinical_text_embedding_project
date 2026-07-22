# Render Figure 4 (mortality trajectories) + Figure S1 (silhouette appendix).
#
# A per-patient mortality-risk trajectory heatmap (raster) grouped by cluster,
# B conditional KM survival from month 60 (left-truncated entry at 60),
# C disease-severity small multiples (% Stage IV, % ICI, mean # met, 10-yr RMST).
# S1: silhouette vs k (cluster-count justification).

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

N_CLUSTERS  <- 4
# prep_figure_4 relabels clusters 0..N-1 by ASCENDING mean risk (cluster 0 =
# lowest risk). Names therefore describe risk LEVEL (the quantity the clusters
# are actually ordered on), not an assumed temporal shape, which the rank does
# not determine. Length must equal N_CLUSTERS.
CLUSTER_NAMES <- c("Lowest Risk", "Low-Intermediate Risk",
                   "High-Intermediate Risk", "Highest Risk")
stopifnot(length(CLUSTER_NAMES) == N_CLUSTERS)

cluster_label <- function(k, n = NA_integer_) {
  # Coerce via character so factors return their underlying integer (not level codes).
  k_int <- suppressWarnings(as.integer(as.character(k)))
  nm <- CLUSTER_NAMES[pmin(k_int + 1L, length(CLUSTER_NAMES))]
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
         title = "Mortality-Risk Trajectories by Cluster") +
    theme_manuscript() +
    theme(panel.grid = element_blank(),
          axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 7))
}


# ============================================================================
# fig4b: conditional KM from month 60 (left-truncated entry)
# ============================================================================
build_fig4b <- function() {
  km <- load_figure_data("fig4_km_data.csv")
  if (nrow(km) == 0) return(placeholder_panel("fig4_km_data.csv empty"))
  km <- km %>%
    mutate(months     = tt_death / 30.44,
           death      = as.integer(death),
           cluster_id = suppressWarnings(as.integer(as.character(cluster)))) %>%
    filter(months > 60, !is.na(cluster_id))
  if (nrow(km) == 0) return(placeholder_panel("no patients survive to month 60"))
  km$entry <- 60

  # Stable cluster-id → (label, color) map; we stratify on the SHORT id to avoid
  # any quirks of strata-name handling on long labels with parens/equals.
  cluster_ids  <- sort(unique(km$cluster_id))
  n_by_id      <- as.integer(table(km$cluster_id)[as.character(cluster_ids)])
  labels_by_id <- setNames(cluster_label(cluster_ids, n_by_id),
                           as.character(cluster_ids))
  colors_by_id <- setNames(CLUSTER_COLORS[seq_along(cluster_ids)],
                           as.character(cluster_ids))

  km <- km %>% mutate(strat = as.character(cluster_id))
  fit <- survfit2(Surv(entry, months, death) ~ strat, data = km)
  td <- ggsurvfit::tidy_survfit(fit) %>%
    mutate(strat_id = sub("^[^=]+=", "", as.character(strata)),
           label    = factor(labels_by_id[strat_id],
                             levels = unname(labels_by_id))) %>%
    # survfit2 emits a synthetic time=0, estimate=1 row per stratum (curve
    # start) that predates the entry=60 left-truncation point; drawing it
    # produces an unstratified flat segment before month 60 once geom_step
    # connects it to the first real event. Drop anything before entry.
    filter(time >= 60)

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
    coord_cartesian(xlim = c(60, 120)) +
    annotate("text", x = 62, y = 0.05,
             label = sprintf("Log-rank p=%.1e", lp),
             hjust = 0, size = 2.7, fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment",
         y = "Overall Survival Probability (conditional)",
         title = "KM Overall Survival by Cluster\n(conditional on survival to month 60)") +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.18), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# fig4c: disease-severity small multiples
# ============================================================================
# All four metrics now come from fig4_cluster_severity.csv. % Stage IV and
# % ICI Treated used to be derived by token-matching the top-N composition
# CSVs, but those top-N filters silently dropped ICI from the panel and tied
# the panel to fragile column-name regexes. The prep computes both directly
# from the raw stage pickle and the per-patient ever-on-ICI flag now.
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
    list(title = "% ICI Treated",    units = "Percentage (%)", vals = by_id("pct_ici"),        is_pct = TRUE),
    list(title = "Mean # met sites", units = "Sites (0-7)",    vals = by_id("mean_met_sites"), is_pct = FALSE),
    list(title = "10-yr RMST",       units = "Months",         vals = by_id("rmst_months"),    is_pct = FALSE)
  )

  panel_for <- function(spec) {
    if (is.null(spec$vals)) return(placeholder_panel(paste("no data:", spec$title)))
    df <- tibble::tibble(cluster = clusters,
                         value = unname(spec$vals[as.character(clusters)]))
    p <- ggplot(df, aes(factor(cluster), value, fill = factor(cluster))) +
      geom_col(width = 0.65, color = "white") +
      scale_fill_manual(values = setNames(CLUSTER_COLORS[clusters + 1],
                                          as.character(clusters)),
                        guide = "none") +
      labs(x = "Cluster", y = spec$units, title = spec$title) +
      theme_manuscript() +
      theme(panel.grid.major.y = element_line(color = "grey90"))
    if (isTRUE(spec$is_pct)) p <- p + coord_cartesian(ylim = c(0, 100))
    p
  }
  ps <- lapply(characteristics, panel_for)
  wrap_plots(ps, nrow = 1) +
    plot_annotation(title = "Disease-Severity Characteristics by Trajectory Cluster") &
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
    geom_vline(xintercept = N_CLUSTERS, color = "#E74C3C",
               linetype = "dashed", linewidth = 1) +
    annotate("point", x = best, y = best_val, color = "#E74C3C", size = 3) +
    annotate("text", x = N_CLUSTERS, y = max(d$silhouette) * 1.04,
             label = sprintf("chosen k=%d", N_CLUSTERS),
             hjust = -0.05, size = 3, fontface = "italic", color = "#E74C3C") +
    annotate("text", x = best, y = best_val + 0.005,
             label = sprintf("best silhouette (k=%d)", best),
             hjust = -0.05, size = 3, fontface = "italic", color = "#E74C3C") +
    labs(x = "Number of clusters (k)", y = "Mean silhouette score",
         title = "Trajectory Cluster-Count Selection") +
    theme_manuscript() +
    theme(panel.grid.major = element_line(color = "grey90"))
}


# ============================================================================
# Compose Figure 4 + Figure S1
# ============================================================================
p4a <- build_fig4a()
p4b <- build_fig4b()
p4c <- build_fig4c()
pS1 <- build_figS1a()

save_panel(p4a, "fig4a",  width = 6.4, height = 4.8)
save_panel(p4b, "fig4b",  width = 6.4, height = 4.8)
save_panel(p4c, "fig4c",  width = 12.0, height = 3.8)
save_panel(pS1, "figS1a", width = 6.0, height = 4.4)

fig4 <- (p4a + p4b) / p4c +
        plot_layout(heights = c(1, 0.85)) +
        plot_annotation(tag_levels = "A") &
        theme(plot.tag = element_text(size = 14, face = "bold"))

save_figure(fig4, "figure4_trajectories", width = 14.0, height = 12.0)
save_figure(pS1, "figureS1_cluster_silhouette", width = 7.0, height = 5.5)
