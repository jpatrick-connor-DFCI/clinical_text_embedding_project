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
CLUSTER_NAMES <- c("Stable Low", "Intermediate", "Stable High",
                   "Rapidly Increasing", "Rebounding")

cluster_label <- function(k, n = NA_integer_) {
  nm <- CLUSTER_NAMES[pmin(as.integer(k) + 1L, length(CLUSTER_NAMES))]
  if (any(!is.na(n))) sprintf("%s (n=%s)", nm, scales::comma(n)) else nm
}

month_to_num <- function(x) {
  as.numeric(stringr::str_extract(as.character(x), "\\d+"))
}

logrank_p <- function(df, time_col, event_col, group_col, start_col = NULL) {
  if (nrow(df) == 0) return(NA_real_)
  rhs <- if (is.null(start_col)) sprintf("Surv(%s, %s)", time_col, event_col)
         else sprintf("Surv(%s, %s, %s)", start_col, time_col, event_col)
  f <- as.formula(paste(rhs, "~", group_col))
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

  ggplot(long, aes(month, row_idx, fill = risk)) +
    geom_raster() +
    scale_fill_viridis_c(option = "A", name = "Std.\nmortality\nrisk", na.value = "grey90") +
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
    mutate(months  = tt_death / 30.44,
           death   = as.integer(death),
           cluster = factor(cluster)) %>%
    filter(months > 60)
  if (nrow(km) == 0) return(placeholder_panel("no patients survive to month 60"))
  km <- km %>% mutate(entry = 60)

  ns <- km %>% group_by(cluster) %>% summarise(n = n(), .groups = "drop")
  cluster_levels <- as.character(ns$cluster)
  km <- km %>% mutate(strat_lbl = cluster_label(cluster, ns$n[match(cluster, ns$cluster)]),
                      strat_lbl = factor(strat_lbl,
                                         levels = cluster_label(ns$cluster, ns$n)))
  fit <- survfit2(Surv(entry, months, death) ~ strat_lbl, data = km)
  td <- ggsurvfit::tidy_survfit(fit) %>%
    mutate(stratum = sub("^[^=]+=", "", as.character(strata)))

  lp <- logrank_p(km, "months", "death", "cluster", start_col = "entry")
  pal <- setNames(CLUSTER_COLORS[seq_along(levels(km$strat_lbl))],
                  levels(km$strat_lbl))

  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = pal, name = NULL) +
    coord_cartesian(xlim = c(60, 120)) +
    annotate("text", x = 62, y = 0.05,
             label = sprintf("Log-rank p=%.1e", lp),
             hjust = 0, size = 2.7, fontface = "italic", color = "#444") +
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
STAGE_IV_PATTERN <- "^(IV|4(\\.0+)?)[A-D]?$"
ICI_PATTERN <- paste0("^(ICI|IMMUNOTHERAPY|PD1|PDL1|PD_?L1|",
                      "IMMUNE[ _]CHECKPOINT[ _]INHIBITORS?|",
                      "CHECKPOINT(?:_INHIBITOR)?)$")

share_by_token <- function(comp, pattern) {
  if (nrow(comp) == 0) return(setNames(numeric(0), character(0)))
  cols <- setdiff(names(comp), c("cluster", "OTHER"))
  match_cols <- cols[grepl(pattern, cols, ignore.case = TRUE, perl = TRUE)]
  if (length(match_cols) == 0) return(setNames(numeric(0), character(0)))
  idx <- comp$cluster
  val <- rowSums(comp[, match_cols, drop = FALSE]) * 100
  setNames(val, as.character(idx))
}

build_fig4c <- function() {
  stage    <- load_figure_data("fig4_cluster_composition_stage.csv")
  treat    <- load_figure_data("fig4_cluster_composition_treatment.csv")
  severity <- load_figure_data("fig4_cluster_severity.csv")

  stage_iv <- share_by_token(stage, STAGE_IV_PATTERN)
  ici      <- share_by_token(treat, ICI_PATTERN)
  met      <- if ("mean_met_sites" %in% names(severity))
                setNames(severity$mean_met_sites, as.character(severity$cluster)) else NULL
  rmst     <- if ("rmst_months" %in% names(severity))
                setNames(severity$rmst_months, as.character(severity$cluster)) else NULL

  clusters <- sort(unique(as.integer(c(names(stage_iv), names(ici), names(met), names(rmst)))))
  if (length(clusters) == 0) return(placeholder_panel("cluster characteristic data empty"))

  characteristics <- list(
    list(title = "% Stage IV",        units = "Percentage (%)", vals = stage_iv, is_pct = TRUE),
    list(title = "% ICI Treated",     units = "Percentage (%)", vals = ici,      is_pct = TRUE),
    list(title = "Mean # met sites",  units = "Sites (0-7)",    vals = met,      is_pct = FALSE),
    list(title = "10-yr RMST",        units = "Months",         vals = rmst,     is_pct = FALSE)
  )

  panel_for <- function(spec) {
    if (length(spec$vals) == 0) return(placeholder_panel(paste("no data:", spec$title)))
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
