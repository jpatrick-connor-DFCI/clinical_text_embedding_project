# Render Figure 4 supplement: within-stage KM curves stratified by mortality-
# trajectory (risk-slope) cluster.
#
# Companion to plot_figure_2_supp.R (which stratifies each stage by overall
# risk-score quartile). Here each stage is stratified by the Fig 4 risk-DYNAMICS
# group (Falling / Stable / Rising), so the panels show that trajectory dynamics
# separate survival within stage-defined strata. Two panels:
#   A  Stage IV patients, by risk-dynamics group
#   B  Stages I-II patients pooled, by risk-dynamics group
# The crossed stage/dynamics comparison is now main Figure 4d.
#
# Conditional on survival to the slope-window landmark (left-truncated entry),
# matching main-figure panel 4b. The curves start at the trajectory-observation
# landmark. Reads fig4_km_data.csv (DFCI_MRN, cluster, death, tt_death, stage,
# landmark_month), all written by prep_figure_4.py.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(scales); library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")
source("R/figure4_utils.R", local = TRUE)


# ----------------------------------------------------------------------------
# Fig 4 risk-dynamics group constants (kept local, mirroring plot_figure_4.R so
# this supplement stands alone — plot_figure_2_supp.R defines its own quartile
# constants the same way rather than importing the main plot script).
# ----------------------------------------------------------------------------
N_SLOPE_GROUPS <- 3
GROUP_NAMES  <- c("Falling Risk", "Stable Risk", "Rising Risk")
GROUP_COLORS <- c(BENEFIT_COLOR, NS_GRAY, HARM_COLOR)  # falling=blue, stable=grey, rising=red
stopifnot(length(GROUP_NAMES) == N_SLOPE_GROUPS)

cluster_label <- function(k, n = NA_integer_) {
  k_int <- suppressWarnings(as.integer(as.character(k)))
  nm <- GROUP_NAMES[pmin(k_int + 1L, length(GROUP_NAMES))]
  if (any(!is.na(n))) sprintf("%s (n=%s)", nm, scales::comma(n)) else nm
}

# ============================================================================
# One KM panel: a stage-defined stratum, conditional on the landmark, by cluster
# ============================================================================
build_stage_dynamics_panel <- function(df, stage_values, stage_label, title_text) {
  # See note in plot_figure_2_supp.R: an empty frame has no `stage` column, so
  # the emptiness check has to come before the filter.
  if (nrow(df) == 0) return(placeholder_panel(paste0("no ", stage_label, " patients")))
  sub <- df %>% filter(stage %in% stage_values)
  if (nrow(sub) == 0) return(placeholder_panel(paste0("no ", stage_label, " patients")))

  # Stratify on the SHORT cluster id (avoids strata-name quirks with parenthesized
  # labels), then map id -> (label, color). Conditional entry at the landmark.
  cluster_ids  <- sort(unique(sub$cluster_id))
  n_by_id      <- as.integer(table(sub$cluster_id)[as.character(cluster_ids)])
  labels_by_id <- setNames(cluster_label(cluster_ids, n_by_id), as.character(cluster_ids))
  colors_by_id <- setNames(GROUP_COLORS[cluster_ids + 1L], as.character(cluster_ids))

  sub <- sub %>% mutate(strat = as.character(cluster_id))
  fit <- survfit2(Surv(entry, months, death) ~ strat, data = sub)
  td  <- tidy_km(fit) %>%
    mutate(label = factor(labels_by_id[stratum], levels = unname(labels_by_id))) %>%
    # Drop the synthetic time=0 curve-start row that predates the landmark
  # left-truncation point (see plot_figure_4.R::build_fig4b for the rationale).
    filter(time >= LANDMARK)
  if (nrow(td) == 0) return(placeholder_panel(
    paste0("no ", stage_label, " events after month ", LANDMARK)))

  pal   <- setNames(unname(colors_by_id), unname(labels_by_id))
  lr    <- logrank_p_lt(sub, "entry", "months", "death", "strat")
  td_ci <- step_ci_df(td, "label")
  ann   <- sprintf("n=%s\nlogrank p=%.1e", scales::comma(nrow(sub)), lr)

  ggplot(td, aes(time, estimate, color = label)) +
    { if (nrow(td_ci) > 0) geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = label),
              inherit.aes = FALSE, alpha = 0.15, color = NA) } +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = pal, name = "Risk dynamics", drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none", drop = FALSE) +
    coord_cartesian(xlim = c(LANDMARK, 120), ylim = c(0, 1.03)) +
    annotate("text", x = 118, y = 1.0, label = ann,
             hjust = 1, vjust = 1, size = MANUSCRIPT_SMALL_TEXT_SIZE,
             fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment",
         y = sprintf("Overall survival (conditional on survival to month %s)", LANDMARK),
         title = title_text) +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.20), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# Compose supplementary figure
# ============================================================================
stage_dynamics <- load_stage_dynamics_data()
d <- stage_dynamics$data
LANDMARK <- stage_dynamics$landmark

pS_iv <- build_stage_dynamics_panel(
  d, "IV", "Stage IV", "Stage IV: survival by risk-dynamics group")
pS_i_ii <- build_stage_dynamics_panel(
  d, c("I", "II"), "Stages I-II", "Stages I-II: survival by risk-dynamics group")

save_panel(pS_iv, "figS_stage4_by_dynamics", group = "figure4", width = 8.6, height = 7.2)
save_panel(pS_i_ii, "figS_stage1_2_by_dynamics", group = "figure4", width = 8.6, height = 7.2)
