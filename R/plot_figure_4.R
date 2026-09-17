# Figure 4: a risk-score heatmap, b survival by risk trajectory,
# c risk-dynamics composition by stage, d stage I-II rising versus stage IV
# falling risk survival. Individual panels and compiled PNG/PDF with panel letters and no plot titles.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(stringr); library(viridisLite)
  library(survival); library(ggsurvfit)
})

source("R/figure_utils.R")
source("R/publication_style.R")
source("R/figure4_utils.R", local = TRUE)

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
  ylabs <- sprintf("%s\n%s / %s", cluster_label(bounds$cluster),
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
      name     = "Risk score\n(z)",
      na.value = "grey90"
    ) +
    scale_x_continuous(expand = c(0, 0), breaks = sort(unique(long$month))) +
    scale_y_reverse(expand = c(0, 0), breaks = yticks, labels = ylabs) +
    geom_hline(data = bounds[-nrow(bounds), ],
               aes(yintercept = top + 0.5),
               color = "white", linewidth = 0.6) +
    labs(x = "Months from first treatment", y = NULL,
         title = "Risk-score trajectories",
         subtitle = "Rows: patients; shown / full group counts") +
    theme_manuscript() +
    theme(panel.grid = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 5.8),
          legend.key.height = unit(5, "mm"), legend.key.width = unit(2.5, "mm"))
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

  risk_times <- sort(unique(c(LANDMARK, seq(24, 120, 24))))
  risk_times <- risk_times[risk_times >= LANDMARK]
  main <- ggplot(td, aes(time, estimate, color = label)) +
    { if (nrow(ci) > 0) geom_rect(data = ci,
                                  aes(xmin = time, xmax = time_next,
                                      ymin = conf.low, ymax = conf.high,
                                      fill = label),
                                  color = NA, alpha = 0.15, inherit.aes = FALSE) } +
    geom_step(linewidth = 0.5) +
    scale_color_manual(values = pal, name = NULL, drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none", drop = FALSE) +
    scale_x_continuous(breaks = risk_times, expand = expansion(mult = SURVIVAL_X_EXPANSION)) +
    coord_cartesian(xlim = c(LANDMARK, 120), ylim = c(0, 1.03)) +
    labs(x = "Months from first treatment",
         y = "Conditional overall survival",
         title = "Survival by risk dynamics",
         subtitle = sprintf("Month-%s landmark; N = %s", LANDMARK, comma(nrow(km)))) +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.18), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))

  p <- main
  # Preserve the model results in the figure legend.
  short_hr <- gsub(" \\(n=[^)]+\\)", "", hr_text)
  attr(p, "caption_detail") <- sprintf(
    "Panel b: landmark %s months, N = %s; Cox score test %s. %sCox analysis (N = %s): %s.",
    LANDMARK, comma(nrow(km)), format_p_inline(lp),
    ifelse(stage_adjusted, "Stage-adjusted ", "Unadjusted "),
    comma(nrow(cx_data)), gsub("\n", "; ", short_hr))
  p
}


# ============================================================================
# fig4c: stage-matched dynamics-group composition (dynamics vs. baseline stage)
# ============================================================================
build_stage_composition <- function() {
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
         title = "Risk dynamics by stage",
         subtitle = sprintf("N = %s; Cramér's V = %.3f",
                            comma(total_n), cramer_v)) +
    theme_manuscript() + theme(legend.position = "bottom")
}


# fig4d: early-stage rising risk versus stage IV falling risk.
build_stage_risk_comparison <- function() {
  cohort <- load_stage_dynamics_data()
  build_crossed_dynamics_panel(
    cohort$data,
    arms = list(
      list(stage_values = c("I", "II"), cluster_id = 2L,
           label = "Stage I-II, Rising Risk", color = GROUP_COLORS[3]),
      list(stage_values = "IV", cluster_id = 0L,
           label = "Stage IV, Falling Risk", color = GROUP_COLORS[1])
    ),
    title_text = "Early-stage rising vs. stage IV falling",
    landmark = cohort$landmark
  )
}

# Compose exactly the four requested mortality-risk dynamics panels.
retire_main_panels(4, letters[1:4])

p4a <- build_fig4a()
p4b <- build_fig4b()
p4c <- build_stage_composition()
p4d <- build_stage_risk_comparison()

.tag <- metric_tag()
save_panel(p4a, paste0("fig4a", .tag), group = "figure4", width = 3.5, height = 3.2, dpi = 600)
save_panel(p4b, paste0("fig4b", .tag), group = "figure4", width = 3.5, height = 3.2, dpi = 600)
save_panel(p4c, paste0("fig4c", .tag), group = "figure4", width = 3.5, height = 3.2, dpi = 600)
save_panel(p4d, paste0("fig4d", .tag), group = "figure4", width = 3.5, height = 3.2, dpi = 600)
save_compiled_figure(
  list(a = p4a, b = p4b, c = p4c, d = p4d),
  number = 4, width = COMPILED_FIGURE_WIDTH, height = COMPILED_FIGURE_HEIGHT
)
