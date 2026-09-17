# Shared stage/risk-dynamics survival comparison. Source after figure_utils.R.
# The main and supplemental renderers use the same landmark-filtered cohort.
load_stage_dynamics_data <- function() {
  d <- load_figure_data("fig4_km_data.csv")
  landmark_values <- if ("landmark_month" %in% names(d)) {
    unique(d$landmark_month[!is.na(d$landmark_month)])
  } else numeric(0)
  landmark <- if (length(landmark_values) == 1) as.numeric(landmark_values) else 12
  if (nrow(d) > 0) {
    d <- d %>%
      mutate(months = tt_death / 30.44, death = as.integer(death),
             cluster_id = suppressWarnings(as.integer(as.character(cluster))),
             stage = as.character(stage)) %>%
      filter(months > landmark, !is.na(cluster_id), !is.na(stage))
    d$entry <- landmark
  }
  list(data = d, landmark = landmark)
}

logrank_p_lt <- function(df, start_col, time_col, event_col, group_col) {
  if (nrow(df) == 0 || length(unique(df[[group_col]])) < 2) return(NA_real_)
  f <- as.formula(sprintf("Surv(%s, %s, %s) ~ %s",
                          start_col, time_col, event_col, group_col))
  cx <- tryCatch(survival::coxph(f, data = df), error = function(e) NULL)
  if (is.null(cx)) return(NA_real_)
  unname(summary(cx)$sctest["pvalue"])
}

build_crossed_dynamics_panel <- function(df, arms, title_text, landmark) {
  # An empty frame (missing CSV) has no columns, so emptiness must be tested
  # before the per-arm filter() below names one.
  if (nrow(df) == 0) {
    return(placeholder_panel("one or both crossed strata have no patients"))
  }
  parts <- lapply(arms, function(a) {
    df %>% filter(stage %in% a$stage_values, cluster_id == a$cluster_id) %>%
      mutate(strat = a$label)
  })
  sub <- bind_rows(parts)
  if (nrow(sub) == 0 || length(unique(sub$strat)) < length(arms)) {
    return(placeholder_panel("one or both crossed strata have no patients"))
  }

  labels_in_order <- vapply(arms, function(a) a$label, character(1))
  sub <- sub %>% mutate(strat = factor(strat, levels = labels_in_order))
  pal <- setNames(vapply(arms, function(a) a$color, character(1)), labels_in_order)

  fit <- survfit2(Surv(entry, months, death) ~ strat, data = sub)
  td  <- tidy_km(fit) %>%
    mutate(label = factor(stratum, levels = labels_in_order)) %>%
    # Drop the synthetic time=0 curve-start row that predates the landmark
    # left-truncation point (see plot_figure_4.R::build_fig4b for the rationale).
    filter(time >= landmark)
  if (nrow(td) == 0) return(placeholder_panel(
    paste0("no events after month ", landmark)))

  lr    <- logrank_p_lt(sub, "entry", "months", "death", "strat")
  td_ci <- step_ci_df(td, "label")
  ann <- sprintf("Cox score test %s", format_p_inline(lr))
  risk_times <- sort(unique(c(landmark, seq(24, 120, 24))))
  risk_times <- risk_times[risk_times >= landmark]

  p <- ggplot(td, aes(time, estimate, color = label)) +
    { if (nrow(td_ci) > 0) geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = label),
              inherit.aes = FALSE, alpha = 0.15, color = NA) } +
    geom_step(linewidth = 0.5) +
    scale_color_manual(values = pal, name = NULL, drop = FALSE) +
    scale_fill_manual(values = pal, guide = "none", drop = FALSE) +
    scale_x_continuous(breaks = risk_times, expand = expansion(mult = SURVIVAL_X_EXPANSION)) +
    coord_cartesian(xlim = c(landmark, 120), ylim = c(0, 1.03)) +
    annotate("text", x = landmark + 2, y = 0.08, label = ann,
             hjust = 0, vjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE,
             color = "#333333") +
    labs(x = "Months from first treatment",
         y = "Conditional overall survival",
         title = title_text,
         subtitle = sprintf("Month-%s landmark; N = %s", landmark, comma(nrow(sub)))) +
    theme_manuscript() +
    theme(legend.position = c(0.02, 0.28), legend.justification = c(0, 0),
          legend.background = element_rect(fill = "white", color = NA))
  group_n <- table(sub$strat)
  attr(p, "caption_detail") <- paste0("Panel d: ", paste(sprintf("%s, n = %s",
    names(group_n), scales::comma(as.integer(group_n))), collapse = "; "), ".")
  p
}
