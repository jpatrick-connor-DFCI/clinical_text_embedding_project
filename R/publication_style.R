# Main-figure typography; source after figure_utils.R.
# Supplements retain their existing theme when they source figure_utils.R anew.
MANUSCRIPT_BASE_SIZE <- 10
MANUSCRIPT_TEXT_SIZE <- 3.2
MANUSCRIPT_SMALL_TEXT_SIZE <- 3
MANUSCRIPT_CAPTION_SIZE <- 9
MANUSCRIPT_WIDTH <- 180 / 25.4
COMPILED_FIGURE_WIDTH <- 10
COMPILED_FIGURE_HEIGHT <- 9
SURVIVAL_X_EXPANSION <- 0.06

SCHEME_LABELS <- c(death_met = "Death + mets", icd3_post = "ICD-10 (3-char)",
                   icd4_post = "ICD-10 (4-char)", phecode_post = "PhecodeX")
MODALITY_DISPLAY["metburden"] <- "Met. burden"

theme_manuscript <- function(base_size = MANUSCRIPT_BASE_SIZE) {
  theme_classic(base_size = base_size, base_family = "sans") +
    theme(plot.title = element_text(size = 11, face = "bold", hjust = 0),
          plot.subtitle = element_text(size = 9, color = "#333333"),
          axis.title = element_text(size = 10, face = "bold"),
          axis.text = element_text(size = 9, face = "bold", color = "#222222"),
          axis.line = element_line(linewidth = 0.6), axis.ticks = element_line(linewidth = 0.6),
          legend.title = element_text(size = 9), legend.text = element_text(size = 9),
          legend.key.size = grid::unit(4, "mm"), legend.spacing = grid::unit(1.5, "mm"),
          legend.margin = margin(1, 1, 1, 1), legend.background = element_blank(),
          legend.key = element_blank(), strip.background = element_blank(),
          strip.text = element_text(size = 10, face = "bold"),
          plot.caption = element_text(size = MANUSCRIPT_CAPTION_SIZE, hjust = 0),
          plot.margin = margin(5, 6, 5, 5))
}

# Risk counts come from the fitted survival object, not from digitized curves.
# Short row labels keep the table aligned with the main survival plot.
manuscript_risk_counts <- function(fit, times, labels) {
  s <- summary(fit, times = times, extend = TRUE)
  strata <- if (is.null(s$strata)) {
    if (length(labels) != 1L) stop("An unstratified survival fit requires one risk-table label")
    rep(names(labels), length(s$time))
  } else sub("^[^=]+=", "", as.character(s$strata))
  if (is.null(names(labels)) || any(!strata %in% names(labels))) {
    stop("Risk-table labels do not match the fitted survival strata")
  }
  tibble::tibble(time = s$time, n_risk = s$n.risk, stratum = strata) %>%
    mutate(row = factor(stratum, levels = rev(names(labels))))
}

add_manuscript_risk_table <- function(plot, fit, times, labels) {
  rt <- manuscript_risk_counts(fit, times, labels)
  table <- ggplot(rt, aes(time, row, label = scales::comma(n_risk))) +
    geom_text(size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#222222") +
    scale_x_continuous(limits = range(times), breaks = times,
                       expand = expansion(mult = c(0.025, 0.025))) +
    scale_y_discrete(labels = labels) +
    labs(title = "Number at risk", x = NULL, y = NULL) + theme_void(base_size = 6) +
    theme(plot.title = element_text(size = 6.5, face = "plain"),
          axis.text.y = element_text(size = 6, color = "#222222"),
          axis.text.x = element_text(size = 6, color = "#222222"),
          plot.margin = margin(1, 4, 2, 3))
  patchwork::wrap_plots(plot, table, ncol = 1, heights = c(1, 0.27))
}
