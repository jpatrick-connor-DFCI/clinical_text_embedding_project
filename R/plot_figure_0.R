# Render Figure 0 (cohort data-availability cascade) in ggplot2.
#
# Single panel: horizontal barplot showing how many patients remain as each
# modality's availability requirement is added, ending in the subset that passes
# every threshold at once (the multi-modal analysis cohort used from Figure 3
# onward). Counts are cumulative, so each bar is a subset of the one above it.
# Output: panel fig0a (png/pdf).

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(scales)
})

source("R/figure_utils.R")

CASCADE_ORDER <- c("full_cohort", "cancer_type", "text", "treatment", "stage",
                   "prs", "somatic", "metburden", "all")
CASCADE_FILL  <- c(full_cohort = "#5B8DB8", text = MODALITY_COLORS[["text"]],
                   cancer_type = "#4E79A7",
                   stage = MODALITY_COLORS[["stage"]], treatment = MODALITY_COLORS[["treatment"]],
                   somatic = MODALITY_COLORS[["somatic"]], prs = MODALITY_COLORS[["prs"]],
                   # metburden usually collapses into the terminal "all" bar via the
                   # dedup filter below, but it survives whenever its count differs
                   # from the final cohort's. As a filled bar (unlike the former
                   # geom_label) a missing key would render grey, so key it here.
                   metburden = MODALITY_COLORS[["metburden"]],
                   all = "#333333")

# ============================================================================
# fig0a: data-availability cascade
# ============================================================================
build_fig0a <- function() {
  d <- load_figure_data("fig0_data_availability.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig0_data_availability.csv empty"))
  d <- d %>%
    mutate(stage = factor(stage, levels = CASCADE_ORDER)) %>%
    arrange(stage) %>%
    # The final row repeats the fully cumulative last modality row; retain one
    # terminal bar and avoid a visually duplicated count.
    filter(!(as.character(stage) == "metburden" &
             n_patients == lead(n_patients, default = -1))) %>%
    mutate(pct = 100 * n_patients / n_total)

  # Bars read top-to-bottom in cascade order, so the y factor is reversed: the
  # eligible cohort sits at the top and the complete-case subset at the bottom.
  d <- d %>% mutate(stage = factor(as.character(stage),
                                   levels = rev(as.character(stage))))

  ggplot(d, aes(x = n_patients, y = stage, fill = as.character(stage))) +
    geom_col(width = 0.62, color = "white") +
    geom_text(aes(label = sprintf("%s (%.0f%%)", scales::comma(n_patients), pct)),
              hjust = -0.1, size = MANUSCRIPT_TEXT_SIZE) +
    scale_fill_manual(values = CASCADE_FILL, guide = "none") +
    scale_y_discrete(labels = setNames(d$label, as.character(d$stage))) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18)), labels = scales::comma) +
    labs(x = "Number of patients", y = NULL,
         title = "Cohort Eligibility and Data Availability",
         subtitle = "Counts are cumulative; each bar is a subset of the bar above it") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          plot.subtitle = element_text(color = "#555555"))
}


# ============================================================================
# Compose
# ============================================================================
p0a <- build_fig0a()

save_panel(p0a, "fig0a", group = "figure0", width = 8.6, height = 5.4)
