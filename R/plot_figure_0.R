# Render Figure 0 (cohort data-availability cascade) in ggplot2.
#
# Single panel: CONSORT-style attrition bar showing how many patients remain
# as each modality's availability requirement is added, ending in the subset
# that passes every threshold at once (the multi-modal analysis cohort used
# from Figure 3 onward).
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
    # terminal box and avoid a visually duplicated count.
    filter(!(as.character(stage) == "metburden" &
             n_patients == lead(n_patients, default = -1))) %>%
    mutate(y = rev(seq_len(n())),
           previous_n = lag(n_patients, default = first(n_patients)),
           retained = 100 * n_patients / previous_n,
           total_pct = 100 * n_patients / first(n_patients),
           box_label = sprintf("%s\n%s patients · %.1f%% of start",
                               label, scales::comma(n_patients), total_pct),
           arrow_label = ifelse(row_number() == 1, "",
                                sprintf("%.1f%% retained", retained)))

  ggplot(d) +
    geom_segment(data = d %>% filter(row_number() > 1),
                 aes(x = 0.5, xend = 0.5, y = y + 0.72, yend = y + 0.32),
                 arrow = arrow(length = unit(0.08, "inches"), type = "closed"),
                 linewidth = 0.5, color = "#555555") +
    geom_label(aes(x = 0.5, y = y, label = box_label,
                   fill = as.character(stage)),
               label.size = 0.4, label.padding = unit(0.22, "lines"),
               size = MANUSCRIPT_TEXT_SIZE, fontface = "bold", lineheight = 0.95) +
    geom_text(data = d %>% filter(row_number() > 1),
              aes(x = 0.73, y = y + 0.51, label = arrow_label),
              hjust = 0, size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "#555555") +
    scale_fill_manual(values = CASCADE_FILL, guide = "none") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0.5, max(d$y) + 0.5), clip = "off") +
    labs(title = "Cohort Eligibility and Data Availability",
         subtitle = "Counts are cumulative; each box is a subset of the preceding box") +
    theme_void(base_size = MANUSCRIPT_BASE_SIZE) +
    theme(plot.title = element_text(face = "bold", size = MANUSCRIPT_BASE_SIZE + 1),
          plot.subtitle = element_text(color = "#555555"))
}


# ============================================================================
# Compose
# ============================================================================
p0a <- build_fig0a()

save_panel(p0a, "fig0a", group = "figure0", width = 9.5, height = 6.2)
