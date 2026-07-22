# Render Figure 0 (cohort data-availability cascade) in ggplot2.
#
# Single panel: CONSORT-style attrition bar showing how many patients remain
# as each modality's availability requirement is added, ending in the subset
# that passes every threshold at once (the multi-modal analysis cohort used
# from Figure 3 onward).
# Output: target_figures/figure0_data_availability.png plus one panel PNG.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(scales)
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

CASCADE_ORDER <- c("full_cohort", "text", "stage", "treatment", "somatic", "prs", "all")
CASCADE_FILL  <- c(full_cohort = "#5B8DB8", text = MODALITY_COLORS[["text"]],
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
    arrange(n_patients) %>%
    mutate(stage = factor(stage, levels = stage),
           pct = 100 * n_patients / n_total)

  ggplot(d, aes(x = n_patients, y = stage, fill = as.character(stage))) +
    geom_col(width = 0.62, color = "white") +
    geom_text(aes(label = sprintf("%s (%.0f%%)", scales::comma(n_patients), pct)),
              hjust = -0.1, size = 3) +
    scale_fill_manual(values = CASCADE_FILL, guide = "none") +
    scale_y_discrete(labels = setNames(d$label, d$stage)) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.18)), labels = scales::comma) +
    labs(x = "Number of patients", y = NULL,
         title = "Cohort Data Availability") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          plot.caption = element_text(size = 6.5, hjust = 1,
                                      face = "italic", color = "#666666"))
}


# ============================================================================
# Compose
# ============================================================================
p0a <- build_fig0a()

save_panel(p0a, "fig0a", width = 8.0, height = 5.2)

save_figure(p0a, "figure0_data_availability", width = 9.0, height = 5.6)
