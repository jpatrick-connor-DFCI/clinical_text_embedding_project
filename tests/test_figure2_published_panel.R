# Rscript tests/test_figure2_published_panel.R (from the repository root)
# Validate main/supplement score selection and a missing combined-model result.
suppressPackageStartupMessages({library(ggplot2); library(dplyr); library(tidyr)})
REPO_ROOT <- normalizePath(".")
MODEL_COLORS <- unlist(jsonlite::fromJSON(file.path(REPO_ROOT, "shared", "palette.json"))$MODEL_COLORS)
theme_combined <- function() theme_classic()
placeholder_panel <- function(reason) structure(list(reason = reason), class = "skipped_panel")
source(file.path(REPO_ROOT, "R", "published_score_utils.R"))

cohort <- tibble(
  score = c(rev(names(PS_MAIN_SCORES)), "imdc_noecog", "mskcc_noecog", "ipi_noecog", "capra_mod", "imdc"),
  source = c(rep("calculated", 14), "regex"),
  anchor = "treatment", lab_window_days = 30, stratum = "all",
  n_patients = c(rep(100L, 14), 34L), n_events = 50L, n_eligible = 150L, n_complete = 120L
)
main <- ps_rows(cohort, scores = PS_MAIN_SCORES)
stopifnot(identical(main$score, names(PS_MAIN_SCORES)), nrow(main) == 10L,
          all(c("imdc_noecog", "mskcc_noecog", "ipi_noecog", "capra_mod") %in% ps_rows(cohort)$score),
          grepl("IMDC (with ECOG)", main$row[main$score == "imdc"], fixed = TRUE))
secondary <- mutate(cohort, anchor = "sequencing")
stopifnot(nrow(ps_rows(secondary, PS_MAIN_SCORES)) == 0L)

cindex <- tidyr::crossing(select(cohort, -n_eligible, -n_complete), model = PS_MODELS) %>%
  mutate(status = "ok", cindex = 0.72, ci_lower = 0.68, ci_upper = 0.76) %>%
  filter(!(score == "albi" & model == "published+text"))
plot <- build_published_cindex(cindex, main, disclose_missing = TRUE)
stopifnot(nrow(plot$data) == 29L,
          identical(levels(plot$data$row), rev(main$row)),
          identical(levels(plot$data$model), PS_MODELS),
          all(plot$data$cindex == 0.72), all(plot$data$ci_lower == 0.68),
          identical(plot$labels$caption, "Unavailable estimates: ALBI (calculated), Published + text."))
stopifnot(length(ggplot_build(plot)$data) == 3L,
          inherits(build_published_cindex(tibble(), main), "skipped_panel"))
message("Figure 2 published-score tests passed")
