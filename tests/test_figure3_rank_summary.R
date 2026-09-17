# Rscript tests/test_figure3_rank_summary.R (from the repository root).
suppressPackageStartupMessages(library(dplyr))

# Load the production helper without rendering figures or reading patient data.
for (expr in parse("R/plot_figure_3.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      identical(expr[[2]], as.name("avg_rank_from_long"))) eval(expr)
}
stopifnot(exists("avg_rank_from_long", mode = "function"))

# Endpoint identity includes scheme: the same event label in different schemes
# must count separately. These are ranks already selected for the figure.
ranks <- tibble(
  scheme = rep(c("death_met", "icd3_post"), each = 4),
  event = rep(rep(c("event1", "event2"), each = 2), 2),
  modality = rep(c("text", "stage"), 4),
  rank = c(1, 2, 2, 1, 1, 2, 1, 2)
)
summary <- avg_rank_from_long(ranks)
text <- filter(summary, modality == "text")
stopifnot(nrow(summary) == 2L, all(summary$n_events == 4L),
          text$mean_rank == 1.25, text$q25_rank == 1, text$q75_rank == 1.25)

# Excluded endpoints must not leak back into the average or the reported N.
restricted <- avg_rank_from_long(filter(ranks, scheme == "icd3_post"))
stopifnot(all(restricted$n_events == 2L),
          restricted$mean_rank[restricted$modality == "text"] == 1)
single <- avg_rank_from_long(filter(ranks, scheme == "death_met", event == "event2"))
stopifnot(all(single$n_events == 1L), all(single$q25_rank == single$q75_rank))
stopifnot(nrow(avg_rank_from_long(NULL)) == 0L,
          nrow(avg_rank_from_long(tibble())) == 0L,
          nrow(avg_rank_from_long(select(ranks, -event))) == 0L)
message("Figure 3 rank-summary tests passed")
