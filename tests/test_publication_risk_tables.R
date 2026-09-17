# Run from the repository root: Rscript tests/test_publication_risk_tables.R
# Verify risk-set counts and stratum mapping, including landmark delayed entry.
suppressPackageStartupMessages({
  library(dplyr)
  library(survival)
})
MODALITY_DISPLAY <- c(metburden = "Metastatic burden")
source("R/publication_style.R")

d <- data.frame(time = c(2, 4, 6, 3, 5, 7),
                event = c(1, 0, 1, 1, 0, 1),
                group = rep(c("A", "B"), each = 3), entry = 1)
labels <- c(A = "First group", B = "Second group")
check_counts <- function(fit) {
  counts <- manuscript_risk_counts(fit, c(1, 3, 5), labels)
  stopifnot(identical(counts$stratum, rep(c("A", "B"), each = 3)),
            identical(counts$time, rep(c(1, 3, 5), 2)),
            all(counts$n_risk == c(3, 2, 1, 3, 3, 2)),
            identical(levels(counts$row), c("B", "A")))
}
check_counts(survfit(Surv(time, event) ~ group, data = d))
check_counts(survfit(Surv(entry, time, event) ~ group, data = d))

fit <- survfit(Surv(time, event) ~ 1, data = subset(d, group == "A"))
counts <- manuscript_risk_counts(fit, c(1, 3, 5), c(A = "First group"))
stopifnot(all(counts$n_risk == c(3, 2, 1)), all(counts$stratum == "A"))
bad <- try(manuscript_risk_counts(fit, c(1, 3, 5), labels), silent = TRUE)
stopifnot(inherits(bad, "try-error"))
message("Publication risk-table tests passed")
