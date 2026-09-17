# Run from the repository root with Rscript tests/test_figure_composition.R.
# Exercises real PNG/PDF composition using synthetic plots, without patient data.
suppressPackageStartupMessages(library(ggplot2))

# Load only the composition helpers so this test does not need survival packages
# or trigger cleanup in the configured manuscript output directory.
helpers <- c("metric_suffix", "metric_tag", "save_panel", "save_compiled_figure",
             "placeholder_panel", "is_skipped_panel", "skip_reason")
for (expr in parse("R/figure_utils.R")) {
  if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
      as.character(expr[[2]]) %in% helpers) eval(expr)
}
METRIC <- "cindex"
FILTER_UNDERPERFORMING_ENDPOINTS <- FALSE
source("R/figure_captions.R")

run_tests <- function() {
  output <- tempfile("figure-composition-")
  dir.create(output)
  on.exit(unlink(output, recursive = TRUE))
  FIGURE_OUT_DIR <<- output
  PNG_OUT_DIR <<- file.path(output, "png")
  PDF_OUT_DIR <<- file.path(output, "pdf")

  # Capture the actual composed plot while still writing both file formats.
  original_save <- save_panel
  captured <- NULL
  save_panel <<- function(plot, ...) {
    captured <<- plot
    original_save(plot, ...)
  }
  on.exit(assign("save_panel", original_save, envir = .GlobalEnv), add = TRUE)

  text_labels <- function(grob) {
    own <- if (inherits(grob, "text")) as.character(grob$label) else character()
    c(own, unlist(lapply(c(grob$grobs, as.list(grob$children)), text_labels)))
  }
  panels <- setNames(lapply(letters[1:13], function(label) {
    ggplot(data.frame(x = 1:3, y = 1:3), aes(x, y)) + geom_point() +
      labs(title = paste("Panel", label))
  }), letters[1:13])
  # Include a nested panel: it should still receive just its outer label.
  panels$h <- patchwork::wrap_plots(panels$a, panels$b)
  design <- "abc\ndde\nfgg\nhhh\nijk\nlmm"

  panels[c("c", "d", "m")] <- list(
    placeholder_panel("cancer C-index unavailable"),
    placeholder_panel("treatment C-index unavailable"),
    placeholder_panel("no eligible phecode event")
  )
  messages <- capture.output(
    files <- save_compiled_figure(panels, 2, width = 9, height = 12, design = design),
    type = "message"
  )
  stopifnot(length(files) == 2L, all(file.info(files)$size > 0))
  labels <- text_labels(patchwork::patchworkGrob(captured))
  retained <- setdiff(letters[1:13], c("c", "d", "m"))
  stopifnot(all(vapply(retained, function(label) sum(labels == label) == 1L, logical(1))),
            !any(c("c", "d", "m") %in% labels),
            "Unavailable panels: c, d, m" %in% labels,
            any(grepl("cancer C-index unavailable", messages)),
            any(grepl("no eligible phecode event", messages)))

  # A completely unavailable figure removes its stale output and returns without
  # throwing, allowing the next figure in the notebook to run.
  skipped <- setNames(rep(list(placeholder_panel("no data")), 13), letters[1:13])
  result <- save_compiled_figure(skipped, 2, width = 9, height = 12, design = design)
  stopifnot(length(result) == 0L, !any(file.exists(files)))
  # Figure 3's spanning lower row previously lost labels b and c.
  files <- save_compiled_figure(list(a = panels$a, b = panels$b, c = panels$h),
                               3, width = 7.09, height = 5.8, design = "ab\ncc")
  labels <- text_labels(patchwork::patchworkGrob(captured))
  stopifnot(length(files) == 2L, all(file.info(files)$size > 0),
            all(vapply(c("a", "b", "c"), function(x) sum(labels == x) == 1L, logical(1))),
            !any(grepl("^Unavailable panels:", labels)),
            file.exists(file.path(FIGURE_OUT_DIR, "captions", "figure3.md")),
            file.exists(file.path(PDF_OUT_DIR, "figure3", "figure3_cindex_captioned.pdf")))

  # NULL is an implementation error, not an expected absence of data.
  result <- try(save_compiled_figure(list(a = panels$a, b = NULL), 4, 8, 4), silent = TRUE)
  stopifnot(inherits(result, "try-error"))
  message("Figure composition tests passed")
}
run_tests()
