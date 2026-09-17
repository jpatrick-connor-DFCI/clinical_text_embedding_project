# Captions are versioned with the renderer; exact counts remain data-derived.
figure_caption <- function(number, panels, missing_labels = character()) {
  path <- file.path("figures", "manuscript_captions.json")
  spec <- jsonlite::fromJSON(path, simplifyVector = FALSE)[[as.character(number)]]
  if (is.null(spec)) return(NULL)
  text <- paste0("Figure ", number, ". ", spec$title, "\n\n", spec$text)
  details <- unlist(lapply(panels, function(p) attr(p, "caption_detail", exact = TRUE)),
                    use.names = FALSE)
  if (length(details)) text <- paste(text, paste(details, collapse = " "), sep = "\n\n")
  if (number %in% 1:3 && isTRUE(FILTER_UNDERPERFORMING_ENDPOINTS)) {
    text <- paste(text, sprintf(paste0("Endpoint analyses use the configured two-sided ",
      "outlier filter: text-minus-base C-index differences outside the mean +/- %s ",
      "standard deviations are excluded."), EVENT_EXCLUSION_SD), sep = "\n\n")
  } else if (number %in% 1:3) {
    text <- paste(text, "The endpoint outlier filter is disabled.", sep = "\n\n")
  }
  if (length(missing_labels)) text <- paste(text,
    paste("Incomplete figure; unavailable panels:", paste(missing_labels, collapse = ", ")),
    sep = "\n\n")
  text
}

save_captioned_figure <- function(plot, caption, name, group, width, height) {
  if (is.null(caption)) return(invisible(NULL))
  dir <- file.path(FIGURE_OUT_DIR, "captions")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  writeLines(caption, file.path(dir, paste0(group, ".txt")), useBytes = TRUE)
  writeLines(caption, file.path(dir, paste0(group, ".md")), useBytes = TRUE)
  # Review copies carry the caption below the art; the uncaptioned vector PDF is
  # retained for journals that require legends as a separate manuscript section.
  caption_size <- 9
  wrapped <- stringr::str_wrap(caption, width = max(60L, floor(width * 16 * 7.5 / caption_size)))
  n_lines <- length(strsplit(wrapped, "\n", fixed = TRUE)[[1]])
  caption_height <- n_lines * caption_size * 1.15 / 72 + 0.20
  caption_grob <- grid::textGrob(wrapped, x = grid::unit(2, "mm"),
                                y = grid::unit(1, "npc") - grid::unit(1, "mm"),
                                just = c("left", "top"),
                                gp = grid::gpar(fontsize = caption_size, fontfamily = "sans", lineheight = 1.15))
  review <- patchwork::wrap_plots(
    patchwork::wrap_elements(full = cowplot::as_grob(plot)),
    patchwork::wrap_elements(full = caption_grob),
    ncol = 1, heights = c(height, caption_height))
  save_panel(review, paste0(name, "_captioned"), group, width, height + caption_height, dpi = 600)
}
