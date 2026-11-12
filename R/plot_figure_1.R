# Figure 1: the complete clinical-text workflow (representation, models, analyses).
# Supplementary Figure 1: a eligible oncology cohort, b outcome endpoints.
# Supplementary Figure 21: a cancer-type composition, b stage distribution.
# Individual panels and compiled PNG/PDF with panel letters and no plot titles.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggsignif); library(cowplot)
})

source("/data/gusev/USERS/jpconnor/code/clinical_text_embedding_project/R/figure_utils.R")
source(file.path(REPO_ROOT, "R", "publication_style.R"))


# ---------------- helpers local to Fig 1 ----------------
# Accepts either Roman (post-_normalize_stage in prep_figure_1.py) or arabic
# (legacy one-hot path); collapses substages and float repr to a major stage.
stage_label <- function(x) {
  x_str   <- as.character(x)
  x_clean <- sub("\\.0+$", "", toupper(x_str))
  x_clean <- sub("^STAGE\\s*", "", x_clean)
  x_clean <- sub("[A-D]$", "", x_clean)                       # IIIA -> III
  roman   <- c("1" = "I", "2" = "II", "3" = "III", "4" = "IV",
               "I" = "I", "II" = "II", "III" = "III", "IV" = "IV")
  ifelse(x_clean %in% names(roman), paste("Stage", roman[x_clean]), x_str)
}


# ============================================================================
# figS1a: cumulative cohort flow (cancer-type availability explicit)
# ============================================================================
build_cohort_flow <- function() {
  d <- load_figure_data("fig0_data_availability.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig0_data_availability.csv empty"))
  order <- c("full_cohort", "cancer_type", "text", "treatment", "stage",
             "prs", "somatic", "metburden", "all")
  d <- d %>% mutate(stage = factor(stage, levels = order)) %>% arrange(stage)
  if (anyNA(d$stage) || anyDuplicated(d$stage) ||
      any(!is.finite(d$n_patients)) || any(d$n_patients < 0) ||
      any(diff(d$n_patients) > 0)) {
    stop("Supplementary Figure 1a requires unique, ordered cumulative cohort counts")
  }
  # Keep named eligibility steps even when no patients are lost. Only collapse
  # the last modality if it duplicates the explicitly named final cohort.
  d <- d %>%
    filter(!(as.character(stage) == "metburden" &
               n_patients == lead(n_patients, default = -1))) %>%
    mutate(ycen = rev(seq_len(n())),
           text = sprintf("%s\nn = %s", label, comma(n_patients)),
           xmin = 0.07, xmax = 0.93, ymin = ycen - 0.31, ymax = ycen + 0.31)
  arrows <- d %>% mutate(yend = lead(ymax)) %>% filter(!is.na(yend))

  # Match PROFILE-testing's render_consort_panel: uniform pale rectangles,
  # centered criterion/count labels, and slate arrows between box edges.
  ggplot(d) +
    geom_rect(aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = "#eef1f5", color = "#5d6d7e", linewidth = 0.4) +
    geom_text(aes(x = 0.5, y = ycen, label = text),
              size = MANUSCRIPT_SMALL_TEXT_SIZE, lineheight = 0.9) +
    geom_segment(data = arrows,
                 aes(x = 0.5, xend = 0.5, y = ymin, yend = yend),
                 arrow = arrow(length = unit(0.12, "cm"), type = "closed"),
                 color = "#5d6d7e", linewidth = 0.45) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0.5, nrow(d) + 0.5), expand = FALSE) +
    labs(title = "Cohort availability") +
    theme_void(base_size = MANUSCRIPT_BASE_SIZE) +
    theme(plot.title = element_text(face = "bold", size = 11, hjust = 0),
          plot.margin = margin(3, 4, 3, 3))
}


# ============================================================================
# figS1b: endpoint counts per scheme
# ============================================================================
# Counts per scheme, recounted from the per-event full-cohort metrics after the
# same exclusion the rest of the manuscript applies. fig1_endpoint_counts.csv is
# pre-aggregated in the Python tier (scheme -> n_endpoints, no event column), so
# it cannot be trimmed in place -- reporting it unchanged would have this panel
# claim more endpoints than any downstream figure actually shows.
#
# Note the two files are not interchangeable: _endpoint_counts() counts every
# trained event, while _full_cohort_metrics() skips events whose result files are
# missing. Recounting therefore also drops those unusable events, which is the
# intended reading of "endpoints this manuscript reports".
trimmed_endpoint_counts <- function() {
  m <- load_figure_data("fig2_full_cohort_metrics.csv")
  if (nrow(m) == 0 || !all(c("scheme", "event") %in% names(m))) {
    message("[figS1b] fig2_full_cohort_metrics.csv unusable (absent, empty, or no ",
            "scheme/event columns) -- cannot trim")
    return(tibble::tibble())
  }
  # The exclusion is judged on the active metric's columns; without them
  # excluded_event_keys() drops nothing and the panel would silently report
  # untrimmed counts, so say so rather than looking like a successful trim.
  needed <- paste0(c("base_", "text_"), metric_suffix(METRIC))
  if (!all(needed %in% names(m))) {
    message(sprintf("[figS1b] fig2_full_cohort_metrics.csv lacks %s -- cannot trim on %s",
                    paste(setdiff(needed, names(m)), collapse = "/"), metric_label(METRIC)))
    return(tibble::tibble())
  }
  n_before <- nrow(dplyr::distinct(m, scheme, event))
  m <- drop_excluded_events(m, excluded_event_keys(m))
  if (nrow(m) == 0) return(tibble::tibble())
  out <- m %>%
    distinct(scheme, event) %>%
    count(scheme, name = "n_endpoints")
  message(sprintf("[figS1b] endpoints trimmed on %s: %d -> %d",
                  metric_label(METRIC), n_before, sum(out$n_endpoints)))
  out
}

build_endpoint_counts <- function() {
  d <- trimmed_endpoint_counts()
  untrimmed <- nrow(d) == 0
  if (nrow(d) == 0) {
    # Fallback: no per-event metrics available, so report the untrimmed counts
    # rather than blanking the panel.
    message("[figS1b] FALLING BACK to untrimmed fig1_endpoint_counts.csv")
    d <- load_figure_data("fig1_endpoint_counts.csv")
  }
  if (nrow(d) == 0) return(placeholder_panel("fig1_endpoint_counts.csv empty"))
  d <- d %>%
    mutate(scheme = factor(scheme, levels = names(SCHEME_LABELS))) %>%
    arrange(scheme) %>%
    mutate(scheme_lbl = SCHEME_LABELS[as.character(scheme)],
           scheme_lbl = factor(scheme_lbl, levels = scheme_lbl))
  p <- ggplot(d, aes(x = scheme_lbl, y = n_endpoints, fill = as.character(scheme))) +
    geom_col(width = 0.62, color = "white") +
    geom_text(aes(label = scales::comma(n_endpoints)),
              vjust = -0.3, size = MANUSCRIPT_TEXT_SIZE) +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                       labels = scales::comma) +
    scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 10)) +
    labs(x = NULL, y = "Endpoints", title = "Outcome endpoints") +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"),
          axis.text.x = element_text(hjust = 0.5, size = 9, face = "plain"))
  if (untrimmed) attr(p, "caption_detail") <- paste(
    "Panel b uses unfiltered aggregate endpoint counts because per-endpoint",
    "results were unavailable; the endpoint filter was not applied to this panel.")
  p
}


# ============================================================================
# fig1: full clinical-text workflow, rendered without patient data
# ============================================================================
build_clinical_text_workflow <- function() {
  path <- file.path(REPO_ROOT, "figures", "workflow", "clinical_text_workflow.png")
  if (!file.exists(path)) {
    return(placeholder_panel("figures/workflow/clinical_text_workflow.png unavailable"))
  }
  if (!requireNamespace("png", quietly = TRUE)) {
    return(placeholder_panel("R package 'png' is required for the workflow review copy"))
  }
  art <- png::readPNG(path)
  # The full workflow already contains its own a/b/c labels. Preserve its native
  # aspect ratio and do not wrap it in the compiled-figure panel-label machinery.
  ggplot() +
    annotation_custom(grid::rasterGrob(art, interpolate = FALSE),
                      xmin = 0, xmax = 1, ymin = 0, ymax = 1) +
    scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
    coord_fixed(ratio = dim(art)[1] / dim(art)[2], expand = FALSE) +
    theme_void() + theme(plot.margin = margin(0, 0, 0, 0))
}

save_clinical_text_workflow <- function() {
  name <- paste0("figure1", metric_tag())
  workflow_dir <- file.path(REPO_ROOT, "figures", "workflow")
  sources <- file.path(workflow_dir, paste0("clinical_text_workflow.", c("png", "pdf")))
  if (!all(file.exists(sources))) {
    stop("Full workflow artwork is missing; run python3 figures/render_workflow.py")
  }
  dirs <- c(file.path(PNG_OUT_DIR, "figure1"), file.path(PDF_OUT_DIR, "figure1"))
  for (dir in dirs) dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  outputs <- file.path(dirs, paste0(name, c(".png", ".pdf")))
  # Keep the submission PDF vector-based and retain the original 600-dpi PNG.
  if (!all(file.copy(sources, outputs, overwrite = TRUE))) {
    stop("Could not export the full clinical-text workflow")
  }
  message(sprintf("[figure1] %s / %s", outputs[1], outputs[2]))
  review <- build_clinical_text_workflow()
  if (!is_skipped_panel(review)) {
    save_captioned_figure(review, figure_caption(1, list(a = review)), name,
                          "figure1", width = 180 / 25.4, height = 166.5 / 25.4)
  } else {
    unlink(c(file.path(dirs, paste0(name, "_captioned", c(".png", ".pdf"))),
             file.path(FIGURE_OUT_DIR, "captions", paste0("figure1.", c("md", "txt")))))
    message("[figure1] Captioned review unavailable: ", skip_reason(review))
  }
  invisible(outputs)
}


# ============================================================================
# figS21a: cancer-type pie of cohort composition
# ============================================================================
build_cancer_pie <- function() {
  d <- load_figure_data("fig1_cancer_type_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_cancer_type_counts.csv empty"))
  # prep_figure_1.py already returns the top-10 types + a pooled "Other" row that
  # sums to the full cohort. Group again after display-label normalization so a
  # stale CSV containing both "OTHER" and "Other" still renders one pooled slice.
  d <- d %>%
    mutate(category = stringr::str_to_title(gsub("_", " ", as.character(category)))) %>%
    group_by(category) %>%
    summarise(n = sum(n), .groups = "drop")
  total <- sum(d$n)
  d <- d %>%
    arrange(desc(n)) %>%
    mutate(pct = 100 * n / total,
           label = sprintf("%s (%s)", category, scales::comma(n)),
           label = factor(label, levels = label))
  pal <- grDevices::hcl.colors(nrow(d), "Set 3")

  ggplot(d, aes(x = 1, y = n, fill = label)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = pi / 2, direction = -1) +
    geom_text(aes(x = 1.35, label = ifelse(pct >= 3, sprintf("%.1f%%", pct), "")),
              position = position_stack(vjust = 0.5),
              size = 2.8, color = "#222222") +
    scale_fill_manual(values = pal, name = NULL) +
    labs(title = "Cancer types", subtitle = sprintf("Patients with text; N = %s", comma(total))) +
    guides(fill = guide_legend(ncol = 2, byrow = TRUE)) +
    theme_void(base_size = MANUSCRIPT_BASE_SIZE) +
    theme(plot.title = element_text(size = 11, face = "bold", hjust = 0),
          plot.subtitle = element_text(size = 9, hjust = 0),
          legend.position = "bottom",
          legend.text = element_text(size = 8.5, margin = margin(l = 3, r = 8)),
          legend.key.size = unit(3.5, "mm"), legend.spacing.y = unit(0, "mm"),
          legend.margin = margin(0, 0, 0, 0), plot.margin = margin(3, 4, 3, 3))
}


# ============================================================================
# figS21b: cancer stage breakdown (with stage label fix: 2.0 -> Stage II)
# ============================================================================
build_stage_counts <- function() {
  d <- load_figure_data("fig1_stage_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_stage_counts.csv empty"))
  d <- d %>%
    mutate(label = stage_label(category)) %>%
    mutate(label = factor(label, levels = paste("Stage", c("I", "II", "III", "IV")))) %>%
    arrange(label)
  stage_n <- sum(d$n, na.rm = TRUE)
  ggplot(d, aes(x = label, y = n)) +
    geom_col(fill = "#5B8DB8", color = "white", width = 0.65) +
    geom_text(aes(label = scales::comma(n)), vjust = -0.3,
              size = MANUSCRIPT_TEXT_SIZE) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                       labels = scales::comma) +
    labs(x = NULL, y = "Patients", title = "Cancer stage",
         subtitle = sprintf("Known stage; N = %s", comma(stage_n))) +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# Compose
# ============================================================================
retire_main_panels(1, character())

pS21a <- build_cancer_pie()
pS21b <- build_stage_counts()
pS1a <- build_cohort_flow()
pS1b <- build_endpoint_counts()

.tag <- metric_tag()
save_clinical_text_workflow()
save_panel(pS21a, paste0("figS21a_cancer_types", .tag), group = "figureS21",
           width = 5, height = 4.5, dpi = 600)
save_panel(pS21b, paste0("figS21b_stage_counts", .tag), group = "figureS21",
           width = 5, height = 4.5, dpi = 600)
save_compiled_figure(
  list(a = pS21a, b = pS21b), number = "S21",
  # Retain the original five-inch cohort panels: the pie's percentage labels
  # collide when the same legend is rendered in a 3.5-inch-wide panel.
  width = 10, height = 4.5, name = paste0("figS21_cohort_composition", .tag)
)

save_panel(pS1a, paste0("figS1a_cohort_flow", .tag), group = "figureS1",
           width = 3.5, height = 4.5, dpi = 600)
save_panel(pS1b, paste0("figS1b_endpoint_counts", .tag), group = "figureS1",
           width = 3.5, height = 4.5, dpi = 600)
save_compiled_figure(
  list(a = pS1a, b = pS1b), number = "S1",
  width = 7.1, height = 4.5, name = paste0("figS1_cohort_endpoints", .tag)
)
