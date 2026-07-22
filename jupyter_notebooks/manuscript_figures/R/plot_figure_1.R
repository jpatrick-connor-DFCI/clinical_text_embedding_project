# Render Figure 1 (cohort & embedding overview) in ggplot2 + patchwork.
#
# Panels: A pipeline schematic, B endpoint counts, C notes/patient by type,
#         D cancer-type pie, E stage breakdown, F first-line treatment breakdown.
# Output: panels fig1a–fig1f (png/pdf).

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggsignif); library(cowplot)
})

script_dir <- local({
  # Notebook path: render_figures.ipynb sets R_DIR in globalenv before sys.source-ing us.
  if (exists("R_DIR", envir = globalenv(), inherits = FALSE)) {
    return(get("R_DIR", envir = globalenv()))
  }
  # Rscript path: --file=... on the command line.
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
  if (length(fa) && nzchar(fa[1])) return(dirname(normalizePath(fa[1])))
  # source() path: ofile lives on the call stack.
  for (n in seq_len(sys.nframe())) {
    ofile <- sys.frame(n)$ofile
    if (!is.null(ofile) && nzchar(ofile)) return(dirname(normalizePath(ofile)))
  }
  stop("Could not determine script directory (set R_DIR in globalenv, run via Rscript, or source() directly)")
})
source(file.path(script_dir, "figure_utils.R"))


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
# fig1a: pipeline schematic (cowplot::ggdraw + grid primitives)
# ============================================================================
build_fig1a <- function() {
  # Layout in [0,1] x [0,1] axes coords.
  boxes <- tibble::tribble(
    ~x,   ~y,    ~w,    ~h,   ~fill,     ~label,
    0.02, 0.72, 0.16, 0.13, "#D6E4F0", "Clinician\nNotes",
    0.02, 0.46, 0.16, 0.13, "#FCE5CD", "Imaging\nReports",
    0.02, 0.20, 0.16, 0.13, "#FFD8A8", "Pathology\nReports",
    0.31, 0.40, 0.20, 0.30, "#9CCBE8", "Clinical-Longformer",
    0.66, 0.66, 0.30, 0.18, "#E6F1FA", "Survival\nPrediction",
    0.66, 0.16, 0.30, 0.18, "#E8F5E9", "ICI Biomarker\nDiscovery"
  )
  arrows <- tibble::tribble(
    ~x,   ~y,    ~xend, ~yend,
    0.18, 0.785, 0.31,  0.55,   # clinician -> longformer
    0.18, 0.525, 0.31,  0.55,   # imaging -> longformer
    0.18, 0.265, 0.31,  0.55,   # pathology -> longformer
    0.51, 0.55,  0.66,  0.75,   # longformer -> survival
    0.51, 0.55,  0.66,  0.25    # longformer -> biomarker
  )
  # Embedding-cluster dots inside the Longformer box (suggesting 768-dim embedding)
  set.seed(0)
  dots <- tibble::tibble(
    x = stats::runif(60, 0.34, 0.48),
    y = stats::runif(60, 0.44, 0.66),
    g = sample(LETTERS[1:5], 60, replace = TRUE)
  )

  ggplot() +
    geom_rect(data = boxes,
              aes(xmin = x, xmax = x + w, ymin = y, ymax = y + h),
              fill = boxes$fill, color = "#3B3B3B", linewidth = 0.5) +
    geom_text(data = boxes,
              aes(x = x + w / 2, y = y + h / 2, label = label),
              size = 3.4, fontface = "bold", lineheight = 0.95) +
    geom_segment(data = arrows,
                 aes(x = x, y = y, xend = xend, yend = yend),
                 arrow = arrow(length = unit(0.12, "inches"), type = "closed"),
                 linewidth = 0.55, color = "#3B3B3B") +
    geom_point(data = dots, aes(x = x, y = y, color = g),
               size = 1.3, alpha = 0.85, show.legend = FALSE) +
    annotate("text", x = 0.41, y = 0.36, label = "768-dim patient embeddings",
             size = 2.8, fontface = "italic", color = "#444444") +
    scale_color_manual(values = unname(MODALITY_COLORS)) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    theme_void()
}


# ============================================================================
# fig1b: endpoint counts per scheme
# ============================================================================
build_fig1b <- function() {
  d <- load_figure_data("fig1_endpoint_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_endpoint_counts.csv empty"))
  d <- d %>%
    mutate(scheme = factor(scheme, levels = names(SCHEME_LABELS))) %>%
    arrange(scheme) %>%
    mutate(scheme_lbl = SCHEME_LABELS[as.character(scheme)],
           scheme_lbl = factor(scheme_lbl, levels = scheme_lbl))
  ggplot(d, aes(x = n_endpoints, y = scheme_lbl, fill = as.character(scheme))) +
    geom_col(width = 0.62, color = "white") +
    geom_text(aes(label = scales::comma(n_endpoints)),
              hjust = -0.15, size = 3) +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
    labs(x = "Number of endpoints", y = NULL, title = "Outcome Endpoints") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          axis.line.y = element_blank(), axis.ticks.y = element_blank())
}


# ============================================================================
# fig1c: notes per patient by type (horizontal boxplot, log scale)
# ============================================================================
build_fig1c <- function() {
  d <- load_figure_data("fig1_notes_per_patient.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_notes_per_patient.csv empty"))
  d <- d %>%
    mutate(note_type = stringr::str_to_title(gsub("_", " ", as.character(note_type))))
  ord <- d %>% group_by(note_type) %>% summarise(m = median(n_notes)) %>%
    arrange(m) %>% pull(note_type)
  d <- d %>% mutate(note_type = factor(note_type, levels = ord))

  ggplot(d, aes(x = n_notes, y = note_type, fill = note_type)) +
    geom_boxplot(outlier.shape = NA, width = 0.55, color = "#333333", alpha = 0.7) +
    scale_x_log10(labels = scales::comma) +
    scale_fill_manual(values = unname(grDevices::hcl.colors(length(ord), "Set 2")),
                      guide = "none") +
    labs(x = "Notes per patient (among patients with ≥1 of type, log scale)",
         y = NULL, title = "Notes per Patient by Type") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"))
}


# ============================================================================
# fig1d: cancer-type pie of cohort composition
# ============================================================================
build_fig1d <- function() {
  d <- load_figure_data("fig1_cancer_type_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_cancer_type_counts.csv empty"))
  # prep_figure_1.py already returns the top-10 types + a pooled "Other" row that
  # sums to the full cohort, so the total below matches the cohort N in Fig 0.
  total <- sum(d$n)
  d <- d %>%
    mutate(category = stringr::str_to_title(gsub("_", " ", as.character(category)))) %>%
    arrange(desc(n)) %>%
    mutate(pct = 100 * n / total,
           label = sprintf("%s (n=%s)", category, scales::comma(n)),
           label = factor(label, levels = label))
  pal <- grDevices::hcl.colors(nrow(d), "Set 3")

  ggplot(d, aes(x = "", y = n, fill = label)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = pi / 2, direction = -1) +
    geom_text(aes(label = ifelse(pct >= 3, sprintf("%.1f%%", pct), "")),
              position = position_stack(vjust = 0.5),
              size = 2.6, color = "white", fontface = "bold") +
    scale_fill_manual(values = pal, name = NULL) +
    labs(title = sprintf("Cohort Composition (N=%s)", scales::comma(total))) +
    theme_void() +
    theme(plot.title = element_text(size = 11, face = "bold", hjust = 0.5),
          legend.text = element_text(size = 7))
}


# ============================================================================
# fig1e: cancer stage breakdown (with stage label fix: 2.0 -> Stage II)
# ============================================================================
build_fig1e <- function() {
  d <- load_figure_data("fig1_stage_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_stage_counts.csv empty"))
  d <- d %>%
    mutate(label = stage_label(category)) %>%
    arrange(n) %>% mutate(label = factor(label, levels = label))
  ggplot(d, aes(x = n, y = label)) +
    geom_col(fill = "#5B8DB8", color = "white", width = 0.65) +
    geom_text(aes(label = scales::comma(n)), hjust = -0.15, size = 2.8) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.12)),
                       labels = scales::comma) +
    labs(x = "Patients", y = NULL, title = "Cancer Stage Breakdown") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          axis.line.y = element_blank(), axis.ticks.y = element_blank())
}


# ============================================================================
# fig1f: first-line treatment breakdown
# ============================================================================
build_fig1f <- function() {
  d <- load_figure_data("fig1_treatment_counts.csv")
  if (nrow(d) == 0) return(placeholder_panel("fig1_treatment_counts.csv empty"))
  d <- d %>%
    mutate(label = gsub("_", " ", as.character(category))) %>%
    arrange(n) %>% mutate(label = factor(label, levels = label))
  ggplot(d, aes(x = n, y = label)) +
    geom_col(fill = "#E8A33D", color = "white", width = 0.65) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.08)),
                       labels = scales::comma) +
    labs(x = "Patients", y = NULL, title = "First-line Treatment Breakdown") +
    theme_manuscript() +
    theme(panel.grid.major.x = element_line(color = "grey90"),
          axis.line.y = element_blank(), axis.ticks.y = element_blank(),
          axis.text.y = element_text(size = 7))
}


# ============================================================================
# Compose
# ============================================================================
p1a <- build_fig1a(); p1b <- build_fig1b(); p1c <- build_fig1c()
p1d <- build_fig1d(); p1e <- build_fig1e(); p1f <- build_fig1f()

save_panel(p1a, "fig1a", group = "figure1", width = 9.6, height = 5.4)
save_panel(p1b, "fig1b", group = "figure1", width = 5.4, height = 4.4)
save_panel(p1c, "fig1c", group = "figure1", width = 7.0, height = 4.4)
save_panel(p1d, "fig1d", group = "figure1", width = 7.0, height = 5.2)
save_panel(p1e, "fig1e", group = "figure1", width = 6.0, height = 4.4)
save_panel(p1f, "fig1f", group = "figure1", width = 7.8, height = 4.4)
