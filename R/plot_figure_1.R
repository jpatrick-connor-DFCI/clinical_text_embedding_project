# Figure 1: a cancer-type pie, b stage counts, c cohort-availability flow,
# d outcome endpoints. Individual panels and compiled PNG/PDF, labeled a-d.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr); library(tidyr)
  library(forcats); library(scales); library(ggsignif); library(cowplot)
})

source("R/figure_utils.R")


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
# fig1c: cumulative cohort flow (cancer-type availability explicit)
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
    stop("Figure 1a requires unique, ordered cumulative cohort counts")
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
    labs(title = "Cohort Eligibility and Data Availability") +
    theme_void(base_size = MANUSCRIPT_BASE_SIZE) +
    theme(plot.title = element_text(face = "bold", size = 12.5, hjust = 0.5),
          plot.margin = margin(8, 12, 8, 12))
}


# ============================================================================
# fig1d: endpoint counts per scheme
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
    message("[fig1d] fig2_full_cohort_metrics.csv unusable (absent, empty, or no ",
            "scheme/event columns) -- cannot trim")
    return(tibble::tibble())
  }
  # The exclusion is judged on the active metric's columns; without them
  # excluded_event_keys() drops nothing and the panel would silently report
  # untrimmed counts, so say so rather than looking like a successful trim.
  needed <- paste0(c("base_", "text_"), metric_suffix(METRIC))
  if (!all(needed %in% names(m))) {
    message(sprintf("[fig1d] fig2_full_cohort_metrics.csv lacks %s -- cannot trim on %s",
                    paste(setdiff(needed, names(m)), collapse = "/"), metric_label(METRIC)))
    return(tibble::tibble())
  }
  n_before <- nrow(dplyr::distinct(m, scheme, event))
  m <- drop_excluded_events(m, excluded_event_keys(m))
  if (nrow(m) == 0) return(tibble::tibble())
  out <- m %>%
    distinct(scheme, event) %>%
    count(scheme, name = "n_endpoints")
  message(sprintf("[fig1d] endpoints trimmed on %s: %d -> %d",
                  metric_label(METRIC), n_before, sum(out$n_endpoints)))
  out
}

build_endpoint_counts <- function() {
  d <- trimmed_endpoint_counts()
  if (nrow(d) == 0) {
    # Fallback: no per-event metrics available, so report the untrimmed counts
    # rather than blanking the panel.
    message("[fig1d] FALLING BACK to untrimmed fig1_endpoint_counts.csv")
    d <- load_figure_data("fig1_endpoint_counts.csv")
  }
  if (nrow(d) == 0) return(placeholder_panel("fig1_endpoint_counts.csv empty"))
  d <- d %>%
    mutate(scheme = factor(scheme, levels = names(SCHEME_LABELS))) %>%
    arrange(scheme) %>%
    mutate(scheme_lbl = SCHEME_LABELS[as.character(scheme)],
           scheme_lbl = factor(scheme_lbl, levels = scheme_lbl))
  ggplot(d, aes(x = scheme_lbl, y = n_endpoints, fill = as.character(scheme))) +
    geom_col(width = 0.62, color = "white") +
    geom_text(aes(label = scales::comma(n_endpoints)),
              vjust = -0.3, size = MANUSCRIPT_TEXT_SIZE) +
    scale_fill_manual(values = SCHEME_COLORS, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0, 0.12)),
                       labels = scales::comma) +
    labs(x = NULL, y = "Number of endpoints", title = "Outcome Endpoints") +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"),
          axis.text.x = element_text(hjust = 0.5, size = 10))
}


# ============================================================================
# fig1a: cancer-type pie of cohort composition
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
           label = sprintf("%s (n=%s)", category, scales::comma(n)),
           label = factor(label, levels = label))
  pal <- grDevices::hcl.colors(nrow(d), "Set 3")

  ggplot(d, aes(x = "", y = n, fill = label)) +
    geom_col(width = 1, color = "white") +
    coord_polar(theta = "y", start = pi / 2, direction = -1) +
    geom_text(aes(label = ifelse(pct >= 3, sprintf("%.1f%%", pct), "")),
              position = position_stack(vjust = 0.5),
              size = MANUSCRIPT_SMALL_TEXT_SIZE, color = "white", fontface = "bold") +
    scale_fill_manual(values = pal, name = NULL) +
    labs(title = sprintf("Cancer Types Among Patients with Text (N=%s)",
                         scales::comma(total))) +
    theme_void() +
    theme(plot.title = element_text(size = 13, face = "bold", hjust = 0.5),
          legend.text = element_text(size = 9))
}


# ============================================================================
# fig1b: cancer stage breakdown (with stage label fix: 2.0 -> Stage II)
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
    labs(x = NULL, y = "Patients", title = "Cancer Stage Breakdown",
         subtitle = sprintf("Patients with a recognized major stage (N=%s)", comma(stage_n))) +
    theme_manuscript() +
    theme(panel.grid.major.y = element_line(color = "grey90"))
}


# ============================================================================
# Compose
# ============================================================================
retire_main_panels(1, letters[1:4])

p1a <- build_cancer_pie()
p1b <- build_stage_counts()
p1c <- build_cohort_flow()
p1d <- build_endpoint_counts()

.tag <- metric_tag()
save_panel(p1a, paste0("fig1a", .tag), group = "figure1", width = 8.4, height = 6.2)
save_panel(p1b, paste0("fig1b", .tag), group = "figure1", width = 7.2, height = 5.4)
save_panel(p1c, paste0("fig1c", .tag), group = "figure1", width = 9.2, height = 7.0)
save_panel(p1d, paste0("fig1d", .tag), group = "figure1", width = 7.2, height = 5.4)
save_compiled_figure(
  list(a = p1a, b = p1b, c = p1c, d = p1d),
  number = 1, width = 20, height = 15
)
