# Shared published-score selection and C-index plotting for Figure 2e and the
# detailed supplement. Source figure_utils.R and within_cancer_utils.R first.

PS_GROUP <- "published_scores"
PS_ANCHOR <- "treatment"
PS_LAB_WINDOW <- 30L
PS_MODELS <- c("published", "text", "published+text")
PS_MODEL_LABELS <- c(published = "Published score", text = "Text",
                     "published+text" = "Published + text")
PS_MODEL_COLORS <- c(published = MODEL_COLORS[["published"]], text = MODEL_COLORS[["text"]],
                     "published+text" = MODEL_COLORS[["published+text"]])
PS_CONTRAST_LABELS <- c(
  "published+text vs. published" = "published+text vs. published",
  "published+text vs. text" = "published+text vs. text",
  "text vs. published" = "text vs. published"
)
PS_SCORES <- c(mgps = "mGPS", rmh = "RMH", lipi = "LIPI", albi = "ALBI", meld = "MELD",
               capra_mod = "CAPRA", ipi = "IPI", ipi_noecog = "IPI (ECOG-free)",
               imdc = "IMDC", imdc_noecog = "IMDC (ECOG-free)",
               mskcc = "MSKCC", mskcc_noecog = "MSKCC (ECOG-free)",
               ecog_only = "ECOG", kps_only = "KPS")
# This catalog is shared with the Python renderer for local report exports.
PS_MAIN_SCORES <- unlist(jsonlite::fromJSON(
  file.path(REPO_ROOT, "shared", "figure2_scores.json")
)$scores)
PS_MIN_PATIENTS <- 50L
PS_SOURCES <- c(calculated = "calculated", regex = "notes (regex)")
PS_COUNT_LEVELS <- c(n_eligible = "Eligible", n_complete = "Scored",
                     n_patients = "Evaluated", n_events = "Deaths")

ps_contrast_label <- function(model, reference) paste(model, "vs.", reference)

# Restrict to the primary anchor/window and the whole-population stratum (the
# only one the prep emits); tables from before the regex source existed have
# no `source` column and are all calculated.
ps_primary <- function(d) {
  if (!nrow(d)) return(d)
  if (!"source" %in% names(d)) d$source <- "calculated"
  d <- filter(d, anchor == PS_ANCHOR, lab_window_days == PS_LAB_WINDOW)
  if ("stratum" %in% names(d)) d <- filter(d, stratum == "all")
  d
}

# One row per (score, source) with at least PS_MIN_PATIENTS evaluated, in
# catalog then source order; `row` is the y-axis label shared by every panel,
# so this is the single place that decides which scores are shown.
ps_rows <- function(cohort, scores = PS_SCORES) {
  d <- ps_primary(cohort)
  if (!nrow(d)) return(tibble::tibble(score = character(), source = character(), row = character()))
  d %>%
    filter(score %in% names(scores), source %in% names(PS_SOURCES),
           !is.na(n_patients), n_patients >= PS_MIN_PATIENTS) %>%
    mutate(
      n_label = ifelse(is.na(n_patients), "n = NA",
                       sprintf("n = %s, %s deaths", scales::comma(n_patients),
                               scales::comma(n_events))),
      row = sprintf("%s, %s\n%s", scores[score], PS_SOURCES[source], n_label),
      order = match(score, names(scores)) * 10 + match(source, names(PS_SOURCES))
    ) %>%
    arrange(order) %>%
    select(score, source, row, n_patients, n_events, n_eligible, n_complete)
}

# Join the shared row labels; top-to-bottom in catalog order.
ps_attach_rows <- function(d, rows) {
  d <- inner_join(d, select(rows, score, source, row), by = c("score", "source"))
  mutate(d, row = factor(row, levels = rev(rows$row)))
}

build_published_cindex <- function(cindex, rows, disclose_missing = FALSE) {
  if (!nrow(cindex) || !nrow(rows)) {
    return(placeholder_panel("no published-score C-index results"))
  }
  d <- ps_primary(cindex) %>%
    filter(status == "ok", model %in% PS_MODELS, is.finite(cindex))
  if (!nrow(d)) return(placeholder_panel("no published-score C-index results"))
  d <- ps_attach_rows(d, rows) %>% mutate(model = factor(model, levels = PS_MODELS))
  if (!nrow(d)) return(placeholder_panel("no published-score C-index results"))
  missing_note <- NULL
  if (isTRUE(disclose_missing)) {
    expected <- tidyr::crossing(select(rows, score, source), model = PS_MODELS)
    missing <- anti_join(expected, d, by = c("score", "source", "model"))
    if (nrow(missing)) {
      missing_note <- paste0("Unavailable estimates: ", paste(
        sprintf("%s (%s), %s", PS_SCORES[missing$score],
                PS_SOURCES[missing$source], PS_MODEL_LABELS[missing$model]),
        collapse = "; "), ".")
    }
  }
  ggplot(d, aes(x = cindex, y = row, color = model)) +
    geom_vline(xintercept = 0.5, color = "grey70", linetype = "dashed") +
    geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), position = position_dodge(width = 0.6),
                   linewidth = 0.6) +
    geom_point(position = position_dodge(width = 0.6), size = 2.2) +
    scale_color_manual(values = PS_MODEL_COLORS, labels = PS_MODEL_LABELS,
                       breaks = PS_MODELS, drop = FALSE, name = NULL) +
    labs(x = "Overall survival C-index", y = NULL,
         title = "Published score, text, and both combined", caption = missing_note) +
    theme_combined() +
    theme(legend.position = "bottom", plot.caption = element_text(hjust = 0, size = 9))
}

