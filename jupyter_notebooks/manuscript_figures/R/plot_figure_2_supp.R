# Render Figure 2 supplement: within-stage KM curves stratified by overall risk score.
#
# Shows that the overall (text) risk score separates survival even WITHIN a single
# clinical stage. Two panels, each a KM of one stage stratified by the patient's
# overall risk-score quartile (the same quartiles used in Figure 2G, defined across
# the known-stage cohort — i.e. NOT re-binned within stage):
#   A  Stage IV patients, by overall risk-score quartile
#   B  Stage I  patients, by overall risk-score quartile
#
# Reuses fig2_km_stage_vs_risk.csv (written by prep_figure_2.py); no new prep needed.

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(dplyr)
  library(scales); library(survival); library(ggsurvfit)
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
source(file.path(script_dir, "figure_utils.R"))  # provides tidy_km, logrank_p, step_ci_df


# ============================================================================
# One KM panel: a single stage, stratified by overall risk-score quartile (+95% CI)
# ============================================================================
RISK_QUARTILE_COLORS <- setNames(ORDINAL4, c("Q1", "Q2", "Q3", "Q4"))

# Within-stage concordance of the overall risk score for OS (higher score = higher risk).
stage_cindex <- function(df) {
  if (nrow(df) < 2 || length(unique(df$death)) < 2) return(NA_real_)
  tryCatch(
    survival::concordance(Surv(months, death) ~ text_risk_score, data = df,
                          reverse = TRUE)$concordance,
    error = function(e) NA_real_
  )
}

build_stage_panel <- function(df, stage_lbl, title_text) {
  sub <- df %>% filter(stage_group == stage_lbl)
  if (nrow(sub) == 0) return(placeholder_panel(paste0("no Stage ", stage_lbl, " patients")))
  sub <- sub %>% mutate(risk_quartile = factor(risk_quartile,
                                               levels = names(RISK_QUARTILE_COLORS)))

  fit <- survfit2(Surv(months, death) ~ risk_quartile, data = sub)
  td  <- tidy_km(fit) %>%
    mutate(stratum = factor(stratum, levels = names(RISK_QUARTILE_COLORS)))
  td_ci <- step_ci_df(td, "stratum")

  lr   <- logrank_p(sub, "months", "death", "risk_quartile")
  cidx <- stage_cindex(sub)
  ann  <- sprintf("n=%s\nc-index=%.3f\nlogrank p=%.1e",
                  scales::comma(nrow(sub)), cidx, lr)

  ggplot(td, aes(time, estimate, color = stratum)) +
    geom_rect(data = td_ci,
              aes(xmin = time, xmax = time_next, ymin = conf.low, ymax = conf.high, fill = stratum),
              inherit.aes = FALSE, alpha = 0.15, color = NA) +
    geom_step(linewidth = 0.9) +
    scale_color_manual(values = RISK_QUARTILE_COLORS, name = "Overall risk") +
    scale_fill_manual(values = RISK_QUARTILE_COLORS, guide = "none") +
    coord_cartesian(xlim = c(0, 60), ylim = c(0, 1.03)) +
    annotate("text", x = 1, y = 0.06, label = ann,
             hjust = 0, vjust = 0, size = 2.6, fontface = "italic", color = "#444444") +
    labs(x = "Months from first treatment", y = "Overall survival", title = title_text) +
    theme_manuscript() +
    theme(legend.position = c(0.98, 0.97),
          legend.justification = c(1, 1),
          legend.background = element_rect(fill = "white", color = NA))
}


# ============================================================================
# Compose supplementary figure
# ============================================================================
d <- load_figure_data("fig2_km_stage_vs_risk.csv")
if (nrow(d) > 0) {
  d <- d %>% mutate(months = tt_death / 30.44, death = as.integer(death))
}

pS_iv <- build_stage_panel(d, "IV", "Stage IV: survival by overall risk-score quartile")
pS_i  <- build_stage_panel(d, "I",  "Stage I: survival by overall risk-score quartile")

save_panel(pS_iv, "figS2_stage4_by_risk", width = 7.2, height = 6.0)
save_panel(pS_i,  "figS2_stage1_by_risk", width = 7.2, height = 6.0)

figS2 <- (pS_iv | pS_i) +
         plot_annotation(tag_levels = "A") &
         theme(plot.tag = element_text(size = 14, face = "bold"))

save_figure(figS2, "figureS2_stage_stratified_risk", width = 14.4, height = 6.4)
