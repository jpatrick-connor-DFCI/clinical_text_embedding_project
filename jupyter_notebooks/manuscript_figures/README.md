# Manuscript figures

Six-figure manuscript layout (cohort data availability → cohort/population characteristics → text vs base → modality comparison → mortality trajectories → ICI biomarker discovery) plus an appendix (Fig S1 silhouette, Fig S2 within-stage risk stratification, and a within-stage risk-dynamics KM supplement). **Data generation is Python; figure rendering is R (`ggplot2 + patchwork`)**. Each R script builds every panel as a standalone ggplot object and saves it individually — there is no composed multi-panel figure; panel assembly (if needed) happens downstream (e.g. in a manuscript layout tool), not in this pipeline.

Figures 2, 3, and S2 support a `MANUSCRIPT_METRIC` switch (`"cindex"` / `"auc"`; see `R/figure_utils.R::METRIC`) that can score panels by Harrell's C-index or mean time-dependent AUC(t) — panel filenames carry a matching suffix (e.g. `fig2a_cindex` / `fig2a_auc`). In practice only **C-index** is rendered; the underlying data for both metrics is still computed/written by the Python prep tier, but `render_figures.ipynb` pins `MANUSCRIPT_METRIC="cindex"` so no `_auc` panels are produced.

Every panel is saved in **both PNG and PDF**, in parallel flat trees — `png/` and `pdf/` — with matching filenames (`<name>.png` / `<name>.pdf`); there is no further nesting.

## Layout

```
manuscript_figures/
├── _figure_utils.py              # shared module for Python preps
├── data_generation/              # Python — compute tier, writes CSVs to figure_data/
│   ├── prep_figure_0.py          # cohort data-availability cascade (text/stage/treatment/somatic/PRS)
│   ├── prep_figure_1.py          # cohort, endpoint, notes/patient, stage/treatment counts
│   ├── prep_figure_2.py          # full-cohort C-index, pan-vs-within models, KMs, stage-vs-risk
│   ├── prep_figure_3.py          # modality C-index, avg-rank, modality ranks (long), joint betas, risk-score corr
│   ├── prep_figure_4.py          # risk-slope (Falling/Stable/Rising) grouping, severity (mean met, RMST, mean slope), group trajectories, slope-by-stage, silhouette
│   └── prep_figure_5.py          # PS predictions, robust hits, KM examples, love-plot SMDs
├── R/                            # R — rendering tier (ggplot2 + patchwork)
│   ├── figure_utils.R            # paths, palettes, theme, IO, stats + KM helpers (tidy_km, logrank_p, step_ci_df);
│   │                              # also the METRIC switch (MANUSCRIPT_METRIC env var) for Figs 2/2-supp/3
│   ├── install_packages.R        # one-time CRAN bootstrap
│   ├── plot_figure_0.R           # 1 panel  → fig0a
│   ├── plot_figure_1.R           # 6 panels → fig1a–fig1f
│   ├── plot_figure_2.R           # 13 panels → fig2a–fig2m ({cindex,auc}-suffixed a–g; h–m C-index only)
│   ├── plot_figure_2_supp.R      # 2 panels → figS2_stage4_by_risk, figS2_stage1_2_by_risk ({cindex,auc})
│   ├── plot_figure_2_supp_events.R # 6 panels (rank-2/3 per-event held-out-risk KM, mets/ICD10/phecodes) → figS_scheme_km_*
│   ├── plot_figure_3.R           # 4 panels → fig3a–fig3d ({cindex,auc}-suffixed)
│   ├── plot_figure_4.R           # 6 panels (A–E + figS1) → fig4a–fig4e, figS1a
│   ├── plot_figure_4_supp.R      # 2 panels → figS_stage4_by_dynamics, figS_stage1_2_by_dynamics
│   └── plot_figure_5.R           # 5 panels → fig5a–fig5e
├── generate_figure_data.ipynb    # Python-kernel notebook — runs all preps
├── render_figures.ipynb          # R-kernel notebook — sources all R plot scripts
├── png/                          # output dir — every panel as PNG (flat, no subfolders)
└── pdf/                          # output dir — every panel as PDF (flat, no subfolders)
```

## Workflow

0. **One-time R bootstrap** (any host with R):

   ```bash
   Rscript jupyter_notebooks/manuscript_figures/R/install_packages.R
   ```

1. **Compute** (cluster, reads from `DATA_PATH`, writes `SURV_PATH/results/figure_data/`):

   ```bash
   for n in 0 1 2 3 4 5; do
     python jupyter_notebooks/manuscript_figures/data_generation/prep_figure_${n}.py
   done
   ```

2. **Render** (anywhere `figure_data/` is reachable; honors the `CLINICAL_FIGURES_OUT`
   environment variable for the output directory). `MANUSCRIPT_METRIC` is pinned to `cindex`
   (mean AUC(t) rendering is disabled — see `R/figure_utils.R::METRIC`):

   ```bash
   export MANUSCRIPT_METRIC=cindex
   for n in 0 1 2 2_supp 2_supp_events 3 4 4_supp 5; do
     Rscript jupyter_notebooks/manuscript_figures/R/plot_figure_${n}.R
   done
   ```

   Each R script emits its individual panels only (no composed multi-panel figure) as both
   `$CLINICAL_FIGURES_OUT/png/<name>.png` and `$CLINICAL_FIGURES_OUT/pdf/<name>.pdf`, C-index-suffixed
   for Figs 2/2-supp/3 (e.g. `fig2a_cindex`). Fig 4's script also emits the appendix
   panel `figS1a.{png,pdf}`.

3. **Notebook orchestration** (two kernels, run in order):

   1. Open [`generate_figure_data.ipynb`](generate_figure_data.ipynb) — **Python kernel** — and
      run all cells. It calls the six `prep_figure_*.py` scripts via the active kernel's
      interpreter (`sys.executable`).
   2. Open [`render_figures.ipynb`](render_figures.ipynb) — **R kernel** (`IRkernel`) — and
      run all cells. It `sys.source()`s each `R/plot_figure_N.R` in its own environment so
      per-script ggplot objects don't leak between scripts, with `MANUSCRIPT_METRIC` pinned
      to `cindex` for all of them.

   The two notebooks are deliberately split so each runs in its native kernel — no subprocess
   bridge between Python and R.

## Key design decisions

- **Cohort data-availability cascade (Fig 0).** A CONSORT-style attrition panel run just ahead of Figure 1: from the full cancer-type cohort (`cancer_type_df.csv.gz` under `clinical_and_genomic_features/`, the same file used as the Fig 1 cohort/cancer-type source), how many patients have text, stage, first-line treatment, somatic, and PRS data, down to how many pass every threshold at once (the set usable for the full multi-modal comparison in Figures 3+). All modalities are read from the raw `clinical_and_genomic_features/` feature files (`complete_somatic_data_df.csv.gz`, `complete_germline_data_df.csv.gz`, `categorical_treatment_data_by_line.csv.gz`, the cancer-stage pickle/`cancer_stage_df.csv.gz`) — the same files `generate_all_non_text_covariates.py` writes — rather than any event-specific held-out risk-score files.
- **`death_met` display label.** The internal scheme key `death_met` is displayed everywhere in figures/captions as **"Death + Mets"** (via `SCHEME_LABELS` in `figure_utils.R`); all filtering logic still keys off the raw string `death_met`, so this is a display-only rename.
- **Compute is separate from plotting.** Prep scripts do the heavy work; plotting scripts are load-and-plot only, so styling can be iterated without re-running risk scoring, Cox refits, clustering, or biomarker aggregation.
- **Plotting targets the agreed panel layouts.** The plot scripts read the
  current `figure_data/` CSVs; the panel contents and final layouts follow the
  reference mockups that were committed under `manuscript_figures/mockups/` in
  commit `e27dfa8` and have since been removed from the working tree (recover them
  from that commit if needed): Figure 1 schematic, Figure 2 heatmap/KM examples,
  Figure 3 modality overlap/comparison, Figure 4 risk-dynamics analysis,
  Figure 5 propensity ROC / biomarker robustness / KM interaction strip.
- **Schema-stable empty outputs.** When an upstream input is missing, prep scripts still emit a header-only CSV with the expected columns, so the corresponding plot script degrades to "no data — skipping" instead of crashing the whole run.
- **Risk-dynamics grouping, not risk level (Fig 4).** Patients are grouped on the per-patient **OLS slope** of their model mortality risk over months `0..L` (`_ols_slopes` in `prep_figure_4.py`), using each patient's observed landmarks (≥3 points, no imputation). `SLOPE_LANDMARK_MONTHS` is the scientific knob (`L = 12` by default; set it to 24 for the longer-window variant). This selects the alive-at-L landmark cohort rather than near-complete survivors through month 60; the KM is left-truncated at the same L. Slopes are clustered into `N_SLOPE_GROUPS = 3` data-driven groups via k-means on the standardized slope. Groups are relabeled by **ascending mean slope** in prep, so cluster 0 is always the most-falling-risk group across reruns: `GROUP_NAMES <- c("Falling Risk", "Stable Risk", "Rising Risk")` in `plot_figure_4.R`. The `GROUP_COLORS` palette (`c(BENEFIT_COLOR, NS_GRAY, HARM_COLOR)`, blue/grey/red) maps consistently to the same semantic group. This replaces the earlier version of the figure, which clustered on the full risk trajectory and labeled clusters by mean risk **level** ("Lowest Risk"…"Highest Risk").
- **Joint-Cox p-values (Fig 3A).** The held-out modality risk scores are written at training time by `run_feature_comp_task.py` (under `<scheme>/held_out_risk_scores/`); sksurv exposes no SEs, so `prep_figure_3` re-fits the joint model per (scheme, event) with `lifelines.CoxPHFitter` on the standardized held-out modality risk scores to derive p-values. Panel A counts endpoints surviving BH-FDR < 0.05 within each fit, restricted to complete-case endpoints (see below).
- **Geometric mean for HR aggregation (Fig 5B).** Hazard ratios are multiplicative; the arithmetic mean of `[0.5, 2.0]` is `1.25` but the geometric mean is `1.0`. Robust-marker ranking uses the latter.
- **Primary-spec top hit (Fig 5C).** Panel C's KM examples are drawn from the primary spec (`cohort2 / covariates_plus_embeddings / ATE`), not whichever sensitivity spec happens to have the smallest p-value.
- **Two cohorts (Fig 5).** Cohort 1 = first-line ICI vs. *all* never-ICI controls (unmatched, discovery); Cohort 2 = ICI lines 1-3 vs. 1:1 matched controls (matched, validation). Panels A and C use the primary Cohort-2 spec (labeled in their subtitles); panel B's robustness columns are grouped by cohort (header + divider) with a definitions caption, and a "spec" = `cohort | ps_model | weight_type`.
- **Complete-case significance (Fig 3A).** Panel A counts significant endpoints only over complete-case endpoints — those where every modality (`MODALITY_ORDER`, labs removed) has a fitted joint beta — so per-modality counts share one denominator. Title shows the n.
- **Average modality rank (Fig 3C).** Panel C ranks modalities per endpoint by the active `MANUSCRIPT_METRIC` (C-index or mean AUC(t), 1 = best) and averages the ranks per modality across the same complete-case endpoints; `prep_figure_3.py` writes both `fig3_modality_avg_rank_{cindex,auc}.csv` / `fig3_modality_ranks_long_{cindex,auc}.csv` so either ranking is available without recomputation.
- **Modality risk-score correlation (Fig 3B).** Panel B is a correlation heatmap of the per-modality held-out risk scores (`fig3_risk_score_corr.csv`, death endpoint) — text is largely orthogonal to genomics/clinical modalities, i.e. it adds independent signal.
- **Standardized coefficients (Fig 3D).** Panel D plots the signed Wald z (β/SE) from the joint Cox fit, not raw log-HR — scale-free and stable (no unpenalized blow-ups, no penalized shrinkage), with ±1.96 reference lines and a display-only Tukey 1.5×IQR trim. Shared `fig3_joint_betas.csv` is untouched.
- **Significance stars on distribution panels.** GraphPad convention (`*` <.05, `**` <.01, `***` <.001, `****` <1e-4, `ns` otherwise). Fig 2B uses Wilcoxon signed-rank vs Δ=0 per scheme, annotated with the **mean** Δ (a diamond ± SD errorbar overlays the mean on the violin; the annotation text also switched from median to mean); Fig 3D uses Wilcoxon signed-rank vs z=0 per modality. Fig 1C shows raw notes-per-patient on a **linear** scale (no log transform, no omnibus test). Fig 3C adds a **Friedman test** (repeated-measures ranks, one block per endpoint, `fig3_modality_ranks_long_{cindex,auc}.csv`) as an omnibus check that modality rank differs at all across the complete-case endpoints. Tests live in the plot scripts (light compute; Fig 3C's per-endpoint ranks are precomputed in `prep_figure_3.py` since they aren't otherwise stored).
- **Event labels by description, not code (Fig 2A).** Top-event labels use a small `event_description()` heuristic in `plot_figure_2.R`: `death_met` events are either the literal `"death"` (labeled "Death") or a metastatic-site name from `MET_SITES` (labeled "Mets: " followed by the site name); other schemes' raw ICD-10/phecode event strings get underscore-cleanup + title-casing. Death and Mets are always distinguishable labels, never a shared raw code.
- **Disease-severity characteristics (Fig 4C).** Panel C is a 2×2 small-multiples grid (% Stage IV, mean # metastatic sites, **10-yr RMST** in months, **mean risk slope**) reading `fig4_cluster_severity.csv`; % ICI treated is no longer displayed. RMST uses `restricted_mean_survival_time` with **τ=120 mo**. The mean-slope panel is the quantity the groups are defined on and is intentionally left un-clamped (no 0–100 range) since it is negative for the Falling group.
- **Mean trajectory + cohort reference band (Fig 4D).** New panel reading `fig4_group_trajectories.csv` (long format: `group`, `month`, `mean_risk`, `q25`, `q75`; `group` is 0/1/2 for the slope groups plus a literal `"cohort"` pseudo-group for the cohort-wide average). `build_fig4d` draws the cohort-wide mean ± IQR band first as a neutral dashed grey reference, then overlays each slope group's mean trajectory ± IQR ribbon in `GROUP_COLORS`, so the reader can see each dynamics group's raw-scale risk path relative to the whole cohort.
- **Stage-matched dynamics composition (Fig 4E).** New panel reading `fig4_slope_by_stage.csv` (`stage`, `cluster`, `n_patients`, `mean_slope`). `build_fig4e` draws `position = "fill"` stacked bars of slope-group composition within each major stage (I–IV), demonstrating that Falling/Stable/Rising dynamics occur across every baseline stage — i.e. risk dynamics are not simply a restatement of stage.
- **Slope-group-count selection (Fig S1, appendix).** `prep_figure_4._silhouette_scan` computes silhouette vs k (2–8) on the standardized **slope** feature (not the full trajectory matrix); `figS1a` plots it and marks the chosen k = `N_SLOPE_GROUPS` = 3.
- **Pan vs. within-stratum models (Fig 2C/2D).** Dumbbell (Cleveland) panels replacing the old cancer×endpoint heatmap: per stratum, grey dot = single pan-cohort embedding model, red dot = stratum-specific model, connected and sorted by Δ, with a dashed line at the overall-pan value. Metric follows the active `MANUSCRIPT_METRIC` (mean time-dependent AUC or Harrell's C-index); both are computed upstream in `within_vs_pan_cancer_models.py` / `within_treatment_vs_pan_treatment_models.py` (AUC uses train-based IPCW, train 5–95th-percentile eval grid clipped to each stratum's follow-up) and passed through by `prep_figure_2.py::_within_vs_pan()`, which also writes an `Overall` row. Death endpoint only; per-stratum rows require n≥30 held-out. Most within dots sitting at/below the dashed line is the intended "a single pan-cohort text model generalizes" message. Each dashed reference line is annotated in-panel with its numeric value (e.g. "Pan avg Mean AUC(t) = 0.71"). Panel widths: Fig 2C ×1.25, Fig 2D ×2 (relative widths `c(1.25, 2)` in the composed row) to give the longer within-stratum dumbbells room to breathe.
- **Overlay mean ± SD on distribution panels (Fig 2B/3D).** Both panels overlay a white diamond at the mean with a ±1 SD errorbar directly on the violins, in addition to the significance-star annotation text; Fig 3D's caption also prints the per-modality mean.
- **KM tertile legend (Fig 2E).** The redundant text/base linetype legend was removed (the panel title already states "(text solid, base dashed)" in plain text); the risk-tertile color legend and its 95% CI bands (`step_ci_df`/`geom_rect`) are retained.
- **Left-truncated KM curves (Fig 4B).** `ggsurvfit::tidy_survfit()` emits a synthetic `time=0, estimate=1` row per stratum for the curve start, which predates the landmark L; left un-filtered it renders as a spurious pre-entry segment once `geom_step` connects it to the first real event. `build_fig4b` reads L from `fig4_km_data.csv`, filters `time >= L`, and adds 95% CI bands (`step_ci_df`/`geom_rect`) matching the other KM panels. Curves are stratified/colored by risk-dynamics group (`GROUP_COLORS`), not risk level.
- **Within-stage risk-dynamics KM supplement (`plot_figure_4_supp.R`).** Two panels — Stage IV (A) and pooled Stages I–II (B) — each a conditional-on-L KM (left-truncated `entry=L`, `x = L–120 mo`, same machinery as Fig 4B) stratified by risk-dynamics group. Reads `fig4_km_data.csv`, which carries `stage` (major stage I–IV from `_major_stage_map` in `prep_figure_4.py`, `NaN` if unavailable) and the constant `landmark_month` so R does not hard-code L. The script defines its own `GROUP_NAMES`/`GROUP_COLORS`/`cluster_label` locally and a left-truncation-aware log-rank (`coxph` score test, since `survdiff` rejects `Surv(start, stop, event)`). Single-metric (no `MANUSCRIPT_METRIC` switch); saved as panels `figS_stage4_by_dynamics` / `figS_stage1_2_by_dynamics`.
- **Within-stage overall-risk KM supplement (`plot_figure_2_supp.R`).** The corresponding overall-risk figure uses the same Stage IV (A) and pooled Stages I–II (B) strata, with curves defined by the cohort-wide risk-score quartiles. Figure 2 prep writes exact pooled `I-II` C-index and mean-AUC(t) rows for the panel annotation; separate Stage I/II estimates are not averaged.
- **Per-scheme Δ C-index top-3 barplots + event KM panels (Fig 2 H–M).** For three event categories — **mets** (`death_met` events excluding the literal `death`), **ICD10** (`icd3_post` and `icd4_post` pooled into one category), and **phecodes** (`phecode_post`) — `prep_figure_2.py::_scheme_delta_topk` ranks events by `delta = text_cindex - base_cindex`, **keeping only `delta > 0`** (a "top" event is never a net-negative regression) and taking the top 3 per category (`fig2_scheme_delta_topk.csv`); a category with fewer than 3 positive-delta events simply yields fewer bars/KM panels. `_scheme_event_km` then pulls each selected event's held-out risk scores (`full_cohort_risk_dir(scheme, event)`) and survival labels (`load_embedding_prediction_df(scheme)` + `filter_event_rows`), bins each model's score into tertiles, and writes one long table (`fig2_scheme_event_km.csv`). `plot_figure_2.R` renders the barplots (`build_scheme_delta_bars`) and the **rank-1** event's KM per category (`build_scheme_event_km`, sharing the `km_tertile_panel` helper extracted from the original Fig 2E renderer, `build_fig2d`) as panels H–M. Ranks 2–3 render the same way in the separate `plot_figure_2_supp_events.R` script as individual panels (`figS_scheme_km_mets2/3`, `figS_scheme_km_icd2/3`, `figS_scheme_km_phecodes2/3`). These panels are **C-index only** — they do not honor `MANUSCRIPT_METRIC` and carry no metric suffix. Per-event held-out risk-score CSVs only exist once `run_full_cohort_risk_scores.py --scheme <s> --event <e>` has been run for that event; until then the affected panel/barplot degrades to a `placeholder_panel`.
- **AUC-by-cancer-type inset (Fig 5A).** The per-cancer-type AUC bar inset sits in the far lower-right corner of the main ROC panel (`cowplot::draw_plot`), with each bar annotated `AUC=x.xx`; the main ROC legend moved to the upper-left to stay clear of it.
- **Overall-risk legend placement (Fig S2).** The risk-quartile legend anchors to the true top-right corner of each stage panel (`legend.position = c(0.98, 0.97)`, `legend.justification = c(1, 1)`); both panels were enlarged to give the KM curves and legend more room.
- **C-index / mean AUC(t) metric switch (Figs 2, 2-supp, 3).** `figure_utils.R` reads `MANUSCRIPT_METRIC` (`"cindex"` or `"auc"`, default `"cindex"`) into a global `METRIC`, plus `metric_label()`/`metric_suffix()`/`metric_tag()` helpers. Nothing new is computed for the switch — `prep_figure_2.py`/`prep_figure_3.py` already write both metrics side by side (`fig2_full_cohort_metrics.csv`'s `{text,base}_{cindex,auc}` columns (plus a precomputed `event_lbl` — real ICD-10 descriptions for `icd3_post`/`icd4_post` via `embed_surv_utils.find_icd_code`, since that lookup isn't available R-side); `fig2_within_vs_pan_*.csv`'s `{cindex,auc}_{pan,within}` columns; `fig2_stage_vs_risk_{cindex,auc}*.csv`; `fig3_modality_avg_rank_{cindex,auc}.csv`/`fig3_modality_ranks_long_{cindex,auc}.csv`) — the R scripts just select whichever columns/CSVs match `METRIC` and suffix their outputs accordingly. FigS2's per-stage C-index (`fig2_stage_vs_risk_cindex_by_stage.csv`) was added to mirror the existing per-stage mean-AUC(t) table (`fig2_stage_vs_risk_auc.csv`), since the pre-existing `fig2_stage_vs_risk_cindex.csv` only had one pooled cohort-wide row, not a per-stage breakdown. The ICD10 (Level 3) outlier dropped from Fig 2A/2B (see per-figure design notes) is always determined from C-index and that same event is excluded from both renderings, so the two metric sets show identical events.
- **Cohort distributions (Fig 1).** The timeline schematic is replaced by population panels: notes-per-patient by type (box/violin, `fig1_notes_per_patient.csv`; shown among patients with ≥1 note of that type), cancer-stage and first-line-treatment breakdowns (`fig1_stage_counts.csv`, `fig1_treatment_counts.csv`), alongside the cancer-type pie. (The embedding UMAP was dropped — it did not read well.)
- **Trajectory heatmap (Fig 4A).** Panel A is the per-patient mortality-risk heatmap (`fig4_trajectories_heatmap.csv`, ≤500 patients/group, ordered by within-group mean risk), with white separators and dynamics-group-name (Falling/Stable/Rising) y-labels.
- **Covariate balance love plot (Fig 5).** `prep_figure_5._love_smd` recomputes stabilized ATE weights from the held-out propensity (`ICI_prediction`, dropping rows missing it) and reports SMD before vs after weighting (`fig5_love_smd.csv`, primary spec, pooled across cancers). Panel A notes the AUC is held-out CV; panel B annotates the denominator (robust hits of markers significant in ≥1 spec — `n_significant_markers`); panel C titles carry the marker×ICI interaction HR + 95% CI (carried through `fig5_km_examples.csv`).

## Per-figure inputs

| Figure | Inputs from `figure_data/` |
|---|---|
| 0 | `fig0_data_availability.csv` |
| 1 | `fig1_endpoint_counts.csv`, `fig1_cancer_type_counts.csv`, `fig1_notes_per_patient.csv`, `fig1_stage_counts.csv`, `fig1_treatment_counts.csv` |
| 2 | `fig2_full_cohort_metrics.csv`, `fig2_within_vs_pan_cancer.csv`, `fig2_within_vs_pan_treatment.csv`, `fig2_km_tertiles.csv`, `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex.csv`, `fig2_stage_vs_risk_auc.csv`, `fig2_scheme_delta_topk.csv` (panels H–J), `fig2_scheme_event_km.csv` (panels K–M; requires per-event `run_full_cohort_risk_scores.py` runs) |
| 3 | `fig3_modality_cindex.csv`, `fig3_modality_avg_rank_cindex.csv`, `fig3_modality_avg_rank_auc.csv`, `fig3_modality_ranks_long_cindex.csv`, `fig3_modality_ranks_long_auc.csv`, `fig3_joint_betas.csv` (includes p-values), `fig3_risk_score_corr.csv` |
| 4 | `fig4_trajectories_heatmap.csv` (panel A), `fig4_km_data.csv` (panel B; incl. `stage` and `landmark_month`), `fig4_group_trajectories.csv` (panel D), `fig4_slope_by_stage.csv` (panel E), `fig4_cluster_severity.csv` (panel C, incl. `mean_slope`), `fig4_silhouette.csv` (appendix Fig S1, slope feature) |
| 5 | `fig5_ps_predictions.csv`, `fig5_robust_hits.csv`, `fig5_km_top_hit.csv`, `fig5_km_examples.csv`, `fig5_top_hit_meta.csv`, `fig5_love_smd.csv`, `fig5_forest_headline.csv` |
| S2 | `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex_by_stage.csv`, `fig2_stage_vs_risk_auc.csv` (all reused from Figure 2 prep; no separate prep script) |
| S (dynamics) | `fig4_km_data.csv` (reused from Figure 4 prep, incl. `stage` + `cluster` + `landmark_month`; no separate prep script) → `plot_figure_4_supp.R` |
| S (scheme events) | `fig2_scheme_delta_topk.csv`, `fig2_scheme_event_km.csv` (both reused from Figure 2 prep; ranks 2–3 per category; no separate prep script) → `plot_figure_2_supp_events.R` |

## Prerequisites for each prep script

- **prep_figure_0**: requires `cancer_type_df.csv.gz` (full cohort + cancer type), the post-text-merge embedding file for `icd3_post` (text availability), the derived cancer-stage pickle (falling back to the one-hot `cancer_stage_df.csv.gz`), `categorical_treatment_data_by_line.csv.gz` (treatment line 1), and the raw `complete_somatic_data_df.csv.gz` / `complete_germline_data_df.csv.gz` feature files — all under `clinical_and_genomic_features/`.
- **prep_figure_1**: nothing beyond standard pipeline outputs.
- **prep_figure_2**: requires `run_full_cohort_event.py` + `run_full_cohort_risk_scores.py` to have completed for `death_met`. The pan-vs-within panels (C/D) read the Pipeline-3 outputs `results/pan_vs_within_cancer/metrics_by_cancer_type.csv` and `results/pan_vs_within_treatment/metrics_by_treatment.csv` (directly under the scheme-agnostic `time-to-event_analysis/results/` dir, so `within_vs_pan_cancer_models.py` and `within_treatment_vs_pan_treatment_models.py` must be run first; they report both C-index and mean time-dependent AUC, both passed through). The stage-vs-risk panel (F) also reads the derived cancer-stage pickle (`STAGE_PATH`), falling back to the one-hot `cancer_stage_df.csv.gz` if it is unavailable. Also writes `fig2_stage_vs_risk_cindex_by_stage.csv` (per-stage C-index) alongside the existing per-stage `fig2_stage_vs_risk_auc.csv`, for FigS2's metric switch. The Δ C-index event KM panels (H–M, and the `plot_figure_2_supp_events.R` supplement) additionally require `run_full_cohort_risk_scores.py --scheme <s> --event <e>` to have been run for each selected top-3 event of each category (mets/ICD10/phecodes) — until then, the affected KM panel degrades to a placeholder rather than failing.
- **prep_figure_3**: requires feature-comp held-out risk scores for all schemes (written at training time by `run_feature_comp_task.py` under `<scheme>/held_out_risk_scores/`; panels A/B/D do a per-(scheme, event) lifelines refit on those scores) and `death_met` for the correlation heatmap (panel C). Also writes `fig3_modality_avg_rank_{cindex,auc}.csv` / `fig3_modality_ranks_long_{cindex,auc}.csv` (per-endpoint modality ranks by each metric, complete-case only) for the Fig 3C Friedman test and metric switch.
- **prep_figure_4**: requires `generate_mortality_trajectories.py` output. Defaults to `decay_param=0.1`; override with `--decay <val>` or `--input <path>`.
- **prep_figure_5**: requires `ICI_train_propensity.py` predictions, `run_IPTW_analysis.py` per-spec outputs, `compile_IPTW_results.py` compiled hits, and IPTW input CSVs.

## Dependencies

- **Python prep** (`data_generation/`): `numpy`, `pandas`, `scipy`, `lifelines`, `scikit-survival`, `scikit-learn`.
- **R rendering** (`R/`, pinned in `R/install_packages.R`): core ggplot stack —
  `ggplot2`, `patchwork`, `scales`, `dplyr`, `tidyr`, `readr`, `forcats`, `tibble`, `stringr`,
  `purrr`; survival — `survival` + `ggsurvfit` (pure-ggplot KMs that compose cleanly with
  patchwork — `add_pvalue()` for log-rank, `add_risktable()` for at-risk strips);
  significance stars — `ggsignif`; correlation heatmap — `ggcorrplot`; schematic + insets —
  `cowplot`, `viridisLite`; top-Δ labels — `ggrepel`. R ≥ 4.1 recommended.
