# Manuscript figures

Six-figure manuscript layout (cohort data availability → cohort/population characteristics → text vs base → modality comparison → mortality trajectories → ICI biomarker discovery) plus an appendix (Fig S1 silhouette, Fig S2 within-stage risk stratification). **Data generation is Python; figure rendering is R (`ggplot2 + patchwork`)**. Each R script builds every panel as a ggplot object and composes the final figure in-memory — there is no separate compose script.

Figures 2, 3, and S2 render as **two parallel figure sets** — one scored by Harrell's C-index, one by mean time-dependent AUC(t) — via a `MANUSCRIPT_METRIC` switch (`"cindex"` / `"auc"`; see `R/figure_utils.R::METRIC`). Output files carry a matching suffix (e.g. `figure2_text_results_cindex.png` / `_auc.png`), so both sets coexist under `target_figures/`.

## Layout

```
manuscript_figures/
├── _figure_utils.py              # shared module for Python preps
├── data_generation/              # Python — compute tier, writes CSVs to figure_data/
│   ├── prep_figure_0.py          # cohort data-availability cascade (text/stage/treatment/somatic/PRS)
│   ├── prep_figure_1.py          # cohort, endpoint, notes/patient, stage/treatment counts
│   ├── prep_figure_2.py          # full-cohort C-index, pan-vs-within models, KMs, stage-vs-risk
│   ├── prep_figure_3.py          # modality C-index, avg-rank, modality ranks (long), joint betas, risk-score corr
│   ├── prep_figure_4.py          # trajectory clustering, severity (mean met, RMST), silhouette
│   └── prep_figure_5.py          # PS predictions, robust hits, KM examples, love-plot SMDs
├── R/                            # R — rendering tier (ggplot2 + patchwork)
│   ├── figure_utils.R            # paths, palettes, theme, IO, stats + KM helpers (tidy_km, logrank_p, step_ci_df);
│   │                              # also the METRIC switch (MANUSCRIPT_METRIC env var) for Figs 2/2-supp/3
│   ├── install_packages.R        # one-time CRAN bootstrap
│   ├── plot_figure_0.R           # 1 panel → figure0_data_availability.png
│   ├── plot_figure_1.R           # 6 panels → figure1_schematic.png
│   ├── plot_figure_2.R           # 7 panels (A–G) → figure2_text_results_{cindex,auc}.png
│   ├── plot_figure_2_supp.R      # within-stage KM by overall risk quartile → figureS2_stage_stratified_risk_{cindex,auc}.png
│   ├── plot_figure_3.R           # 4 panels → figure3_feature_comps_{cindex,auc}.png
│   ├── plot_figure_4.R           # 3 panels + figS1 → figure4_trajectories.png + figureS1_*
│   └── plot_figure_5.R           # 5 panels (A–E) → figure5_biomarkers.png
├── generate_figure_data.ipynb    # Python-kernel notebook — runs all preps
├── render_figures.ipynb          # R-kernel notebook — sources all R plot scripts
└── figures/                      # panel PNGs + target_figures/ composites (output dir)
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
   environment variable for the output directory). Figs 2/2-supp/3 read `MANUSCRIPT_METRIC`
   (`cindex` or `auc`; see `R/figure_utils.R::METRIC`) and must be run once per metric to get
   both parallel figure sets — the other scripts ignore the variable:

   ```bash
   for n in 0 1 4 5; do
     Rscript jupyter_notebooks/manuscript_figures/R/plot_figure_${n}.R
   done
   for n in 2 2_supp 3; do
     for metric in cindex auc; do
       MANUSCRIPT_METRIC=$metric Rscript jupyter_notebooks/manuscript_figures/R/plot_figure_${n}.R
     done
   done
   ```

   Each R script emits one composite PNG (`figureN_*.png`, metric-suffixed for Figs 2/2-supp/3)
   into `$CLINICAL_FIGURES_OUT/target_figures/`, plus individual panel PNGs (`figNx.png`) in
   `$CLINICAL_FIGURES_OUT/` for inspection. Fig 4's script also emits the appendix
   `figureS1_cluster_silhouette.png`.

3. **Notebook orchestration** (two kernels, run in order):

   1. Open [`generate_figure_data.ipynb`](generate_figure_data.ipynb) — **Python kernel** — and
      run all cells. It calls the six `prep_figure_*.py` scripts via the active kernel's
      interpreter (`sys.executable`).
   2. Open [`render_figures.ipynb`](render_figures.ipynb) — **R kernel** (`IRkernel`) — and
      run all cells. It `sys.source()`s each `R/plot_figure_N.R` in its own environment so
      per-script ggplot objects don't leak between scripts, running Figs 2/2-supp/3 twice
      (once per `MANUSCRIPT_METRIC` value) to produce both parallel figure sets.

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
  Figure 3 modality overlap/comparison, Figure 4 three-panel trajectory analysis,
  Figure 5 propensity ROC / biomarker robustness / KM interaction strip.
- **Schema-stable empty outputs.** When an upstream input is missing, prep scripts still emit a header-only CSV with the expected columns, so the corresponding plot script degrades to "no data — skipping" instead of crashing the whole run.
- **Stable cluster labels (Fig 4).** Trajectory clusters are relabeled by ascending mean risk in prep, so Cluster 0 is always the lowest-risk group across reruns. The `CLUSTER_COLORS` palette then maps consistently to the same semantic cluster. The R cluster names (`plot_figure_4.R`) describe risk **level** (the quantity the clusters are ordered on), not an assumed temporal shape.
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
- **Disease-severity characteristics (Fig 4C).** Panel C is a 1×4 small-multiples row (% Stage IV, % ICI treated, mean # metastatic sites, **10-yr RMST** in months) reading `fig4_cluster_severity.csv`. RMST (`restricted_mean_survival_time`, **τ=120 mo** = 5 y past the 60-mo landmark entry requirement, so RMST does not saturate at the entry cap). Stage and ICI tokens accept float repr (`4.0`) and the long form `Immune Checkpoint Inhibitors`.
- **Cluster-count selection (Fig S1, appendix).** `prep_figure_4._silhouette_scan` computes silhouette vs k (2–8) on the same scaled trajectory matrix; `figS1a` plots it and marks the chosen k=4. Composed as `figureS1_cluster_silhouette` (compose key `s1`).
- **Pan vs. within-stratum models (Fig 2C/2D).** Dumbbell (Cleveland) panels replacing the old cancer×endpoint heatmap: per stratum, grey dot = single pan-cohort embedding model, red dot = stratum-specific model, connected and sorted by Δ, with a dashed line at the overall-pan value. Metric follows the active `MANUSCRIPT_METRIC` (mean time-dependent AUC or Harrell's C-index); both are computed upstream in `within_vs_pan_cancer_models.py` / `within_treatment_vs_pan_treatment_models.py` (AUC uses train-based IPCW, train 5–95th-percentile eval grid clipped to each stratum's follow-up) and passed through by `prep_figure_2.py::_within_vs_pan()`, which also writes an `Overall` row. Death endpoint only; per-stratum rows require n≥30 held-out. Most within dots sitting at/below the dashed line is the intended "a single pan-cohort text model generalizes" message. Each dashed reference line is annotated in-panel with its numeric value (e.g. "Pan avg Mean AUC(t) = 0.71"). Panel widths: Fig 2C ×1.25, Fig 2D ×2 (relative widths `c(1.25, 2)` in the composed row) to give the longer within-stratum dumbbells room to breathe.
- **Overlay mean ± SD on distribution panels (Fig 2B/3D).** Both panels overlay a white diamond at the mean with a ±1 SD errorbar directly on the violins, in addition to the significance-star annotation text; Fig 3D's caption also prints the per-modality mean.
- **KM tertile legend (Fig 2E).** The redundant text/base linetype legend was removed (the panel title already states "(text solid, base dashed)" in plain text); the risk-tertile color legend and its 95% CI bands (`step_ci_df`/`geom_rect`) are retained.
- **Left-truncated KM curves (Fig 4B).** `ggsurvfit::tidy_survfit()` emits a synthetic `time=0, estimate=1` row per stratum for the curve start, which predates the `entry=60` left-truncation point; left un-filtered it renders as a spurious unstratified flat segment from month 0–60 once `geom_step` connects it to the first real (post-truncation) event. `build_fig4b` now filters `time >= 60` before plotting, and adds 95% CI bands (`step_ci_df`/`geom_rect`) matching the other KM panels.
- **AUC-by-cancer-type inset (Fig 5A).** The per-cancer-type AUC bar inset sits in the far lower-right corner of the main ROC panel (`cowplot::draw_plot`), with each bar annotated `AUC=x.xx`; the main ROC legend moved to the upper-left to stay clear of it.
- **Overall-risk legend placement (Fig S2).** The risk-quartile legend anchors to the true top-right corner of each stage panel (`legend.position = c(0.98, 0.97)`, `legend.justification = c(1, 1)`); both panels and the composed figure were enlarged to give the KM curves and legend more room.
- **C-index / mean AUC(t) metric switch (Figs 2, 2-supp, 3).** `figure_utils.R` reads `MANUSCRIPT_METRIC` (`"cindex"` or `"auc"`, default `"cindex"`) into a global `METRIC`, plus `metric_label()`/`metric_suffix()`/`metric_tag()` helpers. Nothing new is computed for the switch — `prep_figure_2.py`/`prep_figure_3.py` already write both metrics side by side (`fig2_full_cohort_metrics.csv`'s `{text,base}_{cindex,auc}` columns; `fig2_within_vs_pan_*.csv`'s `{cindex,auc}_{pan,within}` columns; `fig2_stage_vs_risk_{cindex,auc}*.csv`; `fig3_modality_avg_rank_{cindex,auc}.csv`/`fig3_modality_ranks_long_{cindex,auc}.csv`) — the R scripts just select whichever columns/CSVs match `METRIC` and suffix their outputs accordingly. FigS2's per-stage C-index (`fig2_stage_vs_risk_cindex_by_stage.csv`) was added to mirror the existing per-stage mean-AUC(t) table (`fig2_stage_vs_risk_auc.csv`), since the pre-existing `fig2_stage_vs_risk_cindex.csv` only had one pooled cohort-wide row, not a per-stage breakdown. The ICD-3 outlier dropped from Fig 2A/2B (see per-figure design notes) is now determined per-metric, since the largest-Δ event can differ between C-index and AUC(t).
- **Cohort distributions (Fig 1).** The timeline schematic is replaced by population panels: notes-per-patient by type (box/violin, `fig1_notes_per_patient.csv`; shown among patients with ≥1 note of that type), cancer-stage and first-line-treatment breakdowns (`fig1_stage_counts.csv`, `fig1_treatment_counts.csv`), alongside the cancer-type pie. (The embedding UMAP was dropped — it did not read well.)
- **Trajectory heatmap (Fig 4A).** Panel A is the per-patient mortality-risk heatmap (`fig4_trajectories_heatmap.csv`, ≤500 patients/cluster, ordered by within-cluster mean risk), with white separators and cluster-name y-labels.
- **Covariate balance love plot (Fig 5).** `prep_figure_5._love_smd` recomputes stabilized ATE weights from the held-out propensity (`ICI_prediction`, dropping rows missing it) and reports SMD before vs after weighting (`fig5_love_smd.csv`, primary spec, pooled across cancers). Panel A notes the AUC is held-out CV; panel B annotates the denominator (robust hits of markers significant in ≥1 spec — `n_significant_markers`); panel C titles carry the marker×ICI interaction HR + 95% CI (carried through `fig5_km_examples.csv`).

## Per-figure inputs

| Figure | Inputs from `figure_data/` |
|---|---|
| 0 | `fig0_data_availability.csv` |
| 1 | `fig1_endpoint_counts.csv`, `fig1_cancer_type_counts.csv`, `fig1_notes_per_patient.csv`, `fig1_stage_counts.csv`, `fig1_treatment_counts.csv` |
| 2 | `fig2_full_cohort_metrics.csv`, `fig2_within_vs_pan_cancer.csv`, `fig2_within_vs_pan_treatment.csv`, `fig2_km_tertiles.csv`, `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex.csv`, `fig2_stage_vs_risk_auc.csv` |
| 3 | `fig3_modality_cindex.csv`, `fig3_modality_avg_rank_cindex.csv`, `fig3_modality_avg_rank_auc.csv`, `fig3_modality_ranks_long_cindex.csv`, `fig3_modality_ranks_long_auc.csv`, `fig3_joint_betas.csv` (includes p-values), `fig3_risk_score_corr.csv` |
| 4 | `fig4_trajectories_heatmap.csv` (panel A), `fig4_km_data.csv`, `fig4_cluster_severity.csv`, `fig4_silhouette.csv` (appendix Fig S1) |
| 5 | `fig5_ps_predictions.csv`, `fig5_robust_hits.csv`, `fig5_km_top_hit.csv`, `fig5_km_examples.csv`, `fig5_top_hit_meta.csv`, `fig5_love_smd.csv`, `fig5_forest_headline.csv` |
| S2 | `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex_by_stage.csv`, `fig2_stage_vs_risk_auc.csv` (all reused from Figure 2 prep; no separate prep script) |

## Prerequisites for each prep script

- **prep_figure_0**: requires `cancer_type_df.csv.gz` (full cohort + cancer type), the post-text-merge embedding file for `icd3_post` (text availability), the derived cancer-stage pickle (falling back to the one-hot `cancer_stage_df.csv.gz`), `categorical_treatment_data_by_line.csv.gz` (treatment line 1), and the raw `complete_somatic_data_df.csv.gz` / `complete_germline_data_df.csv.gz` feature files — all under `clinical_and_genomic_features/`.
- **prep_figure_1**: nothing beyond standard pipeline outputs.
- **prep_figure_2**: requires `run_full_cohort_event.py` + `run_full_cohort_risk_scores.py` to have completed for `death_met`. The pan-vs-within panels (C/D) read the Pipeline-3 outputs `results/pan_vs_within_cancer/metrics_by_cancer_type.csv` and `results/pan_vs_within_treatment/metrics_by_treatment.csv` (directly under the scheme-agnostic `time-to-event_analysis/results/` dir, so `within_vs_pan_cancer_models.py` and `within_treatment_vs_pan_treatment_models.py` must be run first; they report both C-index and mean time-dependent AUC, both passed through). The stage-vs-risk panel (F) also reads the derived cancer-stage pickle (`STAGE_PATH`), falling back to the one-hot `cancer_stage_df.csv.gz` if it is unavailable. Also writes `fig2_stage_vs_risk_cindex_by_stage.csv` (per-stage C-index) alongside the existing per-stage `fig2_stage_vs_risk_auc.csv`, for FigS2's metric switch.
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
