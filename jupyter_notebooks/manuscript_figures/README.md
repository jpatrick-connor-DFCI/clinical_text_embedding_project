# Manuscript figures

Five-figure manuscript layout (cohort → text vs base → modality comparison → mortality trajectories → ICI biomarker discovery) plus an appendix (Fig S1, silhouette). **Data generation is Python; figure rendering is R (`ggplot2 + patchwork`)**. Each R script builds every panel as a ggplot object and composes the final figure in-memory — there is no separate compose script.

## Layout

```
manuscript_figures/
├── _figure_utils.py              # shared module for Python preps
├── data_generation/              # Python — compute tier, writes CSVs to figure_data/
│   ├── prep_figure_1.py          # cohort, endpoint, notes/patient, stage/treatment counts
│   ├── prep_figure_2.py          # full-cohort C-index, pan-vs-within models, KMs, stage-vs-risk
│   ├── prep_figure_3.py          # modality C-index, avg-rank, joint betas, risk-score corr
│   ├── prep_figure_4.py          # trajectory clustering, severity (mean met, RMST), silhouette
│   └── prep_figure_5.py          # PS predictions, robust hits, KM examples, love-plot SMDs
├── R/                            # R — rendering tier (ggplot2 + patchwork)
│   ├── figure_utils.R            # paths, palettes, theme, IO, stats helpers, KM helper
│   ├── install_packages.R        # one-time CRAN bootstrap
│   ├── plot_figure_1.R           # 6 panels → figure1_schematic.{png,pdf}
│   ├── plot_figure_2.R           # 6 panels → figure2_text_results.{png,pdf}
│   ├── plot_figure_3.R           # 4 panels → figure3_feature_comps.{png,pdf}
│   ├── plot_figure_4.R           # 3 panels + figS1 → figure4_trajectories.{png,pdf} + figureS1_*
│   └── plot_figure_5.R           # 4 panels → figure5_biomarkers.{png,pdf}
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
   for n in 1 2 3 4 5; do
     python jupyter_notebooks/manuscript_figures/data_generation/prep_figure_${n}.py
   done
   ```

2. **Render** (anywhere `figure_data/` is reachable; honors the `CLINICAL_FIGURES_OUT`
   environment variable for the output directory):

   ```bash
   for n in 1 2 3 4 5; do
     Rscript jupyter_notebooks/manuscript_figures/R/plot_figure_${n}.R
   done
   ```

   Each R script emits one composite (`figureN_*.png` + `.pdf`) into
   `$CLINICAL_FIGURES_OUT/target_figures/`, plus individual panel PNGs (`figNx.png`) in
   `$CLINICAL_FIGURES_OUT/` for inspection. Fig 4's script also emits the appendix
   `figureS1_cluster_silhouette.{png,pdf}`.

3. **Notebook orchestration** (two kernels, run in order):

   1. Open [`generate_figure_data.ipynb`](generate_figure_data.ipynb) — **Python kernel** — and
      run all cells. It calls the five `prep_figure_*.py` scripts via the active kernel's
      interpreter (`sys.executable`).
   2. Open [`render_figures.ipynb`](render_figures.ipynb) — **R kernel** (`IRkernel`) — and
      run all cells. It `sys.source()`s each `R/plot_figure_N.R` in its own environment so
      per-script ggplot objects don't leak between scripts.

   The two notebooks are deliberately split so each runs in its native kernel — no subprocess
   bridge between Python and R.

## Key design decisions

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
- **Average modality rank (Fig 3C).** Panel C ranks modalities per endpoint by c-index (1 = best) and averages the ranks per modality across the same complete-case endpoints.
- **Modality risk-score correlation (Fig 3B).** Panel B is a correlation heatmap of the per-modality held-out risk scores (`fig3_risk_score_corr.csv`, death endpoint) — text is largely orthogonal to genomics/clinical modalities, i.e. it adds independent signal.
- **Standardized coefficients (Fig 3D).** Panel D plots the signed Wald z (β/SE) from the joint Cox fit, not raw log-HR — scale-free and stable (no unpenalized blow-ups, no penalized shrinkage), with ±1.96 reference lines and a display-only Tukey 1.5×IQR trim. Shared `fig3_joint_betas.csv` is untouched.
- **Significance stars on distribution panels.** GraphPad convention (`*` <.05, `**` <.01, `***` <.001, `****` <1e-4, `ns` otherwise). Fig 2B uses Wilcoxon signed-rank vs Δ=0 per scheme; Fig 3D uses Wilcoxon signed-rank vs z=0 per modality; Fig 1C uses an omnibus Kruskal-Wallis across note types. Tests live in the plot scripts (light compute, no new CSVs).
- **Disease-severity characteristics (Fig 4C).** Panel C is a 1×4 small-multiples row (% Stage IV, % ICI treated, mean # metastatic sites, **10-yr RMST** in months) reading `fig4_cluster_severity.csv`. RMST (`restricted_mean_survival_time`, **τ=120 mo** = 5 y past the 60-mo landmark entry requirement, so RMST does not saturate at the entry cap). Stage and ICI tokens accept float repr (`4.0`) and the long form `Immune Checkpoint Inhibitors`.
- **Cluster-count selection (Fig S1, appendix).** `prep_figure_4._silhouette_scan` computes silhouette vs k (2–8) on the same scaled trajectory matrix; `figS1a` plots it and marks the chosen k=4. Composed as `figureS1_cluster_silhouette` (compose key `s1`).
- **Pan vs. within-stratum models (Fig 2C/2D).** Dumbbell (Cleveland) panels replacing the old cancer×endpoint heatmap: per stratum, grey dot = single pan-cohort embedding model, red dot = stratum-specific model, connected and sorted by Δ, with a dashed line at the overall-pan value. Metric is **mean time-dependent AUC** (not C-index) to match Fig 2A/Fig 3 and the CV-selection metric; it is computed upstream in `within_vs_pan_cancer_models.py` / `within_treatment_vs_pan_treatment_models.py` (train-based IPCW, train 5–95th-percentile eval grid clipped to each stratum's follow-up), which also write an `Overall` row. Death endpoint only; per-stratum rows require n≥30 held-out. Most within dots sitting at/below the dashed line is the intended "a single pan-cohort text model generalizes" message.
- **Cohort distributions (Fig 1).** The timeline schematic is replaced by population panels: notes-per-patient by type (box/violin, `fig1_notes_per_patient.csv`; shown among patients with ≥1 note of that type), cancer-stage and first-line-treatment breakdowns (`fig1_stage_counts.csv`, `fig1_treatment_counts.csv`), alongside the cancer-type pie. (The embedding UMAP was dropped — it did not read well.)
- **Trajectory heatmap (Fig 4A).** Panel A is the per-patient mortality-risk heatmap (`fig4_trajectories_heatmap.csv`, ≤500 patients/cluster, ordered by within-cluster mean risk), with white separators and cluster-name y-labels.
- **Covariate balance love plot (Fig 5).** `prep_figure_5._love_smd` recomputes stabilized ATE weights from the held-out propensity (`ICI_prediction`, dropping rows missing it) and reports SMD before vs after weighting (`fig5_love_smd.csv`, primary spec, pooled across cancers). Panel A notes the AUC is held-out CV; panel B annotates the denominator (robust hits of markers significant in ≥1 spec — `n_significant_markers`); panel C titles carry the marker×ICI interaction HR + 95% CI (carried through `fig5_km_examples.csv`).

## Per-figure inputs

| Figure | Inputs from `figure_data/` |
|---|---|
| 1 | `fig1_endpoint_counts.csv`, `fig1_cancer_type_counts.csv`, `fig1_notes_per_patient.csv`, `fig1_stage_counts.csv`, `fig1_treatment_counts.csv` |
| 2 | `fig2_full_cohort_metrics.csv`, `fig2_within_vs_pan_cancer.csv`, `fig2_within_vs_pan_treatment.csv`, `fig2_km_tertiles.csv`, `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex.csv` |
| 3 | `fig3_modality_cindex.csv`, `fig3_modality_avg_rank.csv`, `fig3_joint_betas.csv` (includes p-values), `fig3_risk_score_corr.csv` |
| 4 | `fig4_trajectories_heatmap.csv` (panel A), `fig4_cluster_composition_{stage,treatment}.csv`, `fig4_km_data.csv`, `fig4_cluster_severity.csv`, `fig4_silhouette.csv` (appendix Fig S1) |
| 5 | `fig5_ps_predictions.csv`, `fig5_robust_hits.csv`, `fig5_km_top_hit.csv`, `fig5_km_examples.csv`, `fig5_top_hit_meta.csv`, `fig5_love_smd.csv` |

## Prerequisites for each prep script

- **prep_figure_1**: nothing beyond standard pipeline outputs.
- **prep_figure_2**: requires `run_full_cohort_event.py` + `run_full_cohort_risk_scores.py` to have completed for `death_met`. The pan-vs-within panels (C/D) read the Pipeline-3 outputs `results/pan_vs_within_cancer/metrics_by_cancer_type.csv` and `results/pan_vs_within_treatment/metrics_by_treatment.csv` (directly under the scheme-agnostic `time-to-event_analysis/results/` dir, so `within_vs_pan_cancer_models.py` and `within_treatment_vs_pan_treatment_models.py` must be run first; they now report mean time-dependent AUC). The stage-vs-risk panel (F) also reads the derived cancer-stage pickle (`STAGE_PATH`), falling back to the one-hot `cancer_stage_df.csv.gz` if it is unavailable.
- **prep_figure_3**: requires feature-comp held-out risk scores for all schemes (written at training time by `run_feature_comp_task.py` under `<scheme>/held_out_risk_scores/`; panels A/B/D do a per-(scheme, event) lifelines refit on those scores) and `death_met` for the correlation heatmap (panel C).
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
