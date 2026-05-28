# Manuscript figures

Five-figure manuscript layout (cohort → text vs base → modality comparison → mortality trajectories → ICI biomarker discovery). Each panel is a standalone PNG, then `compose_target_figures.py` assembles the target full figures.

## Layout

```
manuscript_figures/
├── _figure_utils.py              # paths, palettes, save_panel, load/save_figure_data
├── data_generation/
│   ├── prep_figure_1.py          # cohort counts, endpoint counts, note volume, composition
│   ├── prep_figure_2.py          # full-cohort C-index, cancer/endpoint heatmap, KM examples
│   ├── prep_figure_3.py          # modality C-index, joint betas, risk-score corr, univ-vs-joint AUC
│   ├── prep_figure_4.py          # trajectory clustering, cluster means, composition, KM merge
│   └── prep_figure_5.py          # PS predictions, robust hits, KM examples, top-hit metadata
├── plot_figure_1_cohort.py       # load + plot only; writes PNG panels
├── plot_figure_2_text_vs_base.py
├── plot_figure_3_modality_comparison.py
├── plot_figure_4_trajectories.py
├── plot_figure_5_biomarkers.py
├── compose_target_figures.py     # assemble current panel PNGs into old-style full figures
├── run_all_figures.ipynb         # one notebook to call prep + plot scripts
└── figures/                      # panel PNGs + target_figures/ composites
```

## Workflow

1. **Compute** (cluster, reads from `DATA_PATH`, writes `SURV_PATH/results/figure_data/`):

   ```bash
   for n in 1 2 3 4 5; do
     python jupyter_notebooks/manuscript_figures/data_generation/prep_figure_${n}.py
   done
   ```

2. **Plot** (anywhere `figure_data/` is reachable):

   ```bash
   for n in 1 2 3 4 5; do
     python jupyter_notebooks/manuscript_figures/plot_figure_${n}_*.py
   done
   ```

   PNGs land in `jupyter_notebooks/manuscript_figures/figures/`.

3. **Compose target figures** (uses the old full-figure layouts from the prior
   `python_scripts/figure_generation/` workflow, but only reads the current panel PNGs):

   ```bash
   python jupyter_notebooks/manuscript_figures/compose_target_figures.py
   ```

   Full PNG/PDF figures land in
   `jupyter_notebooks/manuscript_figures/figures/target_figures/`.

4. **Notebook orchestration**: open `run_all_figures.ipynb` and run all cells. It calls
   every prep and plotting script with the active kernel Python (`sys.executable`), so
   it uses the same conda environment as the notebook kernel rather than shell `python`.

## Key design decisions

- **Compute is separate from plotting.** Prep scripts do the heavy work; plotting scripts are load-and-plot only, so styling can be iterated without re-running risk scoring, Cox refits, clustering, or biomarker aggregation.
- **Plotting targets the historical mockups.** The plot scripts keep reading the
  current `figure_data/` CSVs, but the panel contents and final layouts now follow
  the rendered old mockups committed under `manuscript_figures/mockups/` in
  `e27dfa8` (Figure 1 schematic, Figure 2 heatmap/KM examples, Figure 3 modality
  overlap/comparison, Figure 4 three-panel trajectory analysis, Figure 5
  propensity ROC / biomarker robustness / KM interaction strip).
- **Schema-stable empty outputs.** When an upstream input is missing, prep scripts still emit a header-only CSV with the expected columns, so the corresponding plot script degrades to "no data — skipping" instead of crashing the whole run.
- **Stable cluster labels (Fig 4).** Trajectory clusters are relabeled by ascending mean risk in prep, so Cluster 0 is always the lowest-risk group across reruns. The `CLUSTER_COLORS` palette then maps consistently to the same semantic cluster.
- **Joint-Cox p-values (Fig 3A).** Upstream `feature_risk_score_coxph.py` stores betas only (sksurv does not expose SEs). `prep_figure_3` re-fits the joint model per (scheme, event) with `lifelines.CoxPHFitter` on the standardized held-out modality risk scores to derive p-values; panel A counts endpoints surviving BH-FDR < 0.05 within each fit, restricted to complete-case endpoints (see below).
- **Geometric mean for HR aggregation (Fig 5B).** Hazard ratios are multiplicative; the arithmetic mean of `[0.5, 2.0]` is `1.25` but the geometric mean is `1.0`. Robust-marker ranking uses the latter.
- **Primary-spec top hit (Fig 5C).** Panel C's KM examples are drawn from the primary spec (`cohort2 / covariates_plus_embeddings / ATE`), not whichever sensitivity spec happens to have the smallest p-value.
- **Two cohorts (Fig 5).** Cohort 1 = first-line ICI vs. *all* never-ICI controls (unmatched, discovery); Cohort 2 = ICI lines 1-3 vs. 1:1 matched controls (matched, validation). Panels A and C use the primary Cohort-2 spec (labeled in their subtitles); panel B's robustness columns are grouped by cohort (header + divider) with a definitions caption, and a "spec" = `cohort | ps_model | weight_type`.
- **Complete-case significance (Fig 3A).** Panel A counts significant endpoints only over complete-case endpoints — those where every modality (`MODALITY_ORDER`, labs removed) has a fitted joint beta — so per-modality counts share one denominator. Title shows the n.
- **Average modality rank (Fig 3C).** Panel C ranks modalities per endpoint by c-index (1 = best) and averages the ranks per modality across the same complete-case endpoints.
- **Modality risk-score correlation (Fig 3B).** Panel B is a correlation heatmap of the per-modality held-out risk scores (`fig3_risk_score_corr.csv`, death endpoint) — text is largely orthogonal to genomics/clinical modalities, i.e. it adds independent signal.
- **Standardized coefficients (Fig 3D).** Panel D plots the signed Wald z (β/SE) from the joint Cox fit, not raw log-HR — scale-free and stable (no unpenalized blow-ups, no penalized shrinkage), with ±1.96 reference lines and a display-only Tukey 1.5×IQR trim. Shared `fig3_joint_betas.csv` is untouched.
- **Disease-severity characteristics (Fig 4C).** Panel C is a 1×4 small-multiples row (% Stage IV, % ICI treated, mean # metastatic sites, **10-yr RMST** in months) reading `fig4_cluster_severity.csv`. RMST (`restricted_mean_survival_time`, **τ=120 mo** = 5 y past the 60-mo landmark entry requirement, so RMST does not saturate at the entry cap). Stage and ICI tokens accept float repr (`4.0`) and the long form `Immune Checkpoint Inhibitors`.
- **Cluster-count selection (Fig S1, appendix).** `prep_figure_4._silhouette_scan` computes silhouette vs k (2–8) on the same scaled trajectory matrix; `figS1a` plots it and marks the chosen k=4. Composed as `figureS1_cluster_silhouette` (compose key `s1`).
- **Cohort distributions (Fig 1).** The timeline schematic is replaced by population panels: notes-per-patient by type (box/violin, `fig1_notes_per_patient.csv`; shown among patients with ≥1 note of that type), cancer-stage and first-line-treatment breakdowns (`fig1_stage_counts.csv`, `fig1_treatment_counts.csv`), alongside the cancer-type pie. (The embedding UMAP was dropped — it did not read well.)
- **Trajectory heatmap (Fig 4A).** Panel A is the per-patient mortality-risk heatmap (`fig4_trajectories_heatmap.csv`, ≤500 patients/cluster, ordered by within-cluster mean risk), with white separators and cluster-name y-labels.
- **Covariate balance love plot (Fig 5).** `prep_figure_5._love_smd` recomputes stabilized ATE weights from the held-out propensity (`ICI_prediction`, dropping rows missing it) and reports SMD before vs after weighting (`fig5_love_smd.csv`, primary spec, pooled across cancers). Panel A notes the AUC is held-out CV; panel B annotates the denominator (robust hits of markers significant in ≥1 spec — `n_significant_markers`); panel C titles carry the marker×ICI interaction HR + 95% CI (carried through `fig5_km_examples.csv`).

## Per-figure inputs

| Figure | Inputs from `figure_data/` |
|---|---|
| 1 | `fig1_endpoint_counts.csv`, `fig1_cancer_type_counts.csv`, `fig1_notes_per_patient.csv`, `fig1_stage_counts.csv`, `fig1_treatment_counts.csv` |
| 2 | `fig2_full_cohort_metrics.csv`, `fig2_cancer_endpoint_heatmap.csv`, `fig2_km_tertiles.csv`, `fig2_km_stage_vs_risk.csv`, `fig2_stage_vs_risk_cindex.csv` |
| 3 | `fig3_modality_cindex.csv`, `fig3_modality_avg_rank.csv`, `fig3_joint_betas.csv` (includes p-values), `fig3_risk_score_corr.csv`, `fig3_univariate_vs_joint.csv` |
| 4 | `fig4_trajectories_heatmap.csv` (panel A), `fig4_cluster_composition_{stage,treatment}.csv`, `fig4_km_data.csv`, `fig4_cluster_severity.csv`, `fig4_silhouette.csv` (appendix Fig S1) |
| 5 | `fig5_ps_predictions.csv`, `fig5_robust_hits.csv`, `fig5_km_top_hit.csv`, `fig5_km_examples.csv`, `fig5_top_hit_meta.csv`, `fig5_love_smd.csv` |

## Prerequisites for each prep script

- **prep_figure_1**: nothing beyond standard pipeline outputs.
- **prep_figure_2**: requires `run_full_cohort_event.py` + `run_full_cohort_risk_scores.py` to have completed for `death_met`. The stage-vs-risk panel (E) also reads the derived cancer-stage pickle (`STAGE_PATH`), falling back to the one-hot `cancer_stage_df.csv.gz` if it is unavailable.
- **prep_figure_3**: requires feature-comp held-out risk scores for all schemes (panels A/B/D do a per-(scheme, event) lifelines refit on the held-out scores), `death_met` for the correlation heatmap (panel C), and `feature_risk_score_coxph.py` output in `<scheme_results>/risk_score_coxph/`.
- **prep_figure_4**: requires `generate_mortality_trajectories.py` output. Defaults to `decay_param=0.1`; override with `--decay <val>` or `--input <path>`.
- **prep_figure_5**: requires `ICI_train_propensity.py` predictions, `run_IPTW_analysis.py` per-spec outputs, `compile_IPTW_results.py` compiled hits, and IPTW input CSVs.

## Dependencies

- Plotting scripts: `matplotlib`, `pandas`, `numpy`, `lifelines`
- Prep (data_generation scripts): plotting deps + `scikit-survival`, `scikit-learn`, optionally `umap-learn` (PCA fallback if absent)
