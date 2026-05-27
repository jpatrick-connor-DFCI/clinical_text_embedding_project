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
- **Joint-Cox p-values (Fig 3A/B).** Upstream `feature_risk_score_coxph.py` stores betas only (sksurv does not expose SEs). `prep_figure_3` re-fits the joint model per (scheme, event) with `lifelines.CoxPHFitter` on the standardized held-out modality risk scores to derive p-values; panels A/B then count endpoints surviving BH-FDR < 0.05 within each (scheme, event) fit.
- **Geometric mean for HR aggregation (Fig 5D).** Hazard ratios are multiplicative; the arithmetic mean of `[0.5, 2.0]` is `1.25` but the geometric mean is `1.0`. Robust-marker ranking uses the latter.
- **Primary-spec top hit (Fig 5C).** Panel C's KM examples are drawn from the primary spec (`cohort2 / covariates_plus_embeddings / ATE`), not whichever sensitivity spec happens to have the smallest p-value.

## Per-figure inputs

| Figure | Inputs from `figure_data/` |
|---|---|
| 1 | `fig1_cohort_counts.csv`, `fig1_endpoint_counts.csv`, `fig1_note_volume.csv`, `fig1_cancer_type_counts.csv`, `fig1_stage_counts.csv`, `fig1_treatment_counts.csv`, `fig1_umap_coords.csv` |
| 2 | `fig2_full_cohort_metrics.csv`, `fig2_cancer_endpoint_heatmap.csv`, `fig2_km_examples.csv` |
| 3 | `fig3_modality_cindex.csv`, `fig3_joint_betas.csv` (includes p-values), `fig3_risk_score_corr.csv`, `fig3_univariate_vs_joint.csv` |
| 4 | `fig4_trajectories_with_clusters.csv` (full cohort), `fig4_trajectories_heatmap.csv` (per-cluster downsample for panel A), `fig4_cluster_means.csv`, `fig4_cluster_composition_{cancer,stage,treatment}.csv`, `fig4_km_data.csv` |
| 5 | `fig5_ps_predictions.csv`, `fig5_robust_hits.csv`, `fig5_km_top_hit.csv`, `fig5_km_examples.csv`, `fig5_top_hit_meta.csv` |

## Prerequisites for each prep script

- **prep_figure_1**: nothing beyond standard pipeline outputs.
- **prep_figure_2**: requires `run_full_cohort_event.py` + `run_full_cohort_risk_scores.py` to have completed for `death_met`.
- **prep_figure_3**: requires feature-comp held-out risk scores for all schemes (panels A/B/D do a per-(scheme, event) lifelines refit on the held-out scores), `death_met` for the correlation heatmap (panel C), and `feature_risk_score_coxph.py` output in `<scheme_results>/risk_score_coxph/`.
- **prep_figure_4**: requires `generate_mortality_trajectories.py` output. Defaults to `decay_param=0.1`; override with `--decay <val>` or `--input <path>`.
- **prep_figure_5**: requires `ICI_train_propensity.py` predictions, `run_IPTW_analysis.py` per-spec outputs, `compile_IPTW_results.py` compiled hits, and IPTW input CSVs.

## Dependencies

- Plotting scripts: `matplotlib`, `pandas`, `numpy`, `lifelines`
- Prep (data_generation scripts): plotting deps + `scikit-survival`, `scikit-learn`, optionally `umap-learn` (PCA fallback if absent)
