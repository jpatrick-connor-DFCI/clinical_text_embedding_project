# Notebooks

Grouped by pipeline stage. Directories run in order (`1_data` → `2_models` → `3_biomarkers` →
`4_figures`), and notebooks within a directory are numbered from `01`. Each notebook's header
states what it runs after and what depends on it.

## 1_data — cohort, embeddings, prediction datasets

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`1_data/01_preprocessing.ipynb`](1_data/01_preprocessing.ipynb) | cluster CPU | Cohort build, ICD extraction, non-text covariates, text preprocessing/tokenization. |
| 02 | [`1_data/02_embeddings_gcp.ipynb`](1_data/02_embeddings_gcp.ipynb) | GCP GPU | Copy token batches to the GPU environment and run `generate_clinical_embeddings`, then copy results back. |
| 03 | [`1_data/03_prediction_datasets.ipynb`](1_data/03_prediction_datasets.ipynb) | cluster CPU | Knit embeddings, report data availability, build per-anchor embedding prediction datasets. |

## 2_models — time-to-event training and evaluation

The four notebooks here share one structure: setup → configuration → preconditions → run →
summary. Each wraps `pipelines.*` modules in subprocesses, is resumable, and reports what is on
disk rather than assuming the run it just did is the only one that has happened.

| # | Notebook | Tier | Notes |
|---|---|---|---|
| — | `pipelines.training.build_slurm_manifests` + `slurm/launch_*.sh` | cluster CPU (shell) | Preferred distributed path when Slurm is available — build manifests, then `sbatch` the full-cohort, feature-comparison, and held-out-risk array jobs. When the queue is congested, `2_models/01` can take the light feature-comparison modalities off it. |
| 01 | [`2_models/01_feature_comparison.ipynb`](2_models/01_feature_comparison.ipynb) | allocated Jupyter CPU session | Notebook fallback for the feature-comparison arrays, covering the three cheapest modalities (`stage`, `treatment`, `metburden`) — each pinned to `n_jobs=1` by `run_feature_comp_task` for having under 50 penalized columns. `somatic`, `text` and `prs` stay on the SLURM arrays. Safe to run **alongside** them: with `OVERWRITE = False` each side skips whatever the other has finished. Includes a read-only pre-flight census of how much is already done. Do not execute on a login node. |
| 02 | [`2_models/02_within_vs_pan.ipynb`](2_models/02_within_vs_pan.ipynb) | cluster CPU | Optional within- vs pan-stratum training comparison for cancer type and first-line treatment class, one subprocess per `pipelines.trajectories.*` script, with per-run toggles. Both are long-running and resume from their own per-stratum checkpoints. No longer required by manuscript figures; the within-cancer supplements evaluate existing pooled models and do not use these outputs. |
| 03 | [`2_models/03_mortality_trajectories.ipynb`](2_models/03_mortality_trajectories.ipynb) | cluster CPU | Landmark mortality risk trajectories (months 0–60) from `pipelines.trajectories.generate_mortality_trajectories`, with landmark coverage and at-risk denominators. **Fits one model at month 0 and re-scores it at every later landmark** (hyperparameters matched to the full-cohort runs), so trajectories are comparable across months. Resumable — each landmark is checkpointed as it completes. Must run **before** `4_figures/02` — `figures.prep.figure4` clusters these trajectories. |
| 04 | [`2_models/04_full_cohort_risk_scores.ipynb`](2_models/04_full_cohort_risk_scores.ipynb) | cluster CPU | Held-out risk scores for the full-cohort models, once the SLURM arrays have completed. |

## 3_biomarkers — ICI biomarker discovery

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`3_biomarkers/01_pipeline.ipynb`](3_biomarkers/01_pipeline.ipynb) | cluster CPU | Cohort construction through compiled hits, one subprocess per `pipelines.biomarkers.*` stage, with stage toggles. Stage 5 (`run_IPTW_analysis`) is long-running. Needs `2_models/04`'s full-cohort risk scores; must run **before** `4_figures/02` — `figures.prep.figure5` reads its output. |
| 02 | [`3_biomarkers/02_hit_km_curves.R`](3_biomarkers/02_hit_km_curves.R) | local / cluster (R) | Unweighted and IPTW-weighted KM curves for the compiled biomarker hits. Also writes compiled curve, at-risk, and audit-manifest CSVs. Run with `Rscript notebooks/3_biomarkers/02_hit_km_curves.R` from the repo root. |

## 4_figures — manuscript figures

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`4_figures/01_code_lookups.R`](4_figures/01_code_lookups.R) | local / cluster (R) | **One-time bootstrap**, not a per-run step. Builds the ICD-10→phecode mapping and phecode descriptions in `CODE_PATH` that `figures.prep.figure2` labels its panels from — the only R dependency in the prep tier, split out so `4_figures/02` needs no `Rscript`. Re-run after a cohort rebuild (`1_data/01`) or a Phecode package upgrade. Needs `devtools::install_github("vcastro/Phecode")` plus `arrow`. |
| 02 | [`4_figures/02_figure_data.ipynb`](4_figures/02_figure_data.ipynb) | cluster CPU / local | Runs `figures/prep/figureN.py` and the independent `figures.prep.within_cancer` module to write the CSVs the R tier plots from. **Incremental**: a module whose output CSVs all exist is skipped, so a re-run regenerates only what is missing — set `REGENERATE_ALL` or `FORCE` after anything upstream changes, since the check is presence, not freshness. Pure Python — warns and falls back to raw code labels if `4_figures/01` has not run. |
| 03 | [`4_figures/03_render_figures.R`](4_figures/03_render_figures.R) | local / cluster (R) | Renders manuscript figure panels and supplements from the `4_figures/02` CSVs, including within-cancer text-versus-base scatterplots and text-versus-modality heatmaps. Bootstrap R packages once with `Rscript R/install_packages.R`, then run with `Rscript notebooks/4_figures/03_render_figures.R`. |

The within-cancer supplements require existing matched patient risk scores:
`full_cohort_risk_scores/` for text versus base (generated by `2_models/04` or the
corresponding array jobs), and `held_out_risk_scores/` for text versus other modalities
(generated after feature-comparison training). They evaluate pooled models within
recorded cancer types, without fitting cancer-specific models. C-indices use joint
text/comparator outer-fold blocks weighted by comparable-pair counts. The defaults
require 20 matched patients, 5 events, and 1 comparable pair per comparison; set
`EXTRA_ARGS = {"within_cancer": ["--min-patients", "30", "--min-events", "10"]}` to
change the patient/event thresholds. Use `ONLY = {"within_cancer"}` to prepare just
these supplements, or `FORCE = {"within_cancer"}` to refresh existing outputs.
The module writes both performance CSVs, `fig2_within_cancer_event_counts.csv`,
`fig3_within_cancer_event_counts.csv`, and `within_cancer_audit.csv`. Before evaluating
each endpoint, the count tables report valid patients, observed events, and
`n_non_events` by cancer type, with eligibility, status, and required patient/event
thresholds. They include endpoints without trained risk scores and strata with zero
eligible patients. Full-cohort membership is scheme-specific embedding IDs intersected
with cancer annotations; the modality cohort additionally intersects somatic,
germline, stage, and treatment IDs, matching training. Zero-filled metastatic burden
does not restrict membership. Counts are upper bounds before matching predictions;
final paired cohorts must still meet the thresholds, and undersized strata skip
concordance. Two global progress bars (**Full cohort**, **Modality cohort**) cover
all schemes/endpoints. Warnings are suppressed; skips and missing modality inputs
go to the audit CSV without console logs. Rendering uses eligible rows, applies the
shared manuscript endpoint filter, and reports
descriptive endpoint summaries without significance tests. Images paginate after
12 cancer types; matching summary CSVs and legends are in the output `tables/`
and `captions/` directories.
Risk scores must include `outer_fold`; regenerate legacy files without it using
the corresponding training/risk runner's `--overwrite` option. Even when preparing
only `within_cancer`, rendering with the shared endpoint filter requires
`fig2_full_cohort_metrics.csv` from `figure2`. Set
`MANUSCRIPT_FILTER_UNDERPERFORMING_ENDPOINTS=false` to render without that filter.

## 5_semantic_search — exploratory embedding analyses

An exploratory arm rather than a manuscript pipeline stage, so it does not sit on the
`1_data` → `2_models` → `3_biomarkers` → `4_figures` sequence despite the numbered prefix.
The notebooks here drive the `semantic_search` Python package, which still lives at the repo
root (`semantic_search/`).

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`5_semantic_search/01_aggregate.ipynb`](5_semantic_search/01_aggregate.ipynb) | cluster CPU | Mean-pools progress, imaging, and pathology notes separately. Writes separate 768-dimensional note-type spaces plus the 3×768 concatenated prediction space, one per note window. Skip-if-exists resumable. |
| 02 | [`5_semantic_search/02_pcs.ipynb`](5_semantic_search/02_pcs.ipynb) | cluster CPU / local | Independently applies L2 → StandardScaler → PCA to each note type. Writes patient scores, long-form loadings, fitted transformers, and explained variance. |
| 03 | [`5_semantic_search/03_pc_correlations.ipynb`](5_semantic_search/03_pc_correlations.ipynb) | local / cluster | Tests retained PCs against continuous, categorical, and survival characteristics with family-wise BH-FDR and coverage reporting. |
| 04 | [`5_semantic_search/04_predict.ipynb`](5_semantic_search/04_predict.ipynb) | allocated Jupyter CPU session | Compresses each CV split to 50 progress + 25 imaging + 25 pathology PCs, then runs nested-CV XGBoost for cancer type, stage, first-treatment category, and conventional/AVPC/NEPC. |
| 05 | [`5_semantic_search/5_figures.R`](5_semantic_search/5_figures.R) | local / cluster (R) | Renders PC-space, PC-association, and out-of-fold XGBoost figures from completed semantic-search artifacts, without refitting models. |

Runs after [`1_data/03`](1_data/03_prediction_datasets.ipynb) — it needs the knitted embeddings —
and **nothing depends on it downstream**. It is not part of the `4_figures` manuscript path and does
not need the SLURM arrays or `2_models/04`.

Outputs go to `SEMANTIC_SEARCH_PATH` (`DATA_PATH/semantic_search/`), not to `FIGURE_PATH`. See
[`semantic_search/README.md`](../semantic_search/README.md) for the space/window grid, the method
notes, and the exploratory caveats.

## Splitting feature comparisons between Slurm and a notebook

`slurm/launch_feature_comp.sh` already sizes the six modalities in two classes: `big`
(`text`, `prs`) at 5 CPU / 8G, and `small` (`stage`, `treatment`, `somatic`, `metburden`) at
1 CPU / 4G, because `run_feature_comp_task.py` forces `n_jobs=1` for any modality with fewer than
50 penalized columns. Those four therefore gain nothing from the cluster's parallelism — they
are single-core work sitting in the same queue as the heavy fits.

`2_models/01_feature_comparison.ipynb` takes a **subset** of that class — `stage`, `treatment` and
`metburden` — in an allocated Jupyter session, so it can proceed while the arrays keep their
slots. `somatic` is left on SLURM despite being nominally `small`: its design matrix is a wide
gene-by-alteration panel whose width is derived at runtime from the
`_AMP`/`_DEL`/`_SNV`/`_SV`/`_FUSION` suffixes, so it is not reliably cheap.

The two sides coordinate through the skip logic already in `run_feature_comp_task.py`, which
passes over any scheme/event/modality whose `_test.csv`, `_val.csv`, `_ipcw_reference.csv.gz` and
`_risk_scores.csv` all exist — so with `OVERWRITE = False` neither redoes the other's work.
That check runs once at task start, so a task begun simultaneously on both sides is computed
twice; the outputs are deterministic, so this costs CPU rather than correctness. Because the
`small` array still owns `somatic`, it should keep running rather than being `scancel`-ed; to make
the two sides disjoint instead, re-submit it against a manifest pinned to `somatic` via the
third TSV field.

## Trajectory and biomarker pipelines

Mortality trajectories and biomarkers run before their corresponding modules in
`4_figures/02`. `pipelines.biomarkers.*` is driven by
`3_biomarkers/01_pipeline.ipynb` above and needs `2_models/04`'s full-cohort risk scores.
`pipelines.trajectories.*` is driven by two notebooks. `2_models/02_within_vs_pan.ipynb` runs the
optional within-vs-pan scripts, which fit their own models and read only the `1_data/03` embedding
prediction dataset plus the `1_data/01` covariates, so they can run any time after `1_data/03` and
do not wait on the SLURM arrays or `2_models/04`; manuscript figures no longer read their
outputs. `2_models/03_mortality_trajectories.ipynb` runs
`generate_mortality_trajectories`, which pools the note embeddings itself and so depends only on
`1_data/01`. Neither waits on `2_models/04`. The individual scripts are still runnable directly
with `python -m` from the repo root.
