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
| 02 | [`2_models/02_within_vs_pan.ipynb`](2_models/02_within_vs_pan.ipynb) | cluster CPU | Within- vs pan-stratum model comparison for cancer type and first-line treatment class, one subprocess per `pipelines.trajectories.*` script, with per-run toggles. Both are long-running and resume from their own per-stratum checkpoints. Must run **before** `4_figures/02` — `figures.prep.figure2` reads their metrics CSVs. |
| 03 | [`2_models/03_mortality_trajectories.ipynb`](2_models/03_mortality_trajectories.ipynb) | cluster CPU | Landmark mortality risk trajectories (months 0–60) from `pipelines.trajectories.generate_mortality_trajectories`, with landmark coverage and at-risk denominators. **Fits one model at month 0 and re-scores it at every later landmark** (hyperparameters matched to the full-cohort runs), so trajectories are comparable across months. Resumable — each landmark is checkpointed as it completes. Must run **before** `4_figures/02` — `figures.prep.figure4` clusters these trajectories. |
| 04 | [`2_models/04_full_cohort_risk_scores.ipynb`](2_models/04_full_cohort_risk_scores.ipynb) | cluster CPU | Held-out risk scores for the full-cohort models, once the SLURM arrays have completed. |

## 3_biomarkers — ICI biomarker discovery

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`3_biomarkers/01_pipeline.ipynb`](3_biomarkers/01_pipeline.ipynb) | cluster CPU | Cohort construction through compiled hits, one subprocess per `pipelines.biomarkers.*` stage, with stage toggles. Stage 5 (`run_IPTW_analysis`) is long-running. Needs `2_models/04`'s full-cohort risk scores; must run **before** `4_figures/02` — `figures.prep.figure5` reads its output. |
| 02 | [`3_biomarkers/02_hit_km_curves.ipynb`](3_biomarkers/02_hit_km_curves.ipynb) | local / cluster | KM curves for the compiled biomarker hits. |

## 4_figures — manuscript figures

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`4_figures/01_code_lookups.Rmd`](4_figures/01_code_lookups.Rmd) | local / cluster (R) | **One-time bootstrap**, not a per-run step. Builds the ICD-10→phecode mapping and phecode descriptions in `CODE_PATH` that `figures.prep.figure2` labels its panels from — the only R dependency in the prep tier, split out so `4_figures/02` needs no `Rscript`. Re-run after a cohort rebuild (`1_data/01`) or a Phecode package upgrade. Needs `devtools::install_github("vcastro/Phecode")` plus `arrow`. |
| 02 | [`4_figures/02_figure_data.ipynb`](4_figures/02_figure_data.ipynb) | cluster CPU / local | Runs `figures/prep/figureN.py` to write the CSVs the R tier plots from. **Incremental**: a module whose output CSVs all exist is skipped, so a re-run regenerates only what is missing — set `REGENERATE_ALL` or `FORCE` after anything upstream changes, since the check is presence, not freshness. Pure Python — warns and falls back to raw code labels if `4_figures/01` has not run. |
| 03 | [`4_figures/03_render_figures.Rmd`](4_figures/03_render_figures.Rmd) | local / cluster (R) | Renders manuscript figure panels from the `4_figures/02` CSVs. Bootstrap R packages once with `Rscript R/install_packages.R`, then render with `Rscript -e 'rmarkdown::render("notebooks/4_figures/03_render_figures.Rmd")'`. |

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

Both run before `4_figures/02`. `pipelines.biomarkers.*` is driven by
`3_biomarkers/01_pipeline.ipynb` above and needs `2_models/04`'s full-cohort risk scores.
`pipelines.trajectories.*` is driven by two notebooks. `2_models/02_within_vs_pan.ipynb` runs the
within-vs-pan scripts, which fit their own models and read only the `1_data/03` embedding
prediction dataset plus the `1_data/01` covariates, so they can run any time after `1_data/03` and
do not wait on the SLURM arrays or `2_models/04`. `2_models/03_mortality_trajectories.ipynb` runs
`generate_mortality_trajectories`, which pools the note embeddings itself and so depends only on
`1_data/01`. Neither waits on `2_models/04`. The individual scripts are still runnable directly
with `python -m` from the repo root.
