# v2 notebooks

Numbered in the order they should be run. See [`../../REFACTOR_PLAN.md`](../../REFACTOR_PLAN.md)
for the full pipeline DAG this order is derived from.

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`01_run_preprocessing.ipynb`](01_run_preprocessing.ipynb) | cluster CPU | Cohort build, ICD extraction, non-text covariates, text preprocessing/tokenization. |
| 02 | [`02_generate_embeddings_gcp.ipynb`](02_generate_embeddings_gcp.ipynb) | GCP GPU | Copy token batches to the GPU environment and run `generate_clinical_embeddings`, then copy results back. |
| 03 | [`03_build_prediction_datasets.ipynb`](03_build_prediction_datasets.ipynb) | cluster CPU | Knit embeddings, report data availability, build per-anchor embedding prediction datasets. |
| 04b | `04b_run_training_manifests.ipynb` | allocated Jupyter CPU session | Simple treatment-anchor full-cohort fallback when Slurm submission is unavailable; runs manifest events sequentially with `n_jobs=-1`. Run **04** first; do not execute 04b on a login node. |
| — | `pipelines.training.build_slurm_manifests` + `slurm/launch_*.sh` | cluster CPU (shell) | Preferred distributed path when Slurm is available — build manifests, then `sbatch` the full-cohort, feature-comparison, and held-out-risk array jobs. Run **04** first to catch fitting problems before submitting. |
| 04 | [`04_smoke_test_training.ipynb`](04_smoke_test_training.ipynb) | cluster CPU (interactive) | Rehearses the real training entrypoints on a handful of representative events and reports fitting issues before you `sbatch` the arrays above. |
| 05 | [`05_run_full_cohort_risk_scores.ipynb`](05_run_full_cohort_risk_scores.ipynb) | cluster CPU | Held-out risk scores for the full-cohort models, once the SLURM arrays have completed. |
| 05b | [`05b_run_biomarker_pipeline.ipynb`](05b_run_biomarker_pipeline.ipynb) | cluster CPU | ICI biomarker discovery: cohort construction through compiled hits, one subprocess per `pipelines.biomarkers.*` stage, with stage toggles. Stage 5 (`run_IPTW_analysis`) is long-running. Must run **before** 06b — `figures.prep.figure5` reads its output. |
| 05c | [`05c_diagnose_iptw_run.ipynb`](05c_diagnose_iptw_run.ipynb) | cluster CPU | Diagnostic for 05b stage 5. Run when the screens finish clean but `IPTW_runs_*/` holds zero-row parquets: exercises the guard tests, counts rows per result file, and rebuilds `base_vars` from the saved IPTW datasets to name the covariate that emptied the model frames. Read-only. |
| 06a | [`06a_generate_code_lookups.Rmd`](06a_generate_code_lookups.Rmd) | local / cluster (R) | **One-time bootstrap**, not a per-run step. Builds the ICD-10→phecode mapping and phecode descriptions in `CODE_PATH` that `figures.prep.figure2` labels its panels from — the only R dependency in the prep tier, split out so `06b` needs no `Rscript`. Re-run after a cohort rebuild (01) or a Phecode package upgrade. Needs `devtools::install_github("vcastro/Phecode")` plus `arrow`. |
| 06b | [`06b_generate_figure_data.ipynb`](06b_generate_figure_data.ipynb) | cluster CPU / local | Runs `figures/prep/figureN.py` to write the CSVs the R tier plots from. Pure Python — warns and falls back to raw code labels if `06a` has not run. |
| 07 | [`07_render_figures.Rmd`](07_render_figures.Rmd) | local / cluster (R) | Renders manuscript figure panels from the `06b` CSVs. Bootstrap R packages once with `Rscript v2/R/install_packages.R`, then render with `Rscript -e 'rmarkdown::render("v2/notebooks/07_render_figures.Rmd")'`. |

## Trajectory and biomarker pipelines

Both run after step 05 (full-cohort risk scores) and before step 06b — see `REFACTOR_PLAN.md`.
`pipelines.biomarkers.*` is driven by `05b_run_biomarker_pipeline.ipynb` above; the individual
stages are still runnable directly with `python -m` from `v2/`. `pipelines.trajectories.*` has no
notebook driver and is invoked directly as scripts.
