# v2 notebooks

Numbered in the order they should be run. See [`../../REFACTOR_PLAN.md`](../../REFACTOR_PLAN.md)
for the full pipeline DAG this order is derived from.

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`01_run_preprocessing.ipynb`](01_run_preprocessing.ipynb) | cluster CPU | Cohort build, ICD extraction, non-text covariates, text preprocessing/tokenization. |
| 02 | [`02_generate_embeddings_gcp.ipynb`](02_generate_embeddings_gcp.ipynb) | GCP GPU | Copy token batches to the GPU environment and run `generate_clinical_embeddings`, then copy results back. |
| 03 | [`03_build_prediction_datasets.ipynb`](03_build_prediction_datasets.ipynb) | cluster CPU | Knit embeddings, report data availability, build per-anchor embedding prediction datasets. |
| 04b | `04b_run_training_manifests.ipynb` | allocated Jupyter CPU session | Resumable fallback runner for the full-cohort, feature-comparison, and held-out-risk manifests when Slurm array submission is unavailable. Run **04** first; do not execute 04b on a login node. |
| — | `pipelines.training.build_slurm_manifests` + `slurm/launch_*.sh` | cluster CPU (shell) | Preferred distributed path when Slurm is available — build manifests, then `sbatch` the full-cohort, feature-comparison, and held-out-risk array jobs. Run **04** first to catch fitting problems before submitting. |
| 04 | [`04_smoke_test_training.ipynb`](04_smoke_test_training.ipynb) | cluster CPU (interactive) | Rehearses the real training entrypoints on a handful of representative events and reports fitting issues before you `sbatch` the arrays above. |
| 05 | [`05_run_full_cohort_risk_scores.ipynb`](05_run_full_cohort_risk_scores.ipynb) | cluster CPU | Held-out risk scores for the full-cohort models, once the SLURM arrays have completed. |
| 06 | [`06_generate_figure_data.ipynb`](06_generate_figure_data.ipynb) | cluster CPU / local | Runs `figures/prep/figureN.py` to write the CSVs the R tier plots from. |
| 07 | [`07_render_figures.Rmd`](07_render_figures.Rmd) | local / cluster (R) | Renders manuscript figure panels from the `06` CSVs. Bootstrap R packages once with `Rscript v2/R/install_packages.R`, then render with `Rscript -e 'rmarkdown::render("v2/notebooks/07_render_figures.Rmd")'`. |

## Trajectory and biomarker pipelines

`pipelines.trajectories.*` and `pipelines.biomarkers.*` run after step 05 (full-cohort risk
scores) and are invoked directly as scripts, not from a notebook — see `REFACTOR_PLAN.md`.
