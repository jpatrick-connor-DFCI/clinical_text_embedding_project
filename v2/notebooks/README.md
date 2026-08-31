# v2 notebooks

Numbered in the order they should be run. See [`../../REFACTOR_PLAN.md`](../../REFACTOR_PLAN.md)
for the full pipeline DAG this order is derived from.

| # | Notebook | Tier | Notes |
|---|---|---|---|
| 01 | [`01_run_preprocessing.ipynb`](01_run_preprocessing.ipynb) | cluster CPU | Cohort build, ICD extraction, non-text covariates, text preprocessing/tokenization. |
| 02 | [`02_generate_embeddings_gcp.ipynb`](02_generate_embeddings_gcp.ipynb) | GCP GPU | Copy token batches to the GPU environment and run `generate_clinical_embeddings`, then copy results back. |
| 03 | [`03_build_prediction_datasets.ipynb`](03_build_prediction_datasets.ipynb) | cluster CPU | Knit embeddings, report data availability, build per-anchor embedding prediction datasets. |
| 04b | `04b_run_training_manifests.ipynb` | allocated Jupyter CPU session | Simple treatment-anchor full-cohort fallback when Slurm submission is unavailable; runs manifest events sequentially with `n_jobs=-1`. Run **04** first; do not execute 04b on a login node. |
| 04e | [`04e_run_feature_comp_small.ipynb`](04e_run_feature_comp_small.ipynb) | allocated Jupyter CPU session | Feature-comparison counterpart to 04b, covering the three cheapest modalities (`stage`, `treatment`, `metburden`) — each pinned to `n_jobs=1` by `run_feature_comp_task` for having under 50 penalized columns. `somatic`, `text` and `prs` stay on the SLURM arrays. Safe to run **alongside** them: with `OVERWRITE = False` each side skips whatever the other has finished. Includes a read-only pre-flight census of how much is already done. Do not execute on a login node. |
| — | `pipelines.training.build_slurm_manifests` + `slurm/launch_*.sh` | cluster CPU (shell) | Preferred distributed path when Slurm is available — build manifests, then `sbatch` the full-cohort, feature-comparison, and held-out-risk array jobs. Run **04** first to catch fitting problems before submitting. When the queue is congested, 04e can take the light feature-comparison modalities off it. |
| 04 | [`04_smoke_test_training.ipynb`](04_smoke_test_training.ipynb) | cluster CPU (interactive) | Rehearses the real training entrypoints on a handful of representative events and reports fitting issues before you `sbatch` the arrays above. |
| 04c | [`04c_run_within_vs_pan_models.ipynb`](04c_run_within_vs_pan_models.ipynb) | cluster CPU | Within- vs pan-stratum model comparison for cancer type and first-line treatment class, one subprocess per `pipelines.trajectories.*` script, with per-run toggles. Both are long-running and resume from their own per-stratum checkpoints. Must run **before** 06b — `figures.prep.figure2` reads their metrics CSVs. |
| 04d | [`04d_run_mortality_trajectories.ipynb`](04d_run_mortality_trajectories.ipynb) | cluster CPU | Landmark mortality risk trajectories (months 0–60) from `pipelines.trajectories.generate_mortality_trajectories`, with landmark coverage and at-risk denominators. **Fits one model at month 0 and re-scores it at every later landmark** (hyperparameters matched to the full-cohort runs), so trajectories are comparable across months. Resumable — each landmark is checkpointed as it completes. Must run **before** 06b — `figures.prep.figure4` clusters these trajectories. |
| 05 | [`05_run_full_cohort_risk_scores.ipynb`](05_run_full_cohort_risk_scores.ipynb) | cluster CPU | Held-out risk scores for the full-cohort models, once the SLURM arrays have completed. |
| 05b | [`05b_run_biomarker_pipeline.ipynb`](05b_run_biomarker_pipeline.ipynb) | cluster CPU | ICI biomarker discovery: cohort construction through compiled hits, one subprocess per `pipelines.biomarkers.*` stage, with stage toggles. Stage 5 (`run_IPTW_analysis`) is long-running. Must run **before** 06b — `figures.prep.figure5` reads its output. |
| 05c | [`05c_diagnose_iptw_run.ipynb`](05c_diagnose_iptw_run.ipynb) | cluster CPU | Diagnostic for 05b stage 5. Run when the screens finish clean but `IPTW_runs_*/` holds zero-row parquets: exercises the guard tests, counts rows per result file, and rebuilds `base_vars` from the saved IPTW datasets to name the covariate that emptied the model frames. Read-only. |
| 06a | [`06a_generate_code_lookups.Rmd`](06a_generate_code_lookups.Rmd) | local / cluster (R) | **One-time bootstrap**, not a per-run step. Builds the ICD-10→phecode mapping and phecode descriptions in `CODE_PATH` that `figures.prep.figure2` labels its panels from — the only R dependency in the prep tier, split out so `06b` needs no `Rscript`. Re-run after a cohort rebuild (01) or a Phecode package upgrade. Needs `devtools::install_github("vcastro/Phecode")` plus `arrow`. |
| 06b | [`06b_generate_figure_data.ipynb`](06b_generate_figure_data.ipynb) | cluster CPU / local | Runs `figures/prep/figureN.py` to write the CSVs the R tier plots from. Pure Python — warns and falls back to raw code labels if `06a` has not run. |
| 07 | [`07_render_figures.Rmd`](07_render_figures.Rmd) | local / cluster (R) | Renders manuscript figure panels from the `06b` CSVs. Bootstrap R packages once with `Rscript v2/R/install_packages.R`, then render with `Rscript -e 'rmarkdown::render("v2/notebooks/07_render_figures.Rmd")'`. |

## Splitting feature comparisons between Slurm and a notebook

`slurm/launch_feature_comp.sh` already sizes the six modalities in two classes: `big`
(`text`, `prs`) at 5 CPU / 8G, and `small` (`stage`, `treatment`, `somatic`, `metburden`) at
1 CPU / 4G, because `run_feature_comp_task.py` forces `n_jobs=1` for any modality with fewer than
50 penalized columns. Those four therefore gain nothing from the cluster's parallelism — they
are single-core work sitting in the same queue as the heavy fits.

`04e_run_feature_comp_small.ipynb` takes a **subset** of that class — `stage`, `treatment` and
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

Both run before step 06b — see `REFACTOR_PLAN.md`. `pipelines.biomarkers.*` is driven by
`05b_run_biomarker_pipeline.ipynb` above and needs step 05's full-cohort risk scores.
`pipelines.trajectories.*` is driven by two notebooks. `04c_run_within_vs_pan_models.ipynb` runs the
within-vs-pan scripts, which fit their own models and read only the step 03 embedding prediction
dataset plus the step 01 covariates, so they can run any time after 03 and do not wait on the SLURM
arrays or step 05. `04d_run_mortality_trajectories.ipynb` runs
`generate_mortality_trajectories`, which pools the note embeddings itself and so depends only on
step 01. Neither waits on step 05. The individual scripts are still runnable directly with
`python -m` from `v2/`.
