# v2 pipeline map

`v2/` is the maintained pipeline; `v1/` is an archival reference. Run Python
entry points from `v2/` with `python -m`.

The end-to-end dependency order is:

1. `pipelines.preprocessing.build_cohort`
2. `pipelines.preprocessing.extract_ICD_times`
3. `pipelines.preprocessing.generate_all_non_text_covariates`
4. `pipelines.preprocessing.text_preprocessing_and_tokenization`
5. Copy token batches to the GPU environment and run
   `pipelines.preprocessing.generate_clinical_embeddings`.
6. Copy embedding batches back and run `pipelines.preprocessing.knit_embeddings`.
7. Run `pipelines.preprocessing.report_data_availability`.
8. Run `pipelines.preprocessing.generate_embedding_prediction_datasets` once
   for each requested anchor.
9. Run `pipelines.training.build_slurm_manifests` for the same anchors, then
   launch full-cohort, feature-comparison, and held-out-risk jobs.
10. Run trajectory and biomarker pipelines, then `figures.prep.figure0` through
    `figure5` and the matching scripts under `v2/R/`.

Treatment-anchor filenames retain their historical names. Sequencing-anchor
prediction files use `__sequencing`; result directories use
`anchor_sequencing/`. The launch scripts select the corresponding manifest from
`ANCHOR` automatically.

Embedding-prediction datasets intentionally use a complete-case text cohort:
patients must have pre-anchor Clinician, Imaging, and Pathology embeddings. Each
dataset-generation run prints the retained count so this selection is visible.

Path configuration is centralized in `v2/config.py`. Use `CTEP_DATA_PATH` for
the project data root and `PROFILE_DATA_PATH` for compiled PROFILE inputs.
