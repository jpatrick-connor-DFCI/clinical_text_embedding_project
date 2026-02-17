# Clinical Text Embedding Project

This repository builds clinical-text embeddings from EHR notes using Clinical-Longformer, then trains and evaluates penalized Cox survival models across multiple endpoint schemes (ICD-10 level 3/4, phecodes, death/metastasis). It also includes treatment analysis (ICI propensity scoring via IPTW) and biomarker discovery pipelines.

## Repository Layout

```
clinical_text_embedding_project/
├── python_scripts/
│   ├── data_preprocessing/       # Text processing, embedding generation, covariate creation
│   ├── model_training/           # CoxPH model training with SLURM array jobs
│   ├── model_evaluation/         # Risk scoring, mortality trajectories, model comparisons
│   ├── biomarker_analysis/       # IPTW and risk-based biomarker discovery
│   └── treatment_analysis/       # ICI propensity score modeling
├── python_utils/
│   └── embed_surv_utils/         # Shared preprocessing and Cox model utilities (installable package)
├── bash_scripts/                 # SLURM submission and worker scripts
│   └── slurm_manifests/          # Generated task TSV files
└── jupyter_notebooks/
    └── metrics/                  # Trajectory analysis and visualization
```

---

## File-by-File Reference

### `python_scripts/data_preprocessing/`

| File | Description |
|------|-------------|
| `text_preprocessing_and_tokenization.py` | Extracts clinical notes from JSON files, cleans text (whitespace normalization, special character removal), batches into 50K-note chunks, and tokenizes using the Clinical-Longformer tokenizer. Saves tokenized input IDs and attention masks. |
| `generate_clinical_embeddings.py` | Generates 768-dimensional embeddings from tokenized notes using Clinical-Longformer on GPU. Processes in batches and saves embeddings as PyTorch tensors. |
| `knit_longformer_embeddings.py` | Combines batched embeddings and metadata into unified NumPy arrays. Adds derived columns (note datetime, time relative to treatment start) for downstream temporal analysis. |
| `extract_ICD_times.py` | Extracts ICD-10 diagnosis codes with timestamps from EHR records. Unpacks multiple codes per record and calculates time-to-ICD relative to first treatment date. |
| `generate_embedding_prediction_datasets.py` | Creates survival prediction datasets by merging time-decayed pooled embeddings with clinical outcomes (death, metastasis, ICD-10 events) at three endpoint granularity levels (level 3, level 4, phecodes). |
| `generate_all_non_text_covariates.py` | Compiles non-text clinical and genomic feature matrices (cancer type, stage, somatic mutations, PRS, lab values, treatment types, structural variants) for use as covariates in survival models. |

### `python_scripts/model_training/`

| File | Description |
|------|-------------|
| `build_slurm_manifests.py` | Generates TSV manifest files listing all scheme-event and scheme-event-modality combinations to train. Manifests drive the SLURM array jobs. |
| `run_full_cohort_event.py` | Trains full-cohort penalized CoxPH models (embeddings + base covariates) and baseline CoxPH models (base covariates only) for a single endpoint. Uses 80/20 train+val/test split with 5-fold CV grid search over L1 ratios and regularization alphas. |
| `run_feature_comp_task.py` | Trains a feature-specific CoxPH model for one event and one modality (stage, treatment, labs, somatic, PRS, or text) with hyperparameter tuning. |
| `slurm_array_utils.py` | Shared utilities for loading datasets, identifying valid events per scheme, configuring feature modalities (column selection, PCA settings), and managing output directories. |

### `python_scripts/model_evaluation/`

| File | Description |
|------|-------------|
| `feature_ICD10_level_3_risk_scores.py` | Calculates held-out risk scores for metastatic endpoints and ICD-10 level 3 events using feature-specific CoxPH models trained on individual modalities. |
| `generate_mortality_trajectories.py` | Generates per-patient mortality risk score trajectories over time by fitting penalized CoxPH models at monthly intervals with rolling note windows. |
| `cluster_mortality_trajectories.py` | Clusters mortality trajectories using hierarchical clustering on trajectory features (slope, AUC, extrema) to identify distinct risk progression patterns. |
| `within_vs_pan_cancer_models.py` | Compares pan-cancer vs. cancer-type-specific CoxPH models for mortality prediction. Evaluates whether within-cancer models improve concordance over pooled models. |
| `within_treatment_vs_pan_treatment_models.py` | Compares pan-treatment vs. treatment-specific CoxPH models for mortality prediction. Assesses whether stratifying by treatment improves discrimination. |

### `python_scripts/biomarker_analysis/`

| File | Description |
|------|-------------|
| `generate_IPTW_df.py` | Creates the IPTW analysis dataset by merging propensity scores (from 30-day buffer predictions) with genomic markers (somatic mutations, PRS) and clinical covariates for ICI vs. non-ICI comparison. |
| `generate_risk_based_df.py` | Generates the biomarker discovery dataset for first-line ICI patients, combining text-embedding-derived risk scores with genomic and clinical features. |
| `run_IPTW_analysis.py` | Performs IPTW analysis to identify predictive and prognostic biomarkers for ICI benefit. Applies common support trimming, stabilized ATE weights (truncated at 5th/95th percentile, capped at 20), Cox PH models with marker-treatment interactions, and Benjamini-Hochberg FDR correction. Runs on pan-cancer, SKIN, and LUNG cohorts. |
| `run_risk_based_analysis.py` | Tests whether genomic markers retain significance for mortality after adjustment for text-embedding-derived risk scores in first-line ICI patients. Auto-selects cancer types with sufficient sample size. |

### `python_scripts/treatment_analysis/`

| File | Description |
|------|-------------|
| `ICI_LRs.py` | Trains logistic regression propensity score models predicting first-line ICI vs. never-ICI receipt using clinical note embeddings plus confounders (demographics, cancer type, panel version). Generates held-out propensity scores via stratified 5-fold CV at multiple buffer windows (0, 15, 30, 45 days before treatment). Patients with ICI only at later treatment lines are excluded. |
| `treatment_analysis_common.py` | Shared utilities for loading note embeddings, cohort treatment data, survival outcomes, and confounder matrices (demographics, cancer type, panel version) used across treatment analysis scripts. |

### `python_utils/embed_surv_utils/`

| File | Description |
|------|-------------|
| `__init__.py` | Package initialization; exports all public functions from `preprocessing` and `cox_models` modules. |
| `preprocessing.py` | Text cleaning (`clean_text`, `deduplicate_texts`), ICD-10 code lookup, time-to-event mapping, continuous note window selection (`find_continuous_records_to_analyze`), and embedding pooling (`pool_embedding_series_vectorized` supporting mean, most-recent, and time-decay strategies). Top-level `generate_survival_embedding_df` orchestrates note selection, pooling, and optional survival data merging. |
| `cox_models.py` | CoxPH and CoxnetSurvivalAnalysis model training with cross-validation grid search (`run_grid_CoxPH_parallel`), baseline model fitting (`run_base_CoxPH`), held-out risk scoring (`get_heldout_risk_scores_CoxPH`), and data scaling. Supports PCA with memory-mapped arrays for high-dimensional embeddings and automatic fallback to in-RAM paths. |

### `python_utils/`

| File | Description |
|------|-------------|
| `pyproject.toml` | Build configuration for the `embed_surv_utils` package using setuptools backend. |

### `bash_scripts/`

| File | Description |
|------|-------------|
| `submit_full_cohort_array.sh` | Reads the full-cohort manifest, calculates the number of array tasks needed, and submits the array job with configurable concurrency limits. |
| `array_full_cohort_run.sh` | SLURM worker script for full-cohort training. Processes manifest rows assigned to its array task ID, calling `run_full_cohort_event.py` for each scheme-event combination. Pins threads to 1 to avoid oversubscription. |
| `submit_feature_comp_light_array.sh` | Submits feature-comparison array jobs for lightweight modalities (stage, treatment). Configurable rows per task and max concurrent jobs. |
| `array_feature_comp_light.sh` | SLURM worker for light feature-comparison tasks (6 CPUs, 24 GB). Iterates over assigned manifest rows calling `run_feature_comp_task.py` for each scheme-event-modality triple. |
| `submit_feature_comp_heavy_array.sh` | Submits feature-comparison array jobs for heavyweight modalities (text, PRS, labs, somatic) with lower concurrency due to higher resource requirements. |
| `array_feature_comp_heavy.sh` | SLURM worker for heavy feature-comparison tasks (higher memory/CPU allocation). Same manifest-driven logic as the light variant but with 1 row per task. |

### `jupyter_notebooks/metrics/`

| File | Description |
|------|-------------|
| `analyze_mortality_trajectories.py` | Analysis script for mortality risk trajectories: K-means clustering with elbow plots, spaghetti plots of trajectory clusters, and heatmaps comparing baseline vs. trajectory-adjusted cluster patterns. |

---

## Core Workflow

### 1) Preprocessing

Run in sequence:

```bash
python python_scripts/data_preprocessing/text_preprocessing_and_tokenization.py
python python_scripts/data_preprocessing/generate_clinical_embeddings.py
python python_scripts/data_preprocessing/knit_longformer_embeddings.py
python python_scripts/data_preprocessing/extract_ICD_times.py
python python_scripts/data_preprocessing/generate_embedding_prediction_datasets.py
python python_scripts/data_preprocessing/generate_all_non_text_covariates.py
```

### 2) Build SLURM Task Manifests

```bash
python python_scripts/model_training/build_slurm_manifests.py --schemes icd3 icd4 phecode death_met
```

Writes manifests to `bash_scripts/slurm_manifests/`.

### 3) Submit Training Arrays

```bash
bash bash_scripts/submit_full_cohort_array.sh
bash bash_scripts/submit_feature_comp_light_array.sh
bash bash_scripts/submit_feature_comp_heavy_array.sh
```

### 4) Evaluate Models

Use scripts under `python_scripts/model_evaluation/` for held-out risk scoring, mortality trajectory generation/clustering, and within-vs-pan cohort comparisons.

### 5) Treatment Analysis (ICI Propensity Scoring)

```bash
python python_scripts/treatment_analysis/ICI_LRs.py
```

Generates propensity scores for first-line ICI vs. never-ICI at buffer windows [0, 15, 30, 45] days.

### 6) Biomarker Discovery

```bash
python python_scripts/biomarker_analysis/generate_IPTW_df.py
python python_scripts/biomarker_analysis/run_IPTW_analysis.py
python python_scripts/biomarker_analysis/generate_risk_based_df.py
python python_scripts/biomarker_analysis/run_risk_based_analysis.py
```

## Notes

- Most scripts use hardcoded HPC data paths (configurable via constants at the top of each file).
- The `embed_surv_utils` package should be installed in development mode (`pip install -e python_utils/`).
- Key dependencies: `pandas`, `numpy`, `scikit-learn`, `sksurv`, `lifelines`, `statsmodels`, `torch`, `transformers`, `icd10`.
