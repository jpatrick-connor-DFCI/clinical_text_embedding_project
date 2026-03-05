# Clinical Text Embedding Project

This repository builds clinical-text embeddings from EHR notes using Clinical-Longformer, then trains and evaluates penalized Cox survival models across multiple endpoint schemes (ICD-10 level 3/4, phecodes, death/metastasis). It also includes a line-matched ICI propensity scoring and biomarker discovery pipeline with IPTW-weighted Cox models.

## Repository Layout

```
clinical_text_embedding_project/
├── python_scripts/
│   ├── data_preprocessing/       # Text processing, embedding generation, covariate creation
│   ├── model_training/           # CoxPH model training with SLURM array jobs
│   ├── model_evaluation/         # Risk scoring, mortality trajectories, model comparisons
│   └── biomarker_analysis/       # Line-matched ICI propensity scoring, IPTW biomarker discovery
├── python_utils/
│   └── embed_surv_utils/         # Shared preprocessing and Cox model utilities (installable package)
├── bash_scripts/                 # SLURM submission and worker scripts
│   └── slurm_manifests/          # Generated task TSV files
└── jupyter_notebooks/
    ├── metrics/                  # Survival model performance analysis and trajectory clustering
    ├── biomarker_analysis/       # IPTW KM curves and marker-propensity association testing
    ├── note_embedding_EDA/       # Embedding exploratory data analysis and PCA associations
    ├── manuscript_figures/        # Publication-quality figures
    └── meeting_figures/          # Lab meeting and conference presentations
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
| `generate_icd10_to_phecode_mapping.R` | Builds a comprehensive ICD-10-CM to phecode mapping using the R `Phecode` package (v1.2 primary, PhecodeX v1.0 gap-fill). Outputs a mapping CSV and a list of unmapped codes. |
| `generate_embedding_prediction_datasets.py` | Creates survival prediction datasets by merging time-decayed pooled embeddings with clinical outcomes (death, metastasis, ICD-10 events) at three endpoint granularity levels (level 3, level 4, phecodes). |
| `generate_all_non_text_covariates.py` | Compiles non-text clinical and genomic feature matrices: cancer type, cancer stage, somatic mutations (SNV/AMP/CNV/DEL), structural variants and fusions (from SOMATIC_SV_RESULTS), PRS, mean pre-treatment lab values, treatment classes by line, and panel version. SV/Fusion features use all mutation-tested samples as the universe (positive-only SV data; absence = 0 for tested samples). |

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

The biomarker pipeline uses line-matched cohorts to compare ICI vs. never-ICI patients, with two propensity score models and two analysis tracks run as sensitivity analyses.

**Matching schemes:** 1:1 (one control per case) and 1:k (up to 3 controls per case), matched exactly on (cancer_type, line_category) where line_category bins treatment lines as 1, 2, 3, or 4+.

**Propensity score models:** (1) embeddings-only LR; (2) all-covariates LR (embeddings + demographics + cancer type + panel version + line dummies).

**Analysis tracks:**
- **Track 1** (ICI-only, generalizability-weighted): `S(t) ~ base_vars + line_dummies + marker`. Effect = marker coefficient. IPTW weight = 1/ps to rebalance ICI patients toward the full eligible population.
- **Track 2** (full cohort, IPTW-weighted): `S(t) ~ base_vars + line_dummies + marker + PX_on_ICI + marker×ICI`. Effect = interaction coefficient. Uses stabilized ATE/ATT weights with common-support trimming.

Both tracks run across pan-cancer, SKIN, and LUNG cohorts with FDR correction within mutation type.

| File | Description |
|------|-------------|
| `biomarker_common.py` | Shared utilities for loading note embeddings, survival cohort data, and confounder matrices (demographics, cancer type, panel version). Defines mutation type tags for marker classification. |
| `build_line_matched_cohort.py` | Builds line-matched ICI vs. never-ICI cohorts. For ICI patients, extracts the earliest treatment line with ICI (from `ALL_MEDICATION_LINES.csv`). For never-ICI controls, uses max treatment line reached. Bins lines into 1/2/3/4+ categories and performs exact matching on (cancer_type, line_category) without replacement. Produces both 1:1 and 1:k (k=3) matched cohort CSVs. |
| `ICI_LRs.py` | Trains logistic regression propensity models predicting ICI receipt within a line-matched cohort. Accepts `--matching {1to1,1tok}` argument. For each matching scheme, trains two models: embeddings-only and all-covariates (embeddings + demographics + cancer type + panel version + line dummies). Generates held-out propensity scores via 5-fold stratified CV at multiple buffer windows (0, 15, 30, 45 days). Saves ROC curves for the 30-day buffer. |
| `generate_IPTW_df.py` | Creates the IPTW analysis dataset by merging propensity scores (30-day buffer) with genomic markers (somatic mutations including SV/Fusions), cancer type, panel version, and line category dummies (line 1 as reference). Accepts `--matching` and `--ps_model` arguments to select the specification. |
| `run_IPTW_analysis.py` | Runs both analysis tracks for a given {matching, ps_model} specification. Track 2 (full-cohort interaction) runs with ATE, ATT, and unweighted sensitivity specs. Track 1 (ICI-only) runs weighted (1/ps generalizability) and unweighted. Includes common-support trimming, stabilized IPTW weight truncation, Platt-scaling recalibration for cancer-type subsets (SKIN, LUNG), covariate balance diagnostics (SMD), and effective sample size reporting. Accepts `--matching` and `--ps_model` arguments. |
| `run_biomarker_pipeline.ipynb` | Orchestration notebook that runs the full pipeline in sequence: (1) data regeneration via `generate_all_non_text_covariates.py`, (2) line-matched cohort construction, (3) propensity score generation for both matching schemes, (4) IPTW dataset generation for all 4 specs, (5) Cox model analysis for all 4 specs, (6) results compilation aggregating significant hits across specifications. |

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
| `submit_feature_comp_array.sh` | Submits feature-comparison array jobs. Reads the feature-comparison manifest and submits with configurable rows per task and max concurrent jobs. |
| `array_feature_comp.sh` | SLURM worker for feature-comparison tasks. Iterates over assigned manifest rows calling `run_feature_comp_task.py` for each scheme-event-modality triple. |

### `jupyter_notebooks/metrics/`

| File | Description |
|------|-------------|
| `analyze_landmark_results.ipynb` | Analysis of landmark survival model performance across time horizons. |
| `analyze_mortality_trajectories.ipynb` | Interactive analysis and visualization of mortality risk score trajectories over time. |
| `analyze_mortality_trajectories.py` | Script version: K-means clustering with elbow plots, spaghetti plots of trajectory clusters, and heatmaps comparing baseline vs. trajectory-adjusted cluster patterns. |
| `compile_ICD_level_3_results.ipynb` | Compiles and visualizes full-cohort CoxPH results across ICD-10 level 3 endpoints. |
| `compile_feature_comps_ICD_level_3.ipynb` | Compiles feature-comparison results (text vs. somatic vs. labs vs. PRS vs. stage vs. treatment) for ICD-10 level 3 endpoints. |
| `debug_individual_events.ipynb` | Debugging notebook for inspecting individual event-level model fits. |
| `trajectories_vs_stage.ipynb` | Compares mortality trajectory clusters against cancer stage to assess whether trajectories capture information beyond staging. |

### `jupyter_notebooks/biomarker_analysis/`

| File | Description |
|------|-------------|
| `IPTW_KM_curves.ipynb` | Kaplan-Meier survival curves comparing ICI vs. non-ICI patients, weighted by IPTW propensity scores. |
| `test_marker_propensity_association.ipynb` | Tests for associations between genomic markers and propensity scores to check for residual confounding. |

### `jupyter_notebooks/note_embedding_EDA/`

| File | Description |
|------|-------------|
| `embedding_EDA.ipynb` | Exploratory data analysis of clinical note embeddings: distributions, dimensionality, clustering patterns. |
| `association_testing_for_PCA.ipynb` | Tests associations between PCA components of note embeddings and clinical variables. |

### `jupyter_notebooks/manuscript_figures/`

| File | Description |
|------|-------------|
| `figure_1.ipynb` | Generates Figure 1 for the manuscript. |

### `jupyter_notebooks/meeting_figures/`

| File | Description |
|------|-------------|
| `07_14_2025_lab_meeting.ipynb` | Figures for the July 14, 2025 lab meeting presentation. |
| `ASHG_figures.ipynb` | Figures for the ASHG conference presentation. |

---

## Core Workflow

### 1) Preprocessing

Run in sequence:

```bash
python python_scripts/data_preprocessing/text_preprocessing_and_tokenization.py
python python_scripts/data_preprocessing/generate_clinical_embeddings.py
python python_scripts/data_preprocessing/knit_longformer_embeddings.py
python python_scripts/data_preprocessing/extract_ICD_times.py
Rscript python_scripts/data_preprocessing/generate_icd10_to_phecode_mapping.R
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
bash bash_scripts/submit_feature_comp_array.sh
```

### 4) Evaluate Models

Use scripts under `python_scripts/model_evaluation/` for held-out risk scoring, mortality trajectory generation/clustering, and within-vs-pan cohort comparisons.

### 5) Biomarker Analysis

Run via the orchestration notebook or as individual scripts:

```bash
# Step 1: Regenerate covariates (ensures SV/Fusion columns are correct)
python python_scripts/data_preprocessing/generate_all_non_text_covariates.py

# Step 2: Build line-matched cohorts (1:1 and 1:k)
python python_scripts/biomarker_analysis/build_line_matched_cohort.py

# Step 3: Train propensity models for each matching scheme
python python_scripts/biomarker_analysis/ICI_LRs.py --matching 1to1
python python_scripts/biomarker_analysis/ICI_LRs.py --matching 1tok

# Step 4: Generate IPTW datasets (4 specs: 2 matching x 2 PS models)
python python_scripts/biomarker_analysis/generate_IPTW_df.py --matching 1to1 --ps_model embeddings_only
python python_scripts/biomarker_analysis/generate_IPTW_df.py --matching 1to1 --ps_model all_covariates
python python_scripts/biomarker_analysis/generate_IPTW_df.py --matching 1tok --ps_model embeddings_only
python python_scripts/biomarker_analysis/generate_IPTW_df.py --matching 1tok --ps_model all_covariates

# Step 5: Run Cox models (Track 1 + Track 2 for each spec)
python python_scripts/biomarker_analysis/run_IPTW_analysis.py --matching 1to1 --ps_model embeddings_only
python python_scripts/biomarker_analysis/run_IPTW_analysis.py --matching 1to1 --ps_model all_covariates
python python_scripts/biomarker_analysis/run_IPTW_analysis.py --matching 1tok --ps_model embeddings_only
python python_scripts/biomarker_analysis/run_IPTW_analysis.py --matching 1tok --ps_model all_covariates
```

## Notes

- Most scripts use hardcoded HPC data paths (configurable via constants at the top of each file).
- The `embed_surv_utils` package should be installed in development mode (`pip install -e python_utils/`).
- Key dependencies: `pandas`, `numpy`, `scikit-learn`, `sksurv`, `lifelines`, `statsmodels`, `torch`, `transformers`, `joblib`, `tqdm`.
