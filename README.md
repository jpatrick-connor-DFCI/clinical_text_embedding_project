# Clinical Text Embedding Project

This project investigates whether dense representations of clinical narratives (EHR notes) can improve survival prediction and identify genomic biomarkers of treatment response in oncology. It embeds clinical notes using Clinical-Longformer, trains penalized Cox survival models across hundreds of clinical endpoints, and runs an IPTW-weighted biomarker discovery pipeline for immune checkpoint inhibitor (ICI) response.

## Project Goals

1. **Text-based survival prediction.** Generate 768-dimensional embeddings from pre-treatment clinical notes (clinician notes, imaging reports, pathology reports) using Clinical-Longformer, then evaluate whether these embeddings improve time-to-event prediction for mortality, metastasis, and incident ICD-10/phecode diagnoses beyond standard clinical and genomic covariates.

2. **Feature modality comparison.** Systematically compare the predictive value of six data modalities — text embeddings, somatic mutations (SNV/AMP/DEL), polygenic risk scores (PRS), cancer stage, first-line treatment class, and pre-treatment lab values — for hundreds of clinical endpoints.

3. **Mortality trajectory analysis.** Track how embedding-based mortality risk scores evolve over time by refitting models at rolling monthly windows, then cluster patients into distinct risk progression patterns.

4. **ICI biomarker discovery.** Use text-embedding-derived propensity scores to construct balanced ICI vs. never-ICI cohorts, then screen hundreds of genomic markers for predictive (treatment-modifying) and prognostic associations via IPTW-weighted Cox models with interaction terms.

---

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

## Pipeline 1: Data Preprocessing

Transforms raw EHR data into analysis-ready embedding and covariate matrices. Run in sequence:

```bash
python python_scripts/data_preprocessing/text_preprocessing_and_tokenization.py
python python_scripts/data_preprocessing/generate_clinical_embeddings.py       # GPU required
python python_scripts/data_preprocessing/knit_longformer_embeddings.py
python python_scripts/data_preprocessing/extract_ICD_times.py
Rscript python_scripts/data_preprocessing/generate_icd10_to_phecode_mapping.R
python python_scripts/data_preprocessing/generate_embedding_prediction_datasets.py
python python_scripts/data_preprocessing/generate_all_non_text_covariates.py
```

### Logic and design decisions

**Text processing** (`text_preprocessing_and_tokenization.py`): Reads raw clinical notes from JSON files covering three note types (clinician, imaging, pathology). Filters to cohort MRNs, deduplicates text entries within each note, normalizes whitespace and special characters, and batches into 50K-note chunks. Tokenizes with the Clinical-Longformer tokenizer (max-length truncation + padding). Batching avoids loading the full corpus into memory at once.

**Embedding generation** (`generate_clinical_embeddings.py`): Runs on GPU (originally on GCP). Loads tokenized batches into a PyTorch DataLoader (batch size 64), extracts the [CLS] pooler output (768 dims) from Clinical-Longformer, and saves per-batch embedding tensors. Uses `torch.no_grad()` for inference efficiency.

**Embedding assembly** (`knit_longformer_embeddings.py`): Concatenates per-batch embeddings and metadata into a single NumPy array and CSV. Adds derived temporal columns (note datetime, days relative to first treatment) needed for time-decay pooling downstream.

**ICD extraction** (`extract_ICD_times.py`): Unpacks multi-code diagnosis records (up to 3 ICD-10 codes per row) into long format with one code per row. Computes time-to-ICD relative to first treatment date for survival modeling.

**Phecode mapping** (`generate_icd10_to_phecode_mapping.R`): Builds a comprehensive ICD-10-CM to phecode mapping using Phecode v1.2 as primary, PhecodeX v1.0 to fill gaps, and parent-code fallback for remaining unmapped codes. Outputs the mapping CSV and a list of unmapped codes.

**Survival dataset generation** (`generate_embedding_prediction_datasets.py`): Pools per-patient note embeddings using time-decay weighted mean (decay parameter 0.01, pre-treatment notes only) separately by note type. Creates survival prediction datasets at four endpoint granularity levels:
- **Death + metastasis**: mortality and 7 site-specific metastasis endpoints.
- **ICD-10 level 3**: 3-character ICD-10 codes (e.g., E11), excluding neoplasms (C/D00–D49), external causes (V–Y), and pregnancy (O).
- **ICD-10 level 4**: 4-character codes (e.g., E11.6), same exclusions.
- **Phecodes**: mapped via the R-generated mapping, excluding neoplasm, pregnancy, and injury phecode ranges.

Each level generates two variants: first-ever instance and first post-treatment instance. Endpoints are pre-filtered to those with ≥100 post-baseline events. A shared union of codes across both variants ensures consistent endpoint sets.

**Non-text covariates** (`generate_all_non_text_covariates.py`): Compiles clinical and genomic feature matrices:
- **Cancer type**: One-hot encoded, collapsing types with <500 patients to OTHER.
- **Cancer stage**: Derived from text-extracted staging data, one-hot encoded.
- **Somatic mutations**: Binary carrier matrices for SNV, AMP, and DEL from PROFILE sequencing data (CNV columns dropped as redundant with AMP/DEL). Sample selection uses the sequencing specimen closest to first treatment per patient.
- **Structural variants / fusions**: From SOMATIC_SV_RESULTS. Gene pairs are canonicalized (alphabetically sorted). Genes with a dominant fusion partner (≥50% of SVs, ≥20 cases) get split into `GENE_PARTNER_FUSION` and `GENE_OTHER_SV` features; others get a single `GENE_SV` indicator. All mutation-tested samples serve as the universe (absence = 0). Features with <10 positive samples are dropped.
- **PRS**: Polygenic risk scores from PGS Catalog, linked via cBioPortal sample IDs.
- **Treatment classes**: Medication names mapped to mechanism-of-action categories (GPT-generated classification), one-hot encoded by treatment line.
- **Lab values**: Mean and SD of top-40 most common outpatient labs from the pre-treatment period, with mean imputation and missingness indicators.

---

## Pipeline 2: Survival Model Training

Trains penalized Cox proportional hazards models at scale across all endpoint schemes using SLURM array jobs.

### Logic and design decisions

**Manifest generation** (`build_slurm_manifests.py`): Enumerates all valid (scheme, event) combinations across the active endpoint schemes (`icd3_post`, `icd4_post`, `phecode_post`, `death_met`). Filters to events with ≥50 positive cases and ≥50 censored observations. Supports `--skip-completed` to resume interrupted runs. Outputs TSV manifests consumed by SLURM array scripts.

**Full-cohort training** (`run_full_cohort_event.py`): For each endpoint, trains two models:
1. **Text model**: Base covariates (age, sex) + cancer type dummies (unpenalized) + text embeddings (penalized via elastic net).
2. **Base model**: Age + cancer type only (minimal regularization).

Uses 80/20 stratified train+val/test split. 5-fold stratified CV over a 25-point log-spaced alpha grid × 2 L1 ratios (0.5, 1.0), selecting the hyperparameters maximizing mean time-dependent AUC. Final model is refit on full train+val and evaluated on test.

**Feature comparison** (`run_feature_comp_task.py`): Identical CV pipeline but with one modality at a time: stage, treatment, labs, somatic, PRS, or text. PRS uses randomized PCA (1500 components) before Cox fitting to handle the high-dimensional score matrix. Enables direct modality-vs-modality comparison on held-out data.

**Shared utilities** (`slurm_array_utils.py`): Centralizes dataset loading, event extraction, feature column identification, modality configuration (which columns, PCA settings), output path management, and Cox input validation (constant-column removal, event/censoring checks).

**SLURM orchestration** (`bash_scripts/`):
- `launch_full_cohort.sh` / `array_full_cohort_run.sh`: Submit and execute full-cohort training jobs. Each SLURM array task processes multiple manifest rows (configurable `ROWS_PER_TASK`). Thread pinning (`OMP_NUM_THREADS=1`) prevents oversubscription with joblib parallelism.
- `launch_feature_comp.sh` / `array_feature_comp.sh`: Same pattern for feature-comparison tasks.

### Running

```bash
# 1. Generate manifests
python python_scripts/model_training/build_slurm_manifests.py --schemes icd3_post icd4_post phecode_post death_met

# 2. Submit SLURM arrays
bash bash_scripts/launch_full_cohort.sh
bash bash_scripts/launch_feature_comp.sh
```

---

## Pipeline 3: Model Evaluation

Evaluates trained models through held-out risk scoring, mortality trajectory analysis, and stratified model comparisons.

### Logic and design decisions

**Held-out risk scores** (`feature_ICD10_level_3_risk_scores.py`): For each endpoint, loads the best hyperparameters from each modality's CV results and generates 5-fold held-out risk scores using `get_heldout_risk_scores_CoxPH`. Produces per-patient risk score DataFrames that enable cross-modality comparison on the same held-out patients. Metastatic endpoints additionally exclude patients with baseline metastasis.

**Mortality trajectories** (`generate_mortality_trajectories.py`): Fits a pan-cancer embedding-based Cox model at time 0, then generates risk scores at rolling 3-month intervals (3, 6, ..., 60 months post-treatment). At each interval, re-pools embeddings using all notes up to that time point with time-decay weighting (decay=1.0 for sharper recency focus). Produces a patient × time matrix of risk scores that captures how predicted mortality evolves with accumulating clinical information.

**Trajectory clustering** (`cluster_mortality_trajectories.py`): Extracts 12 trajectory features per patient (start/end risk, slope, AUC, early/mid/late means, min/max values, rebound, early vs late slope). Runs Ward-linkage hierarchical clustering for k=2–10 on both raw and z-scored trajectories. Visualizes cluster mean trajectories with 95% confidence bands.

**Pan vs within-cancer** (`within_vs_pan_cancer_models.py`): Tests whether cancer-type-specific embedding models improve over a single pan-cancer model. Trains one pan-cancer model (with cancer type dummies) and separate within-cancer models (no type dummies), then compares held-out C-indices overall and per cancer type.

**Pan vs within-treatment** (`within_treatment_vs_pan_treatment_models.py`): Same design but stratified by first-line treatment class. Tests whether treatment-specific embedding models outperform a pooled model.

---

## Pipeline 4: Biomarker Analysis

Discovers genomic biomarkers associated with ICI treatment response using propensity-score-weighted survival models. Orchestrated via `run_biomarker_pipeline.ipynb` or individual scripts.

### Design overview

The pipeline addresses confounding in observational ICI data through a multi-specification sensitivity analysis: 1:1 matching × 2 propensity models × 2 analysis tracks × multiple weighting strategies. Markers significant across specifications are considered robust.

### Logic and design decisions

**Line-matched cohort construction** (`build_line_matched_cohort.py`): ICI receipt is confounded by treatment line (later lines more likely to include ICI). Patients are matched exactly on (cancer_type, line_category) where line is binned as 1/2/3/4+. For ICI patients, line = earliest line with ICI; for never-ICI controls, line = max line of therapy reached. Uses 1:1 matching (one control per case, without replacement). This ensures treated and control groups have similar disease severity and treatment history.

**Propensity score generation** (`ICI_train_propensity.py`): Trains logistic regression models predicting ICI receipt within each matched cohort. Two model specifications are used in the current code:
1. **`covariates_only`**: demographics + cancer type (+ line dummies for cohort 2).
2. **`covariates_plus_embeddings`**: the same covariates plus clinical text embeddings.

Uses 5-fold stratified CV to produce held-out propensity scores, avoiding overfitting. The current script is configured for the 30-day buffer used downstream.

**IPTW dataset generation** (`generate_IPTW_df.py`): Merges propensity scores with genomic markers (somatic mutations: SNV, AMP, DEL, SV, fusions), cancer type, panel version, and line category dummies. Produces one analysis-ready CSV per specification.

**IPTW Cox analysis** (`run_IPTW_analysis.py`): Runs two analysis tracks across all cancer types with sufficient data (dynamically discovered from the one-hot cancer type columns, skipping types with too few testable markers):

- **Track 2** (full-cohort interaction): `S(t) ~ base_vars + line_dummies + marker + ICI + marker × ICI`. The interaction coefficient identifies *predictive* markers (differential effect in ICI vs non-ICI). Runs with ATE weights, ATT weights, and unweighted as sensitivity analyses. Uses stabilized weights with common-support trimming and truncation at 1st/99th percentiles.

- **Track 1** (ICI-only generalizability): `S(t) ~ base_vars + line_dummies + marker`. Identifies markers prognostic within ICI-treated patients. Uses generalizability weights (1/ps) to reweight ICI patients toward the full eligible population, plus an unweighted comparison.

Both tracks use robust standard errors, FDR correction within mutation type (SNV, AMP, DEL, SV, FUSION), and flag extreme hazard ratios (>50 or <1/50) indicating possible model separation. Cancer-type-specific analyses recalibrate propensity scores via Platt scaling within each subset.

Markers must pass minimum support thresholds: ≥10 positive cases per treatment arm, ≥10 negative cases per arm, and ≥5 events among marker-positive patients.

**Results compilation** (`compile_IPTW_results.py`): Aggregates significant hits across all specifications. Cross-scheme robustness filtering identifies markers significant in ≥2 specifications with consistent effect direction and no extreme HRs. Cancer types are discovered dynamically from output filenames.

**Shared utilities** (`biomarker_common.py`): Standardized loading functions for embeddings, survival data, and confounders. Defines the canonical mutation type tags (`_SNV`, `_SV`, `_FUSION`, `_DEL`, `_AMP`) used for FDR grouping.

### Running

```bash
# Or run the orchestration notebook: python_scripts/biomarker_analysis/run_biomarker_pipeline.ipynb

# Step 1: Regenerate covariates
python python_scripts/data_preprocessing/generate_all_non_text_covariates.py

# Step 2: Build line-matched cohorts
python python_scripts/biomarker_analysis/build_line_matched_cohort.py

# Step 3: Train propensity models
python python_scripts/biomarker_analysis/ICI_train_propensity.py

# Step 4: Generate IPTW datasets for both PS models
python python_scripts/biomarker_analysis/generate_IPTW_df.py

# Step 5: Run Cox models across cohorts and PS models
python python_scripts/biomarker_analysis/run_IPTW_analysis.py

# Step 6: Compile results
python python_scripts/biomarker_analysis/compile_IPTW_results.py

# Step 7: Build or update the validated findings report
python python_scripts/biomarker_analysis/validate_and_report.py
```

---

## Shared Utilities: `embed_surv_utils`

Installable Python package (`pip install -e python_utils/`) providing core functions used across all pipelines.

### `preprocessing.py`
- `clean_text` / `deduplicate_texts`: Text normalization and deduplication for raw clinical notes.
- `map_time_to_event`: Maps patient IDs to time-to-event and event indicators, handling both observed events and censoring.
- `find_continuous_records_to_analyze`: Identifies continuous note sequences within a time window, splitting on gaps >2 years. Selects the latest pre-treatment segment.
- `pool_embedding_series_vectorized`: Pools per-note embeddings to per-patient vectors using configurable strategies (mean, most-recent, time-decay weighted mean). Computes year-based adjustments (% notes pre-2015) for imaging and pathology.
- `generate_survival_embedding_df`: End-to-end orchestrator: selects notes within a window, pools embeddings, and optionally merges with survival outcomes.

### `cox_models.py`
- `run_grid_CoxPH_parallel`: Parallelized grid search over L1 ratio × alpha for `CoxnetSurvivalAnalysis`. Automatically switches between two execution paths: (a) in-RAM path when no PCA is needed (fastest for dense text embeddings), (b) precomputed fold matrices with memory-mapped arrays when PCA is applied (avoids redundant PCA recomputation). Supports parallelization over folds or L1 values based on grid size.
- `run_base_CoxPH`: Fits an unpenalized `CoxPHSurvivalAnalysis` baseline model with 5-fold CV.
- `get_heldout_risk_scores_CoxPH`: Generates 5-fold held-out risk scores using the same auto-switching logic. Supports both penalized (CoxNet) and unpenalized (CoxPH) models.
- `evaluate_surv_model`: Computes time-dependent AUC, integrated Brier score, and C-index on held-out data.
- `apply_group_pca_np`: Applies randomized PCA to a named group of columns, replacing them with PC components.

---

## Jupyter Notebooks

| Directory | Notebook | Description |
|-----------|----------|-------------|
| `metrics/` | `compile_ICD_level_3_results.ipynb` | Compiles full-cohort CoxPH results across ICD-10 level 3 endpoints. |
| `metrics/` | `compile_feature_comps_ICD_level_3.ipynb` | Compiles feature-comparison results (text vs somatic vs labs vs PRS vs stage vs treatment). |
| `metrics/` | `analyze_landmark_results.ipynb` | Analysis of landmark survival model performance across time horizons. |
| `metrics/` | `analyze_mortality_trajectories.ipynb` | Interactive visualization of mortality risk score trajectories. |
| `metrics/` | `trajectories_vs_stage.ipynb` | Compares trajectory clusters against cancer stage. |
| `metrics/` | `debug_individual_events.ipynb` | Debugging notebook for individual event-level model fits. |
| `biomarker_analysis/` | `IPTW_KM_curves.ipynb` | Kaplan-Meier curves comparing ICI vs non-ICI patients with IPTW weights. |
| `biomarker_analysis/` | `test_marker_propensity_association.ipynb` | Tests for residual confounding between markers and propensity scores. |
| `note_embedding_EDA/` | `embedding_EDA.ipynb` | Exploratory analysis of embedding distributions and clustering patterns. |
| `note_embedding_EDA/` | `association_testing_for_PCA.ipynb` | Tests associations between PCA components and clinical variables. |
| `manuscript_figures/` | `figure_1.ipynb` | Generates Figure 1 for the manuscript. |
| `meeting_figures/` | `07_14_2025_lab_meeting.ipynb` | Figures for the July 14, 2025 lab meeting. |
| `meeting_figures/` | `ASHG_figures.ipynb` | Figures for the ASHG conference presentation. |

---

## Notes

- All scripts use hardcoded HPC data paths (configurable via constants at the top of each file).
- The `embed_surv_utils` package should be installed in development mode: `pip install -e python_utils/`.
- Key dependencies: `pandas`, `numpy`, `scikit-learn`, `scikit-survival`, `lifelines`, `statsmodels`, `torch`, `transformers`, `joblib`, `tqdm`, `icd10`, `matplotlib`, `seaborn`.
- The embedding generation step (`generate_clinical_embeddings.py`) requires GPU access and was originally run on GCP.
- SLURM scripts assume a `clinical_notes_project` conda environment with `embed_surv_utils` installed.
