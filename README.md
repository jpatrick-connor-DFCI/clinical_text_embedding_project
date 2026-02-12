# Clinical Text Embedding Project

This repository builds clinical-text embeddings and structured covariates, then trains and evaluates Cox/Coxnet survival models across multiple endpoint schemes.

## Repository Layout

### `python_scripts/`
- `data_preprocessing/`
  - Tokenization, embedding generation/knitting, ICD extraction, endpoint dataset generation, and non-text covariate generation.
- `model_training/`
  - Array-based training entrypoints:
    - `build_slurm_manifests.py`
    - `run_full_cohort_event.py`
    - `run_feature_comp_task.py`
    - `slurm_array_utils.py`
- `model_evaluation/`
  - Held-out risk scoring, mortality trajectory generation/clustering, and within-vs-pan cohort comparisons.
- `biomarker_analysis/`
  - Biomarker IPTW and risk-based analyses.
- `treatment_analysis/`
  - Treatment line prediction and treatment-specific modeling workflows.

### `python_utils/`
- `embed_surv_utils/`
  - Shared preprocessing helpers and Cox/Coxnet model utilities used across scripts.

### `bash_scripts/`
- SLURM submission scripts for full-cohort and feature-comparison array jobs.
- `slurm_manifests/` for generated task TSV files.

## Core Workflow

### 1) Preprocessing
Run preprocessing scripts in sequence as needed for your analysis:

1. `python_scripts/data_preprocessing/text_preprocessing_and_tokenization.py`
2. `python_scripts/data_preprocessing/generate_clinical_embeddings.py`
3. `python_scripts/data_preprocessing/knit_longformer_embeddings.py`
4. `python_scripts/data_preprocessing/extract_ICD_times.py`
5. `python_scripts/data_preprocessing/generate_embedding_prediction_datasets.py`
6. `python_scripts/data_preprocessing/generate_all_non_text_covariates.py`

Outputs are written to the project data directories configured in each script (many scripts currently use fixed HPC paths).

### 2) Build SLURM Task Manifests

```bash
python3 python_scripts/model_training/build_slurm_manifests.py --schemes icd3 icd4 phecode death_met
```

By default this writes manifests to:
`bash_scripts/slurm_manifests/`

### 3) Submit Training Arrays

```bash
bash bash_scripts/submit_full_cohort_array.sh
bash bash_scripts/submit_feature_comp_light_array.sh
bash bash_scripts/submit_feature_comp_heavy_array.sh
```

`feature_comp_heavy` currently includes `text`, `prs`, `labs`, and `somatic`.

### 4) Evaluate Models

Use scripts under `python_scripts/model_evaluation/` for:
- Feature-specific held-out risk scoring
- Mortality trajectory generation and clustering
- Within-group vs pan-group comparisons

## Notes

- Most scripts are executable scripts (not packaged CLIs) with path constants defined in-file.
- Ensure your runtime environment includes `pandas`, `numpy`, `scikit-learn`, `sksurv`, `lifelines`, `statsmodels`, `torch`, and `transformers` as required by the specific script.
