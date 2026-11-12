# `semantic_search/` data folder

This folder holds two kinds of data, one row per cancer patient:

1. **Patient embeddings.** Each patient's clinical notes are turned into numeric vectors
   (one per note) and averaged into one vector per patient for each note type.
2. **Prediction results.** Classifiers predict clinical labels (cancer type, stage,
   treatment, …) from those patient embeddings. This folder holds each patient's
   predicted labels and the models and metrics behind them.

It also contains a smaller exploratory set of principal components (PCs) and their clinical
associations. Those are summarized [at the end](#4-exploratory-pcs).

```
$DATA_PATH/semantic_search/        # config.SEMANTIC_SEARCH_PATH
```

On the cluster, `DATA_PATH` is `/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/`.
Copy this file to that folder as `README.md`. For the code, see [`README.md`](README.md) in
this directory.

**This folder holds PHI.** Every file is keyed by `DFCI_MRN` (Int64, one row per patient).

```
semantic_search/
├── features/          patient-level embeddings                      ← main data
├── predictions/       per-patient out-of-fold predicted labels      ← main data
├── prediction_meta/   per-run audit record (metrics, settings, cohort hash)
├── models/            final fitted classifiers
├── results/           summary CSVs (metrics, class counts, PC tests)
├── pcs/               exploratory per-note-type PCs
├── clusters/          legacy, old runs only
└── figures/           legacy, always empty
```

---

## 1. Where the embeddings come from

Three upstream steps produce the note embeddings. None of them writes to this folder, but
everything in `features/` is derived from their output.

| Step | Script | What it does |
|---|---|---|
| Collect and clean notes | `pipelines/preprocessing/text_preprocessing_and_tokenization.py` | Loads every note for patients in `$SURV_PATH/cohort_df.parquet` from PROFILE_DATA, then cleans and tokenizes the text |
| Embed | `pipelines/preprocessing/generate_clinical_embeddings.py` (GPU, GCP) | Runs each note through **Clinical ModernBERT** and produces one 768-dim vector per note |
| Knit | `pipelines/preprocessing/knit_embeddings.py` | Joins the batch files into one note-level array with one metadata row per note |

### Notes included

Three note types are used. The `NOTE_TYPE` value in the metadata is shown in parentheses.

| Source table | `NOTE_TYPE` | Contents |
|---|---|---|
| `PROGRESS_NOTES` | `Clinician` | Clinician progress notes, **with discharge summaries merged in** |
| `IMAGING_NOTES` | `Imaging` | Radiology reports |
| `PATHOLOGY_NOTES` | `Pathology` | Pathology reports |

- **Patients.** Only patients in the study cohort (`cohort_df.parquet`) are included.
- **Duplicates.** Notes are already deduplicated by `RPT_ID` upstream; when the same report
  appears in more than one data pull, the newest copy is kept.
- **Missing dates.** Notes with no parseable `EVENT_DATE` are dropped during knitting.

### Text cleaning and tokenization

- Runs of whitespace are collapsed to a single space.
- Every character outside `A–Z a–z 0–9 space . , ? ! ( ) /` is replaced with a space. This
  removes characters such as `:`, `-`, `%`, `+`, `<`, `>`, `#`, and line breaks.
- Text is tokenized with the `Simonlee711/Clinical_ModernBERT` tokenizer, with no padding.
  Notes longer than the model's context window (`model_max_length`) are **truncated**: only
  the beginning of a long note is embedded. Notes are not split into chunks.

### The note embedding

- The model is [`Simonlee711/Clinical_ModernBERT`](https://huggingface.co/Simonlee711/Clinical_ModernBERT).
  It runs in BF16 inference mode, and its weights are **not fine-tuned** on this data.
- Each note's vector is the **mean of the last hidden layer over all of the note's tokens**,
  including the special tokens. The mean is accumulated in FP32.
- Each vector has 768 dimensions and is stored as **float16**. Downstream loaders cast it to
  float32.

### Knitted note-level files (upstream of this folder)

These files are in `$DATA_PATH/batched_datasets/processed_datasets/`:

| File | Contents |
|---|---|
| `full_clinical_notes_embeddings_as_array.npy.zst` | zstd-compressed `.npy` array, *N notes* × 768, float16. Row *i* belongs to metadata row `EMBEDDING_INDEX == i` |
| `full_clinical_notes_embeddings_metadata.parquet` | One row per note (columns below) |

Metadata columns:

- **Note identity:** `DFCI_MRN`, `RPT_ID`, `NOTE_TYPE`.
- **Note descriptors from PROFILE:** `EVENT_DATE`, `PROC_DESC`, `RPT_TYPE`, `INP_RPT_TYPE`,
  `PROVIDER_TYPE`, `ENCOUNTER_TYPE_DESC`.
- **Note time:** `NOTE_DATETIME`, the naive version of `EVENT_DATE`.
- **Anchor dates:** `FIRST_TREATMENT_START_DT` (from the cohort's `first_treatment_date`)
  and `SEQUENCING_DT`.
- **Note time relative to the anchors, in whole days:** `NOTE_TIME_REL_FIRST_TREATMENT_START`
  (also stored as `NOTE_TIME_REL_TREATMENT`) and `NOTE_TIME_REL_SEQUENCING`.
- **Row bookkeeping:** `EMBEDDING_INDEX`, `SUB_BATCH_FILE_ID`, `WITHIN_SUB_BATCH_INDEX`.

The note text is **not** kept in these files.

---

## 2. `features/`: patient-level embeddings

Written by `python -m semantic_search.aggregate_embeddings`.

```
features/{space}_{window}.parquet
```

Each patient's note vectors are averaged into one vector per note type. The average is an
**unweighted mean**: every note counts equally, regardless of its length or age. Notes are
never averaged across note types.

### Spaces

| `space` | Columns after `DFCI_MRN` | Who is included |
|---|---|---|
| `concat` | `CLINICIAN_EMBEDDING_0…767`, `IMAGING_EMBEDDING_0…767`, `PATHOLOGY_EMBEDDING_0…767` (2304 columns, always in this order) | Patients with **at least one note of every type** in the window |
| `clinician` | `CLINICIAN_EMBEDDING_0…767` | Patients with ≥ 1 clinician note in the window |
| `imaging` | `IMAGING_EMBEDDING_0…767` | Patients with ≥ 1 imaging report in the window |
| `pathology` | `PATHOLOGY_EMBEDDING_0…767` | Patients with ≥ 1 pathology report in the window |

All embedding columns are floats. Rows with any non-finite value are removed, so the files
have no nulls. `concat` is the input to the prediction models. The three single-type files
are used for the exploratory PCs, and they include more patients than `concat`.

### Windows

| `window` | Notes averaged |
|---|---|
| `alltime` | Every note the patient has, from before diagnosis to the last note |
| `pretreatment` | Notes with `NOTE_TIME_REL_FIRST_TREATMENT_START < 0` |

The time offset is a whole number of days, rounded toward zero. A note written less than
24 hours before treatment therefore counts as day 0 and is excluded from `pretreatment`.
Patients with no first-treatment date have no time offset and are also excluded.
`pretreatment` therefore contains **fewer, partly different patients** than `alltime`.

### Practical notes

- A patient's vector reflects how much was written about them as well as what was written.
  Patients with many notes are averaged over more text. The note-count columns in
  `results/pc_clinical_associations.csv` (`note_volume` family) can be used to check for
  this.
- `results/feature_summary.csv` gives `n_patients`, `n_features`, and the median and mean
  number of notes per patient for every space and window.
- To load the files:

  ```python
  from semantic_search.common import load_features
  df = load_features("concat", "alltime")   # polars DataFrame
  ```

---

## 3. Prediction outputs

Written by `python -m semantic_search.train_prediction_models`.

### What is predicted

Each target is a patient-level label. A patient missing from a target's source is left
unlabeled; no default label is ever filled in.

| `target` | Classes | Source and how it is defined |
|---|---|---|
| `cancer_type` | Cancer-type categories | `$FEATURE_PATH/cancer_type_df.csv.gz` `CANCER_TYPE` |
| `stage` | `I`–`IV` | Raw stage normalized by `shared.stages.normalize_stage` |
| `first_treatment` | Mechanism-of-action class of the first antineoplastic drug (uppercase `MOA_Category` values) | The cohort's `ANCHOR_DRUG` (first drug by start date) is mapped through `GPT_generated_med_classes.csv` → `MOA_Category`. Unmapped drugs become `OTHER`, and classes with < 25 patients are merged into `OTHER`. With `--treatment-granularity drug` the label is the drug name instead |
| `prostate_subtype` | `CONVENTIONAL` / `AVPC` / `NEPC` | LLM-derived labels at `AVPC_NEPC_LABELS_PATH` (LLM_annotations repo). NEPC overrides AVPC. Prostate patients only |
| `n_lines` | `1 line`, `2 lines`, `3 lines`, `4+ lines` | Lines of therapy observed so far, from `profile_lines.derive_lines_of_therapy`. Drugs that start within 28 days of each other count as one line. Includes patients still in follow-up, with **no censoring adjustment**, and PROFILE records at most 7 drugs per patient |
| `treatment_ici` | `ICI` / `NON_ICI` | **Ever** received a drug of this class, at any line. A drug belongs to the class if its `MOA_Category` matches the class's regex in `drug_classes.py` |
| `treatment_tki` | `TKI` / `NON_TKI` | Same as above |
| `treatment_monoclonal_antibody` | `MONOCLONAL_ANTIBODY` / `NON_…` | Same as above |
| `treatment_adc` | `ADC` / `NON_ADC` | Same as above |
| `treatment_estrogen` | `ESTROGEN` / `NON_ESTROGEN` | Same as above; **female patients only** |
| `treatment_androgen_axis` | `ANDROGEN_AXIS` / `NON_…` | Same as above; **male patients only** |

Patients with no medication record are left out of `n_lines` and the `treatment_*` targets.
They are not counted as negatives.

### Feature spaces used for prediction

| `space` | Model input |
|---|---|
| `concat` (default) | `features/concat_{window}.parquet`. Within each training split, each note-type block is compressed separately (L2-normalize → standardize → PCA): 50 clinician PCs + 25 imaging PCs + 25 pathology PCs = **100 predictors** |
| `concat_full` | The same file, used directly as **all 2304 dimensions** with no PCA. Tests whether compression loses information |
| `cancer_type_baseline` | **No text.** One indicator column per cancer type (`CANCER_TYPE_<NAME>`), built in memory. Shows how much of a target is explained by diagnosis alone. Not run for `cancer_type` |

### How the models are trained

- **Model:** XGBoost (`hist` tree method, 300 trees, learning rate 0.05) with class-balanced sample weights.
- **Cross-validation:** nested. There are 5 stratified outer folds (seed 1234); each outer
  training set is tuned with 3 inner folds.
- **Tuning:** a grid search over `max_depth` {3, 6} × `min_child_weight` {1, 5} ×
  `reg_lambda` {1, 10}, selected by log loss.
- **No leakage:** the PCA and scaling are fitted only on each split's training rows.
- **Final model:** after CV, the model is re-tuned and refit on every patient.
- **Cohort:** in the default `--cohort-mode common`, the run uses only patients who have
  every requested space and a label. `--cohort-from concat` reuses exactly the patients from
  the existing `concat` run (this is how the baseline is paired with it).

### File naming

`predictions/`, `models/`, and `prediction_meta/` share one file stem:

```
{target}__{space}__{window}__{model}       e.g.  treatment_ici__concat__pretreatment__xgboost
```

### `predictions/{stem}.parquet`: one row per patient

This is the main per-patient output. Every prediction comes from the outer fold that held
that patient out, so the model never saw the patient during training.

| Column | Type | Meaning |
|---|---|---|
| `DFCI_MRN` | Int64 | Sorted ascending |
| `fold` | Int16 | Outer fold (1–5) that held this patient out |
| `true_label` | String | The patient's actual label |
| `predicted_label` | String | The most probable class |
| `predicted_probability` | Float64 | Probability of `predicted_label` |
| `class_probabilities` | List[Float64] | Probability of every class, summing to 1. The order matches `classes` in the meta JSON (alphabetical) |

To expand the probabilities into one column per class:

```python
import json, polars as pl
stem = "stage__concat__alltime__xgboost"
pred = pl.read_parquet(f"predictions/{stem}.parquet")
classes = json.load(open(f"prediction_meta/{stem}.json"))["classes"]
pred = pred.with_columns(
    pl.col("class_probabilities").list.to_struct(fields=classes)
).unnest("class_probabilities")
```

### `prediction_meta/{stem}.json`: run record

| Key | Contents |
|---|---|
| `n_patients`, `classes`, `class_counts` | Cohort size and label distribution |
| `pooled_oof_metrics` | Metrics over all out-of-fold predictions (see below) |
| `fold_metrics` | The same metrics for each outer fold, with `n` |
| `per_class_metrics` | Per class: `n`, `precision`, `recall`, `f1`, `ovr_auc`, `average_precision` |
| `fold_tuning`, `final_best_params` | The hyperparameters chosen in each fold and for the final model |
| `n_features`, `n_transformed_features`, `pca_components`, `feature_transform` | Model input dimensions |
| `run_signature` | Hash of the patient → label list and of the feature columns, plus the CV settings, software versions, the feature file's size and mtime, and target-specific details. Those details include collapsed `first_treatment` classes, the med-class file used, the drug-class regexes and sex restriction, and median follow-up per `n_lines` bin (`n_lines_followup_by_bin`) for checking the censoring bias. A rerun without `--overwrite` raises an error if any of these change |

Metrics: `accuracy`, `balanced_accuracy`, `macro_f1`, `weighted_f1`, `log_loss`,
`macro_ovr_auc`, `macro_average_precision`. Binary targets use the positive-class AUC and AP.

### `models/{stem}.joblib`

This is a dict containing the final refit sklearn pipeline (`estimator`, with the block PCA
included), `classes`, `feature_columns` (the input order), `transformed_feature_columns`
(`CLINICIAN_PC1…`, …), and the `target`, `space`, and `window` keys.
`estimator.predict_proba(X)` on a matching `features/` matrix returns the class
probabilities.

### Prediction summary tables in `results/`

| File | One row per | Contents |
|---|---|---|
| `prediction_metrics_summary.csv` | target × space × window × model | Cohort size and number of classes, plus each metric as `_mean` and `_sd` across folds and `_pooled_oof` |
| `prediction_metrics_folds.csv` | … × outer fold | Each metric, plus `n` |
| `prediction_metrics_by_class.csv` | … × class | `n, precision, recall, f1, ovr_auc, average_precision` |
| `prediction_class_counts.csv` | target × space × window × cohort_mode × class | `n` per class in the trained cohort |

A rerun replaces only the rows for the setups it trained. Rows from other targets, spaces,
and windows are kept.

### How to read the predictions

- **`alltime` results measure how much clinical information can be recovered from notes
  already written, not prospective prediction.** Notes can state the diagnosis, stage, or
  drug outright.
- **`pretreatment` excludes notes written after first treatment.** For `first_treatment`,
  this means the model cannot read about the drug once it has started. The `treatment_*`
  and `n_lines` targets cover the whole treatment course, so even `pretreatment` runs
  predict a patient's future treatment, not current information.
- **The two windows have different patients, so their scores are not paired.** To compare
  spaces on the same patients, use the `cancer_type_baseline` or `concat_full` rows from the
  same window.
- **The drug-class labels come from an unvalidated, GPT-generated mapping of mechanism of
  action.** The biomarker arm defines ICI exposure from an explicit drug list instead. Run
  `python -m semantic_search.drug_classes` to see how far the two definitions agree.

---

## 4. Exploratory PCs

These files are written by `semantic_search.compute_pcs` and `semantic_search.correlate_pcs`.
They are separate from the prediction pipeline: PCA is fitted on each of the
`clinician`/`imaging`/`pathology` spaces independently (L2 → standardize → PCA, 50
components, seed 0).

| File | Contents |
|---|---|
| `pcs/{type}_{window}_scores.parquet` | `DFCI_MRN, PC1…PC50` |
| `pcs/{type}_{window}_loadings.parquet` | Long format: `pc, feature, note_type, embedding_dimension, loading, abs_loading` |
| `pcs/{type}_{window}_transformer.joblib` / `_meta.json` | The fitted pipeline, and its explained variance and the source-file signature |
| `results/pc_explained_variance.csv` | Explained variance for each PC, and cumulative |
| `results/pc_clinical_associations.csv` | One row per PC × clinical variable, using Spearman (continuous), Kruskal–Wallis (categorical), or Cox (OS). Columns `effect`/`effect_name`, `p`, `n`, and `fdr` (BH within space × window × variable family) |
| `results/pc_join_coverage.csv` | The fraction of PC patients with data in each variable family. Check this before interpreting a family |

`clusters/` and the `results/cluster_*.csv` and `results/silhouette_scan.csv` files exist
only if the retired KMeans stages were run. `figures/` is always empty; the R report writes
its PNGs elsewhere, to `$SEMANTIC_SEARCH_FIGURES_OUT`.

---

## Regenerating

Run these commands from the repository root. They are also available as notebooks in
`notebooks/5_semantic_search/`.

```bash
python -m semantic_search.aggregate_embeddings --overwrite     # features/
python -m semantic_search.train_prediction_models               # predictions/, models/, prediction_meta/, results/prediction_*
python -m semantic_search.train_prediction_models --spaces cancer_type_baseline --cohort-from concat
python -m semantic_search.compute_pcs                           # pcs/  (exploratory)
python -m semantic_search.correlate_pcs                         # results/pc_*  (exploratory)
```

The data does not include labs, vitals, race/ethnicity, smoking status, ECOG, or PFS. The
only demographics are age and gender, and the only outcome is overall survival.
