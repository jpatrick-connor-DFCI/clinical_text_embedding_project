# semantic_search — patient-level analysis of note embeddings

This arm uses one patient representation: the progress-note, imaging-report, and
pathology-report means concatenated into a 3×768-dimensional vector. Its exploratory branch
computes principal components and tests them against clinical characteristics; its supervised
branch predicts selected clinical labels from the same representation.

## Feature space and windows

| Space | Dim | Meaning |
|---|---:|---|
| `concat` | 2304 | progress, imaging, and pathology patient means side by side |

| Window | Notes included |
|---|---|
| `alltime` | every note a patient has, no anchor |
| `pretreatment` | notes strictly before `first_treatment_date` |

`alltime` is the default. Each note type is mean-pooled separately and retains its own named
768-dimensional block. Patients missing any block are complete-cased out. No across-note-type
mean is constructed.

## Workflow

```text
1_data/03  (knitted embeddings)
    |
    v
01_aggregate  semantic_search.aggregate_embeddings -> features/concat_{window}.parquet
    |
    +--> 02_pcs  semantic_search.compute_pcs         -> pcs/* scores, loadings, transformer, meta
    |       |
    |       v
    |    03_pc_correlations  semantic_search.correlate_pcs
    |                                                -> results/pc_*.csv
    |
    +--> 04_predict  semantic_search.train_prediction_models
                                                     -> predictions/*.parquet
                                                     -> models/*.joblib
                                                     -> prediction_meta/*.json
                                                     -> results/prediction_*.csv
```

Nothing outside `semantic_search` depends on this exploratory arm.

Run from the repository root:

```bash
python -m semantic_search.aggregate_embeddings [--windows ...] [--overwrite]
python -m semantic_search.compute_pcs [--windows ...] [--n-components 50] [--overwrite]
python -m semantic_search.correlate_pcs [--windows ...] [--max-pcs N]
python -m semantic_search.train_prediction_models [--targets ...] [--overwrite]
```

The corresponding notebooks are `01_aggregate.ipynb`, `02_pcs.ipynb`,
`03_pc_correlations.ipynb`, and `04_predict.ipynb`.

## Principal components and clinical associations

Stage 2 applies `L2 row normalization → StandardScaler → PCA`. It defaults to 50 components and
records:

| Path | Contents |
|---|---|
| `pcs/concat_{window}_scores.parquet` | one row per patient, `DFCI_MRN, PC1, ...` |
| `pcs/concat_{window}_loadings.parquet` | long-form feature loadings with note type and embedding dimension |
| `pcs/concat_{window}_transformer.joblib` | fitted normalization, scaling, and PCA pipeline |
| `pcs/concat_{window}_meta.json` | source signature, dimensions, preprocessing, explained variance |
| `results/pc_explained_variance.csv` | per-PC and cumulative explained variance |

Stage 3 tests every retained PC by default; `--max-pcs N` restricts the screen to PC1–PCN.

| Clinical variable | Test | Effect reported |
|---|---|---|
| continuous | Spearman rank correlation | Spearman rho |
| categorical | Kruskal–Wallis omnibus test | epsilon-squared |
| overall survival | univariate Cox model using a standardized PC | hazard ratio per PC SD |

The clinical families are demographics, cancer type, stage, metastatic burden, first-treatment
type, LLM-derived conventional/AVPC/NEPC prostate subtype, broader treatment exposures, somatic
alterations, PRS, note volume, and overall survival. Sparse treatment and somatic binary features
retain the existing minimum-prevalence filters. The PC and prediction workflows use the same
overridable frozen prostate-label artifact.

BH-FDR is applied across all PC–variable tests within each `(space, window, family)`. This treats
the full PC screen as the family's multiplicity burden; correction is not restarted for each PC.
`results/pc_join_coverage.csv` should be checked before interpreting associations from a sparsely
matched clinical family.

## Supervised prediction

The prediction command defaults to `concat/alltime` and fits nested-CV elastic-net logistic
regression and XGBoost models for:

| Target | Label source |
|---|---|
| cancer type | `cancer_type_df.csv.gz:CANCER_TYPE` |
| stage | `cancer_stage_df.csv.gz:CANCER_STAGE` |
| first treatment | `cohort_df.parquet:ANCHOR_DRUG_CATEG` |
| prostate subtype | neighboring LLM pipeline's conventional/AVPC/NEPC artifact |

Five outer folds produce out-of-fold predictions; three inner folds tune by log loss. Final models
are refit on the complete setup cohort. The workflow reports accuracy, balanced accuracy,
macro/weighted F1, log loss, macro one-vs-rest AUROC, average precision, and per-class metrics.

## Interpretation

These analyses are exploratory. With `alltime` embeddings, notes can directly state diagnosis,
stage, treatment, and phenotype evidence. The PC associations and prediction results therefore
measure retrospective clinical-content recovery, not prospective prediction or independent
validation.

The note-volume family is a confound check: associations with note counts, note span, or mean note
year can reveal documentation-intensity or calendar-time structure in a PC. Survival tests on
`alltime` PCs can include notes written after treatment or near the event and must not be read as
prospective prognostic effects. The optional `pretreatment` window remains available as a
sensitivity analysis, but it is not enforced.

Available covariates do not include labs, vitals, race/ethnicity, smoking, ECOG/performance status,
or PFS. Age and gender are the only demographics; OS is the only time-to-event endpoint.
