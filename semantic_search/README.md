# semantic_search — patient-level analysis of note embeddings

This arm mean-pools progress notes, imaging reports, and pathology reports at the patient level.
Its exploratory branch computes and tests principal components separately within each note type;
its supervised branch uses their concatenated 3×768-dimensional representation to predict selected
clinical labels.

## Feature space and windows

| Space | Dim | Meaning |
|---|---:|---|
| `concat` | 2304 | progress, imaging, and pathology patient means side by side |
| `clinician` | 768 | progress-note patient mean; used for exploratory PCA/associations |
| `imaging` | 768 | imaging-report patient mean; used for exploratory PCA/associations |
| `pathology` | 768 | pathology-report patient mean; used for exploratory PCA/associations |

| Window | Notes included |
|---|---|
| `alltime` | every note a patient has, no anchor |
| `pretreatment` | notes strictly before `first_treatment_date` |

`alltime` is the default. Each note type is mean-pooled separately. The `concat` representation
requires all three blocks; each note-type representation requires only that note type. No
across-note-type mean is constructed.

## Workflow

```text
1_data/03  (knitted embeddings)
    |
    v
01_aggregate  semantic_search.aggregate_embeddings -> features/{concat,clinician,imaging,pathology}_{window}.parquet
    |
    +--> 02_pcs  semantic_search.compute_pcs         -> separate clinician/imaging/pathology PCs
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

The corresponding notebooks live in [`notebooks/5_semantic_search/`](../notebooks/5_semantic_search/):
`01_aggregate.ipynb`, `02_pcs.ipynb`, `03_pc_correlations.ipynb`, `04_predict.ipynb`, and
`5_figures.R`. The R script reads completed stage-2--4 artifacts
and renders PC-space, PC-association, and XGBoost-performance panels; it does
not rerun model fitting. Its default `pc_spaces` setting includes clinician,
imaging, and pathology, producing note-type-specific exploratory panels in one
run; its XGBoost panels consistently read the separate `concat` prediction
artifacts.

## Principal components and clinical associations

Stage 2 independently applies `L2 row normalization → StandardScaler → PCA` to clinician,
imaging, and pathology embeddings (50 components per note type by default). It does not fit PCs
across concatenated note types. It records:

| Path | Contents |
|---|---|
| `pcs/{clinician,imaging,pathology}_{window}_scores.parquet` | one row per eligible patient, `DFCI_MRN, PC1, ...` |
| `pcs/{clinician,imaging,pathology}_{window}_loadings.parquet` | long-form loadings for that note type |
| `pcs/{clinician,imaging,pathology}_{window}_transformer.joblib` | fitted normalization, scaling, and PCA pipeline |
| `pcs/{clinician,imaging,pathology}_{window}_meta.json` | source signature, dimensions, preprocessing, explained variance |
| `results/pc_explained_variance.csv` | per-PC and cumulative explained variance |

Stage 3 tests every retained PC by default; `--max-pcs N` restricts the screen to PC1–PCN.

| Clinical variable | Test | Effect reported |
|---|---|---|
| continuous | Spearman rank correlation | Spearman rho |
| categorical | Kruskal–Wallis omnibus test | epsilon-squared |
| overall survival | univariate Cox model using a standardized PC | hazard ratio per PC SD |

The clinical families are demographics, cancer type, stage, metastatic burden, first-treatment
type, LLM-derived conventional/AVPC/NEPC prostate subtype, broader treatment exposures, note
volume, and overall survival. Somatic alterations and PRS are excluded from the PC association
screen. Sparse treatment binary features retain the existing minimum-prevalence filter. The PC and
prediction workflows use the same overridable frozen prostate-label artifact.

BH-FDR is applied across all PC–variable tests within each `(space, window, family)`. This treats
the full PC screen as the family's multiplicity burden; correction is not restarted for each PC.
`results/pc_join_coverage.csv` should be checked before interpreting associations from a sparsely
matched clinical family.

## Supervised prediction

The prediction command defaults to `concat/alltime`. Within every training split it independently
compresses the three embedding blocks to 50 progress-note PCs, 25 imaging PCs, and 25 pathology
PCs, then fits one nested-CV XGBoost model to the resulting 100 predictors for each target:

| Target | Label source |
|---|---|
| cancer type | `cancer_type_df.csv.gz:CANCER_TYPE` |
| stage | `cancer_stage_df.csv.gz:CANCER_STAGE` |
| first treatment | `cohort_df.parquet:ANCHOR_DRUG_CATEG` |
| prostate subtype | neighboring LLM pipeline's conventional/AVPC/NEPC artifact |

Five outer folds produce out-of-fold predictions; three inner folds tune by log loss. Final models
are refit on the complete setup cohort. The workflow reports accuracy, balanced accuracy,
macro/weighted F1, log loss, macro one-vs-rest AUROC, average precision, and per-class metrics.
Hyperparameter tuning defaults to `n_jobs=-1`, using every CPU allocated to the process; each
XGBoost fit is single-threaded so parallel candidate/fold fits do not oversubscribe the allocation.

The saved model keeps blockwise normalization, scaling, and PCA inside its fitted pipeline. During
nested CV, equivalent transforms are learned only from the relevant training partition rather than
from patients later used for validation. Each inner fold's PC matrices are computed once in memory
and reused across all eight XGBoost parameter combinations. This avoids redundant PCA work and
shared filesystem caches while keeping the nested CV leakage-safe.

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
