# semantic_search — patient-level analysis of note embeddings

This arm uses one patient representation directly: the progress-note, imaging-report, and
pathology-report means concatenated into a 3×768-dimensional vector. Its unsupervised branch
clusters patients; its supervised branch predicts clinical labels from the same representation.

## Feature space and windows

| Space | Dim | Meaning |
|---|---|---|
| `concat` | 2304 | progress, imaging, and pathology patient means side by side |

| Window | Notes included |
|---|---|
| `alltime` | every note a patient has, no anchor |
| `pretreatment` | notes strictly before `first_treatment_date` — leak-free against `tt_death` |

`alltime` is the default throughout this arm. `pretreatment` remains an optional sensitivity
window and is never required by the prediction workflow.

Each note type is mean-pooled separately and retains its own named 768-dimensional block. No
across-note-type mean is constructed. Patients missing any of the three blocks are complete-cased
out, so every modeled row has exactly 2,304 finite embedding features.

Pooling is a plain unweighted `mean`, not the production `time_decay_mean`: this arm asks what a
patient's notes say on average, not what they said most recently.

## DAG

```
1_data/03  (knitted embeddings)
    |
    v
01_aggregate     semantic_search.aggregate_embeddings   -> features/concat_{window}.parquet
    |
    v
02_cluster       semantic_search.cluster_patients       -> clusters/concat_{window}_labels.parquet
    |                                                      clusters/concat_{window}_coords.parquet
    |                                                      clusters/concat_{window}_meta.json
    v
03_characterize  semantic_search.characterize_clusters  -> results/*.csv, figures/*.png

01_aggregate
    |
    v
04_predict       semantic_search.train_prediction_models -> predictions/*.parquet
                                                          -> models/*.joblib
                                                          -> prediction_meta/*.json
                                                          -> results/prediction_*.csv
```

Nothing outside `semantic_search` depends on this arm.

Every module is runnable directly from the repo root:

```bash
python -m semantic_search.aggregate_embeddings [--windows ...] [--overwrite] [--limit-mrns N]
python -m semantic_search.cluster_patients [--spaces ...] [--windows ...] [--k N] [--overwrite]
python -m semantic_search.characterize_clusters [--spaces ...] [--windows ...]
python -m semantic_search.train_prediction_models [--targets ...] [--spaces ...] [--overwrite]
```

The same prediction workflow, including pre-flight checks and result summaries, is available in
`semantic_search/notebooks/04_predict.ipynb`.

Stages 1, 2, and 4 are skip-if-exists resumable; pass `--overwrite` to rebuild.

## Where data lands

All under `SEMANTIC_SEARCH_PATH` (`DATA_PATH/semantic_search/`, set in `config.py`):

| Path | Written by | Contents |
|---|---|---|
| `features/concat_{window}.parquet` | 01 | `DFCI_MRN` + three named embedding blocks |
| `clusters/concat_{window}_labels.parquet` | 02 | `DFCI_MRN, cluster` |
| `clusters/concat_{window}_coords.parquet` | 02 | `DFCI_MRN, dim1, dim2` (first two PCs) |
| `clusters/concat_{window}_meta.json` | 02 | k, seed, silhouette, PCA variance, cluster sizes |
| `results/*.csv` | 01–04 | summaries, tests, enrichment, survival, prediction metrics, manifest |
| `figures/*.png` | 03 | notebook-03 panels for every space/window partition |
| `predictions/{target}__{space}__{window}__{model}.parquet` | 04 | one out-of-fold prediction per patient |
| `models/{target}__{space}__{window}__{model}.joblib` | 04 | final full-cohort fitted estimator, classes, and feature order |
| `prediction_meta/{target}__{space}__{window}__{model}.json` | 04 | CV settings, tuning choices, class counts, metrics, artifact paths |
| `results/prediction_*.csv` | 04 | fold, pooled, per-class, and cohort-size comparisons |

## Supervised prediction

The command defaults to the 3×768 `concat` space on `alltime` embeddings; there is no pre-treatment
restriction and no single-note-type or across-note-type alternative.

The four outcomes come from frozen upstream artifacts:

| Target | Label source | Handling |
|---|---|---|
| `cancer_type` | `cancer_type_df.csv.gz:CANCER_TYPE` | uses the upstream ≥500/`OTHER` grouping |
| `stage` | `cancer_stage_df.csv.gz:CANCER_STAGE` | shared normalizer, major stages I–IV |
| `first_treatment` | `cohort_df.parquet:ANCHOR_DRUG_CATEG` | first medication slot; sparse categories become `OTHER` |
| `prostate_subtype` | `avpc_nepc_labels.parquet` from `LLM_clinical_annotations` | NEPC precedence, then AVPC, else conventional |

The prostate loader uses only rows in the LLM pipeline's cohort-complete artifact. It does not
label absent patients conventional. Set `AVPC_NEPC_LABELS_PATH` or pass `--avpc-nepc-labels` to
select a frozen upstream label run. `--treatment-granularity drug` switches the treatment target
from the curated category to the exact first drug.

Each setup uses nested stratified CV: five outer folds produce honest out-of-fold predictions and
three inner folds select hyperparameters by log loss. Elastic net searches `C` and `l1_ratio` after
training-fold standardization. XGBoost searches depth, child weight, and L2 regularization with
histogram trees. The script reports accuracy, balanced accuracy, macro/weighted F1, log loss,
macro one-vs-rest AUROC, macro average precision, and per-class metrics, then tunes and refits one
final estimator on the complete setup cohort.

These default `alltime` runs are retrospective label-recovery experiments: notes can directly state
diagnosis, stage, treatment, and the same phenotype evidence used by the LLM labeler. This is the
intended analysis here, but it should not be interpreted as prospective prediction or independent
clinical validation. A pre-treatment sensitivity analysis remains available with
`--windows pretreatment`, and both windows can be run with `--windows alltime pretreatment`.

## Method notes

**Preprocessing order** is fixed and recorded in each run's meta JSON:
`L2-normalize → StandardScaler → PCA(50) → KMeans(k)`. L2 first because these are transformer
embeddings whose trained geometry is cosine; on unit-norm rows Euclidean KMeans is monotone in
cosine distance. StandardScaler then stops a few high-variance dimensions or one note-type block
from dominating.

**Cluster labels are relabeled by ascending size** (0 = smallest), so a rerun with the same seed
gives not just the same partition but the same integer names. KMeans label integers are otherwise an
artifact of centroid initialization order, which would make per-cluster tables incomparable across
runs.

**k defaults to the silhouette argmax** over `range(2, 13)`; `--k` forces a value. Silhouette is
estimated on a seeded 20k subsample when the cohort is larger, since it is O(n²).

**Coordinates are PCA, not UMAP** — `umap-learn` is not in `environment.yml`, and PCA keeps the
scatter in the same geometry the clustering actually used.

**Adjusted Rand index** compares the all-time and pre-treatment partitions when both windows are
run (`cluster_concordance.csv`, on shared MRNs only).

**BH-FDR is applied within (space, window, family), never pooled**, mirroring
`_fdr_within_mutation_type` in `run_IPTW_analysis.py`. The somatic and PRS families carry hundreds
of tests each; pooling them with the two demographic tests would bury the latter under a correction
they did not earn. Wide families are also capped by a prevalence floor — a marker present in 5
patients cannot separate clusters and only inflates the family's denominator.

## Reading the results: two things to check first

**`cluster_note_volume.csv` is a confound check, not a finding.** An unweighted mean over a
patient's notes encodes *how much* was written about them as well as *what* was written. If clusters
separate mainly on note counts, the partition is a documentation-intensity artifact. The follow-up
is to residualize note count out, or switch to `time_decay_mean` pooling.

**The adjusted Cox is the load-bearing survival number.** Pathology notes name the tumor, so a
cluster/OS association that vanishes after adjusting for age, gender and cancer type is a
restatement of the diagnosis rather than a new axis. `cox_hr` vs `cox_hr_adjusted` in
`cluster_survival.csv` is where that shows up, and the side-by-side scatter in notebook 03 makes it
visible at a glance.

## Caveats

This arm is **exploratory**.

- No held-out validation, and k is chosen on the same data the clusters are then described with.
- FDR is applied within each window and clinical-variable family, not across windows.
- `alltime` is **not leak-free** against survival: it includes notes written after treatment start,
  and after death for decedents. Use `pretreatment` for anything involving `tt_death`.
- Cancer type is the usual explanation for any structure found here. Check it before reading a
  cluster as a novel phenotype.

## What this repo does not have

Absent, and so absent from the characterization: labs, vitals, race/ethnicity, smoking,
ECOG/performance status, and PFS. Age and gender are the only demographics; OS is the only
time-to-event endpoint.
