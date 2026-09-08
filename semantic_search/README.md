# semantic_search — exploratory patient clustering on note embeddings

Every other arm of this project uses the note embeddings **supervised**: pooled per patient, then
fed to a Cox model against a specific endpoint. The one clustering that already exists
(`figures/prep/figure4.py`) clusters *model-derived risk slopes*, not the embeddings.

This arm is unsupervised. It uses the text embeddings directly as patient features, clusters
patients, and asks what the clusters correspond to clinically — and whether different note types
carve the cohort differently.

## The grid: 5 spaces × 2 windows

| Space | Dim | Meaning |
|---|---|---|
| `clinician` | 768 | progress notes only |
| `imaging` | 768 | imaging reports only |
| `pathology` | 768 | pathology reports only |
| `concat` | 2304 | the three per-type means side by side |
| `merged` | 768 | every note pooled, `NOTE_TYPE` ignored |

| Window | Notes included |
|---|---|
| `alltime` | every note a patient has, no anchor |
| `pretreatment` | notes strictly before `first_treatment_date` — leak-free against `tt_death` |

**`merged` is not the average of the three per-type means.** It is a second pooling pass over notes
whose `NOTE_TYPE` has been overwritten with one literal, so it is *note-weighted*: a patient with
200 imaging and 5 pathology notes is imaging-dominated. The average-of-means would be
*type-weighted*, giving those 5 pathology notes equal say. Note-weighted is the intended
"a note is a note" reading — the distinction is real whenever per-type note counts are unequal,
which is nearly always.

Each space is **independently complete-cased**. A patient missing a note type drops out of that
type's space and out of `concat`, but still appears in `merged` and in the spaces for the types they
do have. `concat` is therefore the intersection of all three and is substantially the smallest
cohort; `feature_summary.csv` reports this up front.

Pooling is a plain unweighted `mean`, not the production `time_decay_mean`: this arm asks what a
patient's notes say on average, not what they said most recently.

## DAG

```
1_data/03  (knitted embeddings)
    |
    v
01_aggregate     semantic_search.aggregate_embeddings   -> features/{space}_{window}.parquet
    |
    v
02_cluster       semantic_search.cluster_patients       -> clusters/{space}_{window}_labels.parquet
    |                                                      clusters/{space}_{window}_coords.parquet
    |                                                      clusters/{space}_{window}_meta.json
    v
03_characterize  semantic_search.characterize_clusters  -> results/*.csv, figures/*.png
```

Nothing downstream depends on this arm.

Every module is runnable directly from the repo root:

```bash
python -m semantic_search.aggregate_embeddings [--windows ...] [--overwrite] [--limit-mrns N]
python -m semantic_search.cluster_patients [--spaces ...] [--windows ...] [--k N] [--overwrite]
python -m semantic_search.characterize_clusters [--spaces ...] [--windows ...]
```

Stages 1 and 2 are skip-if-exists resumable; pass `--overwrite` to rebuild.

## Where data lands

All under `SEMANTIC_SEARCH_PATH` (`DATA_PATH/semantic_search/`, set in `config.py`):

| Path | Written by | Contents |
|---|---|---|
| `features/{space}_{window}.parquet` | 01 | `DFCI_MRN` + `EMBEDDING_*` |
| `clusters/{space}_{window}_labels.parquet` | 02 | `DFCI_MRN, cluster` |
| `clusters/{space}_{window}_coords.parquet` | 02 | `DFCI_MRN, dim1, dim2` (first two PCs) |
| `clusters/{space}_{window}_meta.json` | 02 | k, seed, silhouette, PCA variance, cluster sizes |
| `results/*.csv` | 01–03 | summaries, tests, enrichment, survival, manifest |
| `figures/*.png` | 03 | the notebook-03 panels |

## Method notes

**Preprocessing order** is fixed and recorded in each run's meta JSON:
`L2-normalize → StandardScaler → PCA(50) → KMeans(k)`. L2 first because these are transformer
embeddings whose trained geometry is cosine; on unit-norm rows Euclidean KMeans is monotone in
cosine distance. StandardScaler then stops a few high-variance dimensions from dominating, which
matters most for `concat`, where three blocks of different scale sit side by side.

**Cluster labels are relabeled by ascending size** (0 = smallest), so a rerun with the same seed
gives not just the same partition but the same integer names. KMeans label integers are otherwise an
artifact of centroid initialization order, which would make per-cluster tables incomparable across
runs.

**k defaults to the silhouette argmax** over `range(2, 13)`; `--k` forces a value. Silhouette is
estimated on a seeded 20k subsample when the cohort is larger, since it is O(n²).

**Coordinates are PCA, not UMAP** — `umap-learn` is not in `environment.yml`, and PCA keeps the
scatter in the same geometry the clustering actually used.

**Adjusted Rand index** between every pair of runs (`cluster_concordance.csv`, on shared MRNs only)
is the direct answer to "do the note types carve the cohort differently?"

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
- **No multiple-testing correction across spaces** — only within family within space. A variable
  significant in exactly one of the 10 space × window runs is weak evidence.
- `alltime` is **not leak-free** against survival: it includes notes written after treatment start,
  and after death for decedents. Use `pretreatment` for anything involving `tt_death`.
- Cancer type is the usual explanation for any structure found here. Check it before reading a
  cluster as a novel phenotype.

## What this repo does not have

Absent, and so absent from the characterization: labs, vitals, race/ethnicity, smoking,
ECOG/performance status, and PFS. Age and gender are the only demographics; OS is the only
time-to-event endpoint.
