# Complete code review of `v2/` (excluding tokenization + embedding generation)

## Context

`v2/` is the maintained pipeline (`REFACTOR_PLAN.md`) and is now producing manuscript figures,
but it has **never been reviewed end to end** and has **zero test coverage on any file in
scope** — the single test file, `v2/tests/test_generate_clinical_embeddings.py` (160 lines),
covers the one module this review excludes. Every correctness guarantee in the tree today is
either an inline `assert_schema` call or a comment.

Two recent events motivated this review and both are symptomatic rather than isolated:

- A production crash (`PRS values disagree across sample IDs for 1528 patient(s)`) surfaced a
  per-patient collapse whose correctness had never been checked, and which sat behind a
  `raise` that had presumably been passing silently until the input data shifted.
- Building `04_smoke_test_training.ipynb` required documenting five *silent* failure modes in
  the training core — an all-NaN CSV that still exits 0, a `verbose`-gated fold failure, a
  `logger.debug` that no handler ever receives — none of which any consumer checks.

**Scope:** ~11,350 lines of Python (`config.py`, `anchors.py`, `schemes.py`, `data/`,
`shared/`, `figures/` + `figures/prep/`, `pipelines/{preprocessing,training,trajectories,biomarkers}/`,
`survival/` + `survival/cox_models/`), **2,525 lines of R** (`v2/R/`), and **616 lines of shell**
(`v2/slurm/`). Excluded: `text_preprocessing_and_tokenization.py`,
`generate_clinical_embeddings.py`, and their test.

**Deliverable (per user):** a findings report **and** fixes applied, weighting scientific
correctness, silent failures, reproducibility/config, and code quality equally, with R and
shell included fully.

**Intended outcome:** every number that reaches the manuscript is either verified or has a
named, tracked caveat — and the pipeline stops failing silently.

**Decisions taken (from user):**
- **Tier 2 findings are quantified first, then decided individually.** Nothing that moves a
  manuscript number is changed without explicit sign-off; each Tier 2 item is reported with its
  measured blast radius on real data attached.
- **Environments are pinned from the working cluster environment** (`conda env export
  --no-builds` on the cluster is authoritative, including resolving 3.12 → 3.13), not from
  locally-resolved versions. Requires one command run on the cluster.

---

## Step 0: write this plan into the repo

Before any review work, save this document as **`v2/CODE_REVIEW_PLAN.md`** (alongside
`REFACTOR_PLAN.md`, matching the existing convention of top-level planning docs in the repo).
The findings report produced by the review will be appended to it — or written beside it as
`v2/CODE_REVIEW_FINDINGS.md` — so the review's output is version-controlled with the code it
describes rather than living in a scratch file.

---

## Guiding constraint: no test net

With no tests in scope, "apply fixes" cannot mean "edit and hope." Fixes are therefore split
into three tiers, and **only Tier 1 is applied without sign-off**:

| Tier | Kind | Applied? | Verified by |
|---|---|---|---|
| **1 — Safe** | Behavior-preserving: dead code, docstrings contradicting code, duplicate constants, missing `validate=`/assertions that *should* be no-ops | Yes, directly | `pyflakes` + `py_compile`; assertions must not fire on a real cluster run |
| **2 — Behavioral** | Changes numbers in an output file (e.g. re-censoring negative-time met events) | **Proposed with evidence; applied only on explicit approval** | Before/after diff of the affected output on real data |
| **3 — Structural** | Refactors (moving constant-column drop inside folds, unifying `assert_schema`) | **Reported only**, with a recommendation | n/a this pass |

Tier 2 is the important boundary: several findings below change scientific results, and
applying them silently would be worse than the bug.

---

## Findings already confirmed (I verified each against source, not just agent reports)

These are established and go into the report as-is. They also calibrate where the deeper
review should focus.

### Scientific correctness

0. **The Track-2 forest plot's "95% CI" is not a confidence interval — Tier 2, highest severity.**
   `figures/prep/figure5.py:410-412` populates fields *named* `CI95_low` / `CI95_high` with
   `sub["HR_markerxICI"].min()` and `.max()` — the **range of point estimates across
   specifications**. These are rendered in the same visual channel as genuine CIs in a
   manuscript forest plot, where any reader will interpret them as 95% intervals. Real Wald CIs
   are computed elsewhere (`run_IPTW_analysis.py:250,256,368`), so the correct values are
   available. **This is the single most consequential finding in the review: a mislabeled
   statistic on a manuscript figure.** Fix is either to plot the true Wald CI or to rename the
   fields and re-style the glyph as a range — a scientific decision, hence Tier 2.

1. **Metastasis events can be recorded at negative time — Tier 2.**
   `generate_embedding_prediction_datasets.py:279` filters met dates only with
   `dropna(subset=['TIME_TO_MET'])`, while all three ICD/phecode paths filter
   `TIME_TO_ICD > 0` before mapping (e.g. `:424`). `map_time_to_event`
   (`survival/preprocessing.py:131`) takes `.min()` over a patient's rows with no sign check,
   so a metastasis predating t=0 yields a **negative `tt_{site}M` with `event=1`**.
   `_filter_endpoint_events_by_min_post_baseline_count:161` counts only `tt > 0` when deciding
   whether to *keep the column*, but never drops or re-censors the offending rows.
   This contradicts the module's own docstring (`:8-9`).
   *Highest-severity finding in the review.*

2. **`metburden` is trained but invisible to the figures — Tier 1/2.**
   `data_availability.py:23` lists six modalities; `shared/palette.json:2` lists five (no
   `metburden`). `figures/prep/figure3.py` iterates `MODALITY_ORDER` at `:52, :98, :214, :246,
   :262`, and `R/plot_figure_3.R:46` *filters* to rows containing all of `MODALITY_ORDER`. A
   trained modality is silently omitted from the comparison figure.

3. **`PANEL_VERSION` is unreachable under that name — Tier 1.**
   Six live consumers match the prefix `PANEL_VERSION` (`biomarkers/biomarker_common.py:39`,
   `biomarkers/generate_IPTW_df.py:42,112`, `biomarkers/run_IPTW_analysis.py:488,637`,
   `figures/prep/figure5.py:560`) and get empty lists with no warning — dropping a propensity
   confounder. `audit_prs_somatic.py:309-335` already contains the discovery logic; the fix is
   a name reconciliation, not a data recovery.

### Silent failures

4. **`mean_c_index` and `mean_ibs` in every `*_val.csv` are structurally always NaN — Tier 1.**
   `grid_search.py:247,249` (and `:450,452` for the PCA path) append literal `np.nan` into
   those positions; only AUC is computed in CV. Any consumer reading c-index from a `_val.csv`
   gets NaN. (Note: they *are* genuinely computed in `*_test.csv` via `evaluate_surv_model`,
   and `04_smoke_test_training.ipynb:231` correctly reads them only from the test frame — so
   the notebook is sound and this finding is confined to `_val.csv`.)

5. **`error_rate` / `fold_error_flags` do not describe the row they sit on — Tier 1.**
   Both are computed once per l1-path and then written identically into all 25 alpha rows
   (`grid_search.py:243-255`). A single failure at any one alpha marks the whole path.
   Already documented in `04_smoke_test_training.ipynb:31`; needs to be documented in the code
   that produces it.

6. **A winner can be crowned on one surviving fold — Tier 2.**
   `mean_auc(t) = np.nanmean(fold_aucs[:, ai])` (`grid_search.py:248`) with no `min_folds`
   guard; `valid_cv.dropna` only removes grid points where *all* folds failed. Selection
   (`:292`) consults `mean_auc(t)` alone and never `error_rate`. Recommended fix: emit an
   `n_folds_contributing` column and require a threshold — behavioral, so Tier 2.

7. **Fold failures are invisible in production — Tier 1.**
   `heldout.py:191-194, 298-301` print only `if verbose:`, and every SLURM caller leaves
   `verbose=0`. A failed fold writes NaN risk scores for ~20% of patients with zero output.
   Compounded by the absence of any `logging.basicConfig` in `survival/` or
   `pipelines/training/`, which discards the `logger.debug` at `grid_search.py:210,413`.

8. **Blanket warning suppression, permanently.**
   `_common.py:21-28` suppresses *all* `ConvergenceWarning` and *all* `RuntimeWarning`; applied
   at `base.py:49`, `grid_search.py:80`, `heldout.py:119` as bare `filterwarnings` outside any
   `catch_warnings` context, so they permanently mutate process-global filters. Non-convergence
   is currently unmeasurable.

### Data integrity

9. **`knit_embeddings.py:133` is the highest-risk join in the tree — Tier 1 (assertion).**
   `metadata_df` is positionally row-aligned with the embeddings array (`:118-122` length
   assert, `:139` mask, `:142` positional `EMBEDDING_INDEX`). It is then left-joined against
   `cohort_df`. If `cohort_df` ever had a duplicate `DFCI_MRN`, metadata fans out and the
   metadata↔embedding correspondence breaks **silently**. Uniqueness is guaranteed only by
   `build_cohort.py:264`, in a different script at a different time; `knit_embeddings.py:83`
   re-reads with no revalidation. A one-line `assert_schema(..., unique_key=True)` closes this.

10. **Deduped CSVs re-read without revalidation throughout.**
    `generate_embedding_prediction_datasets.py` — the module producing the final training
    inputs — never imports `assert_schema` at all. `slurm_array_utils.py:158-170` merges
    somatic/PRS with no `validate=`. `data_availability.py:40,48,63` returns `set()` on a
    missing file, so missing input reads as "zero coverage" rather than an error.

11. **Two divergent copies of `assert_schema`** (`data/schema.py`, `pipelines/preprocessing/schema.py`),
    selected by `try/except ModuleNotFoundError` at four sites; the preprocessing copy raises
    `ColumnNotFoundError` instead of the intended `ValueError` on a missing key. Which one runs
    depends on `sys.path` — i.e. on how the script was invoked. *Tier 3.*

### Figures / R tier

11b. **An empty result still produces a complete figure set — Tier 1/2.**
   The guard in `figures/io.py:66` rejects only *zero-column* frames, but every prep failure
   path returns `pd.DataFrame(columns=[...])` — columns but no rows — which passes. That writes
   a 0-row CSV; `R/figure_utils.R:108-115` turns a missing/empty file into an empty tibble; the
   scripts branch to `placeholder_panel()`; and `save_panel()` still writes PNG **and** PDF. The
   output directory of a failed run is indistinguishable at the filesystem level from a
   successful one. Recommended: have `save_figure_data` reject or loudly flag zero-row frames,
   and make placeholder panels visually unmistakable.

11c. **`validate_and_report.py` silently prefers a hardcoded laptop path — Tier 1.**
   `pipelines/biomarkers/validate_and_report.py:21` hardcodes
   `/Users/connorpa/Documents/.../compiled_results/`, and `_resolve_compiled_dir():41-42`
   returns it whenever the canonical dir lacks inputs — no warning. On this machine, validation
   can silently read a **stale** result set. (Matches the `compiled_results` path in project
   memory.) Fix: fail loudly, or require `COMPILED_DIR` explicitly.

11d. **Multiple-testing correction is applied twice, independently, in two languages.**
   `run_IPTW_analysis.py:191-201` does BH within `mutation_type` only — not across cancer
   types, cohorts, PS models, or weight specs, all of which are looped over. Separately,
   `R/plot_figure_3.R:24-32` does its own BH within (scheme, event), and
   `replace(p_value, is.na(p_value), 1)` silently converts NA p-values to 1.0 **and keeps them
   in the denominator**, deflating the correction. Needs a single documented strategy.

11e. **Inference lives in the R plotting layer**, so figure p-values are not reproducible from
   the Python outputs alone: `wilcoxon_vs0`/`kruskal_p`/`logrank_p` (`R/figure_utils.R:158-213`),
   `friedman.test` (`plot_figure_3.R:99-107`), `coxph`/`survdiff` (`plot_figure_4.R:48,53`), and
   a hand-rolled `compute_roc` at `plot_figure_5.R:20-32` commented as "matches the Python
   helper" with **no cross-check**. Also `plot_figure_2.R:299-303` makes an aggregation choice
   (n-weighted mean across stages) in the plotting layer. Verifying the duplicated ROC against
   the Python implementation is a concrete Phase 1 task.

11f. **Panel version is adjusted for in neither stage.** `ICI_train_propensity.py:7-9`
   deliberately excludes panel from the PS model *because* it is handled as a Cox confounder —
   but finding 3 shows the Cox side silently gets an empty column list. The two decisions
   combine into an unadjusted confounder, which is why finding 3 outranks a simple rename.

### Reproducibility

12. **Environments are effectively unpinned.** `environment.yml` pins only `python=3.12`,
    `cuda-version`, `cuda-toolkit`, `flash-attn`; `numpy`, `pandas`, `scikit-survival`,
    `lifelines`, `polars`, `xgboost` and the rest float. **The cluster is running Python 3.13**
    (per the crash traceback) against a file pinning 3.12. `v2/R/install_packages.R` pins
    nothing across 18 packages. For a manuscript, this is the finding most likely to make
    results unreproducible a year from now.

13. **Seeds are consistent at `1234` in the modeling core** (`grid_search.py:113,123`,
    `heldout.py:145`, `_common.py:98`, `base.py:65,80`) — good. But `pipelines/biomarkers/`
    and `figures/prep/figure4.py:208` use `42` / `0` inconsistently, and no seed is
    CLI/env-overridable. Note `grid_search.py:113` and `heldout.py:145` share seed 1234 but
    split different frames, so held-out folds do **not** respect the grid search's test set —
    worth stating explicitly since the hyperparameter was tuned on a superset of every fold's
    training data.

14. **Hardcoded absolute `/data/gusev/...` paths** with no env override at `config.py:39,45,59-62,65`.

### Deliberate designs to preserve (do not "fix")

Verified correct; the report should say so explicitly so a future reader doesn't "simplify" them:
- **Fold-scoped preprocessing is correct.** `apply_group_pca_np` (`_common.py:91-138`) does
  `fit_transform` on train / `transform` on test; imputation (`:161-170`) and scaling
  (`:140-159`) are train-only. This is the leakage hypothesis I expected to find and it is
  **disconfirmed**.
- **Two sequencing-date derivations coexist by design** — `build_cohort.py:53-69` (earliest)
  vs `generate_all_non_text_covariates.py:110-125` (treatment-proximate argmin, `>= 0` filter).
  `build_cohort.py:56-58` explicitly warns against unifying them.
- **The `MET_SITE_{site}` leakage guard exists** (`run_feature_comp_task.py:104-111`) — though
  it fires only when `modality == "metburden"`, which the review must confirm is sufficient for
  `--modality all` runs.

---

## Review plan

### Phase 1 — Close the remaining column-semantics gap

All three tiers are now mapped, so Phase 1 is narrow. Two contract-level checks are already
done: **every `figN_*.csv` the R tier reads is written by the Python tier** (no orphans in
either direction), and three files are written but never read
(`fig0_availability_combinations.csv`, `fig2_anchor_cohort_overlap.csv`,
`fig3_modality_cindex.csv` — the last is consumed in-process, so only the first two are dead).

The residual risk is **column semantics**, the class finding 2 belongs to. Remaining work:

- Compare, per figure, the columns `figures/prep/figureN.py` writes against those
  `R/plot_figure_N.R` selects/filters/factors on. Only **one** R script checks column presence
  (`plot_figure_2.R:153-156`); every other site indexes directly, so a renamed column is a
  runtime error or a silently dropped series. Specifically check
  `plot_figure_0.R:29` (`d$n_patients[d$stage == "text"]` can yield `numeric(0)` as a
  denominator) and `plot_figure_2.R:299-303` (assumes `mean_auc` and `n` with no guard).
- Verify the duplicated ROC in `plot_figure_5.R:20-32` against the Python helper it claims to
  match (finding 11e).
- Confirm `shared/palette.json` stays the single source of truth for both tiers —
  `R/figure_utils.R:47` reads the same JSON, which is the right pattern and is why fixing
  finding 2 in the JSON fixes both tiers at once. Note `figure0.py:65` indexes a **5-key**
  `_MODALITY_LABELS` dict over the **6-element** `MODALITY_ORDER`, so adding `metburden` to the
  JSON without also fixing that dict will turn a silent omission into a `KeyError`.

### Phase 2 — Targeted deep review

Reading order, highest expected yield first:

1. `generate_embedding_prediction_datasets.py` — produces the final training inputs, has **no
   schema validation at all**, and already has one confirmed correctness bug (finding 1).
   Check the remaining event paths for the same negative-time class.
2. `survival/cox_models/grid_search.py` + `heldout.py` — the metric and selection core.
3. `pipelines/biomarkers/` — `generate_IPTW_df.py:103-109` chains five unvalidated merges then
   `drop_duplicates(keep='first')`, which *masks* inflation rather than detecting it; plus the
   `PANEL_VERSION` confounder loss.
4. `figures/prep/` ↔ `R/` column-contract audit.
5. `v2/slurm/*.sh` — `set -euo pipefail` is set *after* the prologue (manifest checks, `cd`,
   conda activation) in all three array scripts; `--output`/`--error` paths are relative to the
   submission CWD while the script `cd`s elsewhere; `array_full_cohort_run.sh:95-104` duplicates
   `SCHEMES` from `schemes.py:15-32` as a second source of truth.
6. `config.py` / `anchors.py` / `schemes.py` / `shared/` — smallest and mostly verified.

For each finding record: file:line, what breaks, a concrete failure scenario, tier, and the
proposed fix.

### Phase 3 — Apply Tier 1, propose Tier 2

Apply Tier 1 in small thematic commits (validation assertions; docstring/comment corrections;
duplicate-constant unification; logging enablement). Tier 2 findings each get a written
before/after argument and wait for approval. Tier 3 is reported with a recommendation only.

The two environment files should be pinned from a **known-good cluster environment**
(`conda env export --no-builds`, `renv::snapshot()` or a dated MRAN/Posit snapshot) — pinning to
locally-resolved versions would encode an environment that has never run the pipeline. Resolving
the 3.12/3.13 discrepancy needs a decision from you about which is authoritative.

---

## Verification

Because there is no test net, verification is layered:

- **Static:** `pyflakes` on every touched Python file (per project convention, `py_compile`
  alone misses undefined names); `Rscript -e 'lapply(list.files("v2/R", full.names=TRUE), parse)'`
  for R syntax; `bash -n` on each `slurm/*.sh`.
- **Tier 1 assertions must be no-ops.** Every added `assert_schema` / `validate=` is a claim
  that an invariant already holds. Run `01_run_preprocessing` on the cluster; **any assertion
  that fires is a finding, not a regression** — that is the point of adding them.
- **Tier 2 changes get a before/after diff.** For finding 1, report how many patient-events
  currently carry `tt <= 0` with `event = 1` per met endpoint *before* changing anything; that
  count is both the severity evidence and the acceptance test.
- **Figure reproduction is the end-to-end test.** Re-run `06_generate_figure_data.ipynb` and
  `07_render_figures.Rmd`, and diff the output CSVs and panel files against a pre-review run.
  Tier 1 changes must produce **byte-identical** figure data; any diff means a fix was
  misclassified. This is the strongest signal available given the absence of unit tests.
  **Caveat (finding 11b):** a rendered figure set does *not* prove success today — placeholder
  panels are written as normal PNG/PDF. So the diff must be taken over the **CSVs**, and the
  baseline run must first be checked for 0-row files. Fixing 11b early makes every later
  verification step trustworthy, so it should be the **first** fix applied.
- **Capture the baseline before touching anything.** Record the pre-review figure CSVs, the
  row counts of each, and the `tt <= 0 & event == 1` counts per met endpoint. Without a
  baseline captured up front, none of the diffs above are interpretable.
- **`04_smoke_test_training.ipynb`** already exercises the training entrypoints and reports the
  silent-failure surfaces; use it as the regression check for any `survival/` change.
- Consider it a secondary outcome of this review to leave behind a minimal `pytest` file for the
  pure functions that are trivially testable (`shared/icd10.py`, `shared/stages.py`,
  `anchors.py`) — the tree currently has no place to put a regression test.
