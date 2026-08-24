# Code review findings — `v2/` (per `CODE_REVIEW_PLAN.md`)

Status: two passes so far, both **without cluster access**. Every static check that could run
locally (`pyflakes`, `py_compile`, `bash -n`, R `parse()`) passed on all touched files. The
plan's cluster-dependent verification steps — assertions confirmed as no-ops on a real run,
before/after figure-CSV diffs, `04_smoke_test_training.ipynb` as a regression check — have
**not** been run and remain outstanding; they require a session with cluster access.

Tier 2 blast-radius quantification (findings 0, 1, 6) was explicitly deferred both passes per
user decision — not attempted, not estimated.

---

## Pass 2 — additional fixes applied (with explicit user sign-off where noted)

1. **Slurm launcher `--output`/`--error` paths.** All three array scripts
   (`array_full_cohort_run.sh`, `array_feature_comp.sh`, `array_full_cohort_risk_scores.sh`)
   declare `#SBATCH --output=v2/slurm/.../%A_%a.out` as a *relative* path, but the script only
   `cd`s into `$V2_ROOT` after the job starts — `#SBATCH` directives are resolved by Slurm at
   submission time relative to the submission CWD, so log location silently depended on where
   the user happened to invoke the launcher from. Fixed in all three `launch_*.sh` wrappers by
   passing `--output`/`--error` as absolute `sbatch` CLI flags (which override the script's
   `#SBATCH` defaults), built from `$PROJECT_ROOT`, plus a `mkdir -p` for the log directory
   before submission. `launch_feature_comp.sh` submits two jobs (big/small) against the same
   array script — both `sbatch` calls were updated.

2. **`set -euo pipefail` set after the prologue.** In all three array scripts, manifest
   existence checks, `cd "$V2_ROOT"`, `module load miniforge3`, and
   `eval "$(conda shell.bash hook)"` ran *before* `set -euo pipefail`, so a failure in any of
   them that wasn't already explicitly guarded could go unnoticed. Rather than moving
   `set -euo pipefail` to the very top of the file (risky to verify without a cluster —
   `module`/`conda` machinery is a common source of `-u`-related breakage on cluster module
   systems, and this session has no way to test that), added an explicit `|| { echo ...;
   exit 1; }` guard to the two previously-unguarded steps (`cd` and `module load`) so nothing
   in the prologue can fail silently even before strict mode is enabled. `set -euo pipefail`
   itself stays where it was, now documented as deliberate rather than unexplained.

3. **Finding 2 — `metburden` added to `shared/palette.json` and `figure0.py`'s
   `_MODALITY_LABELS`, with explicit user sign-off (Tier 2, this is the approval the plan's
   gate requires).** `shared/palette.json`'s `MODALITY_ORDER`/`MODALITY_COLORS`/
   `MODALITY_DISPLAY` did not include `metburden`, even though
   `pipelines/preprocessing/data_availability.py`'s own independent `MODALITY_ORDER` already
   had all six modalities. Added `metburden` to all three palette.json dicts (color
   `#1F77B4`, label `"Met. Burden"` — **chosen without design guidance, flag for review**) and
   to `figure0.py`'s `_MODALITY_LABELS` (`"With Met. Burden"`), closing the exact `KeyError`
   risk the original review flagged (figure0.py:65 indexes `_MODALITY_LABELS` positionally
   against the 6-entry `MODALITY_ORDER` from `data_availability.py`). Confirmed the R side
   (`R/plot_figure_3.R:46`, `build_fig3a`'s `all(MODALITY_ORDER %in% s)` complete-case filter)
   needs no code change — it already reads the same `palette.json` and will now correctly
   require all six modalities including metburden. **Not yet verified against real figure
   data** — this is a content change to a manuscript figure and should be confirmed with a
   real `06b_generate_figure_data.ipynb` / `07_render_figures.Rmd` run before trusting the
   rendered Figure 3.

## Investigated, remains blocked (Finding 3 — `PANEL_VERSION`)

User confirmed the real column name in `GENOMIC_SPECIMEN.parquet` is literally
`PANEL_VERSION` — the same string all six consumers already match on. This **contradicts**
the plan's original theory that a naming mismatch was the cause of finding 3. Traced the
column's path: `profile_sources.load_genomic_specimen()` does no column projection (returns
every parquet column), `build_somatic_data_df` (`generate_all_non_text_covariates.py:99-138`)
takes `metadata_cols` dynamically from `selected_sample.columns` post-join, so `PANEL_VERSION`
should structurally survive into `complete_somatic_data_df.csv.gz` unless something else
(casing at read time, a later column drop, the `SOMATIC_GROUP_KEY` join, or the column
genuinely being absent/null-only in a given data pull) is responsible. **No local copy of
`GENOMIC_SPECIMEN.parquet` or `complete_somatic_data_df.csv.gz` exists in this session**, so
this cannot be resolved further without either the data or the actual output of
`audit_prs_somatic.py`'s `_audit_panel_version()` run on the cluster. Did not touch any of the
six consumer sites — there is nothing to rename if the name already matches; the real defect,
if any, is elsewhere and unidentified. **This should be the first thing run on the next
cluster session** — it's a five-minute check that will either close finding 3 outright (if
`PANEL_VERSION` is in fact reaching the consumers fine) or redirect the investigation.

---

## Pass 3 — Tier 2 fixes applied with explicit sign-off (findings 0, 1, 6)

1. **Finding 0 — Track-2 forest-plot "CI" was an inter-spec range, not a CI. Fixed: real
   Wald CI added and plotted.** `run_IPTW_analysis.py`'s `_fit_track2_marker` already computed
   `se_mx = sqrt(V.loc[mx, mx])` for the marker×ICI interaction term (`beta_mx`) but discarded
   it — the docstring in `figure5.py` claiming "Track-2 upstream emits no SE for the
   interaction term" was **incorrect**; the SE existed, it just wasn't propagated to an output
   column. Fixed at the source: added `CI95_markerxICI_low`/`_high` (`exp(beta_mx ± 1.96*se_mx)`)
   to `_fit_track2_marker`'s result dict and `TRACK2_RESULT_COLS`
   (`run_IPTW_analysis.py:208-211,245-248,266-268`) — flows through
   `compile_IPTW_results.py` unchanged (no column projection there) into
   `track2_all_significant_hits.csv`. `figures/prep/figure5.py`'s `_track2_row` now reads
   `CI95_markerxICI_low/high` directly instead of `sub["HR_markerxICI"].min()/.max()`.
   `R/plot_figure_5.R`'s `build_fig5e` comment and x-axis label (previously "T2: interaction HR
   (inter-spec range)") updated to reflect that both T1 and T2 brackets are now genuine Wald
   95% CIs — no plotting-code change needed since it already draws a generic `CI95_low/high`
   error bar. **Not yet verified against real data** — this changes what the Track-2 side of
   Figure 5's synthesis forest panel (fig5e) shows; confirm with a real
   `06b_generate_figure_data.ipynb` / `07_render_figures.Rmd` run before trusting it.

2. **Finding 1 — negative-time metastasis events. Fixed: sign filter added, matching the
   ICD/phecode paths.** `generate_embedding_prediction_datasets.py`'s `_add_metastatic_events`
   only did `met_date_df.dropna(subset=['TIME_TO_MET'])`, unlike `icd_data_base.loc[...>
   0]`/`icd4_data_base.loc[...>0]`/`phe_data.loc[...>0]` used for every other endpoint class —
   so a metastasis recorded before the anchor date produced a valid-looking `tt_{site}M < 0,
   event=1` row that would be fit into a Cox model. Added
   `met_date_df = met_date_df.loc[met_date_df['TIME_TO_MET'] > 0]` immediately after the
   existing `dropna`, mirroring the ICD/phecode pattern exactly (line ~279-282). Confirmed
   `map_time_to_event` (`survival/preprocessing.py:131`) takes `.min()` over its `df_events`
   argument with no sign check of its own, so filtering at the caller (as the other three event
   classes already do) is sufficient — no second guard needed inside `map_time_to_event`.
   **Not yet verified against real data** — no before/after count of affected rows was taken
   (blast-radius quantification explicitly skipped per your instruction); the fix is applied
   on the strength of the code-level argument alone, matching the ICD/phecode precedent in the
   same file.

3. **Finding 6 — single-fold winner selection. Fixed: `n_folds_contributing` column and
   selection log line added; no `min_folds` gate (explicitly not wanted).** Both the no-PCA
   and PCA paths in `grid_search.py` (`evaluate_l1_path_no_pca`/`evaluate_l1_path_with_pca`)
   now append `n_folds_contributing = count of non-NaN fold_aucs for that alpha` as a new
   column in `cv_results_df`, and the selection step (`opt = valid_cv.sort_values(...).iloc[0]`
   in both paths) now logs the chosen `l1_ratio`/`alpha`/`mean_auc(t)` together with
   `n_folds_contributing` out of the total fold count via `logger.info`. This makes a 1-fold
   win visible in the run log and in the CV output table itself, without changing which
   hyperparameters get selected — `mean_auc(t)` is still the sole sort key, exactly as you
   asked (fine to have only one succeeding fold; no threshold enforced).

All three pass-3 files (`run_IPTW_analysis.py`, `generate_embedding_prediction_datasets.py`,
`figures/prep/figure5.py`, `survival/cox_models/grid_search.py`) pass `pyflakes` and
`py_compile` (pyflakes flags six pre-existing, unrelated f-string warnings at
`run_IPTW_analysis.py:554+`, outside anything touched this pass). `R/plot_figure_5.R` parses
clean via `Rscript`.

---

## Applied (Tier 1)

1. **Finding 4/5 — `mean_c_index`/`mean_ibs`/`fold_error_flags`/`error_rate` semantics
   undocumented in code.** `survival/cox_models/grid_search.py` (both the no-PCA and PCA
   `rows.append([...])` blocks): added inline comments on each affected column stating (a)
   `mean_c_index`/`mean_ibs` are always `np.nan` in CV — only computed on `*_test.csv` — and
   (b) `fold_error_flags`/`error_rate` are computed once per l1-path and repeated identically
   across every alpha row, so they describe the path, not the row. No values changed.

2. **Finding 7 — fold failures invisible.** `survival/cox_models/heldout.py`: both
   `if verbose: print(...)` failure sites (no-PCA and PCA paths) now unconditionally call
   `logger.warning(...)`, naming how many patients get NaN risk scores. Python's logging
   module emits WARNING+ via its lastResort handler even with no `logging.basicConfig()`
   configured anywhere in `survival/` — so these are now visible in production without
   requiring `verbose=1` or a logging setup change. `grid_search.py`'s existing per-alpha
   `logger.debug` calls were left as debug (correctly gated per-alpha noise); the
   `error_rate`/`fold_error_flags` columns (finding 5) already carry that signal at
   per-path granularity.

3. **Finding 8 — blanket warning suppression mutated global state.** Three bare
   `warnings.filterwarnings("ignore", ...)` calls applied outside any `catch_warnings()`
   context permanently mutated process-global warning filters:
   - `survival/cox_models/base.py:49` (`run_base_CoxPH`)
   - `survival/cox_models/grid_search.py:80` (`run_grid_CoxPH_parallel`)
   - `survival/cox_models/heldout.py:119` (`get_heldout_risk_scores_CoxPH`)

   All three now run inside `with warnings.catch_warnings():`, scoped to the function body
   (or, in `grid_search.py`/`heldout.py`, to the setup segment before dispatch — the
   downstream per-fold fits already suppress the same warnings unconditionally inside their
   own `catch_warnings()` blocks, so the outer scope only needed to cover
   `train_test_split`/`StratifiedKFold`). Suppression behavior during each call is unchanged;
   the filters no longer leak into the calling process afterward. Non-convergence is now only
   as unmeasurable as it was before *during* these calls — still worth a real convergence
   audit later (Tier 3, not attempted).

4. **Finding 9 — `knit_embeddings.py`'s highest-risk join had no uniqueness check.**
   `pipelines/preprocessing/knit_embeddings.py`: added an assertion immediately before the
   `metadata_df.join(cohort_df...)` call that `cohort_df["DFCI_MRN"]` is unique, raising
   `ValueError` with the duplicate count if not. `metadata_df` is positionally row-aligned
   with the embeddings array; a duplicate `DFCI_MRN` in `cohort_df` would fan out that
   left join and silently break the correspondence. Per the plan's verification note, this
   assertion is a **claim that the invariant already holds** — if it fires on a real cluster
   run, that is a finding, not a regression, and must be treated as such rather than reverted.

5. **Finding 11b — empty-but-columned figure-data frames pass the write guard silently.**
   `figures/io.py`'s `save_figure_data` rejected only zero-*column* frames; the common
   `pd.DataFrame(columns=[...])` failure-path pattern used across `figures/prep/figure2.py`,
   `figure4.py`, etc. (zero rows, real columns) passed through, and downstream R renders a
   placeholder panel indistinguishable on disk from a real one. Did **not** make this raise
   (that would turn every legitimate "no data for this arm" path into a hard crash — a
   behavioral change requiring the Tier 2 sign-off the plan calls for). Instead added a
   `logger.warning(...)` that fires whenever a 0-row frame is written, naming the file. This
   makes the condition visible in run logs without changing what gets written to disk.
   **Recommend applying the stronger Tier 2 fix (reject, or a `--strict` flag) once the
   sign-off gate is cleared** — this Tier 1 change is a floor, not the intended end state.

6. **Finding 11c — hardcoded laptop path silently preferred on missing inputs.**
   `pipelines/biomarkers/validate_and_report.py`'s `_resolve_compiled_dir()` fell back to a
   hardcoded `/Users/connorpa/.../compiled_results/` path with no signal when the canonical
   dir lacked inputs. Left the fallback itself in place (removing it outright is a behavior
   change some workflows may depend on) but added `logger.warning(...)` naming both paths and
   stating explicitly that this can read a stale result set, plus the fix (`COMPILED_DIR` env
   var) in the message.

All six of the above compile clean and pass `pyflakes` with zero findings.

---

## Confirmed but not applied this pass (blocked or requires sign-off)

- **Finding 3 — `PANEL_VERSION` unreachable.** Six consumers match the literal prefix
  `PANEL_VERSION`; the real column name in `GENOMIC_SPECIMEN.parquet` is only discoverable by
  running `audit_prs_somatic.py`'s existing `_audit_panel_version()` against real data (no
  local copy of `GENOMIC_SPECIMEN.parquet` / `complete_somatic_data_df.csv.gz` was available
  this session). **Do not guess the rename** — run the audit script on the cluster first, then
  apply the six-site rename as a small, mechanical Tier 1 fix once the true name is known.

- **Finding 2 — `metburden` missing from `shared/palette.json` / `figure0.py`'s
  `_MODALITY_LABELS`.** Technically a one-line JSON addition, but adding it changes what
  appears in the manuscript comparison figure (`figures/prep/figure3.py`,
  `R/plot_figure_3.R`) — this is a content change to a figure, i.e. Tier 2, not Tier 1,
  despite the small diff. **Not applied.** Also note `figure0.py:65` indexes a 5-key
  `_MODALITY_LABELS` dict positionally against `MODALITY_ORDER`; if `metburden` is added to
  the JSON without also adding it to `_MODALITY_LABELS`, the currently-silent omission becomes
  a `KeyError` instead — both edits must land together.

- **Findings 0, 1, 6 — applied in Pass 3, see above.** Blast-radius quantification was
  explicitly skipped for all three (per your decision); fixes were applied on code-level
  argument alone, not verified against real data or a before/after diff.

- **Finding 11b's stronger "reject" behavior, findings 11d/11e
  (duplicated multiple-testing correction / inference in R)** — Tier 2/3, unchanged.

- **Finding 11 — two divergent `assert_schema` copies** and **finding 10 — deduped CSVs
  re-read without revalidation** — Tier 3 (structural), reported only, no fix attempted.

- **Environment pinning (finding 12) and seed/path findings (13, 14)** — require a
  `conda env export --no-builds` run on the cluster per the plan's explicit decision; not
  attempted this session.

---

## Verification status

- `pyflakes` + `py_compile`: clean on all Python files touched across both passes, including
  `figures/prep/figure0.py` (pass 2).
- `bash -n` on all six touched `v2/slurm/*.sh` files: clean.
- `Rscript -e 'lapply(...)'` parse check on all 13 `v2/R/*.R` files: clean, including
  `R/plot_figure_3.R` (consumes the new `metburden` palette entry).
- `shared/palette.json`: validated as well-formed JSON.
- **Cluster-only verification (assertion no-op check, figure-CSV before/after diff,
  `04_smoke_test_training.ipynb` regression run) is outstanding** and must be completed before
  these Tier 1 changes are considered fully verified per the plan's own standard. In
  particular, the new `knit_embeddings.py` uniqueness assertion (item 4 above) has not yet
  been run against real `cohort_df.parquet` — if it fires, that's a genuine data problem to
  report, not a bug in the assertion.
