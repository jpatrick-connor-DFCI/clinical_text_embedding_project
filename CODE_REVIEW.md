# Code Review — Clinical Text Embedding Project

**Date:** 2026-05-31
**Scope:** Full repository (~12.5k lines: Python pipelines, `embed_surv_utils` package, R figure-rendering tier, SLURM scripts, notebooks).
**Goal context:** This codebase backs a manuscript; the five figures under `jupyter_notebooks/manuscript_figures/` are the publication targets. Findings are therefore prioritized by whether they could change a number, panel, or conclusion that reaches the paper.

## How this review was conducted

Eight module-scoped reviewers read every source file in parallel (data preprocessing, model training, mortality trajectories, biomarker analysis, the `embed_surv_utils` core library, the Python figure-prep tier, the R rendering tier, and the bash/notebook glue). A dedicated pass mapped every file to its references to find orphans. **Every candidate finding was then independently re-read by a separate adversarial verifier** that defaulted to skepticism and only confirmed issues backed by concrete code evidence.

Raw findings: 53 → verified: 41 → **confirmed: 36** (17 false positives/duplicates dropped). The items below are the confirmed set.

> The single most reassuring result: the core CV machinery in `embed_surv_utils/cox_models.py` is leakage-safe. PCA, continuous-variable scaling, and mean imputation are all fit **inside each training fold** (never on the full data or test fold), and the pre-treatment-only note window in `preprocessing.py` correctly prevents future information from entering the embeddings. The issues below are mostly at the edges — second-stage models, cohort construction, figure labeling, and hyperparameter reuse — not in the inner fitting loop.

---

## Publication-critical issues

These can change reported results or mislabel published panels. **All five have been addressed** — see the resolution note under each. H1, H3, H4, H5 are code fixes; H2 was resolved by deleting deprecated scripts. Re-run the affected pipelines/figures to regenerate results.

### H1 · Train/eval split is inverted — models trained on 25%, evaluated on 75% — ✅ FIXED
`python_scripts/mortality_trajectories/within_vs_pan_cancer_models.py:49–51`
`python_scripts/mortality_trajectories/within_treatment_vs_pan_treatment_models.py:54–56`

`eval_mrns = full_df['DFCI_MRN'].sample(frac=0.75, ...)` draws **75%** of patients, but those MRNs are routed to the *held-out* set while `train_df` is the 25% complement. Both the pan and within models are therefore fit on a quarter of the data and scored on three-quarters. Consequences: (1) the `len(sub_df) < 100` guard then drops many cancer-type / treatment subsets from the within arm entirely, biasing the headline within-vs-pan comparison; (2) high-dimensional embedding Cox fits lose most of their power. The variable name `eval_mrns` plus `frac=0.75` shows the intent was a 75/25 *train*/test split.
**Resolution:** both files now sample `frac=0.25` into `held_mrns` (75% train / 25% held-out). Re-run both comparison scripts.

### H2 · Metastasis endpoints scored twice, full-cohort pass overwrites the met-free pass — ✅ RESOLVED (script deleted)
`python_scripts/mortality_trajectories/feature_ICD10_level_3_risk_scores.py:63, 130–175`

The exclusion `col not in met_events` compared full column names like `tt_brainM` against bare names like `brainM`, so it never matched and the metastasis events were *not* excluded from the second (full-cohort) loop. The first loop deliberately scored met endpoints on the metastasis-free cohort (and additionally dropped brain-cancer patients for `brainM`); the second loop then re-scored the same endpoints on the full cohort and **overwrote** the correctly-filtered `*_held_out_preds.csv`.
**Resolution:** `feature_ICD10_level_3_risk_scores.py` (and the companion `feature_risk_score_coxph.py`) were **deleted** — both duplicated work the training scripts already do. Per-modality held-out risk scores are produced at training time by `run_feature_comp_task.py` (under `<scheme>/held_out_risk_scores/`), and the second-stage joint Cox refit is done in `prep_figure_3.py` with `lifelines`. The dead `_univariate_vs_joint` reader in `prep_figure_3.py` was removed too. ⚠️ If any metastasis `*_held_out_preds.csv` were previously written by the buggy loop, regenerate them via the training path.

### H3 · Immortal-time bias in the matched ICI cohort (Cohort 2) — ✅ FIXED (verify on real data)
`python_scripts/biomarker_analysis/generate_IPTW_df.py` (survival clock)

For Cohort 2, ICI can be received at line 1, 2, or 3, but the survival clock (`tt_death`) was anchored at **first-line time zero**. A patient who receives ICI at line 3 must survive long enough to reach line 3 — that guaranteed survival was credited to the ICI arm (classic immortal-time bias). This feeds Figure 5's primary (Cohort-2) panels.
**Resolution:** `generate_IPTW_df.py` now re-anchors every patient's survival clock to their line-specific landmark (`treatment_start_date`, the propensity/exposure anchor): patients who did not survive to their landmark are dropped (removing immortal time) and `tt_death` is measured from the landmark. For Cohort 1 (all first-line) this is a no-op. The note/embedding window already uses the landmark, so it is now consistent. ⚠️ **Verify when you run it:** the script prints the landmark-shift distribution — for Cohort 2 it must be `>0` for the line-2/3 patients. If the shift is ~0 everywhere, then `treatment_start_date` in `prediction_times.csv.gz` is *not* the line-specific date and the landmark must be sourced from the line-start date instead. (A full delayed-entry/left-truncation Cox via lifelines `entry_col` is the stricter alternative if you prefer modeling the risk set explicitly.)

### H4 · Figure 2 & 5: cancer-type labels mis-recovered from `drop_first` one-hot — ✅ FIXED (regenerate covariates)
`jupyter_notebooks/manuscript_figures/data_generation/prep_figure_2.py:223–235`
`jupyter_notebooks/manuscript_figures/data_generation/prep_figure_5.py:107–117`

Cancer type was recovered with `idxmax` over `drop_first=True` dummy columns. Rows belonging to the dropped **reference** category are all-zero, so `idxmax` silently returned the first dummy column — mislabeling every reference-category patient as some other cancer. This corrupted the cancer-type breakdowns in the Fig 2 heatmap and the Fig 5 PS-prediction annotation. (`generate_all_non_text_covariates.py:33` confirmed `drop_first=True`.)
**Resolution:** `generate_all_non_text_covariates.py` now also writes the raw collapsed `CANCER_TYPE` string column into `cancer_type_df.csv.gz` (named without a trailing `_` so it is never picked up as a feature column). `prep_figure_5._patient_cancer_type` prefers that raw label and only falls back to the dummy `idxmax` (with a warning) for legacy files. (The `prep_figure_2._cancer_type_labels` consumer was subsequently **removed** when the cancer×endpoint heatmap panel was retired — see M8 — so H4 now affects only `prep_figure_5`.) ⚠️ **Regenerate `cancer_type_df.csv.gz`** (re-run `generate_all_non_text_covariates.py`) so the raw column exists before re-running the figure preps.

### H5 · Figure 4 cluster names assert trajectory shapes that aren't tested — ✅ FIXED
`jupyter_notebooks/manuscript_figures/R/plot_figure_4.R:29–38`

Clusters are *ordered and labeled by ascending mean risk* in prep, but the R names asserted **shapes** ("Stable Low", "Rapidly Increasing", "Rebounding", etc.). A cluster's mean-risk rank does not imply a temporal shape, so a published cluster could be named "Rapidly Increasing" while actually flat. (Related: `CLUSTER_NAMES` had 5 entries for `N_CLUSTERS=4` — issue L15.)
**Resolution:** `CLUSTER_NAMES` is now four risk-**level** labels ("Lowest Risk" … "Highest Risk") matching the actual ascending-mean-risk ordering, with `stopifnot(length(CLUSTER_NAMES) == N_CLUSTERS)` so a future k-change fails loudly (also closes L15).

### H6 · `rerun_negative_delta_events.ipynb`: keep-if-improved is selection-on-noise — ✅ RESOLVED (deleted)
`jupyter_notebooks/rerun_negative_delta_events.ipynb` (cell 5, `improved = new_delta > old_delta`)

The notebook re-ran events where the text model underperformed the base model, but **only overwrote the stored result when the re-run's text-vs-base delta improved**. Conditioning the keep/replace decision on the outcome itself biases the reported text-vs-base improvement upward — a garden-of-forking-paths error. (It also used a different `MAX_ITER` than production, so the re-run wasn't even a like-for-like refit.)
**Resolution:** this one-off notebook has been **removed from the repository** (recover from git history at `8774b85` if needed). ⚠️ **Important:** if any full-cohort `text_vs_base` results were overwritten by a prior run of this notebook, those events should be **re-run once with the production `COXNET_MAX_ITER` and overwritten unconditionally** so the saved results no longer carry the selection bias. Verify the `death_met` / ICD `text_vs_base` deltas feeding Figure 2 were not contaminated.

---

## Medium issues

### M1 · Second-stage Cox reports in-sample C-index — ✅ RESOLVED (script deleted)
`python_scripts/mortality_trajectories/feature_risk_score_coxph.py:185–198` — the joint/univariate models over modality risk scores reported apparent (in-sample) C-index with no held-out evaluation. This script was **deleted** (see H2); the Figure 3 joint refit now lives in `prep_figure_3.py`. ⚠️ Note: that refit is still apparent-performance — if a joint-model C-index is reported in the manuscript, cross-validate it or label it as in-sample.

### M2 · Family-wide multiplicity not controlled across biomarker specifications
`run_IPTW_analysis.py:193–203, 287–290`; `compile_IPTW_results.py:76–99`; `validate_and_report.py:716–751` — FDR is applied *within* each specification, but hits are then pooled across many specs (2 cohorts × 2 PS models × multiple weightings × 2 tracks × cancer types) with no global correction. The "significant in ≥2 specs" filter helps but is not a formal multiplicity control.
**Fix:** pre-register one primary specification per marker, or apply a hierarchical/global correction, and always report the number of specifications tested.

### M3 · Extreme-HR / separation flag computed but never enforced
`run_IPTW_analysis.py:294–301, 361–367` — `extreme_hr_flag` (HR > 50 or < 1/50, indicating likely Cox separation) is written but `compile_IPTW_results.py` never uses it to drop or down-weight hits, so separation-driven artifacts can surface as "robust" markers.
**Fix:** exclude or separately label flagged rows in compilation and the Fig 5 volcano (see L13).

### M4 · Embedding "prognostic-score" adjustment advertised but unused
`run_IPTW_analysis.py:423–426, 448, 472, 622` — embedding columns are loaded and the docstrings/prints describe an embedding-based prognostic adjustment, but no Cox model actually includes them. Methods and implementation disagree.
**Fix:** either add the embedding adjustment to `base_vars`, or remove the claim from docstrings/outputs.

### M5 · Cohort 1 mixes first-line ICI cases with all-line controls, unadjusted
`build_line_matched_cohort.py:117–130` — Cohort 1 (discovery) compares first-line ICI cases against *all* never-ICI controls regardless of line, with no line-of-therapy adjustment, confounding the comparison by disease stage/severity.
**Fix:** restrict Cohort-1 controls to first-line, or include control line of therapy as a covariate in the Cohort-1 PS and Cox models.

### M6 · `evaluate_surv_model` time grid can exceed eval-fold follow-up
`embed_surv_utils/cox_models.py:66–76, 359–360, 412–414, 607–609` — `eval_times` are drawn from *training-fold* survival percentiles; if an eval fold's max follow-up is shorter, `cumulative_dynamic_auc`/`integrated_brier_score` raise and the fold silently becomes NaN, biasing CV means toward folds with longer follow-up.
**Fix:** clip `eval_times` to each eval fold's support (and log when a metric is NaN for time-grid reasons vs. true failure). *(Still open in `cox_models.py`. Note: the new mean-AUC(t) code added to the two within-vs-pan scripts already applies this clip per stratum, so it does not exhibit the bug.)*

### M7 · `brainM` eligibility count ignores the brain-cancer exclusion
`python_scripts/model_training/build_slurm_manifests.py:22–29` — the manifest builder counts `brainM` cases on the un-merged cohort (no `CANCER_TYPE_BRAIN` column), so the ≥50-case gate uses a cohort that differs from the one training actually uses after excluding brain-cancer patients.
**Fix:** build the same merged cohort used at training time before counting.

### M8 · Fig 2C heatmap stars are a cosmetic threshold, not significance — ✅ RESOLVED (panel removed)
`jupyter_notebooks/manuscript_figures/R/plot_figure_2.R:131` — cells got a `*` when `text_cindex >= 0.60`, reading as a significance marker but actually a fixed cutoff. **Resolution:** the cancer×endpoint heatmap panel was **removed** from Figure 2 (its `build_fig2c`, the `_cancer_endpoint_heatmap` prep, and `fig2_cancer_endpoint_heatmap.csv` are gone) and replaced by the pan-vs-within-model dumbbell panels (Fig 2C/2D), which carry no cosmetic stars.

---

## Low / minor issues

| # | File:line | Issue | Fix |
|---|-----------|-------|-----|
| L1 | `preprocessing.py:112–120` | `map_time_to_event` copies censoring time from `tt_death` without checking an event time doesn't exceed it (negative/zero survival if a diagnosis post-dates recorded death). | Validate/cap event time against `tt_death`; flag or censor violations. |
| L2 | `preprocessing.py:275–282` | Pooled-embedding `DFCI_MRN` is cast to **float** by `np.concatenate` with float embedding columns, risking silent merge-key mismatch against int MRNs. | Assign the int MRN column separately after building the float matrix. |
| L3 | `generate_all_non_text_covariates.py:49` | Somatic specimen merge uses `CANCER_TYPE` as a join key; label disagreement between sources silently drops sequenced samples. | Merge on `['DFCI_MRN','sample_id']` only. |
| L4 | `run_full_cohort_risk_scores.py:114–130` | Hyperparameters tuned on the 80% split are reused to generate OOF risk scores over **100%** of patients → mild selection optimism. | Nested CV, or score only the held-out 20%, or disclose in methods. |
| L5 | `run_feature_comp_task.py:145–164` | Same hyperparameter-reuse optimism for the feature-comparison held-out risk scores. | Same as L4. |
| L6 | `cox_models.py:742–799` | `get_heldout_risk_scores_CoxPH` runs OOF scoring at fixed hyperparameters that were chosen on overlapping patients (root cause of L4/L5). | Offer a nested-CV mode, or document. |
| L7 | `slurm_array_utils.py:146–186` | Labs use mean-imputed values **plus** missingness indicators, both penalized; coupling is fine (imputation is per-fold) but undocumented. | Document the value+indicator encoding; no code change needed. |
| L8 | `within_vs_pan_cancer_models.py:86–108` | Continuous vars are StandardScaler-scaled outside the helper, which **re-scales per fold** internally → double scaling. | Pick one scaling site. |
| L9 | ~~`feature_ICD10_level_3_risk_scores.py:124–125, 172–173`~~ | `lab_scores` omitted from the `reduce` merge. | **Moot — script deleted (H2).** |
| L10 | `cluster_mortality_trajectories.py:116–128, 211–223` | `max_val`/`time_to_max` computed but unused; `rebound = end_risk − min_val` doesn't measure post-minimum rebound as the comment claims. | Drop dead features; redefine rebound over the post-argmin segment. |
| L11 | `run_IPTW_analysis.py:235, 330` | `CoxPHFitter(penalizer=0.01)` ridge-penalizes the **tested marker/interaction** terms, shrinking the very HRs/p-values being reported. | Fit the primary term unpenalized, or verify the penalty doesn't move marker estimates. |
| L12 | `biomarker_common.py:26–27` | `load_survival_cohort` appends `.gz` unconditionally, double-appending if a caller passes a name already ending in `.csv.gz`. | Append `.gz` only if absent. |
| L13 | `preprocessing.py:254–261` | `time_decay_mean` doesn't renormalize weights when an embedding cell is NaN (`nansum` zeroes it but keeps its weight). | Per-dimension renormalize excluding NaN cells, or assert NaN-free. |
| L14 | `prep_figure_5.py:146–163` | Fig 5 volcano ignores `extreme_hr_flag`, so separation-driven HRs can appear as significant points. | Drop/de-emphasize flagged rows (match the `_robust_hits` cleaning). |
| L15 | `plot_figure_4.R:29–37` | `CLUSTER_NAMES` has 5 names for `N_CLUSTERS=4`; the 5th is dead and the cap hides a future mismatch. | Make length == `N_CLUSTERS` and assert it. |
| L16 | `plot_figure_1.R:141–144` | Fig 1C prints literal `"Kruskal-Wallis p=NA  n/a"` when the test can't run (no-op `ifelse`). | Only draw the annotation when `is.finite(kw_p)`. |
| L17 | `plot_figure_3.R:187,195` (and `plot_figure_2.R:99`) | Star y-position uses `ymax * 1.04`; if all values are negative the label moves *down* off the panel. | Use an additive offset: `max(z) + 0.04*diff(range(z))`. |
| L18 | `extract_ICD_times.py:19, 55` | Hard-coded `datetime.strptime` format strings crash on any off-format date. | Use `pd.to_datetime(..., errors='coerce')` then drop `NaT`. |
| L19 | `debug_individual_events.ipynb` (cell 1) | `PROJECT_ROOT = getcwd()/../..` overshoots the repo root → imports fail. | Walk parents until `python_utils`/`python_scripts` exist (or use one `..`). |

---

## Cross-cutting themes

1. **Hyperparameter-selection optimism (L4–L6).** Across full-cohort and feature-comparison risk scoring, OOF scores are generated at hyperparameters tuned on overlapping patients. Each instance is individually minor, but it is systematic. Either move to nested CV in `get_heldout_risk_scores_CoxPH`, or add one sentence to the methods stating that per-patient risk scores are not fully nested.
2. **Observational-ICI confounding (H3, M2, M5, M3, L11).** The biomarker pipeline is the most inferentially delicate component. The immortal-time anchoring (H3) is the highest-impact; the multiplicity (M2), separation handling (M3), unadjusted Cohort 1 (M5), and marker penalization (L11) compound it. Recommend a focused methods review of this pipeline with a statistician before the biomarker claims go in.
3. **Figure-label fidelity (H4, H5, M8, L14–L17).** Several figure issues are about labels asserting more than the data shows (shape names, significance stars) or edge-case rendering. None corrupt the underlying CSVs, but they are exactly what reviewers scrutinize.

## Status & remaining work

**Done (this pass):** H1–H5 addressed (H1/H3/H4/H5 code fixes; H2 + M1 + L9 resolved by deleting `feature_ICD10_level_3_risk_scores.py` and `feature_risk_score_coxph.py`; H6 resolved earlier by deleting `rerun_negative_delta_events.ipynb`). **Re-run to materialize:** the two within-vs-pan scripts (H1), `generate_all_non_text_covariates.py` → figure preps (H4), `generate_IPTW_df.py` → Figure 5 (H3, **and verify the printed landmark-shift**), and `plot_figure_4.R` (H5).

**Still open (you opted to leave these unfixed):**

1. **M2, M5, M3, L11 (biomarker statistics)** — multiplicity across specs, unadjusted Cohort 1, separation-flag enforcement, marker penalization. Best reviewed with a statistician, then re-run Figure 5.
2. **M8, L16, L17 (figure render polish)** — cosmetic-significance stars, NA annotations, negative-value label placement. Cheap; touch published panels.
3. **L4–L6 (hyperparameter-selection optimism)** — systematic but minor; either nested CV in `get_heldout_risk_scores_CoxPH` or a one-line methods disclosure.
4. **Medium/low remainder** — fold into the next cleanup pass; several (L12, L18, L19) are one-line fixes.

*All file references are clickable from the repository root. The raw machine-readable finding set is preserved in the review transcript.*
