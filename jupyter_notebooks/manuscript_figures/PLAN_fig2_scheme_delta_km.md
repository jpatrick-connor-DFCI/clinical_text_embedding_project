# Figure 2 — Δ C-index barplots + text-vs-base held-out-risk KM panels (per event category)

## Context

Figure 2 currently shows text-vs-base full-cohort performance (scatter, Δ violins,
pan-vs-within dumbbells, and OS-only KM curves). It does **not** show, per clinical-event
category, *which specific events* the text model improves most, nor the held-out risk
stratification for those events. This change adds, for each of three event categories —
**mets**, **ICD10**, **phecodes** — a barplot of the top-3 events by Δ C-index (text − base),
plus Kaplan–Meier curves of held-out risk (text vs base) for those events.

**Placement (confirmed with user):** the **barplots + the single top KM curve per category**
go on the **main Figure 2**; the **remaining two KM curves per category** go into a **new
supplement**. The new panels are **C-index only** — they do **not** honor the
`MANUSCRIPT_METRIC` switch (they always rank/label by Δ C-index and produce one
un-suffixed figure).

**Category → scheme mapping (confirmed, "mets / ICD10 / phecode"):**
- **mets** = `death_met` events excluding the literal `death` (metastatic-site events).
- **ICD10** = `icd3_post` **and** `icd4_post` pooled into one category.
- **phecodes** = `phecode_post`.

## Data model (already in place)

- **Δ C-index source**: `fig2_full_cohort_metrics.csv` (`prep_figure_2.py::_full_cohort_metrics`)
  — one row per `(scheme, event)` with `event_lbl, text_cindex, base_cindex`;
  `delta = text_cindex - base_cindex`.
- **Held-out risk scores per event**: `full_cohort_risk_dir(scheme, event)` (`_figure_utils.py`)
  → `text_risk_scores.csv` (`DFCI_MRN, text_risk_score`) and `base_risk_scores.csv`
  (`DFCI_MRN, base_risk_score`), written by `run_full_cohort_risk_scores.py` per `(scheme, event)`.
- **Survival labels per event**: `load_embedding_prediction_df(scheme)` (`slurm_array_utils`)
  provides `<event>` (0/1 flag) and `tt_<event>`; filter with `filter_event_rows(df, event)`.
- Reusable helpers already present: `_merge_risk_with_surv` (death-only; to be generalized),
  `_safe_tertiles`, `save_figure_data`; R-side `survfit2`, `tidy_km`, `logrank_p`, `step_ci_df`,
  `SCHEME_COLORS`, `SCHEME_LABELS`, `RISK_COLORS`, `MODEL_COLORS`, `theme_manuscript`.

> **Pipeline prerequisite (flag, don't fix in training code):** per-event held-out risk-score
> CSVs only exist for `(scheme, event)` pairs where `run_full_cohort_risk_scores.py --scheme <s>
> --event <e>` has been run. Today the prep reads them only for `death`. The prep must degrade
> gracefully — skip an event / emit header-only rows — when a selected top-3 event lacks
> risk-score files, so the figure still renders before the cluster backfills those runs. This plan
> does **not** modify the training scripts.

## Changes

### 1. `data_generation/prep_figure_2.py` — two new outputs

- **Category tagging** helper mirroring the R logic:
  `death_met & event!="death" → "mets"`, `icd3_post|icd4_post → "ICD10"`,
  `phecode_post → "phecodes"`, `death_met & event=="death" → None` (dropped).
  Add `CATEGORY_ORDER = ["mets", "ICD10", "phecodes"]`.

- **`_scheme_delta_topk(metrics_df, k=3)` → `fig2_scheme_delta_topk.csv`**
  `delta = text_cindex - base_cindex`, tag category, drop NaN deltas and the `death` row,
  **keep only `delta > 0`** (largest *positive* improvements only — never surface a
  net-negative event as a "top" hit), then `groupby(category).delta.nlargest(3)`. A category
  with fewer than 3 positive-delta events yields fewer bars (and correspondingly fewer KM
  panels). Columns:
  `category, rank(1..3), scheme, event, event_lbl, text_cindex, base_cindex, delta`.

- **`_scheme_event_km(topk_df) → fig2_scheme_event_km.csv`**
  Generalize `_merge_risk_with_surv` to arbitrary `(scheme, event)`: read
  `full_cohort_risk_dir(scheme, event)/{text,base}_risk_scores.csv`, load survival via
  `load_embedding_prediction_df(scheme)` + `filter_event_rows(df, event)`, merge on `DFCI_MRN`,
  keep `tt_<event> > 0`, drop NaN, bin each model's score with `_safe_tertiles`. Emit one long
  table over the ≤9 selected events, renaming per-event `<event>`/`tt_<event>` to constant
  `event_flag`/`tt`:
  `category, scheme, event, event_lbl, DFCI_MRN, text_risk_score, base_risk_score, event_flag, tt, text_tertile, base_tertile`.
  Skip (printed note) any event missing risk-score files; header-only CSV if none survive.
  Fold the existing death-only `_merge_risk_with_surv` into the general helper (`_km_tertiles`
  calls it with `death`).

- Wire both into `main()`; update the module docstring's output list with both schemas.

### 2. `R/plot_figure_2.R` — 3 barplots + 3 top-1 KM panels

All builders guarded by `placeholder_panel`.

- **`build_scheme_delta_bars(topk, category)`** — horizontal `geom_col` of `delta` for the 3 top
  events of one category, ordered by delta, `event_lbl` y-labels, value labels via `geom_text`,
  filled with the category's `SCHEME_COLORS` (mets→`death_met`, ICD10→`icd3_post`,
  phecodes→`phecode_post`), title = category, x = "Δ C-index (Text − Base)".

- **`km_tertile_panel(km, title)`** — extract the KM rendering currently inside `build_fig2d`
  (text-solid/base-dashed tertile curves + 95% CI bands + log-rank p) into a shared helper.
  `build_fig2d` calls it for OS; the new per-event panels call it too.

- **`build_event_km(km_df, topk, category)`** — subset `fig2_scheme_event_km.csv` to the
  **rank-1** event of the category, `months = tt/30.44`, call `km_tertile_panel` with
  `title = event_lbl`.

- **Compose**: add two rows below `risk_row` — a barplot row (`bars_mets | bars_icd | bars_phe`)
  and a top-KM row (`km_mets1 | km_icd1 | km_phe1`). Add `save_panel(...)` per new panel
  (fixed names, **no `.tag`**); bump `save_figure(fig2, ...)` height; keep
  `plot_annotation(tag_levels = "A")`.

### 3. New `R/plot_figure_2_supp_events.R` — remaining 6 KM panels

New script (separate from the existing stage-stratified supp). Same bootstrap +
`source(figure_utils.R)`; local copy of `km_tertile_panel` (as `plot_figure_4_supp.R` keeps its
own helpers). Read `fig2_scheme_event_km.csv` + `fig2_scheme_delta_topk.csv`, build a KM panel for
**ranks 2 and 3** of each category (6 panels), compose as a 3×2 grid (rows = category,
cols = rank 2 / rank 3), `plot_annotation(tag_levels = "A")`, single un-suffixed output
`figureS_scheme_event_km.png`.

### 4. `README.md`

- Update `plot_figure_2.R` tree line (panel count); add the new supp script line.
- Add per-figure design notes: Δ C-index top-3 barplots (category mapping, C-index-only rationale)
  and the per-event held-out-risk KM panels (top-1 in main, ranks 2–3 in supp).
- Add `fig2_scheme_delta_topk.csv` + `fig2_scheme_event_km.csv` to the Figure 2 inputs table and
  the new supp row; note the per-event `run_full_cohort_risk_scores.py` prerequisite.
- Add the new supp to the **no-metric-switch** render loop (it is C-index-only).

## Reuse (do not re-implement)

- Category/mets logic mirrors `plot_figure_2.R::build_fig2a` (`FIG2A_GROUP_*`) and
  `prep_figure_2.py::MET_SITES`.
- KM rendering: refactor `build_fig2d` → shared `km_tertile_panel()`; reuse `tidy_km`, `logrank_p`,
  `step_ci_df`, `RISK_COLORS`, `MODEL_COLORS`.
- Tertile binning: `_safe_tertiles`. IO: `load_figure_data`/`save_figure_data`,
  `full_cohort_risk_dir`, `load_embedding_prediction_df`, `filter_event_rows`.

## Verification

This machine cannot run the pipeline (no cluster data; `py_compile`/`Rscript` parse are
syntax-only — see project-memory notes).

1. **Python**: `python -m py_compile prep_figure_2.py`, then **grep** every new symbol
   (`_scheme_delta_topk`, `_scheme_event_km`, category helper, `CATEGORY_ORDER`) for undefined
   references (`py_compile` does not catch those).
2. **R**: `Rscript -e 'parse("R/plot_figure_2.R")'` and the new supp; confirm `km_tertile_panel`
   is defined before first use and the composite still assembles.
3. **Schema-empty degradation**: confirm both new prep generators return header-only frames on
   missing risk-score files, and each R builder falls back to `placeholder_panel` on empty CSV.
4. **On-cluster (user, later)**: run `run_full_cohort_risk_scores.py --scheme <s> --event <e>` for
   the selected top-3 events, then `python prep_figure_2.py`, then `Rscript R/plot_figure_2.R` +
   `Rscript R/plot_figure_2_supp_events.R`; inspect
   `target_figures/figure2_text_results_cindex.png` and `figureS_scheme_event_km.png`.
