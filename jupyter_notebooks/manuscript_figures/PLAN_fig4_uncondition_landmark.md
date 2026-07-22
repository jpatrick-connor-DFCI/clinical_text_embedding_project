# Figure 4 — Un-condition the mortality-trajectory analysis (landmark L, prep-only)

## Context
Figure 4 groups patients by the **slope** of their model-estimated mortality
risk over 0–60 months, then contrasts groups with a conditional-on-survival KM.
The question addressed here: **can the trajectory analysis be done WITHOUT
conditioning on survival to month 60?**

Investigation found the "conditioning" has **two independent sources**:

1. **Generator** (`python_scripts/mortality_trajectories/generate_mortality_trajectories.py`,
   lines 76–95): at each landmark `month_adj`, `tt_death` is re-anchored
   (`preprocessing.py:331` subtracts `max_note_window = month_adj*30`), then
   `monthly_data.loc[monthly_data['tt_death'] > 0]` (line 83) **drops** patients
   who have died by that landmark, and `prev_mrns = risk_scores['DFCI_MRN'].unique()`
   (line 92) means dropouts never return. So late-month risk scores are
   **structurally NaN** for early-dying patients — a full 0–60 trajectory cannot
   exist for them.
2. **Prep** (`data_generation/prep_figure_4.py:162-164`, `_scaled_trajectory_matrix`):
   `missing_rowwise > 18` keeps only patients present in ≥19 of 21 landmarks
   (≈ survived to ~month 54–60). **This is the filter that turns the slope into a
   "conditioned-on-60mo-survival" quantity.**

**Decisions (confirmed):**
- **Relax the PREP layer only** — no generator re-run. Keep the existing
  `survival_trajectories_w_decay_param_0.1.csv`.
- **Align the KM landmark to the slope window** — group patients by their risk
  dynamics over months `0..L`, and left-truncate survival at `entry = L`
  (measure survival forward from L), instead of `entry = 60`.
- **Parameterize L** as a module constant; default **L = 12** (largest
  unconditioned cohort, early-dynamics signal). L = 24 selectable by changing one
  value.

Outcome: the dynamics grouping and KM are computed on patients **alive at month
L** (not month 60), each patient's slope taken over the landmarks they actually
have in `0..L` (≥3 observed points, no impute). This removes the near-full-survival
selection while keeping a clean, immortal-time-safe landmarked KM.

## Files to modify

### 1. `data_generation/prep_figure_4.py`
- **New module constants:**
  - `SLOPE_LANDMARK_MONTHS = 12` — the L knob (set to 24 to switch variant).
  - `MIN_SLOPE_POINTS = 3` — minimum observed landmarks in `0..L` to keep a patient.
- **`_scaled_trajectory_matrix` — the core change.** Replace the
  `missing_rowwise > 18` full-completeness filter and the ffill/bfill impute with
  a **windowed, no-impute** selection:
  - Restrict `months_to_keep` to landmarks with month number `<= SLOPE_LANDMARK_MONTHS`
    (reuse `_month_nums` on the column names).
  - Keep patients with **≥ `MIN_SLOPE_POINTS` non-NaN** values within that window
    (`missing_rowwise = (~traj[window].isna()).sum(axis=1); pxs_to_keep = idx[missing_rowwise >= MIN_SLOPE_POINTS]`).
  - **Do NOT ffill/bfill.** Leave NaNs in place; the slope is computed per-patient
    over each patient's own observed points (below). `X_z` is still needed for the
    silhouette scan / any KMeans on standardized *slope*, so build it from the
    **slope feature**, not the raw matrix — i.e. move standardization to operate on
    the slope vector (see `_cluster_trajectories`), and have this function return
    the windowed (un-imputed) `traj_sub` + `months_to_keep`. Adjust the return
    contract/callers accordingly.
- **`_ols_slopes` — make it per-patient-window-aware.** The current vectorized
  form (`_ols_slopes`, lines 183–196) assumes every row is complete over
  `months_to_keep`. Rewrite to compute each patient's slope over **their own
  non-NaN** subset of the window: for each row, `mask = ~isnan(y)`;
  `slope = cov(x[mask], y[mask]) / var(x[mask])` (closed form); require
  `mask.sum() >= MIN_SLOPE_POINTS` (guaranteed by the prep filter) and
  `var(x[mask]) > 0`. Vectorization optional — a per-row loop over a few-thousand
  patients is fine; or group rows by identical missingness pattern and batch.
- **`_cluster_trajectories`:** unchanged in spirit — compute slopes via the new
  `_ols_slopes`, `StandardScaler` the 1-D slope, `KMeans(n_clusters=N_SLOPE_GROUPS,
  random_state=0, n_init=10)`, relabel by ascending mean slope (existing idiom).
  `traj_sub["slope"]` persisted as before. The empty/`< N_SLOPE_GROUPS` guard stays.
- **`_km_data`:** add the landmark to the contract. Emit the existing columns
  (`DFCI_MRN, cluster, death, tt_death, stage`) **plus** a constant column
  `landmark_month = SLOPE_LANDMARK_MONTHS` so R reads L from data rather than
  hard-coding it. The cluster assignment now only exists for patients alive at L,
  which is exactly the intended cohort.
- **`_group_trajectories` / `_slope_by_stage` / `_cluster_severity` / heatmap
  downsample:** these consume `months_to_keep` + `slope` + `cluster`; they keep
  working, but note `months_to_keep` is now the **`0..L` window** (heatmap/mean-
  trajectory panels now span 0..L, not 0..60). `_heatmap_downsample` must tolerate
  NaNs (no impute upstream) — prefer a display-only local ffill so the heatmap
  stays readable, with a comment that grouping used un-imputed slopes.
- Update the module docstring: slope computed over months `0..L`
  (L = `SLOPE_LANDMARK_MONTHS`) on each patient's observed landmarks with
  `MIN_SLOPE_POINTS` minimum; cohort = patients alive at L, not survivors to 60.

### 2. `R/plot_figure_4.R`
- **`build_fig4b` (conditional KM):** replace the hard-coded `entry = 60` / `time >= 60`
  / `xlim(60,120)` with a `LANDMARK` value read from the data
  (`unique(d$landmark_month)`, fallback constant `12`). `Surv(entry = LANDMARK,
  months, death) ~ strat`; `filter(time >= LANDMARK)`; `coord_cartesian(xlim = c(LANDMARK, 120))`.
  Update axis label to "…conditional on survival to month {LANDMARK}".
- **fig4a heatmap / fig4d mean-trajectory / fig4e stage bars:** x-axis now spans
  `0..L`; no logic change beyond whatever axis limits/breaks are hard-coded to 60.
  Update fig4d/fig4a titles if they mention "60".
- Keep `N_SLOPE_GROUPS`, `GROUP_NAMES`, `GROUP_COLORS`, `cluster_label` as-is.

### 3. `R/plot_figure_4_supp.R`
- Mirror the KM landmark change: read `LANDMARK` from `fig4_km_data.csv`
  (`unique(d$landmark_month)`, fallback 12). Replace every `60`/`entry <- 60`/
  `filter(months > 60)`/`filter(time >= 60)`/`xlim(60,120)` and the
  "conditional on survival to month 60" y-axis label with the landmark value.
  `build_stage_dynamics_panel` uses `Surv(entry, months, death)` where
  `d$entry <- LANDMARK`.

### 4. `README.md`
- Update the Fig 4 + Fig 4 supplement bullets: slope now over months `0..L`
  (default L = 12) on each patient's **observed** landmarks (≥3 points, no impute),
  cohort = alive-at-L rather than survivors-to-60; KM left-truncated at L.
  Note the `SLOPE_LANDMARK_MONTHS` knob (12 vs 24) and the added `landmark_month`
  column in `fig4_km_data.csv`.

## Reuse (do not reinvent)
- `_month_nums` (prep_figure_4.py:178) to get month numbers from column names.
- Ascending-mean-slope relabel idiom already in `_cluster_trajectories`.
- Stage normalization `_major_stage_map` / `_normalize_major_stage` (already added).
- R: `load_figure_data`, `save_panel`, `save_figure`, `placeholder_panel`,
  `step_ci_df`, `tidy_km`, `logrank_p_lt` (supp), palette constants.

## Trade-offs baked into this design (for the record)
- **Prep-only ⇒ generator NaNs remain**, so the *latest* landmark any patient can
  have is bounded by their survival — but since we now only need `0..L` with L≪60
  and ≥3 points, nearly all patients alive at L qualify. The residual selection is
  "alive at L", which the KM's `entry = L` left-truncation correctly accounts for
  (immortal-time-safe).
- **No impute ⇒ slope uses real observed risk only**; a patient with points at
  {0,6,12} gets an honest 3-point slope rather than a bfill-fabricated one.
- **L is a genuine scientific knob**: L=12 maximizes the early-death-inclusive
  cohort; L=24 gives steadier slopes. Committed default 12; switch via one constant.

## Verification
Local machine has **no `/data/gusev` mount** — cannot run the pipeline. Verify
locally by syntax only, and **grep for stale `60` / removed symbols** (py_compile
and Rscript-parse do NOT catch name-resolution):
- `python3 -m py_compile data_generation/prep_figure_4.py`
- `Rscript -e 'invisible(parse("R/plot_figure_4.R")); invisible(parse("R/plot_figure_4_supp.R")); cat("OK\n")'`
- `grep -n "> 18\|ffill\|bfill\|\b60\b" data_generation/prep_figure_4.py R/plot_figure_4*.R`
  to confirm no hard-coded 60-month / full-completeness logic survives.
- AST undefined-name scan on prep_figure_4.py (as done for the N_CLUSTERS fix).

On the cluster: run `python data_generation/prep_figure_4.py` (default
`--decay 0.1`) then the render notebook / `Rscript R/plot_figure_4.R` &
`plot_figure_4_supp.R`, and confirm:
- cohort size jumps vs. the old survivors-to-60 cohort,
- three monotone-slope groups still form,
- fig4b / supp KM curves start at month L and still separate falling < rising,
- re-running with `SLOPE_LANDMARK_MONTHS = 24` reproduces the L=24 variant.
