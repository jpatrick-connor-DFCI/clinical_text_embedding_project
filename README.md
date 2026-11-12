# Clinical Text Embedding Project

This project investigates whether dense representations of clinical narratives (EHR notes) can
improve survival prediction and identify genomic biomarkers of treatment response in oncology.

## Layout

The tree is organized by pipeline stage rather than by file type:

- `pipelines/` — `preprocessing/`, `training/`, `trajectories/`, `biomarkers/`, `note_transformer/`
- `survival/` — Cox model fitting, checkpointing, evaluation
- `figures/prep/` — main-figure and supplemental preparation modules, which write the CSVs the R tier plots from
- `R/` — figure rendering and shared plot utilities
- `notebooks/` — thin driver notebooks, grouped by stage
- `shared/` — palette and stage logic used across pipelines and figures
- `data/` — versioned lookup tables and their loaders
- `manuscript_drafts/` — main manuscript and figure/table assets

The main rough draft (reviewed 13 October 2026) is
[Main Clinical Text Embedding Project Manuscript](<manuscript_drafts/Main Clinical Text Embedding Project Manuscript.docx>).
The drafted Introduction and Discussion are merged into this document with
tracked changes and one consolidated bibliography. Results and Methods retain
their reviewed text apart from citation renumbering; figures and captions are
unchanged. Substantive review suggestions are in comments. The superseded
standalone Introduction/Discussion draft has been removed from the repository.
The current figure export uses the full clinical-text workflow for Figure 1,
published-score comparisons in Figure 2e, modality comparisons in Figure 3, and
mortality trajectories in Figure 4. The current curated package is
[`manuscript_drafts/manuscript_export/`](manuscript_drafts/manuscript_export/),
mirrored to the Desktop export. Supplementary Table 1 combines cohort eligibility,
data availability, cancer types, and stage distributions; endpoint counts remain
in Supplementary Table 2. The main manuscript contains Figures 1–4 and
Supplementary Figures S1–S11; the corresponding curated supplementary files are
`Figures/Supplementary/fig_s1.png` through `fig_s11.png`, with matching captions.
Table files remain separate, with their existing identifiers. Historical export
manifests preserve their original source filenames and numbering.
Superseded manuscript drafts and review documents, including archived copies,
were removed on 13 October 2026. Figure and table assets and their archives remain.
The cohort-table renderer reads the preserved consolidated CSV by default, so
regeneration does not depend on a removed manuscript draft.

Render manuscript panels and compiled Figures 1–4 with:

```bash
R/render_all_figures.sh
```

All performance figures use C-index; AUC variants and the former Figure 5a
propensity ROC panel are retired. Rendering removes their old PNG/PDF files from
the configured output directory. Figures 1–4 are also assembled with lowercase
panel letters and without individual plot titles, saved as `png/figureN/figureN_cindex.png` and
`pdf/figureN/figureN_cindex.pdf` beneath `CLINICAL_FIGURES_OUT`. Individual panels
are retained. Missing panels are skipped with their reasons logged; available
panels retain their original file identifiers, and the compiled figure lists unavailable
panels in its caption. If every panel is missing, that figure is skipped and the
remaining figures still render. This renderer produces working source artwork;
the export script applies the curated supplementary selection.

The main figures use the following panel order. Individual files are
`figNa_cindex`, `figNb_cindex`, etc.; these identifiers follow reading order in
the compiled figures (Figure 2e spans the third row; Figure 3c spans the bottom row).

| Figure | a | b | c | d | e |
| --- | --- | --- | --- | --- | --- |
| 1 | Patient text representation | Endpoint prediction | Main manuscript analyses | — | — |
| 2 | Base vs. text C-index | Delta-C-index violin | Stage KM | Text-risk quartile KM | Published score, text, and combined C-index |
| 3 | Modality rank | Significant endpoints | Joint Cox violins | — | — |
| 4 | Risk-score heatmap | KM by risk trajectory | Risk dynamics by stage | Stage I–II rising vs. stage IV falling risk KM | — |

The earlier cohort/data-availability and cancer-type/stage plots remain as working
source artwork under `manuscript_figures/figureS1` and `figureS21` (within the PNG/PDF
directories). They are excluded from the curated export. The consolidated
Supplementary Table 1 preserves the source counts and reported percentages, with
separate denominators for eligibility (65,080), cancer types (46,914), and known
stage (30,827). Original Supplementary Tables 1, 3, and 4 are archived after
consolidation; the remaining table identifiers are preserved.
Figure 1 uses the full `figures/workflow/clinical_text_workflow.png` and vector PDF;
regenerate them with `python3 figures/render_workflow.py`.

Figure 2e selects mGPS, RMH, LIPI, ALBI, MELD, IPI, IMDC and MSKCC with ECOG,
ECOG, and KPS. It retains calculated and note-documented sources with at least
50 evaluated patients and excludes ECOG-free variants and CAPRA. Absent model
estimates are labeled unavailable. The broader score report remains available
separately. The default renderer omits the legacy ICI biomarker Figure 5.

To refresh a local export from saved vector artwork and the aggregate score table
when the preparation data are unavailable, run:

```bash
python3 figures/render_workflow.py
python3 figures/render_published_score_panel.py --table manuscript_drafts/manuscript_figures/tables/figPS_published_scores.csv --output-stem /tmp/clinical-figure-revision/fig2e_cindex
# macOS PDF rasterizer (the main PDFs retain vector artwork):
swiftc figures/render_pdf.swift -o /tmp/render_clinical_pdf
python3 figures/revise_saved_figures.py --score-panel /tmp/clinical-figure-revision/fig2e_cindex.pdf --pdf-renderer /tmp/render_clinical_pdf
python3 figures/render_cohort_summary_table.py
python3 figures/curate_manuscript_supplements.py --package manuscript_drafts/manuscript_export --revision-date 2026-10-12
python3 figures/curate_manuscript_supplements.py --package "/path/to/existing/Desktop/export" --revision-date 2026-10-12
```

The saved-artwork revision archives prior Figures 1/2 before replacing them, uses
original vector panels, and marks the absent ALBI combined estimate. It requires
the existing Figure 1 archive in `manuscript_drafts/archive/`. The exporter updates
four main figures, retains only the selected supplementary figures, archives
excluded artwork, removes the retired main Figure 5, and refreshes the index and
checksums. Run `figures/curate_manuscript_supplements.py` to install the consolidated
table and update table metadata before refreshing the figure export. The source
Word manuscript is preserved.

Within-versus-pan training comparisons and top-event plots are retired, including
the top-event supplement. Their inputs are no longer required by Figure 2 preparation.

Within-cancer supplements evaluate the existing pooled models separately within
each recorded cancer type, without retraining. `figures.prep.within_cancer` pairs
text and base scores from `full_cohort_risk_scores/` and text and other-modality
scores from `held_out_risk_scores/`. Both models are evaluated on matched patients
with concordance calculated within joint text/comparator outer-fold blocks and
weighted by comparable-pair counts, avoiding comparisons across fitted risk scales.
The default minimum is 20 matched patients, 5 events, and 1 comparable pair per
endpoint/cancer/comparator; override the first two with `--min-patients` and
`--min-events`. The preparation writes `fig2_within_cancer_cindex.csv`,
`fig3_within_cancer_cindex.csv`, `fig2_within_cancer_event_counts.csv`,
`fig3_within_cancer_event_counts.csv`, `within_cancer_audit.csv`, and
`fig3_within_cancer_modality_cindex.csv` to `FIGURE_DATA_DIR`. The last scores every
modality on one shared set of patients and comparable pairs per endpoint and cancer type
(blocks joint over all modalities' outer folds), so modality ranks share one footing.

`figures.prep.within_cancer_joint` instead refits the Figure 3 joint Cox model within each
selected cancer type (`SELECTED_CANCER_TYPES` in `shared/palette.json`), standardizing the
held-out modality risk scores within the cancer type and using the same eligibility rules and
fit variants as `fig3_joint_betas.csv`. It writes `fig3_within_cancer_joint_betas.csv` and a
per-stratum status table, `fig3_within_cancer_joint_fits.csv`.

`figures.prep.figure3_combined` fits stacked Cox models on the held-out modality risk scores
for every endpoint: each non-text modality alone and with the text score, text alone, all
modalities except text, and all modalities. Each score is standardized within its own outer
fold, the stacking Cox fit is cross-fitted over the text model's outer folds, and every model
for an endpoint is scored on one shared cohort and set of comparable pairs (fold-block C-index).
It writes `fig3_combined_cindex.csv`, plus overall-survival patient-bootstrap intervals
(`--n-boot`, default 1000) in `fig3_combined_os_cindex.csv` and `fig3_combined_os_delta.csv`.

`figures.prep.published_scores` compares published within-cancer-type prognostic scores
(MDCalc-style: mGPS, RMH, LIPI, ALBI, MELD, CAPRA-mod, IPI, IMDC and MSKCC) against the
held-out full-cohort text risk score for overall survival. For each score it evaluates three
models — the published score alone, text alone, and the two combined — on the same eligible,
lab-observable, complete-case patients and comparable pairs (fold-block C-index, blocks over the
text model's outer folds). It reads `published_scores_df*.csv.gz`, written per (anchor, lab
window) by `pipelines.preprocessing.build_published_scores`, itself fed by the read-only
`pipelines.preprocessing.audit_published_score_inputs` feasibility audit. It writes
`pubscore_cindex.csv`, `pubscore_delta.csv` (the three paired contrasts with bootstrap CIs),
`pubscore_cox.csv` (text HR per SD and published HR per point, each adjusted for the other, with
LRT p-values), `pubscore_km.csv`, and `pubscore_cohort.csv`. Only the primary treatment-anchor,
30-day lab-window run is plotted (`R/plot_figure_published_scores.R`, `published_scores` output
group); the sequencing-anchor and 90-day-window runs are sensitivity results kept in the report
tables only. No structured performance-status source exists, so the full IPI/IMDC/MSKCC take
ECOG (or KPS converted to ECOG) regex-extracted from progress notes 0-90 days before the anchor by
`pipelines.preprocessing.extract_note_scores`, which must run before the builder; ECOG >= 2 scores
the item (KPS < 80% for IMDC/MSKCC). Their `*_noecog` variants drop that item and are lower bounds
scored with the original cutpoints, over the larger population without documented ECOG.
Every score is evaluated from two sources, each on its own cohort: `calculated` (the builder's
structured-data score) and `regex` (the score as documented in progress notes 0-90 days before
the anchor, from `pipelines.preprocessing.extract_note_scores`, reviewed in
`notebooks/1_data/01c_published_scores_note_regex.ipynb`). The regex source runs only for the
primary lab-window key, and not for the `*_noecog` variants. Every output carries `source`; `pubscore_cohort.csv` adds
`n_both_sources` (patients scored by both) and the figure labels each row with its evaluated
patients and deaths, with a separate panel for eligible, scored, and evaluated counts.

`figures.prep.within_cancer_km` splits the Figure 2c/d cohort (patients with a known major stage
and a full-cohort overall-survival text risk score) by selected cancer type, keeping the pan-cancer
text risk quartiles of Figure 2d (cut over the whole cohort before the split). It writes `fig2_km_stage_vs_risk_by_cancer.csv` and
`fig2_stage_vs_risk_cindex_by_cancer.csv` (stage and text-risk C-index per cancer type, with a
status for types below `--min-patients` 20 / `--min-events` 5).

Before evaluating each endpoint, the count tables report valid patients, observed
events, and non-events (`n_non_events`) by cancer type, including endpoints with no
trained risk scores and strata with zero eligible patients. Full-cohort membership
is the intersection of scheme-specific embedding IDs and cancer annotations;
modality membership also intersects somatic, germline, stage, and treatment IDs,
as in training. Metastatic burden is zero-filled and adds no membership restriction.
The count tables record eligibility, status, and the required patient/event thresholds.
These counts are upper bounds before matching predictions; the final paired cohort
must still meet the thresholds. Concordance is skipped for undersized strata.
The module displays two global progress bars, **Full cohort** and **Modality cohort**,
across all schemes/endpoints. Warnings are suppressed; skipped comparisons and
missing modality inputs are recorded in the audit CSV without console logs.

Supplement S2 shows text versus base C-indices in cancer-type facets with endpoint
counts and median paired differences. Supplement S3 shows a cancer-by-modality
heatmap of median paired text-minus-comparator differences, with median absolute
C-indices and endpoint counts in each cell. A median paired difference need not
equal the difference of the two medians. Only eligible comparisons are plotted;
unavailable combinations are not assigned zero. The summaries are descriptive,
without endpoint-level significance tests. PNG/PDF files are named
`figS2_within_cancer_cindex` and `figS3_within_cancer_cindex` in the `figure2` and
`figure3` output groups, with up to 12 cancer types per page (`_page2`, etc.).
Matching summary CSVs and legends are written under `tables/` and `captions/`.
Each supplement also writes a single-page selected-cancer version
(`figS2_within_cancer_selected_cindex`, `figS3_within_cancer_selected_cindex`) limited to
Breast, Leukemia, Lung, Bowel, Brain, Skin, Pancreas, Lymphoma, and CUP
(`SELECTED_CANCER_TYPES` in `R/within_cancer_utils.R`); in the heatmap, a selected type
with no eligible endpoints keeps a grey "Unavailable" row. S2 also writes
`figS2_within_cancer_selected_v2_cindex`: every recorded cancer type except the pooled OTHER
category on one page, 7 panels per row. The Figure 3 supplement also
writes within-cancer modality ranks, mirroring Figure 3a: for each endpoint and cancer type
with every modality evaluable, modalities are ranked by their shared-cohort C-index from
`fig3_within_cancer_modality_cindex.csv` (1 = best; ties averaged). Heatmaps of
mean rank [IQR] are saved as `figS3_within_cancer_rank_cindex` (all cancer types, paged) and
`figS3_within_cancer_rank_selected_cindex` (selected cancer types). `R/plot_figure_os_within_cancer.R`
writes `figOS_within_cancer_cindex` to the `figure_os` group: overall-survival
(`death_met`/`death`) C-indices for the same cancer types, one panel per comparison
(text versus base, then text versus each other modality), with a matching table and legend.
`R/plot_figure_3_supp_cancer_joint.R` renders the per-cancer joint Cox refits to the `figure3`
group: `figS3_within_cancer_joint_betas` (coefficient violins per cancer type, as in Figure 3c)
and `figS3_within_cancer_joint_significant` (BH-FDR significant endpoints per cancer and
modality, as in Figure 3b), both on complete-case endpoints within each cancer type and the
`MANUSCRIPT_JOINT_COX_VARIANT` fit.
`R/plot_figure_3_supp_combined.R` renders those models to the `figure3` group as
`figS3_combined_models` (panels also saved separately): overall survival for each modality with
and without text, for text / all except text / all modalities, their paired differences, and
those differences across all endpoints (shared endpoint filter applied).
Two slide-sized (16:9) pages plot per-endpoint C-indices as in Figure 2a (dotted y = x):
`figS3_combined_scatter_modality_text` (each modality alone vs. with text, five facets in one
row at full slide width) and `figS3_combined_scatter_all_text` (all except text vs. all modalities, beside all
except text vs. text alone); their caption and table are written under `figS3_combined_scatter`.
`R/plot_figure_2_supp_cancer_km.R` renders Figure 2c/d per selected cancer type to the `figure2`
group: `figS2_within_cancer_km_stage` (overall survival by stage) and
`figS2_within_cancer_km_risk` (by pan-cancer text risk quartile, within-type patients), 3×3 grids annotated with the
per-cancer C-index and log-rank p.
Risk-score files must contain `outer_fold`; older files are skipped with an audit
entry and can be regenerated with the corresponding training/risk runner's
`--overwrite` flag. With the shared endpoint filter enabled, rendering also needs
`fig2_full_cohort_metrics.csv` from `figures.prep.figure2`.

Figure 1 preserves the workflow artwork at its native size. Figure 2 uses a full-width
score panel in the third row, below two paired rows; Figures 3 and 4 retain their existing layouts.
Compiled exports include 600-dpi PNGs and vector PDFs. KM panels omit number-at-risk tables.
Captions identify panels by letter. Standalone panels retain their plot titles and filenames.
The renderer also writes `_captioned.png` / `_captioned.pdf` review copies and
standalone legends in `CLINICAL_FIGURES_OUT/captions/`. Caption wording is maintained
in `figures/manuscript_captions.json`; readable base legends are in
[`figures/figure_captions.md`](figures/figure_captions.md). Sample sizes and model
statistics are appended from the current run. The original downloaded figures
were reviewed in [`figures/review/manuscript_review.md`](figures/review/manuscript_review.md).

Figure rendering defaults to a two-sided endpoint filter: text-minus-base
C-index differences outside the pooled mean ± 3 standard deviations are excluded.
Set `MANUSCRIPT_FILTER_UNDERPERFORMING_ENDPOINTS=false` to retain every endpoint.
The setting is shared by Figures 1–3 and their supplements and is disclosed in
the generated main-figure captions.

Single sources of truth: paths in `config.py`, the scheme registry in `schemes.py`, and the
time-zero anchor registry in `anchors.py`.

Run with `python -m` from the repo root — no install step, since that puts the repo root on
`sys.path` automatically:

```bash
python -m pipelines.training.run_full_cohort_event --scheme death_met --event death
python -m figures.prep.figure2
python -m figures.prep.within_cancer
python -m figures.prep.within_cancer_joint
python -m figures.prep.within_cancer_km
python -m figures.prep.figure3_combined
```

The published-score comparison (audit, note scores, builder, prep) is a separate standalone pipeline —
run it via [`notebooks/1_data/01b_published_scores.ipynb`](notebooks/1_data/01b_published_scores.ipynb)
or the equivalent commands directly:

```bash
python -m pipelines.preprocessing.audit_published_score_inputs --anchor treatment
python -m pipelines.preprocessing.extract_note_scores --anchor treatment  # before the builder: ECOG
python -m pipelines.preprocessing.build_published_scores --anchor treatment
python -m figures.prep.published_scores
```

`within_cancer` (per-endpoint C-index evaluation, both cohorts), `figure3` and
`within_cancer_joint` (joint Cox fits) and `figure3_combined` (stacked Cox models) run their CPU-bound work on a process pool
(`figures/prep/parallel.py`). The worker count is `FIGURE_PREP_N_JOBS` if set, otherwise
16 capped at the CPU allocation (`SLURM_CPUS_PER_TASK`, else the core count); `within_cancer`
also takes `--n-jobs`. Each worker runs one polars/BLAS thread (`FIGURE_PREP_WORKER_THREADS`
overrides). On Linux the worker count is also reduced, with a notice, to fit the per-user
thread limit (`ulimit -u`, 900 on some cluster nodes) given the threads you already have
running; exceeding it crashes workers with "Resource temporarily unavailable". Results are collected in
submission order, so the output CSVs are the same as a serial run (`FIGURE_PREP_N_JOBS=1`).
`within_cancer` writes the full-cohort outputs as soon as that phase finishes, then the
modality-cohort outputs. A rerun after an interruption reloads a completed phase (tracked in
`FIGURE_DATA_DIR/.within_cancer_checkpoint`, keyed on the thresholds) and evaluates only the rest;
`--restart` discards that checkpoint, and the notebook passes it when the module is forced.

## Note-level survival transformer (OS)

`pipelines/note_transformer/` adapts Intae Moon's note-survival transformer, which models each
patient's sequence of note embeddings instead of one pooled vector, to overall survival
(`death_met:death`). The Cox text cohort and its exact 20% test split are reused, with a validation set
carved from the training patients. Landmarks are 0, 90, 180 and 365 days, and a note counts at a landmark
only if its fractional elapsed time from the raw timestamps is at most the landmark.

```bash
python -m pipelines.note_transformer.prepare_os_inputs --anchor treatment
python -m pipelines.note_transformer.run_os_transformer --anchor treatment --stage all
```

The test stage compares the transformer with the Cox text and base out-of-fold scores on the same patient
pairs within Cox folds, a descriptive comparison because the models train on different partitions.
`notebooks/2_models/07_note_transformer_os.ipynb` drives both modules, and
`slurm/launch_note_transformer.sh` submits them as one job (GPU via `GPU_GRES`). The code is not licensed
for redistribution; see [`pipelines/note_transformer/README.md`](pipelines/note_transformer/README.md).

## Where to start

- Understanding the pipeline DAG: [`notebooks/README.md`](notebooks/README.md) walks the stages in
  run order, from cohort build through to the rendered figures.
- Reproducing or extending the manuscript figures: `figures/prep/figure0.py` … `figure5.py`
  plus `figures/prep/within_cancer.py`, `within_cancer_joint.py` and `within_cancer_km.py`,
  rendered via the R scripts in `R/`. `notebooks/4_figures/` drives both steps.
- Running on the cluster: `slurm/launch_*.sh` build manifests and submit the array jobs. They
  default `PROJECT_ROOT` to the cluster checkout path; override it to run elsewhere.

## Configuration and validation

Set `CTEP_DATA_PATH` to override the project data root and `PROFILE_DATA_PATH`
to override compiled PROFILE inputs. Create the environment with
`conda env create -f environment.yml`; run checks with `python -m pytest`.

Prediction-dataset generation uses patients with all three pre-anchor note
modalities (Clinician, Imaging, and Pathology). The stage reports the resulting
complete-case cohort size.
