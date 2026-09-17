# Clinical Text Embedding Project

This project investigates whether dense representations of clinical narratives (EHR notes) can
improve survival prediction and identify genomic biomarkers of treatment response in oncology.

## Layout

The tree is organized by pipeline stage rather than by file type:

- `pipelines/` — `preprocessing/`, `training/`, `trajectories/`, `biomarkers/`
- `survival/` — Cox model fitting, checkpointing, evaluation
- `figures/prep/` — `figure0.py` … `figure5.py`, which write the CSVs the R tier plots from
- `R/` — figure rendering and shared plot utilities
- `notebooks/` — thin driver notebooks, grouped by stage
- `shared/` — palette and stage logic used across pipelines and figures
- `data/` — versioned lookup tables and their loaders

Render manuscript panels and compiled Figures 1–4 with:

```bash
R/render_all_figures.sh
```

All performance figures use C-index; AUC variants and the former Figure 5a
propensity ROC panel are retired. Rendering removes their old PNG/PDF files from
the configured output directory. Figures 1–4 are also assembled without printed
panel letters, saved as `png/figureN/figureN_cindex.png` and
`pdf/figureN/figureN_cindex.pdf` beneath `CLINICAL_FIGURES_OUT`. Individual panels
are retained. Missing panels are skipped with their reasons logged; available
panels retain their original file identifiers, and the compiled figure lists unavailable
panels in its caption. If every panel is missing, that figure is skipped and the
remaining figures still render. Figure 1c follows the PROFILE-testing cohort
flowchart style, with cumulative patient counts in connected rectangular boxes.

The main figures use the following panel order. Individual files are
`figNa_cindex`, `figNb_cindex`, etc.; these identifiers follow reading order in
the compiled figures (Figure 3c spans the bottom row).

| Figure | a | b | c | d |
| --- | --- | --- | --- | --- |
| 1 | Cancer-type pie | Stage barplot | Cohort availability flowchart | Outcome endpoints |
| 2 | Base vs. text C-index | Delta-C-index violin | Stage KM | Text-risk quartile KM |
| 3 | Modality rank | Significant endpoints | Joint Cox violins | — |
| 4 | Risk-score heatmap | KM by risk trajectory | Risk dynamics by stage | Stage I–II rising vs. stage IV falling risk KM |

Within-versus-pan comparisons and top-event plots are retired, including the
top-event supplement. Their inputs are no longer required by Figure 2 preparation.

Compiled Figures 1, 2, and 4 render at 10 × 9 inches; Figure 3 renders at
10 × 8.2 inches, with 600-dpi PNGs. KM panels omit number-at-risk tables.
Captions identify panels by position, and standalone panel filenames are unchanged.
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
```

## Where to start

- Understanding the pipeline DAG: [`notebooks/README.md`](notebooks/README.md) walks the stages in
  run order, from cohort build through to the rendered figures.
- Reproducing or extending the manuscript figures: `figures/prep/figure0.py` … `figure5.py`,
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
