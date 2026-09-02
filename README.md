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
