# Clinical Text Embedding Project

This project investigates whether dense representations of clinical narratives (EHR notes) can
improve survival prediction and identify genomic biomarkers of treatment response in oncology.
See [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) for the rationale behind the `v1/`/`v2/` split below
and the phase-by-phase migration record.

## `v1/` vs `v2/`

- **`v2/`** is the maintained tree. Start here. Organized by pipeline
  (`pipelines/preprocessing`, `pipelines/training`, `pipelines/trajectories`,
  `pipelines/biomarkers`) rather than by file type, with single sources of truth for paths
  (`config.py`), scheme registry (`schemes.py`), and shared palette/stage logic (`shared/`).
  Run with `python -m` from inside `v2/`, e.g.:

  ```bash
  cd v2
  python -m pipelines.training.run_full_cohort_event --scheme death_met --event death
  python -m figures.prep.figure2
  ```

  No install step — `python -m` from `v2/` puts `v2/` on `sys.path` automatically.

- **`v1/`** is the frozen, byte-for-byte original tree (organized by file type:
  `python_scripts/`, `python_utils/`, `bash_scripts/`, `jupyter_notebooks/`). It is the reference
  that produced current manuscript results and remains fully runnable, but is not edited going
  forward — treat it as archive-only once `v2/` passes verification. A few components that don't
  feed the manuscript figures (the mortality-model-comparison benchmark, some diagnostic
  notebooks) only exist in `v1/`; see `REFACTOR_PLAN.md` for the full list.

## Where to start

- Reproducing or extending the manuscript figures: `v2/figures/prep/figure0.py` … `figure5.py`,
  rendered via the R scripts in `v2/R/`. `v2/notebooks/` has thin driver notebooks for both steps.
- Understanding the pipeline DAG: `REFACTOR_PLAN.md`'s Context section traces raw data through to
  the figures.
- Digging into what changed during the refactor: `REFACTOR_PLAN.md`'s phase-by-phase plan, or
  diff a `v2/` file against its `v1/` counterpart directly.
