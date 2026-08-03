# Restructure into `v1/` (frozen) + `v2/` (clean rewrite)

> **Status:** Phases 1-5 complete locally (`v1/` frozen, `v2/` built, split, and migrated, tagged
> `pre-refactor` before the freeze). Local verification (Verification section, "Local" subsection)
> passes: `compileall`, `pyflakes`, the `sys.path`/`/data/gusev` grep gates, `Rscript` parse checks,
> and `python -m <module> --help` smoke tests for every training/biomarker entry point. Phase 0's
> cluster-side baseline copy and the "On the cluster" verification (the real A/B — diffing
> `figure_data` and re-rendering figures) have **not** been run — this machine cannot execute the
> pipeline (no access to `/data/gusev`). That remains the outstanding gate before `v1/` is
> considered archive-only.

## Context

The repo is organized **by file type** (`python_scripts/`, `python_utils/`, `bash_scripts/`,
`jupyter_notebooks/`) rather than by pipeline, even though the real unit of work is the four-stage
DAG that terminates in the manuscript figures. Only `embed_surv_utils` is installable, so every
other shared module is reached via `sys.path` surgery. That one decision drives most of the mess:

| Problem | Evidence |
|---|---|
| `DATA_PATH` hardcoded, identical, in 12 files | `slurm_array_utils.py:9`, `biomarker_common.py:14`, `_figure_utils.py:34`, `extract_ICD_times.py:7`, both preprocessing `.R` scripts, … |
| `sys.path` hacks to reach sibling modules | 8 sites: `prep_figure_{0,1,2,4,5}.py`, `_figure_utils.py:61`, 3 notebooks |
| Scheme→(file, results dir) map defined 3× | `SCHEME_CONFIG` (`slurm_array_utils.py:13`), `SCHEME_RESULT_DIRS` + `EMBEDDING_FILES` (`_figure_utils.py:45-57`) |
| Figure palettes defined twice, in two languages | `_figure_utils.py:68-83` vs `R/figure_utils.R:73-81` — silent drift risk **in the deliverable** |
| Stage loader copy-pasted 4× | `STAGE_PATH` in `prep_figure_{0,1,2,4}.py`; `_normalize_stage` 3×. `prep_figure_4.py:90` *documents* the duplication in a comment instead of extracting it |
| Data embedded in code | `VALIDATION_REF` — ~580 lines of hand-curated literature inside `validate_and_report.py:69` |
| Very long functions | `run_grid_CoxPH_parallel` 464 lines, `_load_or_init_report_doc` 340, `_run_marker_screen` 261 |
| Dead code | `compile_all_scheme_results.ipynb` imports `publication_plot_utils`, **deleted from git**; README links a non-existent `CODE_REVIEW.md` |

**Approach:** rather than mutating a working, in-flight codebase in place, freeze it as `v1/` and
build `v2/` clean alongside. v1 stays runnable and untouched as the reference that produced current
results; v2 becomes the maintained tree. Both can run side by side, which makes the correctness
check trivial: run each, diff the outputs.

**Setup stays simple — no installable package, no `pip install -e`.** v2 is run with `python -m`
from the `v2/` directory, which puts `v2/` on `sys.path` automatically. Shared code is plain
importable modules. The only environment change is one `PYTHONPATH` export in the SLURM scripts.

---

## Target layout

```text
clinical_text_embedding_project/
├── REFACTOR_PLAN.md          # this document
├── README.md                 # short: what v1/v2 are, where to start
├── v1/                       # ENTIRE current tree, moved verbatim, then frozen
│   ├── python_scripts/  python_utils/  bash_scripts/  jupyter_notebooks/  README.md
└── v2/
    ├── config.py             # THE path module — DATA_PATH and everything derived
    ├── schemes.py            # THE scheme registry
    ├── shared/
    │   ├── palette.json      # one palette, read by BOTH Python and R
    │   ├── palette.py
    │   └── stages.py         # ONE stage loader (replaces the 4 copies)
    ├── survival/             # was embed_surv_utils
    │   ├── preprocessing.py  checkpoint.py
    │   └── cox_models/       # 977-line module split (Phase 4)
    ├── pipelines/
    │   ├── preprocessing/    # pipeline 1
    │   ├── training/         # pipeline 2
    │   ├── trajectories/     # pipeline 3
    │   └── biomarkers/       # pipeline 4
    ├── figures/              # PROMOTED — the deliverable, not a "notebook"
    │   ├── io.py             # save_figure_data / load_figure_data / dir helpers
    │   └── prep/figure0.py … figure5.py
    ├── R/                    # plot_figure_*.R, figure_utils.R
    ├── notebooks/            # thin drivers only
    ├── slurm/                # was bash_scripts/
    └── data/validation_reference.csv
```

Running v2, from `v2/`:

```bash
python -m pipelines.training.run_full_cohort_event --scheme death_met --event death
python -m figures.prep.figure2
```

`R/` and `notebooks/` sit at v2 top level because they are not Python modules.

---

## What goes into v2 — and what stays behind in v1

You said what matters is *what filters through to the manuscript figures*. I traced the inputs of
every `prep_figure_*.py`. Three components are **not** in the figure path and should stay in v1
rather than be carried forward:

| Component | Lines | Why it stays in v1 |
|---|---|---|
| `jupyter_notebooks/mortality_model_comparison/` (`survival_benchmark.py` + 3 notebooks + README) | ~1,376 | Nothing anywhere reads its outputs — verified by grep. Standalone XGBoost/RSF benchmark. |
| `mortality_trajectories/cluster_mortality_trajectories.py` | 322 | Writes only exploratory PNGs to a `model_metrics/` dir. `prep_figure_4.py` does its own clustering (`_silhouette_scan`, `_scaled_trajectory_matrix`). |
| `compile_all_scheme_results.ipynb` | 607 | **Already broken** — imports `publication_plot_utils`, deleted from git. |
| `debug_individual_events.ipynb`, `propensity_score_evaluation.ipynb` | 594 | Diagnostic/evaluation tools, no figure output. |

That is ~2,900 lines v2 never has to carry. They remain runnable in `v1/` if needed.

**Confirmed in the figure path, so these do move to v2:** all of Pipeline 1 (incl. both `.R`
scripts — `prep_figure_2.py` reads their `phecode_descriptions.csv` / `icd10_to_phecode_mapping.csv`
output), all of Pipeline 2, `generate_mortality_trajectories.py` (→ `prep_figure_4`),
`within_vs_pan_cancer_models.py` + `within_treatment_vs_pan_treatment_models.py`
(→ `prep_figure_2._within_vs_pan`), and all of Pipeline 4.

`validate_and_report.py` is a judgment call: it does **not** feed any figure (it writes the standalone
docx report), but it is an active deliverable, so it moves to v2.

---

## Phase 0 — Baseline (on the cluster, before anything moves)

Byte-identical verification needs a baseline, and **this Mac cannot run the pipeline** — the data
lives at `/data/gusev/...`.

```bash
cp -r <SURV_PATH>/results/figure_data              ~/refactor_baseline/figure_data
cp -r <DATA_PATH>/biomarker_analysis/compiled_results ~/refactor_baseline/compiled_results
```

Then `git tag pre-refactor`.

## Phase 1 — Freeze v1

`git mv` the four current top-level dirs plus `README.md` into `v1/`. Rename-only, one commit, no
content edits — the diff must read as pure renames so history is preserved. v1 is not touched
again after this.

## Phase 2 — Build v2 skeleton + single sources of truth

Copy (not move) from v1 into the v2 layout, then collapse the duplication:

1. **`v2/config.py`** — `DATA_PATH` defined once, overridable via `CTEP_DATA_PATH` env var with the
   current cluster path as default. Derive `SURV_PATH`, `FEATURE_PATH`, `NOTES_PATH`,
   `RESULTS_PATH`, `BIOMARKER_PATH`, `FIGURE_DATA_DIR`, `CODE_PATH`. Also hoist the third-party
   inputs currently inline mid-file: `MED_LINES_FILE`, `IO_START_FILE`, `TREATMENT_FILE`, the PRS
   matrix (`generate_all_non_text_covariates.py:96`), `INTAE_DATA_PATH`, `METS_PROJECT`, `STAGE_PATH`.
2. **`v2/schemes.py`** — one `SCHEMES` registry (`embedding_file` + `results_dir` per scheme);
   express the old `SCHEME_RESULT_DIRS` / `EMBEDDING_FILES` as views over it. Keep `_ensure_scheme`,
   `get_output_dir`, and the `full_cohort_*` / `feature_*` dir helpers here.
3. **`v2/shared/palette.json`** — `MODALITY_ORDER`, `MODALITY_COLORS`, `MODALITY_DISPLAY`,
   `MODEL_COLORS`, `CLUSTER_COLORS`. `shared/palette.py` loads it; `R/figure_utils.R` loads the same
   file via `jsonlite::fromJSON()`. **Diff the two current definitions before merging** — if they
   already disagree, that is a live figure bug to report, not a merge to resolve by preference.
4. **`v2/shared/stages.py`** — one `load_stage_map()` + `normalize_stage()`. Reconcile the 4 copies:
   `prep_figure_4.py` carries an extra `_STAGE_IV_TOKEN` (keep as a separate helper);
   `prep_figure_2.py` lacks `STAGE_ORDER`. Confirm the three `_normalize_stage` bodies are
   equivalent before collapsing.
5. **`v2/data/validation_reference.csv`** — extract `VALIDATION_REF` to
   `gene,mutation_type,cancer_type,level,notes` (empty `cancer_type` = pan-cancer). The loader
   rebuilds the same dict keyed by 2- and 3-tuples, so lookup code downstream is unchanged.

## Phase 3 — Wire up v2

- Replace every `sys.path` insert with a real import (`from config import SURV_PATH`) — all 8 sites.
- SLURM: `python "$PROJECT_ROOT/python_scripts/model_training/run_feature_comp_task.py"` →
  `cd "$V2_ROOT" && python -m pipelines.training.run_feature_comp_task`. The
  `import embed_surv_utils` guard (`array_*.sh:33`) becomes `import config`. Export
  `PYTHONPATH="$V2_ROOT"` once at the top so non-`-m` invocations also work.
- Add `if __name__ == "__main__":` guards to the 8 biomarker/preprocessing scripts that currently
  execute on import — required for them to be importable at all.
- Drop the README's dangling `CODE_REVIEW.md` link (or restore the file).

## Phase 4 — Split oversized units (mechanical, no logic change)

- `cox_models.py` (977) → `survival/cox_models/`. The 464-line `run_grid_CoxPH_parallel`
  (`cox_models.py:281-745`) contains two cleanly separable execution paths — in-RAM vs.
  memmapped-PCA — extract each behind the existing signature. Re-export from `__init__.py` so
  `from survival import run_grid_CoxPH_parallel` still works.
- `validate_and_report.py` (1108) drops to ~500 once `VALIDATION_REF` leaves; then split the
  340-line `_load_or_init_report_doc` into `report.py` (docx assembly) vs `validate.py` (logic).
- `run_IPTW_analysis.py`: lift the 261-line `_run_marker_screen` body into per-track helpers
  alongside the existing `_fit_track1_marker` / `_fit_track2_marker`.

## Phase 5 — Conventions cleanup (own commit, last)

Per the documented convention (`.csv.gz` = large inputs; results = plain `.csv`), the biomarker
compiled outputs still violate it: `track1_all_significant_hits.csv.gz`,
`track2_all_significant_hits.csv.gz`, `all_findings_with_validation.csv.gz`
(`prep_figure_5.py:155,218,427-428`, `validate_and_report.py:50`). Migrate to `.csv`. Separate
commit — it changes filenames the figure prep reads, so it must be verified on its own.

---

## Verification

Local (this Mac — structure only; **cannot execute the pipeline**):

```bash
cd v2
python -m compileall -q .
python -c "import config, schemes, shared.palette, shared.stages"
grep -rn "sys.path" . --include=*.py --include=*.ipynb   # must return nothing
grep -rn "/data/gusev" . --include=*.py | grep -v config.py   # must return nothing
Rscript -e 'source("R/figure_utils.R")'                  # parses + palette.json loads
for m in pipelines.training.run_full_cohort_event pipelines.training.run_feature_comp_task \
         pipelines.training.build_slurm_manifests; do python -m $m --help >/dev/null || echo "FAIL $m"; done
```

> `compileall` and `--help` catch syntax and import errors but **not** undefined names in
> unexecuted branches. After any rename, grep for every reference.

On the cluster — the real gate. Because v1 still exists, this is a true A/B:

```bash
cd v2 && for i in 0 1 2 3 4 5; do python -m figures.prep.figure$i; done
diff -r ~/refactor_baseline/figure_data <SURV_PATH>/results/figure_data   # MUST be empty
```

Then re-render and compare panels:

```bash
cd v2 && Rscript -e 'source("R/plot_figure_2.R")'   # etc.
```

Panels should be visually identical; small PNG byte differences from a different R session are fine,
differing **data** is not. If the `figure_data` diff is non-empty, bisect by phase — each phase is
its own commit — rather than debugging forward.

---

## Risks

- **In-flight manuscript.** v1 remains fully runnable throughout, so there is always a working tree
  that reproduces current results. Nothing is deleted, only copied forward.
- **Two trees can drift.** Mitigate by treating v1 as read-only from Phase 1 onward and finishing
  v2 in one push. Once v2 passes the cluster diff, v1 becomes archive-only.
- **The palette and stage-loader merges assume the copies are equivalent.** They are only
  *probably* equivalent — there are known small differences noted above. Diff each before
  collapsing; a real divergence is a finding to report, not a preference call.
- **Cluster-only verification.** Nothing here can be numerically validated from this machine.
