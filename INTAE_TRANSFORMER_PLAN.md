# Integrate the note-survival transformer for OS prediction

## Context

`intae-note-survival-transformer-main/` (untracked, repo root) is Intae Moon's standalone PyTorch package: a landmark-conditioned causal Transformer over per-note embeddings, trained with a sampled Cox partial likelihood and selected by validation-only C-index. The repo's existing OS models (`death_met:death`) are Coxnet fits on *patient-pooled* embedding vectors; the transformer instead consumes each patient's note *sequence*, so it needs its own input preparation from the note-level store rather than the prebuilt prediction parquet.

Goal: (1) vendor the package under `pipelines/`, (2) a prepare script that maps the repo's data onto the package's five-file contract for OS, plus a thin orchestrator for search / train / held-out test that also compares against the existing Cox text and base models on identical test patients, (3) `notebooks/2_models/07_note_transformer_os.ipynb` driving it in house style.

## Defaults and invariants

| Decision | Choice | Why |
|---|---|---|
| Landmarks | `0,90,180,365` days | treatment-start baseline + 3/6/12-month updates; matches the trajectory grid |
| Cohort | `text` = the Cox text-model cohort (embedding prediction df ⋈ cancer_type_df, `tt_death>0`) | results comparable to `full_cohort/death/text_test.csv`; `--cohort all` uses every `death_met_surv_df` patient |
| Split | for `text`, reproduce the Cox 20% test set exactly (seed 1234); carve `val` = 20% of train+val, stratified, seed 1234 | transformer test patients == Cox test patients; `all` uses its own split and makes no identical-test-cohort claim |
| Static covariates | on: `GENDER`, `age_col(anchor)`, `CANCER_TYPE_*` → `patients.csv` | same base covariates as Cox |
| Note availability | derive fractional elapsed days from raw note and anchor timestamps; include only `days_from_index <= landmark` | integer `NOTE_TIME_REL_*` truncates post-landmark hours into the landmark day; a note at anchor +12h must be excluded at baseline |
| Cox comparison | reuse OOF scores, comparing only patient pairs scored within the same Cox folds | independent Cox fits have different score origins/scales; raw cross-fold rankings cannot define the comparison |
| Device | `auto` (cuda if available else cpu); threads from `--num-threads` / `SLURM_CPUS_PER_TASK` / cpu_count-1 | no GPU precedent in `slurm/`; GPU partition name unknown → env overrides |
| Vendored extras | keep `docs/*.md` (minus QUICKSTART/SHARING/BENCHMARKING); drop wrappers, pyproject, MANIFEST, package .gitignore, benchmarks | they document a standalone install that no longer exists |

## Verified facts the design depends on

- Cox test split (`survival/cox_models/grid_search.py:283-289`): `train_test_split(np.arange(n), test_size=0.2, stratify=event, random_state=1234)` over rows of `filter_event_rows(build_full_prediction_df("death_met", anchor), "death")` (`pipelines/training/run_full_cohort_event.py:58-105`; `validate_cox_inputs` drops only constant *columns*), followed by the caller's all-finite feature filter. Replay the same calls, but do not assume the polars inner join preserves parquet row order. **Verify membership**: `full_cohort/death/text_ipcw_reference.csv.gz` row `eval_data=="test"` holds `reference_ids` = the train+val MRNs (`grid_search.py:128-136, 311`).
- `text_risk_scores.csv` is nested-CV OOF; `base_risk_scores.csv` is ordinary OOF with no hyperparameter tuning (`run_full_cohort_risk_scores.py:122-156`). Both carry `outer_fold` and independently fitted scores. Preserve these fold IDs and label results `cox_*_oof`. These are descriptive OOF comparisons on the common test patients, with different training partitions from the transformer; do not pool raw scores across folds. Base fold failures can produce NaNs (`heldout.py:300-302`).
- Sequencing anchor: `death_met_surv_df__sequencing.parquet` already aliases `tt_death` from `tt_death_sequencing` (`generate_embedding_prediction_datasets.py:195`) → always use `tt_death`.
- Note store: `NOTES_PATH/full_clinical_notes_embeddings_metadata.parquet` retains `DFCI_MRN, NOTE_TYPE, EMBEDDING_INDEX, NOTE_DATETIME, FIRST_TREATMENT_START_DT, SEQUENCING_DT`, plus integer `NOTE_TIME_REL_*` columns; the latter are derived with `.dt.total_days()` (`knit_embeddings.py:141-159`) and are unsuitable for inclusive landmark cutoffs. Use the raw timestamps for fractional elapsed days. Precision: `NOTE_DATETIME` comes from the tz-aware `EVENT_DATE` timestamp converted to naive UTC (`knit_embeddings.py:67-76`), so it can carry time-of-day; `FIRST_TREATMENT_START_DT`/`SEQUENCING_DT` are date columns cast to midnight datetimes, so a same-day note with any time-of-day lands at a positive fractional offset and is excluded at landmark 0. Vectors are in `..._as_array.npy.zst` (float16 `[N,768]`, zstd). Embedding model id `Simonlee711/Clinical_ModernBERT`.
- Package: relative imports only (nestable); `load_data` rejects extra `outcomes.csv` columns; output dirs must be new unless `--resume`; fingerprints sha256 every input file; `seed_everything` sets `torch.use_deterministic_algorithms(True)` and `torch.set_num_threads(num_threads)`.
- `environment.yml` already has pytorch-gpu, lifelines, zstandard, pandas, pyarrow, pytest → nothing to add.

## File operations

**Move** (untracked source → plain `mv`)
- `intae-note-survival-transformer-main/note_survival_transformer/` → `pipelines/note_transformer/note_survival_transformer/` (code unchanged, keep its README.md)
- `.../docs/{ARCHITECTURE,DATA_FORMAT,HYPERPARAMETER_SEARCH,METHODOLOGY_ALIGNMENT,TRAINING_OBJECTIVE,VALIDATION}.md`, `validation_summary.json` → `pipelines/note_transformer/docs/`
- `.../tests/test_note_survival.py` → `tests/test_note_survival_transformer.py` (rewrite ~10 import lines and the 4 `patch("note_survival_transformer....")` strings at L364/372/383/387 to `pipelines.note_transformer.note_survival_transformer....`; add the `REPO_ROOT` sys.path block other tests use)

**Create**
- `pipelines/note_transformer/README.md` (provenance: adapted from Intae Moon's private package, no license, do not redistribute; model summary; input contract table; how prepare maps repo data; stage CLI; output layout; comparison semantics; device/threads/determinism notes)
- `pipelines/note_transformer/configs/os_config.json` = `example_config.json` with `landmarks [0,90,180,365]`, `num_epochs 30`, `patience 6`, `scheduler "cosine"`; `configs/os_search_space.json` = `example_search_space.json` verbatim
- `pipelines/note_transformer/prepare_os_inputs.py`, `pipelines/note_transformer/run_os_transformer.py`
- `notebooks/2_models/07_note_transformer_os.ipynb`
- `slurm/note_transformer_job.sh`, `slurm/launch_note_transformer.sh`
- `tests/test_note_transformer_prepare.py`
- `tests/test_note_transformer_runner.py`

**Modify**: `README.md` (Layout list adds `note_transformer/`; ~10-line section "Note-level survival transformer (OS)" before "Where to start"); `notebooks/README.md` (add 2_models rows 06, 01b, 07; 1_data rows 01b, 01c).

**Delete** after moves: the remainder of `intae-note-survival-transformer-main/`. No `__init__.py` in `pipelines/note_transformer/` (namespace, like sibling stages).

## `prepare_os_inputs.py`

CLI: `--anchor {treatment,sequencing}` `--cohort {text,all}` `--static/--no-static` `--val-frac 0.2` `--seed 1234` `--inputs-dir` `--max-patients N` (smoke; default dir `inputs_smoke<N>`, skips split assertion) `--chunk-rows 65536` `--allow-split-mismatch` `--overwrite`.
Default inputs dir: `scheme_results_dir("death_met", anchor)/note_transformer/inputs/` (`inputs_all/` for `--cohort all`). Skip with `[skip]` only when the manifest matches the requested preparation and sources and the exact expected output set is complete: `notes.npy`, `notes.csv`, `outcomes.csv`, `splits.csv`, `prepare_manifest.json`, plus `patients.csv` only for static inputs. Compare schema/preparation version, anchor, cohort, static mode, validation fraction, split seeds, smoke limit, split-check policy, timestamp policy, and source signatures; chunk size does not change data semantics. Missing/mismatched manifests, incomplete outputs, or unexpected `patients.csv` require `--overwrite` or a new inputs directory, with the differences reported.

For `--overwrite`, build and validate the complete bundle in a fresh sibling staging directory, then replace the previous bundle with rollback on failure. Never overwrite tables piecemeal into the old directory. A successful `--no-static` preparation must leave no `patients.csv`; retain the previous bundle if preparation fails.

Functions (polars throughout; numpy/sklearn at boundaries; CSVs written by polars, package reads them with pandas):
- `build_cox_text_cohort(anchor)` — exact replay: `build_full_prediction_df` → `filter_event_rows(df,"death")` → `validate_cox_inputs(df,"death","tt_death", base_vars+type_cols+embed_cols)` → the same all-finite row filter. Returns rows **in Cox order** with `DFCI_MRN, death, tt_death, GENDER, age_col, CANCER_TYPE_*`.
- `build_all_cohort(anchor)` — `death_met_surv_df{suffix}.parquet` → `filter_event_rows` → left-join cancer types.
- `split_patients(cohort, val_frac, seed)` — Cox call (`test_size=0.2, stratify=event, random_state=1234`, fixed) then `train_test_split(trainval, test_size=val_frac, stratify=event[trainval], random_state=seed)`.
- `cox_trainval_ids(anchor)` / `assert_split_matches_cox(...)` — for the full `text` cohort, parse `text_ipcw_reference.csv.gz` test row `reference_ids`; raise with counts on mismatch (unless `--allow-split-mismatch`, recorded in manifest). Mark smoke and `all` cohorts explicitly as not checked against the Cox split.
- `load_note_metadata(anchor, mrns)` — filter `NOTE_TYPE ∈ {Clinician, Imaging, Pathology}` and cohort MRNs; require `NOTE_DATETIME` and the selected raw anchor column (`FIRST_TREATMENT_START_DT` or `SEQUENCING_DT`). Compute `days_from_index` as elapsed microseconds divided by `86_400_000_000.0`, retaining float64 precision. Drop rows with missing timestamps and report their counts; missing timestamp columns are an actionable error requiring regenerated metadata, never a fallback to integer `NOTE_TIME_REL_*`. Sort by `EMBEDDING_INDEX`. Inclusivity applies to the represented timestamps: equality is allowed, any positive offset after a landmark is excluded; date-only sources provide no within-day availability claim.
- `load_zst_npy_readonly(path)` — decompress once, parse the npy header, `np.frombuffer` → float16 view (one copy; the trajectories script's `.astype(float32)` pattern would be 3×).
- `write_cohort_embeddings(src, source_rows, out_path, chunk_rows)` — `np.lib.format.open_memmap(tmp, "w+", float32, (n, D))`, chunked `astype(float32)` + finiteness check, `os.replace(tmp, out)`.
- `write_tables(...)` — `notes.csv` (`patient_id=str(DFCI_MRN)`, `note_id=str(EMBEDDING_INDEX)`, float64 `days_from_index` without integer rounding, `embedding_row=int_range`, `embedding_model_id`), `outcomes.csv` (exactly `patient_id,time,event`), `splits.csv`, `patients.csv` (omitted with `--no-static` in the fresh bundle).
- `preparation_signature(...)` / `validate_existing_inputs(...)` — implement the manifest/configuration and expected-file checks above before deciding to skip. Track source paths, sizes, and mtimes, including presence/signature of the optional Cox IPCW reference, so a newly available reference triggers revalidation.
- `write_manifest(...)` — `prepare_manifest.json`: preparation signature and expected files; counts (patients, events, notes, notes exactly at the anchor, notes in `0 < days_from_index < 1`, post-anchor notes, dropped missing timestamps, per-split patients/events); static mode/columns; timestamp derivation and inclusive cutoff policy; `split_check ∈ {matched_cox_ipcw_reference, reference_absent, skipped_smoke, not_applicable_all_cohort, mismatch_allowed}`; cohort rule; source signatures. Do not label every legacy integer day-0 note as available at baseline.
- `main(argv=None)`.

## `run_os_transformer.py`

CLI: `--anchor` `--stage {search,train,test,all}` `--config` `--search-space` `--num-trials 20` `--search-seed 42` `--landmarks 0,90,180,365` `--n-epochs` `--device auto|cpu|cuda` `--num-threads` `--inputs-dir` `--tag default` `--checkpoint` `--resume` `--overwrite` `--allow-nondeterministic`.
Layout: `scheme_results_dir("death_met", anchor)/note_transformer/<tag>/{search,single_run,test}/`.

Imports `Config`, `load_data`, `fit`, `run_search`, `final_test_evaluation`, and `safe_c_index` from their vendored submodules (Python API, not subprocess). Functions: `resolve_device`, `resolve_threads`, `run_dirs`, `load_inputs` (validates the preparation manifest and static-file contract, passes `patients.csv` only when the manifest enables static covariates, `embedding_model_id` constant), `build_config` (`Config.from_json` → `dataclasses.replace(landmarks, device, num_threads, num_epochs)` → `.validate()`), `prepare_output_dir` (rmtree of the requested stage's output only on `--overwrite`; clear error if exists without `--resume`), `stage_search`, `stage_train`, `pick_checkpoint`, `stage_test`, `build_comparison`, `main`.

`pick_checkpoint`: for standalone `--stage test`, use an explicit `--checkpoint` if supplied; otherwise select the sole existing candidate among `search/best_model.pt` and `single_run/best_model.pt`. Fail with both paths and instructions to pass `--checkpoint` when both exist; fail clearly when neither exists. `all` = search → test and passes the checkpoint returned by that search directly, without directory precedence. Restrict `--checkpoint` to standalone `test` so it cannot redirect an `all` run to an unrelated model. Print and retain the selected checkpoint path and model ID in test outputs. `--allow-nondeterministic` temporarily wraps `torch.use_deterministic_algorithms` to force `warn_only=True` throughout search, fitting, and test restoration (which reseeds); restore the original function/state afterward, with no vendored-code edit.

`build_comparison(test_predictions_csv, anchor, out_csv)` uses the following contract:

- Load available `full_cohort_risk_dir(...)/{text,base}_risk_scores.csv`, rename MRNs to string `patient_id`, and retain separately named `outer_fold` columns. Validate required columns and unique patient rows; diagnose an unreadable file, missing fold column, or duplicate IDs as `invalid_cox_file` and treat that optional comparator as unavailable. Exclude individual rows with invalid fold assignments in the common-subset check below.
- At each landmark, use one common patient subset with finite scores and valid fold IDs for every available Cox comparator. Transformer outcomes and eligibility come from `test_predictions.csv`. Record the all-test count, missing-score, nonfinite-score, and invalid-fold exclusions, retained patients/events, and which comparators were available; use sequential exclusion counts so totals reconcile. Missing Cox files produce null rows with `c_index_reason=missing_cox_file` and a `[note]`; they do not remove patients from the available comparator's analysis. If no Cox comparator is usable, retain the all-test transformer metric and null comparison rows with explicit unavailability reasons.
- Stratify the common subset by the tuple of available Cox fold IDs. When text/base assignments agree this is their shared outer fold; when they differ, use their joint strata and report that choice. Only compare pairs within a stratum, using the **same strata and patient pairs for the transformer and every available Cox model**. Never compute a pooled C-index from raw OOF scores across folds.
- Within each stratum, use the vendored `safe_c_index(time_from_landmark, score, event)`. Count admissible pairs as `sum_t(events_at_t * (patients_after_t + censored_at_t))`, using grouped/sorted times; this includes same-time event/censor pairs and excludes same-time event/event pairs. Aggregate as `sum(c_index_s * n_pairs_s) / sum(n_pairs_s)` over strata with comparable pairs, retaining lifelines' half credit for tied risk scores. Weight by admissible pairs, not patient counts or an unweighted mean of fold C-indices. Record `n_strata`, `n_evaluable_strata`, and `n_comparable_pairs`. The existing joint-fold comparison in `figures/prep/within_cancer.py` and its tests provide the repository precedent; retain the package's lifelines scorer here without importing figure-preparation code.
- Emit rows `{landmark, evaluation_scope=within_cox_folds, model ∈ {transformer, cox_text_oof, cox_base_oof}, n_patients, n_events, n_comparable_pairs, c_index, c_index_reason, ...audit counts}`. Empty intersections, fewer than two patients, no events, or zero comparable pairs produce null C-indices with reasons, never a comparison failure. Invalid Cox scores are excluded jointly before scoring, so all compared models retain identical denominators.
- Preserve the package's per-landmark all-test transformer metrics separately with `evaluation_scope=all_test`, plus its `weighted_test_c_index` summary. Report existing `{text,base}_test.csv` landmark-0 metrics under a separate `evaluation_scope=reported_cox_holdout`; do not present these as the within-fold comparison or select a hyperparameter row using its test score. Print the table and write `test/comparison.csv`, including reasons/exclusions even when a Cox comparison is unavailable.

The OOF comparison remains descriptive because Cox and transformer training partitions differ. A future comparison from one Cox fit on the original train+val set would support ordinary concordance across the matched held-out test set; this plan reuses the existing OOF artifacts and therefore restricts comparisons to within-fold pairs.

## Notebook `07_note_transformer_os.ipynb` (nbformat 4.5, python3, no outputs)

0. md header: runs after `1_data/03` (and `2_models/04` for the Cox comparison); tier: GPU-capable Jupyter session preferred, CPU works but hours per search; outputs per stage; fractional timestamp availability, within-fold OOF comparison, and different training-partition caveats.
1. shared setup cell copied verbatim from `03_mortality_trajectories.ipynb`.
2–3. `## Configuration`: `ANCHOR, COHORT, STATIC, VAL_FRAC, SPLIT_SEED, TAG, STAGES=["prepare","search","test"], NUM_TRIALS, LANDMARKS, DEVICE, NUM_THREADS, OVERWRITE, RESUME, MAX_PATIENTS, CHECKPOINT=None`; dirs from `schemes.scheme_results_dir`, `full_cohort_event_dir`, `full_cohort_risk_dir`; pass the same resolved inputs directory to preparation and every runner stage.
4–5. `## Preconditions`: `check_inputs` on metadata parquet, `.npy.zst`, surv parquet, `embedding_file`, cancer_type csv; check raw note/anchor timestamp columns; torch version + `cuda.is_available()` printed, not required; `report_outputs` on Cox files labelled optional.
6–7. `## 1. Prepare inputs`: `run_module("pipelines.note_transformer.prepare_os_inputs", args)`; print manifest counts.
8–9. `## 2. Search / train`: `run_module(MODULE, [*common, "--stage", "search", "--num-trials", ...])` or `"train"`.
10–11. `## 3. Held-out test`: `--stage test` with the explicit checkpoint from the search/train stage successfully run in this invocation; for test-only execution use configured `CHECKPOINT` or the runner's sole-candidate rule. Never prefer an old search checkpoint over a newly trained model merely because both exist.
12–13. `## Summary` (polars in `pl.Config`): selected checkpoint/model ID; per-landmark eligible/events/C-index from `test_metrics.json`; top-10 search leaderboard from `hparam_search_results.csv`; `comparison.csv` separated by evaluation scope, showing comparable-pair counts, exclusions, and null reasons; `report_outputs`. All reads guarded by `os.path.exists`.

## Slurm

`slurm/note_transformer_job.sh`: single job (`--cpus-per-task=8 --mem=64G --time=24:00:00`, output under `slurm/note_transformer/`), same `PROJECT_ROOT`/`PYTHONPATH`/`CONDA_ENV_PREFIX`/`PYTHON_BIN` block as `slurm/array_full_cohort_run.sh:17-40`; env `ANCHOR COHORT TAG NUM_TRIALS STAGE=all DEVICE=auto`, optional `CHECKPOINT` for test-only runs; runs prepare then `run_os_transformer --num-threads "$SLURM_CPUS_PER_TASK"`. Resolve and pass the same inputs directory to both commands, including `inputs_all/` for `COHORT=all`.
`slurm/launch_note_transformer.sh`: `PARTITION=${PARTITION:-normal}`, `GPU_GRES=${GPU_GRES:-}` (adds `--gres` when set), `CPUS_PER_TASK MEMORY TIME_LIMIT`; one `sbatch --export=ALL,...`. Header documents `GPU_GRES=gpu:1 PARTITION=<gpu-partition> bash slurm/launch_note_transformer.sh`.

## Tests

- `tests/test_note_survival_transformer.py`: moved suite (25 unittest tests), imports rewritten only.
- `tests/test_note_transformer_prepare.py` (pytest, rng 1234): fixture writes a tiny note store (float16 npy → zstd) with raw note/anchor timestamps, `death_met_surv_df.parquet`, `death_met_embedding_prediction_df.parquet` (~80 patients, two `EMBEDDING_*` cols), `cancer_type_df.csv.gz` (one MRN missing to exercise the join) under `tmp_path/data`; runs prepare as a subprocess with `CTEP_DATA_PATH` set (avoids stale module constants). Assert package `load_data` accepts outputs; `notes.npy` rows equal source rows cast to float32; test split equals in-process replay of the Cox functions + `train_test_split(..., 1234)` (with `schemes.SURV_PATH` / `slurm_array_utils.FEATURE_PATH` monkeypatched); train/val/test partition outcomes exactly; a fabricated mismatching IPCW reference fails for the full text cohort.
- Preparation regressions: for both anchors, notes just before, exactly at, and just after each default landmark retain fractional timestamps through CSV loading and yield the expected package histories; explicitly cover anchor +12h and landmark +12h and reject missing raw timestamp columns. An identical rerun skips; changed static mode, validation fraction, seed, timestamp policy/version, or source signature cannot skip. Overwriting static inputs with `--no-static` removes `patients.csv` and yields zero static dimensions; failed preparation retains the prior complete bundle. Cover smoke/`all` split-check statuses.
- `tests/test_note_transformer_runner.py`: checkpoint selection covers neither/one/both candidate paths, explicit override, stale search plus new single-run checkpoint, and `all` passing its own search result even when another checkpoint exists. Assert the test evaluator receives the intended checkpoint; test selection without an explicit path fails when both candidates exist.
- Comparison regressions: hand-made predictions and Cox scores with fold IDs verify admissible-pair weighting and identical pair sets across models; adding a constant to one Cox fold leaves its comparison C-index unchanged. Cover matching and differing text/base fold assignments, tied times/risk scores, one uninformative fold, empty intersections, no events, no comparable pairs, nonfinite Cox scores, missing optional files, and duplicate IDs. Assert null reasons and exclusion counts, and keep all-test transformer metrics unchanged. Validate the pair counter against a small brute-force oracle using lifelines tie rules and ordinary `safe_c_index` when only one stratum is present.

## Verification

1. `python -m pytest tests/test_note_survival_transformer.py -q` — relocated package works under the new dotted path.
2. `python -m pipelines.note_transformer.note_survival_transformer.synthetic_example --output_dir <scratchpad>/nst_demo --run_demo` — CPU end-to-end search/resume/fit/test/inference.
3. `python -m pytest tests/test_note_transformer_prepare.py tests/test_note_transformer_runner.py -q`, then `python -m pytest -q` (whole suite green); `pyflakes` on the two new scripts.
4. `bash -n slurm/launch_note_transformer.sh slurm/note_transformer_job.sh`; `nbformat.validate` on the notebook.
5. Cluster smoke: `prepare_os_inputs --anchor treatment --max-patients 300` then `run_os_transformer --inputs-dir .../inputs_smoke300 --tag smoke --stage all --num-trials 2 --n-epochs 2 --device auto`; check checkpoint provenance and `test/comparison.csv` scopes, pair counts, exclusions, and null reasons. Sparse late landmarks must not turn a successful test evaluation into a failed job.
6. Full `text` prepare on the cluster: `prepare_manifest.json` must say `split_check: matched_cox_ipcw_reference`; train+val IDs must equal IPCW `reference_ids`, test IDs must equal the cohort complement, and per-split patient/event counts must match the generated splits joined to outcomes. `text_test.csv` does not contain patient/event denominators. Confirm raw timestamp derivation and audit the positive offsets within the first 24h that are excluded from baseline histories.

## Risks

- `notes.npy` for the cohort is float32 (3 KB/note; e.g. 2M notes → 6 GB); written in one sequential memmap pass. Peak RAM ≈ one float16 copy of the full store.
- Every `load_data` sha256s and finiteness-scans `notes.npy` (tens of seconds at GB scale).
- CPU training: minutes-to-hours per epoch for thousands of events; `NUM_TRIALS=20` is a GPU budget — use `--num-trials 5 --n-epochs 10` on CPU; `--num-threads` matters (package default 1).
- CUDA + `use_deterministic_algorithms(True)` can raise on some kernels → `--allow-nondeterministic`.
- Split reproduction relies on stable polars inner-join row order; the IPCW assertion catches drift (fail loudly rather than silently mis-split).
- `patients.csv` columns are frozen into the checkpoint: a `--no-static` input set cannot be tested with a static-trained checkpoint.
- Fractional elapsed time prevents truncation leakage but cannot recover within-day availability from date-only source records; document source precision and do not claim every same-day note was known at treatment start.
- Within-fold OOF comparisons use fewer admissible pairs than ordinary all-test concordance, especially when text/base folds differ; report the pair counts and keep evaluation scopes distinct. They do not establish performance under identical training partitions.
- Changed preparation options require a new input bundle or explicit overwrite; training/test fingerprints should reject checkpoints made with the prior bundle. Ambiguous checkpoint directories require explicit selection.
