# Note-level survival transformer (overall survival)

A landmark-conditioned causal Transformer over each patient's sequence of note embeddings, trained with a
sampled Cox partial likelihood and selected by validation-only C-index. This directory wraps it for the
repo's overall-survival endpoint (`death_met:death`).

## Provenance

`note_survival_transformer/` and `docs/` are adapted from Intae Moon's private
`note-survival-transformer` package. The package carries no license, so do not redistribute it outside
the project. The vendored prediction API has a local adaptation to accept a runtime thread count
during checkpoint restoration and final test evaluation. Its own README and the files in `docs/` describe the model, training
objective, data format and validation in detail. The package's tests live in
`tests/test_note_survival_transformer.py`.

## What the model does

At each landmark `t0` the model reads the patient's notes written on or before `t0` (the most recent `max_notes`,
in time order), each as an embedding plus its elapsed time, attends causally over them, and emits one
log-risk. Training draws a risk set per event and optimises the Cox partial likelihood over sampled
controls. Validation selects hyperparameters and checkpoints by the C-index averaged over landmarks,
weighted by eligible patients. Predictions rank patients by risk and are not calibrated probabilities.

## Input contract

The package reads five files. `prepare_os_inputs.py` writes them from the repo's data.

| File | Columns | Source |
|---|---|---|
| `notes.npy` | float32 `[N, 768]` | rows of `full_clinical_notes_embeddings_as_array.npy.zst` for cohort patients, cast from float16 |
| `notes.csv` | `patient_id, note_id, days_from_index, embedding_row, embedding_model_id` | note metadata; `note_id` is the source `EMBEDDING_INDEX` |
| `outcomes.csv` | `patient_id, time, event` | `tt_death`, `death` |
| `splits.csv` | `patient_id, split` | train, val, test |
| `patients.csv` | `patient_id` plus numeric columns | optional: `GENDER`, age at anchor, `CANCER_TYPE_*` |

`prepare_manifest.json` records the preparation settings, source file signatures, counts and the split
check. Run `prepare_os_inputs` from the repo root with `python -m pipelines.note_transformer.prepare_os_inputs`.

## How preparation maps the repo's data

- **Cohort.** `--cohort text` (default) replays the Cox text-model cohort in Cox row order. `--cohort all`
  uses every patient in `death_met_surv_df` with a valid `tt_death` and does not claim an identical test
  set.
- **Split.** For `text`, the 20% test set reproduces the Cox split (`random_state=1234`, stratified by
  event). Validation is 20% of the rest, stratified, seeded by `--seed`. The Cox train+val IDs are
  checked against `text_ipcw_reference.csv.gz`. The manifest `split_check` is one of
  `matched_cox_ipcw_reference`, `reference_absent`, `skipped_smoke`, `not_applicable_all_cohort` or
  `mismatch_allowed`. A mismatch raises unless `--allow-split-mismatch` is passed.
  The runner requires `--anchor` to match the prepared bundle's anchor before loading its data.
- **Notes.** Clinician, Imaging and Pathology notes of cohort patients, Clinical ModernBERT embeddings.
  Patients without notes stay in the cohort with empty histories.
- **Timing.** `days_from_index` is elapsed microseconds between the raw `NOTE_DATETIME` and the raw
  anchor column (`FIRST_TREATMENT_START_DT` or `SEQUENCING_DT`) divided by 86,400,000,000. The integer
  `NOTE_TIME_REL_*` columns are never used because they truncate hours after a landmark into the
  landmark day. A note is available at a landmark when `days_from_index <= landmark`. The anchor columns
  are dates, so a same-day note with any time of day sits at a positive offset and is excluded at
  landmark 0. Missing raw timestamp columns are an error, not a fallback.
- **Reruns.** An identical complete bundle is skipped. Any change to the settings, sources or file set
  stops with the differences listed unless `--overwrite` is given. An overwrite is built in a staging
  directory and swapped in with rollback, so a failed run leaves the previous bundle intact.
  `--chunk-rows` must be positive; invalid values are rejected before writing output.
- **Smoke.** `--max-patients N` writes a subsample to `inputs_smoke<N>/` and skips the Cox split check.

## Running

```bash
python -m pipelines.note_transformer.prepare_os_inputs --anchor treatment
python -m pipelines.note_transformer.run_os_transformer --anchor treatment --stage all
```

`run_os_transformer` stages:

| Stage | What it does |
|---|---|
| `search` | random search over `configs/os_search_space.json` (`--num-trials`, `--search-seed`), validation only |
| `train` | one fit of `configs/os_config.json` |
| `test` | evaluates one checkpoint on the test patients and writes the comparison |
| `all` | `search` then `test`, passing the search checkpoint directly |

Standalone `test` uses `--checkpoint` if given, otherwise the only existing checkpoint among
`search/best_model.pt` and `single_run/best_model.pt`, and fails if both exist. `--checkpoint` is rejected
for other stages. Other flags: `--config`, `--landmarks`, `--n-epochs`, `--tag`, `--inputs-dir`,
`--resume`, `--overwrite` (mutually exclusive) and `--allow-nondeterministic`.

For the cluster, `slurm/launch_note_transformer.sh` submits `slurm/note_transformer_job.sh`. The notebook
`notebooks/2_models/07_note_transformer_os.ipynb` drives the same modules.

## Outputs

Under `scheme_results_dir("death_met", anchor)/note_transformer/`:

```
inputs/                 prepared bundle (inputs_all/, inputs_smoke<N>/ for other bundles)
<tag>/search/           trials, hparam_search_results.csv, best_config.json, search_summary.json, best_model.pt
<tag>/single_run/       training_log.jsonl, result.json, best_model.pt
<tag>/test/             test_metrics.json, test_predictions.csv, comparison.csv
```

## Comparison with the Cox models

`test/comparison.csv` has three evaluation scopes.

- **`within_cox_folds`.** The transformer and the existing Cox text and base models, scored on the same
  patients and the same pairs. The Cox scores are the out-of-fold scores in
  `full_cohort_risk_scores/death/`. Each model was fit on different partitions, so scores are never
  pooled across folds. Patients are grouped by their Cox fold IDs and only pairs inside a group are
  compared. Each group's C-index is weighted by its number of admissible pairs, counted as events times
  later or same-time censored patients. Tied risks get half credit. Patients with missing, nonfinite or
  invalid-fold Cox scores are dropped for every model, and the exclusion counts are reported.
  This is a descriptive comparison, not one between models trained on identical partitions.
- **`all_test`.** The transformer's own per-landmark metrics over every eligible test patient.
- **`reported_cox_holdout`.** The Cox models' landmark-0 metrics from `text_test.csv` and `base_test.csv`.
  Not comparable pair for pair with the other scopes.

Missing or unreadable Cox files give null rows with `c_index_reason` set. They never fail the run.
Landmarks with no eligible patients, no events or no comparable pairs likewise retain rows with null
C-indices and a reason. The notebook displays an unavailable weighted C-index without stopping its summary.

## Device, threads and determinism

`--device auto` uses CUDA when available, otherwise CPU. Threads come from `--num-threads`, then
`SLURM_CPUS_PER_TASK`, then the CPU count minus one. This runtime budget also applies to standalone
test evaluation, while the checkpoint retains its original training configuration. The package enables
`torch.use_deterministic_algorithms(True)`. If a CUDA kernel rejects that, `--allow-nondeterministic`
downgrades it to a warning for the run. On CPU a 20-trial search takes hours, so use about 5 trials and
fewer epochs there.

## Limits

- `notes.npy` is float32, about 3 KB per note. Preparation holds one float16 copy of the note store in
  memory.
- Every data load hashes `notes.npy`, which takes tens of seconds at multi-gigabyte scale.
- A checkpoint records its static covariate columns, so a `--no-static` bundle cannot evaluate a model
  trained with static covariates, and a checkpoint trained on one bundle is rejected on another.
- Date-only anchors cannot show whether a same-day note was written before or after treatment started.
