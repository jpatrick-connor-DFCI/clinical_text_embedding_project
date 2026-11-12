# Reproducible hyperparameter search

```bash
python search_note_survival_transformer.py \
  --note_embeddings_path notes.npy \
  --note_metadata_path notes.csv \
  --outcomes_path outcomes.csv \
  --splits_path splits.csv \
  --landmarks=-7,3,7,30,90 \
  --search_space_json configs/example_search_space.json \
  --num_trials 30 --search_seed 42 \
  --output_dir results/note_survival
```

Those landmarks only illustrate the CLI; choose clinically appropriate anchors.
Optional `--config` defines fixed training settings. `--n_epochs`, `--patience`
and `--device` override those settings. Search uses fixed patient splits and
validation-only ranking. Each trial has an independently seeded model and epoch
sampler. Test prediction is **never** performed by search.

Search fields support candidate lists, continuous uniform/loguniform distributions,
and inclusive integer distributions, for example:

```json
{
  "d_model": [32, 64],
  "n_heads": [1, 2],
  "n_layers": {"type": "int", "low": 1, "high": 2},
  "dropout": {"type": "uniform", "low": 0.0, "high": 0.2},
  "lr": {"type": "loguniform", "low": 0.0001, "high": 0.003}
}
```

Other supported fields: `dim_ff`, `weight_decay`, `batch_size`,
`risk_set_max_controls`, `max_notes`, `time_scale_days`, `landmark_scale_days`,
and `gap_cap_days`. Incompatible configurations, including nondivisible hidden
dimensions/heads, are rejected and redrawn. Rejections are recorded in the search
manifest; impossible spaces fail after a bounded number of attempts. Independent
draws can repeat a configuration with a different trial seed.

Each `trials/trial_XXXX/` directory contains configuration, training/checkpoints,
logs and a `trial_result.json` tracking status, validation metrics, runtime and
errors. Consolidated outputs are `hparam_search_results.csv`,
`search_manifest.json`, `search_summary.json`, `best_config.json`, and
`best_model.pt`. Results include per-landmark eligible/event/death counts and C-index.
The earliest trial wins exact score ties.

Rerun with `--resume`: successful trials are verified/skipped; interrupted trials
resume; failed trials are retried. Changing files/splits, search space/seed or base
configuration is rejected. Increasing `--num_trials` preserves the saved prefix
of trials and appends new ones. Decreasing it is rejected. Run only one writer per
output directory.

## Validation and interruption recovery

AdamW, gradient clipping, and an optional epoch-stepped cosine scheduler are
supported. Validation evaluates full eligible cohorts independently at every
configured landmark. The selection score is the eligible-patient-count-weighted
mean primary-event Harrell C-index over computable landmarks. Undefined landmarks
have explicit reasons and are excluded from the weighted score. If all are
undefined, training fails instead of saving an arbitrarily selected model.
Early stopping uses `patience`, `min_epochs`, and `min_delta`; no test scores enter
selection. Full-cohort validation Cox loss is event-normalized. Its magnitude
cannot be compared directly with the sampled training loss.

Output files: `run_manifest.json`, inferred `config.json`,
`training_log.jsonl`, `best_model.pt`, `last_model.pt`, and `result.json`.
The manifest records risk-set exclusions and singleton groups.
Add `--resume` with the same inputs/configuration to resume a stopped run. Input
file hashes, split assignments, encoder ID, config and preprocessing must match.
Epoch-boundary optimizer, scheduler and RNG state are saved atomically. A process
killed mid-epoch replays that epoch from the previous saved boundary; resuming a
completed run returns its existing result. To change settings, use a new directory.

## Inspecting results

Open `hparam_search_results.csv` to compare successful trials, their configurations,
validation scores, runtimes, errors and per-landmark eligible/event/death counts.
`config_json` records the complete inferred model configuration.
`best_config.json` is the validation-selected configuration and `best_model.pt`
is its self-contained checkpoint. Freeze that selection before the explicit
held-out test operation; do not revise search settings based on test scores.
