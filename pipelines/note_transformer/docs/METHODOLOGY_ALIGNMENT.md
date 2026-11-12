# Methodology alignment and extraction audit

Audited source: `codes/train_sequence_time_varying_vte_model_simple_v3_with_procedure.py`
and `codes/run_final_vte_model_comparison.py` in the original research codebase. Source filenames and line
numbers below are historical provenance; those files are not needed or included
in this repository. The original VTE repository is left unchanged.

## Active definitions and documented design

The training script defines several symbols twice. Python uses the later
definitions, except that `Preprocessor` at line 7477 inherits the earlier class
through `_BasePreprocessor` (line 6215). Imported helpers and earlier evaluation
functions resolve active module globals at runtime. Copying an arbitrary earlier
block would therefore extract a different model.

The research design described two layers and two attention heads with an
approximately 128-token history budget. Active `Config` (6705) defaults to one
layer, two heads, and a 512-token model budget. The previously audited selected checkpoint
records two layers, one head, hidden width 128, FFN width 256, and budget 128.
These are different configurations, not contradictory architecture requirements.
The portable example uses two layers/two heads and a 128-note budget; every
dimension is configurable. A note vector represents a whole note, not an EHR token.

## Alignment matrix

| Specification | Active source implementation | Alignment and portable treatment |
|---|---|---|
| Compact causal transformer | `EHRTransformerBackbone` 8234; `forward` 8283 | Causal encoder, LayerNorm, dropout and time channels match. Replace token/type/value and procedure embeddings with a linear note projection. Retain configurable heads/layers/FFN/history budget. |
| Landmark conditioning | `_compute_time_features_after_trunc` 4063; `_overwrite_landmark_token_in_prefix` 4179 | Relative-time tanh, landmark tanh and capped log-gap match. Use a learned landmark token without a vocabulary. Compute gaps after filtering and capping. |
| Global event-time risk pool | `LandmarkEventRiskSetDataset` 3530; `_risk_set_indices` 3694 | Queries all represented training patients with `T >= T_event`; does not restrict controls to the minibatch. Portable pool is the entire validated training split, including patients with no notes. |
| Uniform controls without replacement | `_sample_controls_uniform` 3723 | Matches. Resample each epoch; default weights one; optional uniform inclusion weight `N/m`. Remove `_sample_controls_hybrid` 3755 and cancer assumptions. |
| Random eligible landmark | `_sample_landmark_time` 3659 | Uniform eligible-anchor sampling matches when anchors exist. Legacy fallback can return `t0 >= T_event`; exclude and count such events instead. Read settings directly from the run configuration, not `preproc.cfg`. Retain fractional anchors; omit clinical jitter. |
| Common landmark for all members | `LandmarkEventGroupCollator` 8033 | Matches: every member's history is truncated at group `t0`. Preserve this explicitly. |
| No future histories | `_effective_len_upto_t0` 3896; `build_patient_sequences` 7614 | Prefix cutoff assumes sorted histories; preprocessing caps globally before landmark filtering. Portable loader sorts deterministically and caps only after cutoff. Embedding provenance must independently guarantee availability-time validity. |
| Empty histories | Active sequence builder 7614; fixed-landmark dataset 3440 | Global exclusions depend on selected EHR events; retained patients can become context-only at early landmarks. Portable population includes every eligible patient, using the learned landmark token for zero notes. |
| Sampled Cox contribution | Active `landmark_groupwise_cox_loss` 8676 | Case-first score and weighted `logsumexp` match. Separate pure score loss from model execution; preserve case weight one. |
| Primary-event batch mean | Same loss; collator cause weights | Legacy pools VTE/death groups with `lambda_death` and weight-sum normalization plus procedure regularization. Portable version averages primary-event groups only and has no death head or procedure penalty. |
| Breslow ties | Inclusive risk query; `cox_ph_loss_static` 4957 | Tied cases and censoring at the event time remain at risk. Separate contributions recover Breslow at a fixed landmark with full controls. No Efron correction. Add a separate full-cohort reference. |
| First-event cause-specific outcome | Inherited `build_patient_labels` 2972; helper `_get_shared_landmark_outcomes` in `utils_training.py` 7022 | First VTE/death cause is used, but differing cause-specific censoring fields can create inconsistent times/flags. Require one explicit finite observed first-event/censoring time and mutually exclusive indicators. Death ends follow-up even without a death head. Reject delayed-entry/recurrent schemas. |
| Validation ranking | `evaluate_landmark_metrics` 5040; `train_and_evaluate_config` 9452 | Per-landmark Harrell C and eligible-count-weighted mean match. Legacy training and validation death-loss normalizations differ. Portable full primary Cox evaluation is event-normalized; sampled training is a group mean. Those losses differ by sampled versus full denominators and must not be numerically compared. |
| Undefined C-index | `safe_concordance_index` 2503 | Guards no events/small cohorts but can raise on no comparable pairs. Return NaN with an explicit reason and omit undefined landmarks from weighted selection. |
| Checkpoint/inference | Training checkpoint 9452; `predict_landmark_df` 882; `predict_on_test_by_landmark` 985 | Legacy inference requires external vocabulary/genetics/preprocessor state. Save portable model config, static transforms, provenance, fingerprints and weights; predict on outcome-free patients. |
| Hyperparameter search | Active `run_hyperparam_search` 9582 | Seeded configurations, atomic saves and validation ranking are useful. Legacy resume primarily keys on experiment/trial IDs. Portable resume verifies input/config/search fingerprints and never evaluates test data while searching. |
| Prediction interpretation | `risk_from_repr` 8610; export in comparison script 907 | Scores are unconstrained cause-specific log-relative risks, not horizon probabilities. Keep this interpretation and avoid probability labels. |

## Mathematical target

For a primary event at `T_i`, draw one `t0` uniformly from configured anchors
strictly below `T_i`, then controls uniformly without replacement from all other
training patients with `T_j >= T_i`. Encode every member at that same `t0`:

`loss_i = -eta_i(t0) + logsumexp_j(eta_j(t0) + log(w_ij))`.

The case has weight one. Controls have weight one by default, or `N/m` with
optional inclusion weighting. IPW estimates the full denominator before taking
its logarithm; it does not make the log partial likelihood exactly unbiased.
Sampling one anchor per case weights each eligible case-anchor pair by
`1 / len(L_i)`, rather than equally weighting all pairs. This is deliberate.

This is a first-event landmark cause-specific sampled Cox objective. The legacy
name `time_varying_ag` does not provide general recurrent-event Andersen–Gill
start/stop semantics. Neither version is Fine–Gray, and neither produces
calibrated survival or cumulative-incidence probabilities.

## Dependencies and deliberate removals

`utils_training.py` imports `models.py`, `transformer.py`, `utils.py`, and
`utils_data.py`; the utilities also import each other. These bring unrelated
survival models, plotting, clinical baselines and data downloads into an import.
The active sampled loss is defined in the training script, not these helpers.
`utils_training.get_cox_log_partial_likelihood` (2287) exponentiates scores and
returns a differently normalized log likelihood; its cumulative sorted Cox
helpers (4679/4697) are not the tie-aware reference chosen here.
`transformer.TransformerModel` (43) is a different positional/noncausal,
mean-pooling architecture and has an opposite mask convention.

Exclude genetics, all fusion/MoE branches (8343–8639), additive modality hazard
combination, EHR vocabulary/feature selection, labs/medications/procedures,
Khorana, cancer/context derivation, Dropbox paths, VTE caches, comparison
orchestration, dashboards and clinical reporting. Generic static covariates use
only a small MLP. The standalone package imports none of these source modules.

## Preserved correctness boundaries

- Patient splits precede landmark expansion and risk-set construction.
- All times share one index origin; `t0` is a prediction day relative to it.
- `T > t0` defines retrospective landmark eligibility; `T_j >= T_i` defines controls.
- Information availability, not retrospective service date, bounds histories.
- Death/censoring before an event excludes that patient from its risk set.
- Ties use inclusive risk membership and Breslow, without Efron correction.
- Outcomes never enter the model forward input.
- C-index is evaluated separately at landmarks; cross-landmark raw scores are not pooled.
- Validation selects the model; test evaluation is an explicit final operation.
- Model checkpoints are independent of original EHR artifacts.

## Implemented verification

The extracted package and three root CLI scripts import no original VTE modules.
The delivered suite passes 25 CPU tests, including all-control Breslow equivalence,
global sampling, strict landmark eligibility, leakage/history-cap invariance,
empty histories, group-mean microbatch gradients, exact interrupted epoch replay,
validation-only search selection, and interrupted/successful search resumption.
The synthetic demo and four-file CLI workflows completed, and a wheel installed
outside the repository reproduced source-CLI predictions. Current independent-repository verification is recorded in
[VALIDATION.md](VALIDATION.md). Generated data, checkpoints and logs are excluded
from version control. Original source files are unchanged;
these tests establish implementation behavior, not clinical validity.
