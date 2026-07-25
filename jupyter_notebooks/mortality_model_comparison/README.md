# Mortality model comparison

These notebooks compare XGBoost with a Cox loss, elastic-net Cox, and random
survival forests (RSF) for the full-cohort `death_met/death` endpoint. Each
algorithm is fit with two feature sets:

- `baseline`: gender, age at treatment start, and cancer-type indicators.
- `baseline_text`: the baseline variables plus the existing pooled text
  embedding columns.

## Design

`01_tune_and_test.ipynb` creates one event-stratified 80/20 train/test split
shared by all six model variants. On the 80% training portion, it performs
5-fold stratified cross-validation and selects the candidate with the highest
mean cumulative/dynamic AUC. It then refits on all training records and writes
held-out mean AUC(t), Harrell c-index, integrated Brier score, and wall-clock
times for tuning, refitting, and test prediction/evaluation.

`02_generate_oof_risk_scores.ipynb` reads the selected parameters and performs
5-fold cross-fitting on the entire cohort. It writes one wide table containing
the six requested held-out risk-score columns and six corresponding
single-score files. To make scores trained in different folds comparable, each
validation-fold score is centered and scaled using the scores from that fold's
training patients only.

`03_feature_comparison_all_models.ipynb` re-runs the existing elastic-net Cox
modality comparison and extends it to XGBoost-Cox and RSF for mortality. It
fits baseline plus one of six modalities (stage, treatment, labs, somatic, PRS,
or text), selects hyperparameters within the shared 80% training cohort,
reports test metrics and timing, and produces 18 whole-cohort held-out
risk-score vectors. For elastic net, baseline variables remain unpenalized,
matching the existing Cox implementation. Lab imputation and the existing
1,500-component PRS PCA are fitted within each training fold.

The complete-case cohort is fixed before splitting, so all six comparisons use
identical patients. Test data are untouched during hyperparameter selection.
The c-index uses the full test set. The IPCW metrics use test records whose
follow-up lies within training follow-up support; the exact sample count and
time interval are included in the output.

## Dependencies and execution

The environment needs `numpy`, `pandas`, `scikit-learn`, `scikit-survival`,
`xgboost`, `matplotlib`, and Jupyter. The notebooks reuse
`python_scripts/model_training/slurm_array_utils.py`, so the source files must
be available at the data paths configured there.

```bash
python -m pip install -r jupyter_notebooks/mortality_model_comparison/requirements.txt
```

Run notebook 1 before notebook 2. Notebook 3 is independent of notebooks 1–2.
By default the first two notebooks write outputs to:

```text
<SURV_PATH>/results/death_met_results/mortality_model_comparison/
```

Notebook 3 writes to:

```text
<SURV_PATH>/results/death_met_results/mortality_feature_comparison_all_models/
```

Set `N_JOBS` in the configuration cell to match the allocated CPUs. The grids
contain 8 XGBoost, 12 elastic-net, and 6 RSF candidates per feature set. Edit
`PARAM_GRIDS` in notebook 1 if a pilot run indicates a narrower or wider search
is warranted.
