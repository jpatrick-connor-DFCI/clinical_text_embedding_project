"""Generate IPTW dataset for biomarker analysis (ICI vs never-ICI).

Builds on cohort-specific propensity scores from ICI_LRs.py.
Cohort 2: line_category dummy-coded (LINE_2, LINE_3; line 1 as reference).
Cohort 1: no line dummies (all ICI patients are first-line; line_category dropped).
Also includes clinical text embeddings for prognostic score adjustment.

Notebook-ready: loops over all cohort x ps_model combinations automatically.
"""

import os
import random

import pandas as pd

from config import BIOMARKER_PATH, DATA_PATH, SURV_PATH

# ============================================================
# Configuration — edit these to control which combinations to run
# ============================================================
COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']


def main() -> None:
    random.seed(42)

    os.makedirs(BIOMARKER_PATH, exist_ok=True)

    # === Load shared data (only needs to happen once) ===
    tt_death_df = pd.read_parquet(os.path.join(SURV_PATH, 'death_met_surv_df.parquet'))
    tt_death_df['first_treatment_date'] = pd.to_datetime(tt_death_df['first_treatment_date'])
    tt_death_df = tt_death_df[['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                                'GENDER', 'AGE_AT_TREATMENTSTART']].copy()

    cancer_type_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/cancer_type_df.csv.gz'))

    somatic_df = pd.read_csv(
        os.path.join(DATA_PATH, 'clinical_and_genomic_features/complete_somatic_data_df.csv.gz'))
    mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
    panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
    mutation_cols = [col for col in somatic_df.columns if any(tag in col.upper() for tag in mutation_tags)]
    somatic_keep_cols = ['DFCI_MRN'] + panel_cols + mutation_cols
    somatic_keep_cols = list(dict.fromkeys(somatic_keep_cols))
    somatic_df = somatic_df[somatic_keep_cols].copy()

    # ============================================================
    # Main loop: iterate over cohort x ps_model combinations
    # ============================================================
    for COHORT in COHORTS:
        print(f"\n{'='*60}")
        print(f"[generate_IPTW_df] Cohort: {COHORT}")
        print(f"{'='*60}")

        PRED_BASE = os.path.join(DATA_PATH, f'treatment_prediction/{COHORT}/')
        PRED_DATA_PATH = os.path.join(PRED_BASE, 'prediction_data/')

        # === Load prediction times (includes line_category) ===
        prediction_times = pd.read_csv(os.path.join(PRED_DATA_PATH, 'prediction_times.csv.gz'))
        prediction_times['treatment_start_date'] = pd.to_datetime(prediction_times['treatment_start_date'])
        prediction_times = prediction_times.drop_duplicates(subset='DFCI_MRN', keep='first')

        # === Load clinical text embeddings (30-day buffer) for prognostic score ===
        EMBED_BUFFER = 30
        embed_file = os.path.join(PRED_DATA_PATH, f'w_{EMBED_BUFFER}_day_buffer/',
                                  f'ICI_prediction_df_w_{EMBED_BUFFER}_day_buffer.csv.gz')
        embed_df = pd.read_csv(embed_file)
        embedding_cols = [c for c in embed_df.columns
                          if ('IMAGING' in c) or ('PATHOLOGY' in c) or ('CLINICIAN' in c)]
        embed_df = embed_df[['DFCI_MRN'] + embedding_cols].drop_duplicates(subset='DFCI_MRN', keep='first')
        print(f"  Loaded {len(embedding_cols)} embedding columns from {embed_file}")

        # === Compute common patient set across PS models ===
        # Intersect MRNs from all PS model predictions so both specifications
        # use exactly the same patients.
        ps_pred_dfs = {}
        common_mrns = None
        for PS_MODEL in PS_MODELS:
            PS_PATH = os.path.join(PRED_BASE, f'{PS_MODEL}_propensity/w_30_day_buffer/')
            preds = pd.read_csv(os.path.join(PS_PATH, 'predictions.csv.gz'))
            required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
            if not required_pred_cols.issubset(set(preds.columns)):
                raise ValueError(f"predictions.csv must contain columns: {sorted(required_pred_cols)}")
            preds = preds[['DFCI_MRN', 'ground_truth', 'model_probs']].dropna().copy()
            preds['ground_truth'] = preds['ground_truth'].astype(int)
            ps_pred_dfs[PS_MODEL] = preds
            model_mrns = set(preds['DFCI_MRN'])
            common_mrns = model_mrns if common_mrns is None else common_mrns & model_mrns

        print(f"  Common patients across PS models: {len(common_mrns)}")

        # === Build IPTW dataset for each PS model using the common patient set ===
        cohort_mrn_sets = {}
        for PS_MODEL in PS_MODELS:
            print(f"\n  --- PS model: {PS_MODEL} ---")

            preds = ps_pred_dfs[PS_MODEL]
            preds = preds[preds['DFCI_MRN'].isin(common_mrns)].copy()

            # === Build unified patient dataframe ===
            # Inner join on embeddings ensures identical patients across PS model specifications
            patient_df = (tt_death_df
                          .merge(prediction_times, on='DFCI_MRN')
                          .merge(somatic_df, on='DFCI_MRN')
                          .merge(cancer_type_df, on='DFCI_MRN')
                          .merge(preds, on='DFCI_MRN')
                          .merge(embed_df, on='DFCI_MRN')
                          .drop_duplicates(subset=['DFCI_MRN'], keep='first'))

            # One-hot encode panel version
            if 'PANEL_VERSION' in patient_df.columns:
                patient_df = pd.get_dummies(patient_df, columns=['PANEL_VERSION'], drop_first=True)

            # Cohort 2: dummy-code line_category (line 1 as reference → LINE_2, LINE_3)
            # Cohort 1: all ICI are first-line — drop line_category entirely
            if COHORT != 'cohort1':
                patient_df = pd.get_dummies(patient_df, columns=['line_category'], prefix='LINE', drop_first=True, dtype=int)
            else:
                patient_df = patient_df.drop(columns=['line_category'], errors='ignore')

            # === Landmark re-anchoring to remove immortal-time bias ===
            # `tt_death` is measured from the first-line treatment start
            # (`first_treatment_date`), but in the matched cohort ICI can begin at
            # line 2 or 3. The time a patient must survive to reach that line is
            # "immortal" and would otherwise be wrongly credited to the ICI arm.
            # Re-anchor every patient's survival clock to their line-specific
            # landmark (`treatment_start_date`, the propensity/exposure anchor) and
            # require survival to that landmark, so ICI and matched controls are
            # compared from the line at which exposure is defined.
            # NOTE: for cohort 1 (all first-line) the shift is ~0, so this is a
            # no-op there. When you run this on the real data, check the diagnostic
            # below: a near-zero shift for cohort 2 would mean `treatment_start_date`
            # is NOT line-specific and the re-anchoring must be sourced differently.
            landmark_shift = (
                (patient_df['treatment_start_date'] - patient_df['first_treatment_date'])
                .dt.days.clip(lower=0)
            )
            patient_df = patient_df.assign(_landmark_shift=landmark_shift)
            n_pre = len(patient_df)
            print(f"  Landmark shift (days, first-line→line start): "
                  f"median={landmark_shift.median():.0f}, max={landmark_shift.max():.0f}, "
                  f">0 for {(landmark_shift > 0).sum()}/{n_pre} patients")

            # Drop patients who did not survive to their landmark (removes immortal
            # time), then re-anchor tt_death so time 0 = landmark. The strict '>'
            # also enforces positive post-landmark survival.
            pre_landmark = patient_df['tt_death'] <= patient_df['_landmark_shift']
            if pre_landmark.any():
                print(f"  Dropping {int(pre_landmark.sum())} patients who died/censored before their landmark")
            patient_df = patient_df.loc[~pre_landmark].copy()
            patient_df['tt_death'] = patient_df['tt_death'] - patient_df['_landmark_shift']
            patient_df = patient_df.drop(columns='_landmark_shift')

            # === Assign treatment group and propensity scores ===
            patient_df['PX_on_ICI'] = patient_df['ground_truth'].astype(int)
            patient_df['ICI_prediction'] = patient_df['model_probs']

            # === Select final columns ===
            required_cols = ['DFCI_MRN', 'tt_death', 'death']
            base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
            line_cols = [col for col in patient_df.columns if col.startswith('LINE_')]
            meta_cols = ['PX_on_ICI', 'ICI_prediction', 'first_treatment_date', 'treatment_start_date',
                         'ground_truth', 'model_probs']
            drop_cols = set(required_cols + base_vars + line_cols + meta_cols + embedding_cols)
            biomarker_cols = [col for col in patient_df.columns if col not in drop_cols]

            output_cols = (required_cols + base_vars + line_cols + biomarker_cols +
                           embedding_cols + ['PX_on_ICI', 'ICI_prediction'])
            print(f"  {len(embedding_cols)} embedding cols included for prognostic score")
            interaction_ICI_df = patient_df[output_cols].copy()

            interaction_ICI_df = interaction_ICI_df.dropna(subset=['ICI_prediction', 'tt_death', 'death']).copy()
            interaction_ICI_df['PX_on_ICI'] = interaction_ICI_df['PX_on_ICI'].astype(int)
            interaction_ICI_df['death'] = interaction_ICI_df['death'].astype(int)

            output_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{COHORT}_{PS_MODEL}.csv.gz')
            interaction_ICI_df.to_csv(output_file, index=False)
            print(f"  Saved {len(interaction_ICI_df)} patients to {output_file}")
            print(f"  ICI: {interaction_ICI_df['PX_on_ICI'].sum()}, "
                  f"Controls: {(~interaction_ICI_df['PX_on_ICI'].astype(bool)).sum()}")
            print(f"  Line dummies: {line_cols}")

            cohort_mrn_sets[PS_MODEL] = set(interaction_ICI_df['DFCI_MRN'])

        # === Verify patient consistency across PS models ===
        if len(cohort_mrn_sets) == 2:
            models = list(cohort_mrn_sets.keys())
            set_a, set_b = cohort_mrn_sets[models[0]], cohort_mrn_sets[models[1]]
            if set_a == set_b:
                print(f"\n  [CHECK] Patient sets identical across PS models ({len(set_a)} patients)")
            else:
                only_a = set_a - set_b
                only_b = set_b - set_a
                raise ValueError(
                    f"Patient mismatch in {COHORT}: {len(only_a)} only in {models[0]}, "
                    f"{len(only_b)} only in {models[1]}. "
                    f"This should not happen — check ICI_LRs.py common patient restriction."
                )


if __name__ == "__main__":
    main()
