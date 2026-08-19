"""Generate IPTW dataset for biomarker analysis (first-line ICI vs non-ICI).

Builds on cohort-specific propensity scores from ICI_LRs.py.
Cohort 2: line_category dummy-coded (LINE_2, LINE_3; line 1 as reference).
Cohort 1: no line dummies (all ICI patients are first-line; line_category dropped).
Also includes clinical text embeddings for prognostic score adjustment.

Somatic markers and cancer type are built from PROFILE_DATA in-process rather
than read from the pre-baked feature CSVs, so the somatic data can be anchored
at each patient's line-specific landmark (`treatment_start_date`) instead of at
`first_treatment_date` — a sequencing report issued after the ICI landmark must
not enter the marker set.

Notebook-ready: loops over all cohort x ps_model combinations automatically.
"""

import os
import random

import polars as pl

from config import BIOMARKER_PATH, DATA_PATH, SURV_PATH
from pipelines.preprocessing.generate_all_non_text_covariates import (
    build_cancer_type_df,
    build_somatic_data_df,
)
from shared.polars_utils import filter_finite_rows

# ============================================================
# Configuration — edit these to control which combinations to run
# ============================================================
COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']


def main() -> None:
    random.seed(42)

    os.makedirs(BIOMARKER_PATH, exist_ok=True)

    # === Load shared data (only needs to happen once) ===
    tt_death_df = pl.read_parquet(os.path.join(SURV_PATH, 'death_met_surv_df.parquet'))
    if tt_death_df['first_treatment_date'].dtype == pl.Utf8:
        tt_death_df = tt_death_df.with_columns(pl.col('first_treatment_date').str.to_datetime(strict=False))
    tt_death_df = tt_death_df.select(['DFCI_MRN', 'first_treatment_date', 'tt_death', 'death',
                                       'GENDER', 'AGE_AT_TREATMENTSTART'])

    # Cancer type is built on the full cohort, not the biomarker subset, so the
    # >=500-patient OTHER collapse produces the same labels the rest of the
    # project uses.
    full_cohort_df = pl.read_parquet(os.path.join(SURV_PATH, 'cohort_df.parquet'))
    cancer_type_df = build_cancer_type_df(full_cohort_df)

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
        prediction_times = pl.read_csv(os.path.join(PRED_DATA_PATH, 'prediction_times.csv.gz'))
        prediction_times = prediction_times.with_columns(
            pl.col('treatment_start_date').str.to_datetime(strict=False))
        prediction_times = prediction_times.unique(subset='DFCI_MRN', keep='first', maintain_order=True)

        # === Somatic markers, anchored at each patient's line landmark ===
        # REPORT_DT in GENOMIC_SPECIMEN is a pl.Date, so the landmark is cast to
        # Date for the `REPORT_DT <= anchor` eligibility filter.
        landmark = prediction_times.select(
            'DFCI_MRN', pl.col('treatment_start_date').cast(pl.Date))
        somatic_df = build_somatic_data_df(landmark, anchor_col='treatment_start_date')
        mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
        panel_cols = [col for col in somatic_df.columns if col.upper().startswith('PANEL_VERSION')]
        if not panel_cols:
            raise ValueError(
                "No PANEL_VERSION* column survived build_somatic_data_df; the panel-version "
                "confounder would be silently dropped. Check GENOMIC_SPECIMEN.parquet's column "
                f"names. Have: {sorted(somatic_df.columns)[:40]}"
            )
        mutation_cols = [col for col in somatic_df.columns if any(tag in col.upper() for tag in mutation_tags)]
        somatic_keep_cols = list(dict.fromkeys(['DFCI_MRN'] + panel_cols + mutation_cols))
        somatic_df = somatic_df.select(somatic_keep_cols).unique(subset='DFCI_MRN', keep='first')
        print(f"  Somatic: {somatic_df.height} patients, {len(mutation_cols)} marker columns "
              f"(anchored at treatment_start_date)")

        # === Load clinical text embeddings (30-day buffer) for prognostic score ===
        EMBED_BUFFER = 30
        embed_file = os.path.join(PRED_DATA_PATH, f'w_{EMBED_BUFFER}_day_buffer/',
                                  f'ICI_prediction_df_w_{EMBED_BUFFER}_day_buffer.csv.gz')
        embed_df = pl.read_csv(embed_file)
        embedding_cols = [c for c in embed_df.columns
                          if ('IMAGING' in c) or ('PATHOLOGY' in c) or ('CLINICIAN' in c)]
        embed_df = embed_df.select(['DFCI_MRN'] + embedding_cols).unique(subset='DFCI_MRN', keep='first', maintain_order=True)
        print(f"  Loaded {len(embedding_cols)} embedding columns from {embed_file}")

        # === Compute common patient set across PS models ===
        # Intersect MRNs from all PS model predictions so both specifications
        # use exactly the same patients.
        ps_pred_dfs = {}
        common_mrns = None
        for PS_MODEL in PS_MODELS:
            PS_PATH = os.path.join(PRED_BASE, f'{PS_MODEL}_propensity/w_30_day_buffer/')
            preds = pl.read_csv(os.path.join(PS_PATH, 'predictions.csv.gz'))
            required_pred_cols = {'DFCI_MRN', 'ground_truth', 'model_probs'}
            if not required_pred_cols.issubset(set(preds.columns)):
                raise ValueError(f"predictions.csv must contain columns: {sorted(required_pred_cols)}")
            preds = preds.select(['DFCI_MRN', 'ground_truth', 'model_probs']).drop_nulls('DFCI_MRN')
            preds = filter_finite_rows(preds, ['ground_truth', 'model_probs'])
            preds = preds.with_columns(pl.col('ground_truth').cast(pl.Int64))
            ps_pred_dfs[PS_MODEL] = preds
            model_mrns = set(preds['DFCI_MRN'].to_list())
            common_mrns = model_mrns if common_mrns is None else common_mrns & model_mrns

        print(f"  Common patients across PS models: {len(common_mrns)}")

        # === Build IPTW dataset for each PS model using the common patient set ===
        cohort_mrn_sets = {}
        for PS_MODEL in PS_MODELS:
            print(f"\n  --- PS model: {PS_MODEL} ---")

            preds = ps_pred_dfs[PS_MODEL]
            preds = preds.filter(pl.col('DFCI_MRN').is_in(common_mrns))

            # === Build unified patient dataframe ===
            # Inner join on embeddings ensures identical patients across PS model specifications
            patient_df = (tt_death_df
                          .join(prediction_times, on='DFCI_MRN', how='inner')
                          .join(somatic_df, on='DFCI_MRN', how='inner')
                          .join(cancer_type_df, on='DFCI_MRN', how='inner')
                          .join(preds, on='DFCI_MRN', how='inner')
                          .join(embed_df, on='DFCI_MRN', how='inner')
                          .unique(subset=['DFCI_MRN'], keep='first', maintain_order=True))

            # One-hot encode panel version
            if 'PANEL_VERSION' in patient_df.columns:
                panel_dummies = patient_df.to_dummies(columns=['PANEL_VERSION'])
                panel_cols_raw = [c for c in panel_dummies.columns if c.startswith('PANEL_VERSION_')]
                keep_panel_cols = sorted(panel_cols_raw)[1:]
                drop_panel_cols = [c for c in panel_cols_raw if c not in keep_panel_cols]
                patient_df = panel_dummies.drop(drop_panel_cols)

            # Cohort 2: dummy-code line_category (line 1 as reference → LINE_2, LINE_3)
            # Cohort 1: all ICI are first-line — drop line_category entirely
            if COHORT != 'cohort1':
                line_dummies = patient_df.to_dummies(columns=['line_category'])
                line_cols_raw = [c for c in line_dummies.columns if c.startswith('line_category_')]
                keep_line_cols = sorted(line_cols_raw)[1:]
                drop_line_cols = [c for c in line_cols_raw if c not in keep_line_cols]
                rename_map = {c: f'LINE_{c[len("line_category_"):]}' for c in keep_line_cols}
                patient_df = line_dummies.drop(drop_line_cols).rename(rename_map)
                patient_df = patient_df.with_columns(
                    [pl.col(c).cast(pl.Int64) for c in rename_map.values()])
            else:
                patient_df = patient_df.drop([c for c in ['line_category'] if c in patient_df.columns])

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
            patient_df = patient_df.with_columns(
                (pl.col('treatment_start_date') - pl.col('first_treatment_date'))
                .dt.total_days().clip(lower_bound=0).alias('_landmark_shift')
            )
            landmark_shift = patient_df['_landmark_shift']
            n_pre = patient_df.height
            print(f"  Landmark shift (days, first-line→line start): "
                  f"median={landmark_shift.median():.0f}, max={landmark_shift.max():.0f}, "
                  f">0 for {(landmark_shift > 0).sum()}/{n_pre} patients")

            # Drop patients who did not survive to their landmark (removes immortal
            # time), then re-anchor tt_death so time 0 = landmark. The strict '>'
            # also enforces positive post-landmark survival.
            pre_landmark = patient_df['tt_death'] <= patient_df['_landmark_shift']
            if pre_landmark.sum() > 0:
                print(f"  Dropping {int(pre_landmark.sum())} patients who died/censored before their landmark")
            patient_df = patient_df.filter(~pre_landmark)
            patient_df = patient_df.with_columns(
                (pl.col('tt_death') - pl.col('_landmark_shift')).alias('tt_death')
            )
            patient_df = patient_df.drop('_landmark_shift')

            # === Assign treatment group and propensity scores ===
            patient_df = patient_df.with_columns([
                pl.col('ground_truth').cast(pl.Int64).alias('PX_on_ICI'),
                pl.col('model_probs').alias('ICI_prediction'),
            ])

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
            interaction_ICI_df = patient_df.select(output_cols)

            interaction_ICI_df = filter_finite_rows(
                interaction_ICI_df, ['ICI_prediction', 'tt_death', 'death', 'PX_on_ICI']
            )
            interaction_ICI_df = interaction_ICI_df.with_columns([
                pl.col('PX_on_ICI').cast(pl.Int64),
                pl.col('death').cast(pl.Int64),
            ])

            output_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{COHORT}_{PS_MODEL}.csv.gz')
            interaction_ICI_df.write_csv(output_file, compression='gzip')
            print(f"  Saved {interaction_ICI_df.height} patients to {output_file}")
            n_ici = interaction_ICI_df['PX_on_ICI'].sum()
            print(f"  ICI: {n_ici}, "
                  f"Controls: {interaction_ICI_df.height - n_ici}")
            print(f"  Line dummies: {line_cols}")

            cohort_mrn_sets[PS_MODEL] = set(interaction_ICI_df['DFCI_MRN'].to_list())

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
