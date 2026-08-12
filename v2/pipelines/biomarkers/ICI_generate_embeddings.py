"""Generate embedding datasets for ICI propensity score analysis.

For each cohort and buffer window, pools note embeddings relative to treatment
start and saves prediction-ready DataFrames to disk.

Run this once (or when inputs change), then use ICI_train_propensity.py to
train propensity models on the generated data.
"""

import os

import polars as pl
from tqdm import tqdm

from config import DATA_PATH, MATCHED_COHORT_PATH
from survival import generate_survival_embedding_df
from shared.polars_utils import filter_finite_rows

from pipelines.biomarkers.biomarker_common import load_note_embeddings

# ============================================================
# Configuration
# ============================================================
COHORTS = ['cohort1', 'cohort2']
BUFFERS = [30]


def main() -> None:
    # === Load shared data ===
    notes_meta, embeddings_data = load_note_embeddings()
    note_types = ['Clinician', 'Imaging', 'Pathology']
    pool_fx = {nt: 'time_decay_mean' for nt in note_types}

    # ============================================================
    # Main loop
    # ============================================================
    for COHORT in COHORTS:
        print(f"\n{'='*60}")
        print(f"[ICI_generate_embeddings] Cohort: {COHORT}")
        print(f"{'='*60}")

        OUTPUT_BASE = os.path.join(DATA_PATH, f'treatment_prediction/{COHORT}/')
        EMBEDDING_DATA_PATH = os.path.join(OUTPUT_BASE, 'prediction_data/')
        os.makedirs(EMBEDDING_DATA_PATH, exist_ok=True)

        # === Load matched cohort ===
        cohort_df = pl.read_csv(os.path.join(MATCHED_COHORT_PATH, f'matched_cohort_{COHORT}.csv.gz'))
        cohort_df = cohort_df.with_columns(pl.col('treatment_start_date').str.to_datetime())

        n_ici = cohort_df['PX_on_ICI'].sum()
        n_ctrl = len(cohort_df) - n_ici
        print(f"Cohort: {int(n_ici)} ICI + {int(n_ctrl)} controls = {len(cohort_df)} total")

        # === Generate embedding datasets for each buffer ===
        for buffer in tqdm(BUFFERS, desc="Generating embedding datasets"):
            buffer_path = os.path.join(EMBEDDING_DATA_PATH, f'w_{buffer}_day_buffer/')
            os.makedirs(buffer_path, exist_ok=True)

            cohort_treatment_dates = cohort_df.select(['DFCI_MRN', 'treatment_start_date'])
            cohort_mrn_set = set(cohort_df['DFCI_MRN'].to_list())
            notes_meta_sub = notes_meta.filter(pl.col('DFCI_MRN').is_in(cohort_mrn_set)).join(
                cohort_treatment_dates, on='DFCI_MRN', how='left'
            )
            note_dt = (
                pl.col('NOTE_DATETIME').str.to_datetime(strict=False)
                if notes_meta_sub.schema['NOTE_DATETIME'] == pl.Utf8
                else pl.col('NOTE_DATETIME').cast(pl.Datetime)
            )
            notes_meta_sub = notes_meta_sub.with_columns(
                (note_dt - pl.col('treatment_start_date').cast(pl.Datetime))
                .dt.total_days().alias('NOTE_TIME_REL_PRED_START_DT')
            )

            embedding_vals = generate_survival_embedding_df(
                notes_meta=notes_meta_sub, survival_df=None, embedding_array=embeddings_data,
                note_types=note_types, note_timing_col="NOTE_TIME_REL_PRED_START_DT",
                max_note_window=-buffer, pool_fx=pool_fx, decay_param=0.01, continuous_window=False)

            embedding_cols = [c for c in embedding_vals.columns if c != 'DFCI_MRN']
            embedding_vals_pl = filter_finite_rows(
                embedding_vals.drop_nulls('DFCI_MRN'), embedding_cols
            )
            full_dataset = cohort_df.join(embedding_vals_pl, on='DFCI_MRN')
            full_dataset.write_csv(
                os.path.join(buffer_path, f'ICI_prediction_df_w_{buffer}_day_buffer.csv.gz'),
                compression='gzip')

        # Save prediction times for downstream use
        cohort_df.select(['DFCI_MRN', 'treatment_start_date', 'PX_on_ICI', 'line_category']).write_csv(
            os.path.join(EMBEDDING_DATA_PATH, 'prediction_times.csv.gz'), compression='gzip')

        print(f"[ICI_generate_embeddings] Done with {COHORT}. Data in {EMBEDDING_DATA_PATH}")


if __name__ == "__main__":
    main()
