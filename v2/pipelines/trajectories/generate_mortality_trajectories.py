"""Generate Mortality Trajectories script for model evaluation workflows."""

import io
import os

import numpy as np
import polars as pl
import zstandard as zstd
from tqdm import tqdm

from config import FEATURE_PATH, NOTES_PATH, SURV_PATH
from schemes import scheme_results_dir
from survival import generate_survival_embedding_df, get_heldout_risk_scores_CoxPH, run_grid_CoxPH_parallel


def main() -> None:
    os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

    trajectory_path = os.path.join(scheme_results_dir("death_met"), 'mortality_trajectories/')
    os.makedirs(trajectory_path, exist_ok=True)

    # Load datasets
    notes_meta = pl.read_parquet(NOTES_PATH + 'full_clinical_notes_embeddings_metadata.parquet')
    with open(NOTES_PATH + 'full_clinical_notes_embeddings_as_array.npy.zst', 'rb') as f:
        embeddings_data = np.load(io.BytesIO(zstd.decompress(f.read())))
    embeddings_data = embeddings_data.astype(np.float32)
    events_data = pl.read_parquet(SURV_PATH + 'death_met_surv_df.parquet')
    cancer_type_df = pl.read_csv(os.path.join(FEATURE_PATH, 'cancer_type_df.csv.gz'))

    event = 'death'
    alphas_to_test = np.logspace(-5, 0, 25)
    l1_ratios = [0.5, 1.0]

    # regenerate predication dataframe at time 0 with new decay parameter
    decay_param = 0.1
    note_types = ['Clinician', 'Imaging', 'Pathology']
    full_prediction_df = (generate_survival_embedding_df(notes_meta, events_data, embeddings_data,
                                                         note_types=note_types, pool_fx={key: 'time_decay_mean' for key in note_types},
                                                         decay_param=decay_param, max_note_window=0)
                              .join(cancer_type_df, on='DFCI_MRN').drop_nulls())
    full_prediction_df = full_prediction_df.filter(pl.col(f'tt_{event}') > 0)

    # Define model columns
    base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']
    events = [col.split('_', 1)[1] for col in full_prediction_df.columns if col.startswith('tt')]
    tt_events = [f"tt_{event}" for event in events]

    # Column groups
    embed_cols = [c for c in full_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
    continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols
    type_cols = [c for c in full_prediction_df.columns if c.startswith('CANCER_TYPE_')]

    ## Train the baseline cancer model
    _, embed_val_results, pan_cancer_model = run_grid_CoxPH_parallel(
        full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
        l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}',
        max_iter=3000, verbose=5)

    opt_row = embed_val_results.sort('mean_auc(t)', descending=True).row(0, named=True)
    opt_l1_ratio, opt_alpha = opt_row['l1_ratio'], opt_row['alpha']

    ## Generate monthly data frames
    months_to_test = [i * 3 for i in range(1, 21)]

    cohort_mrns = full_prediction_df['DFCI_MRN'].unique().to_list()
    trajectory_predictions_df = pl.DataFrame(
        {'DFCI_MRN': cohort_mrns} |
        {f'plus_{month_adj}_months_data': [np.nan for _ in range(len(cohort_mrns))] for month_adj in months_to_test}
    )

    risk_scores = get_heldout_risk_scores_CoxPH(full_prediction_df, base_vars + type_cols, continuous_vars, embed_cols,
                                                event_col=event, tstop_col=f'tt_{event}', id_col='DFCI_MRN', penalized=True,
                                                l1_ratio=opt_l1_ratio, alpha=opt_alpha, max_iter=3000, verbose=5)

    risk_map = dict(zip(risk_scores['DFCI_MRN'].to_list(), risk_scores['risk_score'].to_list()))
    trajectory_predictions_df = trajectory_predictions_df.with_columns(
        pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan).alias('plus_0_months_data')
    )

    prev_mrns = cohort_mrns
    failed_landmarks = []
    for month_adj in tqdm(months_to_test):
        notes_meta_copy = notes_meta.filter(pl.col('DFCI_MRN').is_in(prev_mrns))
        events_data_copy = events_data.filter(pl.col('DFCI_MRN').is_in(prev_mrns))
        monthly_data = (generate_survival_embedding_df(notes_meta_copy, events_data_copy, embeddings_data, note_types=note_types,
                                                      pool_fx={key: 'time_decay_mean' for key in note_types}, decay_param=decay_param,
                                                      max_note_window=month_adj * 30)
                        .select(['DFCI_MRN', event, f'tt_{event}'] + base_vars + embed_cols)
                        .join(cancer_type_df, on='DFCI_MRN').drop_nulls())
        monthly_data = monthly_data.filter(pl.col(f'tt_{event}') > 0)

        try:
            risk_scores = get_heldout_risk_scores_CoxPH(monthly_data, base_vars + type_cols, continuous_vars, embed_cols,
                                                        event_col=event, tstop_col=f'tt_{event}', id_col='DFCI_MRN', penalized=True,
                                                        l1_ratio=opt_l1_ratio, alpha=opt_alpha, max_iter=3000, verbose=0)
            risk_map = dict(zip(risk_scores['DFCI_MRN'].to_list(), risk_scores['risk_score'].to_list()))
            trajectory_predictions_df = trajectory_predictions_df.with_columns(
                pl.col('DFCI_MRN').replace_strict(risk_map, default=np.nan).alias(f'plus_{month_adj}_months_data')
            )
            prev_mrns = risk_scores['DFCI_MRN'].unique().to_list()

        except Exception as exc:
            print(f"Trajectory landmark {month_adj} months failed: {exc}")
            failed_landmarks.append((month_adj, str(exc)))
            continue

    trajectory_predictions_df.write_csv(
        os.path.join(trajectory_path, f'survival_trajectories_w_decay_param_{decay_param}.csv'))
    if failed_landmarks:
        failed_months = ", ".join(str(month) for month, _ in failed_landmarks)
        raise RuntimeError(f"Trajectory generation failed at month landmarks: {failed_months}")


if __name__ == "__main__":
    main()
