import os
import pickle
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from statsmodels.stats.multitest import multipletests
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/PROFILE/CLINICAL/irAE/PATRICK/cleaned_data/'
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
OUTPUT_PATH = os.path.join(RESULTS_PATH, 'phecode_model_comps_final')
os.makedirs(OUTPUT_PATH, exist_ok=True)

os.environ["JOBLIB_DEFAULT_WORKER_TIMEOUT"] = "600"

# Load datasets
cancer_type_df = pd.read_csv('/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
                             usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group' : 'CANCER_TYPE'})

notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
embeddings = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))

tt_phecodes_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-phecode/tt_vte_plus_phecodes.csv'))
irAE_df = pd.read_csv(os.path.join(IO_PATH, 'full_irAE_survival_data.csv')).rename(columns={'PATIENT_ID' : 'DFCI_MRN'})

IO_mrns = irAE_df['DFCI_MRN'].unique()
tt_phecodes_df_wo_IO_mrns = tt_phecodes_df.loc[~tt_phecodes_df['DFCI_MRN'].isin(IO_mrns)]

tstart_dict_w_IOs = dict(zip(irAE_df['DFCI_MRN'], irAE_df['IO_START'])) | \
                    dict(zip(tt_phecodes_df_wo_IO_mrns['DFCI_MRN'], tt_phecodes_df_wo_IO_mrns['first_treatment_date']))

notes_meta['IO_ANALYSIS_START_DT'] = notes_meta['DFCI_MRN'].map(tstart_dict_w_IOs)
notes_meta['NOTE_TIME_REL_IO_ANALYSIS_START_DT'] = (pd.to_datetime(notes_meta['NOTE_DATETIME']) - pd.to_datetime(notes_meta['IO_ANALYSIS_START_DT'])).dt.days

irAE_df['death'] = irAE_df['event'].map({'death' : 1, 'censor' : 0, 'irAE' : 0})
irAE_df['tt_death'] = irAE_df['tstop']

tt_phecodes_df_w_IO_mrns = pd.concat([tt_phecodes_df_wo_IO_mrns[['DFCI_MRN', 'death', 'tt_death']], 
                                      irAE_df[['DFCI_MRN', 'death', 'tt_death']]])

note_types = ['Clinician', 'Imaging', 'Pathology']
IO_prediction_df = pd.get_dummies(generate_survival_embedding_df(notes_meta, tt_phecodes_df_w_IO_mrns, embeddings, note_types=note_types,
                                                                 pool_fx={key : 'time_decay_mean' for key in note_types}, decay_param=0.01,
                                                                 note_timing_col='NOTE_TIME_REL_IO_ANALYSIS_START_DT')
                                  .merge(tt_phecodes_df[['DFCI_MRN', 'GENDER', 'AGE_AT_TREATMENTSTART']], on='DFCI_MRN')
                                  .merge(cancer_type_df, on='DFCI_MRN'), columns=['CANCER_TYPE'], drop_first=True).dropna()

IO_prediction_df.dropna(inplace=True)

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + [col for col in IO_prediction_df if col.startswith('CANCER_TYPE')]
embed_cols = [c for c in IO_prediction_df.columns if 'EMBEDDING' in c or '2015' in c]
continuous_vars = ['AGE_AT_TREATMENTSTART'] + embed_cols

IO_mrns = list(set(irAE_df['DFCI_MRN'].unique()).intersection(set(IO_prediction_df['DFCI_MRN'].unique())))

somatic_df = (pd.read_csv(os.path.join(DATA_PATH, 'VTE_cohort_somatic_mutations.csv'), index_col=0)
              .drop(columns=['death', 'tt_death', 'text_based_risk_score']))

# event='death'
# alphas_to_test = np.logspace(-5, 0, 30)
# l1_ratios = [0.5, 1.0]

# _, IO_val_results, _ = run_grid_CoxPH_parallel(
#     IO_prediction_df, base_vars, continuous_vars, embed_cols,
#     l1_ratios, alphas_to_test, event_col=event, tstop_col=f'tt_{event}', verbose=5)

# IO_l1_ratio, IO_alpha = IO_val_results.sort_values(by='mean_auc(t)', ascending=False).iloc[0][['l1_ratio', 'alpha']]

# trained_IO = (get_heldout_risk_scores_CoxPH(IO_prediction_df, base_vars, continuous_vars, embed_cols,
#                                             event_col=event, tstop_col=f'tt_{event}', penalized=True,
#                                             l1_ratio=IO_l1_ratio, alpha=IO_alpha)
#               .rename(columns={'risk_score' : 'IO_risk_score'}))

# biomarker_df = pd.get_dummies(irAE_df[['DFCI_MRN', 'tstop', 'event', 'GENDER', 'AGE_AT_TREATMENTSTART']]
#                 .merge(cancer_type_df, on='DFCI_MRN')
#                 .merge(somatic_df, on='DFCI_MRN')
#                 .merge(trained_IO, on='DFCI_MRN'), columns=['CANCER_TYPE', 'PANEL_VERSION'], drop_first=True)
# biomarker_df.to_csv(os.path.join(SURV_PATH, 'IO_biomarker_discovery.csv'), index=False)

from lifelines import CoxPHFitter

biomarker_df = pd.read_csv(os.path.join(SURV_PATH, 'IO_biomarker_discovery.csv'))

death_df = biomarker_df.copy()
death_df['event'] = death_df['event'].map({'death' : 1, 'censor' : 0, 'irAE' : 0})

genomics_cols = [col for col in somatic_df if not col in ['DFCI_MRN', 'PANEL_VERSION']]
base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART'] + \
            [col for col in death_df if col.startswith('CANCER_TYPE')] + \
            [col for col in death_df if col.startswith('PANEL_VERSION')]

marker_dfs = []
for test_col in tqdm(genomics_cols):
    base_cph = CoxPHFitter().fit(death_df[['tstop', 'event'] + base_vars + [test_col]], duration_col='tstop', event_col='event')
    base_cph_summary_df = base_cph.summary.reset_index()
    base_entry = base_cph_summary_df.loc[base_cph_summary_df['covariate'] == test_col]
    base_entry.columns = [col + '_without_text_risk' for col in base_entry.columns]

    risk_cph = CoxPHFitter().fit(death_df[['tstop', 'event'] + base_vars + ['IO_risk_score', test_col]], duration_col='tstop', event_col='event')
    risk_cph_summary_df = risk_cph.summary.reset_index()
    risk_entry = risk_cph_summary_df.loc[risk_cph_summary_df['covariate'] == test_col]
    risk_entry.columns = [col + '_with_text_risk' for col in risk_entry.columns]

    complete_entry = pd.concat([base_entry.reset_index(drop=True), risk_entry.reset_index(drop=True)], axis=1)
    complete_entry.insert(0, 'covariate', test_col)
    complete_entry.insert(1, 'c_index_without_text_risk', base_cph.concordance_index_)
    complete_entry.insert(1, 'c_index_with_text_risk', risk_cph.concordance_index_)

    marker_dfs.append(complete_entry)

full_IO_marker_df = pd.concat(marker_dfs)

reject, pvals_corrected, _, _ = multipletests(full_IO_marker_df["p_without_text_risk"], alpha=0.05, method="fdr_bh")
risk_reject, risk_pvals_corrected, _, _ = multipletests(full_IO_marker_df["p_with_text_risk"], alpha=0.05, method="fdr_bh")

full_IO_marker_df['corrected_p_without_text_risk'] = pvals_corrected
full_IO_marker_df['corrected_p_with_text_risk'] = risk_pvals_corrected

full_IO_marker_df['significant_without_text_risk'] = reject
full_IO_marker_df['significant_with_text_risk'] = risk_reject

all_sig_hits = full_IO_marker_df.loc[(full_IO_marker_df['significant_without_text_risk']) |
                                        (full_IO_marker_df['significant_with_text_risk'])]

import seaborn as sns
import matplotlib.pyplot as plt

all_sig_hits['sig_category'] = 'none'
all_sig_hits.loc[all_sig_hits['significant_without_text_risk'] & all_sig_hits['significant_with_text_risk'], 'sig_category'] = 'Both'
all_sig_hits.loc[all_sig_hits['significant_without_text_risk'] & ~all_sig_hits['significant_with_text_risk'], 'sig_category'] = 'w/o Text'
all_sig_hits.loc[~all_sig_hits['significant_without_text_risk'] & all_sig_hits['significant_with_text_risk'], 'sig_category'] = 'w/ Text'

plt.figure(figsize=(10, 8))
ax = sns.scatterplot(all_sig_hits, x='exp(coef)_without_text_risk', y='exp(coef)_with_text_risk', hue='sig_category')

# Add error bars manually
for _, row in all_sig_hits.iterrows():
    ax.errorbar(
        row["exp(coef)_without_text_risk"], row["exp(coef)_with_text_risk"],
        xerr=row["se(coef)_without_text_risk"], yerr=row["se(coef)_with_text_risk"],
        fmt="none", ecolor="gray", alpha=0.7, capsize=3
    )

# Add x=y line
lims = [
    min(all_sig_hits["exp(coef)_without_text_risk"].min(), all_sig_hits["exp(coef)_with_text_risk"].min()) - 0.1,
    max(all_sig_hits["exp(coef)_without_text_risk"].max(), all_sig_hits["exp(coef)_with_text_risk"].max()) + 0.1
]
ax.plot(lims, lims, "k--", alpha=0.7, zorder=0)  # black dashed line
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title('log(HR)s for Significant Hits')