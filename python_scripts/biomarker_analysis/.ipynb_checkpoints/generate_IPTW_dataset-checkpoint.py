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
from lifelines import CoxPHFitter
from embed_surv_utils import run_grid_CoxPH_parallel, get_heldout_risk_scores_CoxPH, generate_survival_embedding_df
from statsmodels.stats.multitest import multipletests
from scipy import stats

random.seed(42)  # set seed for reproducibility

# Paths
IO_PATH = '/data/gusev/USERS/mjsaleh/'
PROJ_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/'
FIGURE_PATH = os.path.join(PROJ_PATH, 'figures/model_metrics/')
DATA_PATH = os.path.join(PROJ_PATH, 'data/')
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/VTE_data/processed_datasets/')
STAGE_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/'
MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')

# Load datasets
cancer_type_df = pd.read_csv('/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/first_treatments_dfci_w_inferred_cancers.csv',
                             usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group' : 'CANCER_TYPE'})
tt_phecodes_df = pd.read_csv(os.path.join(SURV_PATH, 'time-to-phecode/tt_vte_plus_phecodes.csv'))
biomarker_df = pd.read_csv(os.path.join(SURV_PATH, 'IO_biomarker_discovery.csv')).drop_duplicates(subset=['DFCI_MRN'], keep='first')

treatment_df = pd.read_csv("/data/gusev/USERS/mjsaleh/profile_lines_of_rx/profile_rxlines.csv")
somatic_df = pd.read_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/PROFILE_2024_MUTATION_CARRIERS.csv'))

ICI_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/treatment_prediction/ICI_propensity/w_30_day_buffer/'
line1_preds = pd.read_csv(os.path.join(ICI_PATH, 'line_1_predictions.csv'))

line1_mrns = line1_preds['DFCI_MRN'].unique()

line1_IO_data = biomarker_df.loc[biomarker_df['DFCI_MRN'].isin(line1_mrns)]
line1_non_IO_data = pd.get_dummies(tt_phecodes_df.loc[(tt_phecodes_df['DFCI_MRN'].isin(line1_mrns)) & 
                                       (~tt_phecodes_df['DFCI_MRN'].isin(line1_IO_data['DFCI_MRN']))]
                     .merge(somatic_df, on='DFCI_MRN')
                     .merge(cancer_type_df, on='DFCI_MRN'), columns=['CANCER_TYPE'], drop_first=True).drop_duplicates(subset=['DFCI_MRN'], keep='first')

cols_to_include = list(set(line1_IO_data.columns) & set(line1_non_IO_data.columns))
required_cols = ['DFCI_MRN', 'tt_death', 'death']

cancer_type_cols = [col for col in cols_to_include if col.startswith('CANCER')]
panel_version_cols = [col for col in cols_to_include if col.startswith('PANEL_VERSION')]

base_vars = ['GENDER', 'AGE_AT_TREATMENTSTART']

biomarker_cols = [col for col in cols_to_include if col not in (required_cols + base_vars)]
line1_IO_data['PX_on_IO'] = 1
line1_non_IO_data['PX_on_IO'] = 0

# ---------------------------------------
# Build analysis dataframe (as before)
# ---------------------------------------
interaction_IO_df = pd.concat([
    line1_IO_data[required_cols + base_vars + biomarker_cols + ['PX_on_IO']],
    line1_non_IO_data[required_cols + base_vars + biomarker_cols + ['PX_on_IO']]
], ignore_index=True)

# Map PS from LR predictions
ps_map = dict(zip(line1_preds['DFCI_MRN'], line1_preds['model_probs']))
interaction_IO_df['IO_prediction'] = interaction_IO_df['DFCI_MRN'].map(ps_map)

interaction_IO_df = interaction_IO_df.dropna(subset=['IO_prediction','tt_death','death']).copy()
interaction_IO_df['death'] = interaction_IO_df['death'].astype(int)

interaction_IO_df.to_csv(os.path.join(MARKER_PATH, 'IPTW_IO_interaction_runs_df.csv'), index=False)