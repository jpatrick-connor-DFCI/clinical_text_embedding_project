import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = '/data/gusev/USERS/jpconnor/clinical_text_project/data/'
DIAGNOSTICS_PATH = '/data/gusev/PROFILE/CLINICAL/OncDRS/ALL_2025_03/'
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
SURV_PATH = os.path.join(DATA_PATH, 'survival_data/')
RESULTS_PATH = os.path.join(SURV_PATH, 'results/')

vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))
labs_df = pd.read_csv(os.path.join(DIAGNOSTICS_PATH, 'OUTPT_LAB_RESULTS_LABS.csv'), 
                      usecols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'TEST_TYPE_DESCR', 'NUMERIC_RESULT'])

test_count_df = labs_df['TEST_TYPE_DESCR'].value_counts().reset_index()
test_count_df['rank_index'] = range(len(test_count_df))

tests_to_include = test_count_df.loc[test_count_df['rank_index'] < 40]

lab_subset_df = (labs_df.merge(tests_to_include['TEST_TYPE_DESCR'], on=["TEST_TYPE_DESCR"], how="inner")
                 .merge(vte_data[['DFCI_MRN', 'first_treatment_date']]))
lab_subset_df = lab_subset_df.loc[lab_subset_df['NUMERIC_RESULT'] != 9999999.00].dropna()

lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] = (pd.to_datetime(lab_subset_df['SPECIMEN_COLLECT_DT']) - pd.to_datetime(lab_subset_df['first_treatment_date'])).apply(lambda x : x.days)

mean_lab_df = (
    lab_subset_df.loc[lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] < 0]
    .groupby(["DFCI_MRN", "TEST_TYPE_DESCR"])["NUMERIC_RESULT"]
    .agg(["mean", "std"])
    .reset_index()
    .pivot(index="DFCI_MRN", columns="TEST_TYPE_DESCR")
)

# flatten column names
mean_lab_df.columns = [f"{lab}_{stat}" for lab, stat in mean_lab_df.columns]
mean_lab_df = mean_lab_df.reset_index()

X = mean_lab_df.drop(columns=["DFCI_MRN"])

X_imp = (
    X
    .fillna(X.mean())
    .astype("float32")
)

final_mean_lab_df = pd.concat(
    [X_imp, X.isna().astype("int8").add_suffix("_missing")],
    axis=1
)

final_mean_lab_df.insert(0, "DFCI_MRN", mean_lab_df["DFCI_MRN"])

final_mean_lab_df.to_csv(os.path.join(DATA_PATH, 'clinical_and_genomic_features/mean_lab_vals_pre_first_treatment.csv'), index=False)