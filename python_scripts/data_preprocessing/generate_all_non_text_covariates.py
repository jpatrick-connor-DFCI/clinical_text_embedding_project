import os
import pickle
import pandas as pd
import numpy as np
from functools import reduce
from collections import defaultdict

# === Paths ===
PROFILE_PATH = '/data/gusev/PROFILE/CLINICAL/'
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
CLINICAL_FEATURE_PATH = os.path.join(DATA_PATH, 'clinical_and_genomic_features/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
INTAE_DATA_PATH = os.path.join(PROFILE_PATH, 'robust_VTE_pred_project_2025_03_cohort/data/')
DIAGNOSTICS_PATH = os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/')
STAGE_PATH = os.path.join(PROFILE_PATH, 'OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/derived_files/cancer_stage/')

os.makedirs(CLINICAL_FEATURE_PATH, exist_ok=True)

# === Shared cohort data loads ===
vte_data = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'), usecols=['DFCI_MRN', 'first_treatment_date'])
time_decayed_events_df = pd.read_csv(os.path.join(SURV_PATH, 'level_3_ICD_embedding_prediction_df.csv'), usecols=['DFCI_MRN'])
cancer_type_df = pd.read_csv(
    os.path.join(INTAE_DATA_PATH, 'first_treatments_dfci_w_inferred_cancers.csv'),
    usecols=['DFCI_MRN', 'med_genomics_merged_cancer_group']).rename(columns={'med_genomics_merged_cancer_group': 'CANCER_TYPE'})

# === Cancer types ===
cancer_type_sub = cancer_type_df.loc[cancer_type_df['DFCI_MRN'].isin(time_decayed_events_df['DFCI_MRN'].unique())]
cancer_type_counts = cancer_type_sub['CANCER_TYPE'].value_counts()
types_to_keep = cancer_type_counts[cancer_type_counts >= 500].index.tolist()
cancer_type_sub['CANCER_TYPE'] = cancer_type_sub['CANCER_TYPE'].where(cancer_type_sub['CANCER_TYPE'].isin(types_to_keep), 'OTHER')
cancer_type_sub = pd.get_dummies(cancer_type_sub, columns=['CANCER_TYPE'], drop_first=True)

# === Cancer stage ===
mrn_stage_dict = pickle.load(open(os.path.join(STAGE_PATH, 'dfci_cancer_mrn_to_derived_cancer_stage.pkl'), 'rb'))
mrn_stage_df = pd.get_dummies(pd.DataFrame({'DFCI_MRN' : mrn_stage_dict.keys(),
                                            'CANCER_STAGE' : mrn_stage_dict.values()}),
                              columns=['CANCER_STAGE'], drop_first=True)

cancer_type_sub.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'cancer_type_df.csv'), index=False)
mrn_stage_df.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'cancer_stage_df.csv'), index=False)

# === Genomics ===
idmap = pd.read_csv(os.path.join(PROFILE_PATH, 'PROFILE_2024_idmap.csv'), usecols=['DFCI_MRN', 'sample_id', 'CANCER_TYPE', 'cbio_sample_id', 'PANEL_VERSION'])
px_metadata = (pd.read_csv(os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/SOMATIC_SPECIMEN.csv'),
                           usecols=['DFCI_MRN', 'SAMPLE_BL_NBR', 'CANCER_TYPE', 'SAMPLE_COLLECTION_DT'])
               .rename(columns={'SAMPLE_BL_NBR' : 'sample_id'})
               .merge(idmap, on=['DFCI_MRN', 'sample_id', 'CANCER_TYPE'])
               .merge(vte_data, on='DFCI_MRN'))
px_metadata['SAMPLE_COLLECTION_DT'] = pd.to_datetime(px_metadata['SAMPLE_COLLECTION_DT'])
px_metadata['first_treatment_date'] = pd.to_datetime(px_metadata['first_treatment_date'])
px_metadata['days_from_sequencing_to_first_treatment'] = (px_metadata['first_treatment_date'] - px_metadata['SAMPLE_COLLECTION_DT']).apply(lambda x : x.days)
px_metadata_min = (px_metadata
                   .loc[px_metadata
                        .groupby('DFCI_MRN')['days_from_sequencing_to_first_treatment']
                        .idxmin()]
                   .reset_index(drop=True))

dfs_to_merge = []
for mut_type in ['AMP', 'CNV', 'DEL', 'SNV']:
    if mut_type == 'SNV':
        cur_df = pd.read_csv(os.path.join(PROFILE_PATH, 'PROFILE_2024.MUTATION_CARRIERS.csv'), index_col=0)
    else:
        cur_df = pd.read_csv(os.path.join(PROFILE_PATH, f'PROFILE_2024.ALL_{mut_type}_CARRIERS.csv'), index_col=0)

    cur_df.columns = [col + f'_{mut_type}' for col in cur_df.columns]
    cur_df['sample_id'] = cur_df.index
    cur_df = cur_df.loc[cur_df['sample_id'].isin(px_metadata_min['sample_id'].unique())].reset_index(drop=True)
    dfs_to_merge.append(cur_df)

complete_mutation_data = (reduce(lambda left, right : pd.merge(left, right, on='sample_id', how='inner'), dfs_to_merge)
                          .dropna(axis=1, how='all')
                          .fillna(0)
                          .merge(px_metadata_min, on='sample_id'))

mutation_columns = [col for col in complete_mutation_data.columns if ('_AMP' in col) or ('_CNV' in col) or ('_DEL' in col) or ('_SNV' in col)]
metadata_columns = px_metadata_min.columns.tolist()

complete_mutation_data = complete_mutation_data[metadata_columns + mutation_columns]
complete_mutation_data[mutation_columns] = complete_mutation_data[mutation_columns].astype(int)

complete_mutation_data.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'complete_somatic_data_df.csv'), index=False)

prs_df = (pd.read_csv('/data/gusev/USERS/mjsaleh/PRS_PGScatalog/pgs_matrix_with_avg.tsv', sep='\t')
          .rename(columns={'IID' : 'cbio_sample_id'})
          .merge(px_metadata_min[['cbio_sample_id', 'DFCI_MRN']], on='cbio_sample_id'))
prs_df.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'complete_germline_data_df.csv'), index=False)

# === Structural variant features ===
SV_DOMINANCE_THRESHOLD = 0.5
SV_MIN_CASES = 20
SV_MIN_POSITIVE_SAMPLES = 10

sv_data = pd.read_csv(
    os.path.join(PROFILE_PATH, 'OncDRS/ALL_2025_03/SOMATIC_SV_RESULTS.csv'),
    usecols=['DFCI_MRN', 'SAMPLE_BL_NBR', 'SV_TYPE', 'PARTNER1_HUGO_GENE_NM', 'PARTNER2_HUGO_GENE_NM'],
).rename(columns={'SAMPLE_BL_NBR': 'sample_id'})

sv_data = sv_data.loc[sv_data['sample_id'].isin(px_metadata_min['sample_id'].unique())].copy()

# Canonicalize fusion partners by sorting gene names per row
genes = (
    sv_data[['PARTNER1_HUGO_GENE_NM', 'PARTNER2_HUGO_GENE_NM']]
    .astype('string')
    .apply(lambda x: x.str.strip().str.upper())
)
sorted_genes = np.sort(genes.fillna('').to_numpy(), axis=1)
sv_data['GENE_A'] = pd.Series(sorted_genes[:, 0], index=sv_data.index).replace('', pd.NA)
sv_data['GENE_B'] = pd.Series(sorted_genes[:, 1], index=sv_data.index).replace('', pd.NA)

gene_long = pd.concat(
    [
        sv_data[['sample_id', 'GENE_A', 'GENE_B']].rename(columns={'GENE_A': 'GENE', 'GENE_B': 'PARTNER'}),
        sv_data[['sample_id', 'GENE_A', 'GENE_B']].rename(columns={'GENE_B': 'GENE', 'GENE_A': 'PARTNER'}),
    ],
    ignore_index=True,
).dropna(subset=['sample_id', 'GENE', 'PARTNER'])

partner_counts = (
    gene_long
    .groupby(['GENE', 'PARTNER'])['sample_id']
    .nunique()
    .reset_index(name='n')
)
gene_totals = partner_counts.groupby('GENE')['n'].sum().reset_index(name='total_n')
partner_counts = partner_counts.merge(gene_totals, on='GENE')
partner_counts['fraction'] = partner_counts['n'] / partner_counts['total_n']

major_partners = partner_counts.loc[
    (partner_counts['fraction'] >= SV_DOMINANCE_THRESHOLD) &
    (partner_counts['n'] >= SV_MIN_CASES)
].copy()

genes_with_major = set(major_partners['GENE'])
all_genes = set(gene_long['GENE'].unique())
genes_without_major = all_genes - genes_with_major

feature_dict = defaultdict(set)

# For genes with a dominant partner, split into dominant fusion and "other SV" features
for _, row in major_partners.iterrows():
    gene = row['GENE']
    partner = row['PARTNER']

    major_name = f'{gene}_{partner}_FUSION'
    other_name = f'{gene}_OTHER_SV'

    major_mask = (
        ((sv_data['GENE_A'] == gene) & (sv_data['GENE_B'] == partner)) |
        ((sv_data['GENE_A'] == partner) & (sv_data['GENE_B'] == gene))
    )
    gene_mask = (sv_data['GENE_A'] == gene) | (sv_data['GENE_B'] == gene)

    major_samples = set(sv_data.loc[major_mask, 'sample_id'])
    gene_samples = set(sv_data.loc[gene_mask, 'sample_id'])

    feature_dict[major_name].update(major_samples)
    feature_dict[other_name].update(gene_samples - major_samples)

# For genes without a dominant partner, use a single gene-level SV indicator
for gene in genes_without_major:
    colname = f'{gene}_SV'
    gene_mask = (sv_data['GENE_A'] == gene) | (sv_data['GENE_B'] == gene)
    feature_dict[colname].update(set(sv_data.loc[gene_mask, 'sample_id']))

all_samples = sv_data['sample_id'].dropna().unique()
feature_df = pd.DataFrame(0, index=all_samples, columns=list(feature_dict.keys()), dtype='int8')
for col, sample_set in feature_dict.items():
    feature_df.loc[list(sample_set), col] = 1
feature_df = feature_df.reset_index().rename(columns={'index': 'sample_id'})

if feature_df.shape[1] > 1:
    pos_counts = feature_df.iloc[:, 1:].sum()
    valid_cols = pos_counts.index[pos_counts >= SV_MIN_POSITIVE_SAMPLES].tolist()
else:
    valid_cols = []

complete_sv_data = (
    feature_df[['sample_id'] + valid_cols]
    .merge(px_metadata_min, on='sample_id', how='inner')
)
sv_columns = [col for col in complete_sv_data.columns if col.endswith('_SV') or col.endswith('_FUSION')]
complete_sv_data = complete_sv_data[metadata_columns + sv_columns]
if sv_columns:
    complete_sv_data[sv_columns] = complete_sv_data[sv_columns].astype(int)

complete_sv_data.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'complete_sv_data_df.csv'), index=False)

# === Categorical treatment by line ===
med_classes = pd.read_csv(os.path.join(DATA_PATH, 'GPT_generated_med_classes.csv'))

treatment_df = (pd.read_csv('/data/gusev/USERS/mjsaleh/profile_lines_of_rx/ALL_MEDICATION_LINES.csv')
                .rename(columns={'MRN': 'DFCI_MRN', 'MED_START_DT': 'treatment_start_date'}))
treatment_df['treatment_start_date'] = pd.to_datetime(treatment_df['treatment_start_date'])
treatment_df = treatment_df.sort_values(['DFCI_MRN', 'treatment_start_date'])
treatment_df['treatment_line'] = treatment_df.groupby('DFCI_MRN').cumcount() + 1

med_class_dict = dict(zip(med_classes['MED_NAME'], med_classes['MOA_Category']))

treatments_long = (
    treatment_df['MED_NAME']
    .fillna('')
    .str.split(';')
    .explode()
    .str.strip())
treatments_long = treatments_long[treatments_long != '']

classes_long = treatments_long.map(med_class_dict).fillna('OTHER')
dummies = (pd.get_dummies(classes_long, prefix='PX_on', dtype=int)
    .groupby(level=0).max())

dummies = dummies.reindex(treatment_df.index, fill_value=0)
treatment_df = pd.concat([treatment_df, dummies], axis=1)

line_by_line_treatment_data = treatment_df.drop(columns=['TPLAN_TYPE', 'MED_NAME', 'THERAPY_TYPE', 'HAS_ICI', 'LINE',
                                                         'THERAPY_TYPES', 'ICI_SUBTYPES', 'type_of_rx', 'type_of_rx_sub', 'TPLAN_DX_NAME'])
line_by_line_treatment_data.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'categorical_treatment_data_by_line.csv'), index=False)

# === Mean lab values ===
labs_df = pd.read_csv(os.path.join(DIAGNOSTICS_PATH, 'OUTPT_LAB_RESULTS_LABS.csv'),
                      usecols=['DFCI_MRN', 'SPECIMEN_COLLECT_DT', 'TEST_TYPE_DESCR', 'NUMERIC_RESULT'])

test_count_df = labs_df['TEST_TYPE_DESCR'].value_counts().reset_index()
test_count_df['rank_index'] = range(len(test_count_df))
tests_to_include = test_count_df.loc[test_count_df['rank_index'] < 40]

lab_subset_df = (labs_df.merge(tests_to_include['TEST_TYPE_DESCR'], on=['TEST_TYPE_DESCR'], how='inner')
                 .merge(vte_data[['DFCI_MRN', 'first_treatment_date']]))
lab_subset_df = lab_subset_df.loc[lab_subset_df['NUMERIC_RESULT'] != 9999999.00].dropna()

lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] = (pd.to_datetime(lab_subset_df['SPECIMEN_COLLECT_DT']) - pd.to_datetime(lab_subset_df['first_treatment_date'])).apply(lambda x : x.days)

mean_lab_df = (
    lab_subset_df.loc[lab_subset_df['LAB_TIME_REL_FIRST_TREATMENT_START'] < 0]
    .groupby(['DFCI_MRN', 'TEST_TYPE_DESCR'])['NUMERIC_RESULT']
    .agg(['mean', 'std'])
    .reset_index()
    .pivot(index='DFCI_MRN', columns='TEST_TYPE_DESCR')
)

mean_lab_df.columns = [f'{lab}_{stat}' for lab, stat in mean_lab_df.columns]
mean_lab_df = mean_lab_df.reset_index()

X = mean_lab_df.drop(columns=['DFCI_MRN'])
X_imp = (
    X
    .fillna(X.mean())
    .astype('float32')
)

final_mean_lab_df = pd.concat(
    [X_imp, X.isna().astype('int8').add_suffix('_missing')],
    axis=1
)
final_mean_lab_df.insert(0, 'DFCI_MRN', mean_lab_df['DFCI_MRN'])
final_mean_lab_df.to_csv(os.path.join(CLINICAL_FEATURE_PATH, 'mean_lab_vals_pre_first_treatment.csv'), index=False)
