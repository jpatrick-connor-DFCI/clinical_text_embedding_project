"""Generate Embedding Prediction Datasets script for data preprocessing workflows."""

import os
import re
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from embed_surv_utils import generate_survival_embedding_df, map_time_to_event

# Paths
DATA_PATH = '/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/'
CODE_PATH = os.path.join(DATA_PATH, 'code_data/')
SURV_PATH = os.path.join(DATA_PATH, 'time-to-event_analysis/')
NOTES_PATH = os.path.join(DATA_PATH, 'batched_datasets/processed_datasets/')
INTAE_DATA_PATH = '/data/gusev/PROFILE/CLINICAL/robust_VTE_pred_project_2025_03_cohort/data/'
METS_PROJECT = '/data/gusev/Recurrent_Mets_Project/'
PROCESSED_DATA_PATH = os.path.join(METS_PROJECT, 'clinical_to_ag/')

# Shared columns/config
BASE_INPUT_COLS = [
    'DFCI_MRN',
    'AGE_AT_FIRST_TREAT',
    'BIOLOGICAL_SEX',
    'first_treatment_date',
    'death_date',
    'last_contact_date',
    'tt_death',
    'death',
    'tt_vte',
    'vte',
]
BASE_OUTPUT_COLS = ['DFCI_MRN', 'first_treatment_date', 'last_contact_date', 'AGE_AT_TREATMENTSTART', 'GENDER']
NOTE_TYPES = ['Clinician', 'Imaging', 'Pathology']
MET_SITES = ['brain', 'bone', 'adrenal', 'liver', 'lung', 'node', 'peritoneal']


def _normalize_icd10_undotted(code: str) -> Optional[str]:
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9.]", "", code)
    code = code.replace(".", "")
    return code if code else None


def _to_icd10_level_3(code: str) -> Optional[str]:
    code = _normalize_icd10_undotted(code)
    if code is None or len(code) < 3:
        return None
    return code[:3]


def _to_icd10_level_4(code: str) -> Optional[str]:
    code = _normalize_icd10_undotted(code)
    if code is None or len(code) < 3:
        return None
    if len(code) == 3:
        return code
    return f"{code[:3]}.{code[3]}"


def _normalize_phecode(code: str) -> Optional[str]:
    if pd.isna(code):
        return None
    code = str(code).strip()
    code = re.sub(r"[^0-9.]", "", code)
    if not code:
        return None

    if code.count('.') > 1:
        left, right = code.split('.', 1)
        code = f"{left}.{right.replace('.', '')}"

    if '.' in code:
        left, right = code.split('.', 1)
        right = right.rstrip('0')
        return left if right == '' else f"{left}.{right}"
    return code


def _resolve_column(df: pd.DataFrame, expected: str) -> str:
    col_map = {col.strip().lower(): col for col in df.columns}
    if expected not in col_map:
        raise ValueError(f"Expected column '{expected}' not found. Available columns: {list(df.columns)}")
    return col_map[expected]


def _dedupe_in_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


# ICD-10 chapter exclusions: external causes (V-Y), pregnancy (O), neoplasms (C, D00-D49)
_ICD10_EXCLUDED_PREFIXES = {'V', 'W', 'X', 'Y', 'O', 'C'}


def _is_excluded_icd10(code: str) -> bool:
    """Return True if an ICD-10 code falls in an excluded chapter."""
    if not code:
        return True
    first = code[0].upper()
    if first in _ICD10_EXCLUDED_PREFIXES:
        return True
    # D00-D49 are neoplasms; D50+ are blood/immune disorders (keep those)
    if first == 'D' and len(code) >= 3:
        try:
            num = int(code[1:3])
            if num <= 49:
                return True
        except ValueError:
            pass
    return False


# Phecode range exclusions: neoplasms (140-239), pregnancy (635-677), injuries/external (800-999)
_PHECODE_EXCLUDED_RANGES = [(140, 239.99), (635, 677.99), (800, 999.99)]


def _is_excluded_phecode(code: str) -> bool:
    """Return True if a phecode falls in an excluded range."""
    if not code:
        return True
    try:
        val = float(code)
    except (ValueError, TypeError):
        return True
    return any(lo <= val <= hi for lo, hi in _PHECODE_EXCLUDED_RANGES)


def _filter_endpoint_events_by_min_post_baseline_count(
    vte_data_sub: pd.DataFrame,
    endpoint_events: list[str],
    min_events: int = 100,
) -> list[str]:
    kept_events = []
    for event in endpoint_events:
        tt_col = f'tt_{event}'
        if event not in vte_data_sub.columns or tt_col not in vte_data_sub.columns:
            continue

        post_baseline_events = ((vte_data_sub[event] == 1) & (vte_data_sub[tt_col] > 0)).sum()
        if int(post_baseline_events) >= min_events:
            kept_events.append(event)
    return kept_events


def _load_shared_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    base_vte_data_sub = pd.read_csv(os.path.join(INTAE_DATA_PATH, 'follow_up_vte_df_cohort.csv'))[BASE_INPUT_COLS].copy()
    split_ehr_icd_subset = pd.read_csv(os.path.join(SURV_PATH, 'timestamped_icd_info.csv'))
    embeddings_data = np.load(os.path.join(NOTES_PATH, 'full_VTE_embeddings_as_array.npy'))
    notes_meta = pd.read_csv(os.path.join(NOTES_PATH, 'full_VTE_embeddings_metadata.csv'))
    return base_vte_data_sub, split_ehr_icd_subset, embeddings_data, notes_meta


def _prefilter_events_by_min_count(
    event_data: pd.DataFrame,
    cohort_mrns: set,
    event_col: str,
    time_col: str,
    events_to_analyze: list[str],
    min_events: int = 100,
) -> list[str]:
    """Fast pre-filter: keep only codes with >= min_events unique patients
    in the cohort who have a post-baseline (time > 0) occurrence."""
    # restrict to cohort and post-baseline
    mask = event_data['DFCI_MRN'].isin(cohort_mrns) & (event_data[time_col] > 0)
    counts = event_data.loc[mask].drop_duplicates(
        subset=['DFCI_MRN', event_col]
    )[event_col].value_counts()
    kept = [e for e in events_to_analyze if counts.get(e, 0) >= min_events]
    n_dropped = len(events_to_analyze) - len(kept)
    print(f"  Pre-filter: {len(kept)}/{len(events_to_analyze)} codes have >= {min_events} post-baseline events ({n_dropped} dropped)")
    return kept


def _map_events_to_columns(
    vte_data_sub: pd.DataFrame,
    event_data: pd.DataFrame,
    events_to_analyze: list[str],
    event_col: str,
    time_col: str,
    progress_desc: str,
) -> pd.DataFrame:
    mapped_cols: dict[str, pd.Series] = {}
    for event in tqdm(events_to_analyze, desc=progress_desc):
        event_data_sub = event_data.loc[event_data[event_col] == event]
        tt_series, event_series = map_time_to_event(
            event_data_sub, vte_data_sub, 'DFCI_MRN', event, time_col
        )
        mapped_cols[f'tt_{event}'] = tt_series
        mapped_cols[event] = event_series
    return pd.DataFrame(mapped_cols, index=vte_data_sub.index)


def _add_metastatic_events(vte_data_sub: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dfs_to_concat = [
        pd.read_csv(os.path.join(PROCESSED_DATA_PATH, f'clinical_to_{site}_met.csv'))
        .loc[lambda df: df['event'] == 1, ['dfci_mrn', 'date', 'type']]
        for site in MET_SITES
    ]
    met_date_df = pd.concat(dfs_to_concat, ignore_index=True)
    met_date_df.rename(columns={'dfci_mrn': 'DFCI_MRN', 'date': 'MET_DATE', 'type': 'MET_LOCATION'}, inplace=True)

    met_date_df = met_date_df.loc[met_date_df['DFCI_MRN'].isin(vte_data_sub['DFCI_MRN'])].copy()
    mrn_tstart_dict = dict(zip(vte_data_sub['DFCI_MRN'], pd.to_datetime(vte_data_sub['first_treatment_date'], errors='coerce')))
    met_date_df['first_treatment_date'] = met_date_df['DFCI_MRN'].map(mrn_tstart_dict)
    met_date_df['MET_DATE'] = pd.to_datetime(met_date_df['MET_DATE'].astype(str).str.split(' ').str[0], errors='coerce')
    met_date_df['TIME_TO_MET'] = (met_date_df['MET_DATE'] - met_date_df['first_treatment_date']).dt.days
    met_date_df = met_date_df.dropna(subset=['TIME_TO_MET'])

    met_events_added = []
    met_event_cols: dict[str, pd.Series] = {}
    for met_loc in sorted(met_date_df['MET_LOCATION'].dropna().unique()):
        cur_met_data_sub = met_date_df.loc[met_date_df['MET_LOCATION'] == met_loc]
        tt_series, event_series = map_time_to_event(
            cur_met_data_sub, vte_data_sub, 'DFCI_MRN', met_loc, 'TIME_TO_MET'
        )
        met_event_cols[f'tt_{met_loc}'] = tt_series
        met_event_cols[met_loc] = event_series
        met_events_added.append(str(met_loc))

    if met_event_cols:
        vte_data_sub = pd.concat([vte_data_sub, pd.DataFrame(met_event_cols, index=vte_data_sub.index)], axis=1)

    return vte_data_sub, met_events_added


def _finalize_base_covariates(vte_data_sub: pd.DataFrame) -> None:
    vte_data_sub['AGE_AT_TREATMENTSTART'] = vte_data_sub['AGE_AT_FIRST_TREAT']
    vte_data_sub['GENDER'] = vte_data_sub['BIOLOGICAL_SEX'].map({'MALE': 0, 'FEMALE': 1})
    vte_data_sub.drop(columns=['AGE_AT_FIRST_TREAT', 'BIOLOGICAL_SEX', 'death_date'], inplace=True)


def _write_outputs(
    vte_data_sub: pd.DataFrame,
    endpoint_events: list[str],
    surv_filename: str,
    embedding_filename: str,
    pooled_embedding_df: pd.DataFrame,
) -> None:
    event_cols = [event for event in endpoint_events if event in vte_data_sub.columns]
    event_cols = _dedupe_in_order(event_cols)
    tt_event_cols = [f'tt_{event}' for event in event_cols]

    events_data_sub = vte_data_sub[BASE_OUTPUT_COLS + event_cols + tt_event_cols]
    events_data_sub.to_csv(os.path.join(SURV_PATH, surv_filename), index=False)

    monthly_data = events_data_sub.merge(pooled_embedding_df, on='DFCI_MRN', how='left')
    embedding_cols = [col for col in pooled_embedding_df.columns if col != 'DFCI_MRN']
    monthly_data = monthly_data.dropna(subset=embedding_cols)
    monthly_data.to_csv(os.path.join(SURV_PATH, embedding_filename), index=False)


def _write_death_met_outputs(
    base_vte_data_sub: pd.DataFrame,
    pooled_embedding_df: pd.DataFrame,
    surv_filename: str = 'death_met_surv_df.csv',
    embedding_filename: str = 'death_met_embedding_prediction_df.csv',
    min_events: int = 100,
) -> None:
    vte_data_sub = base_vte_data_sub.copy()
    vte_data_sub, met_events_added = _add_metastatic_events(vte_data_sub)
    _finalize_base_covariates(vte_data_sub)

    met_events_added = _filter_endpoint_events_by_min_post_baseline_count(
        vte_data_sub, met_events_added, min_events=min_events
    )

    event_cols = ['death'] + [event for event in met_events_added if event in vte_data_sub.columns]
    event_cols = _dedupe_in_order(event_cols)
    tt_event_cols = [f'tt_{event}' for event in event_cols]

    events_data_sub = vte_data_sub[BASE_OUTPUT_COLS + event_cols + tt_event_cols]
    events_data_sub.to_csv(os.path.join(SURV_PATH, surv_filename), index=False)

    monthly_data = events_data_sub.merge(pooled_embedding_df, on='DFCI_MRN', how='left')
    embedding_cols = [col for col in pooled_embedding_df.columns if col != 'DFCI_MRN']
    monthly_data = monthly_data.dropna(subset=embedding_cols)
    monthly_data.to_csv(os.path.join(SURV_PATH, embedding_filename), index=False)


# Shared input loading (once)
base_vte_data_sub, split_ehr_icd_subset, embeddings_data, notes_meta = _load_shared_inputs()

# Pool embeddings once; merge into each endpoint-specific survival table below.
pooled_embedding_df = generate_survival_embedding_df(
    notes_meta=notes_meta,
    survival_df=None,
    embedding_array=embeddings_data,
    note_types=NOTE_TYPES,
    note_timing_col='NOTE_TIME_REL_FIRST_TREATMENT_START',
    continuous_window=False,
    pool_fx={key: 'time_decay_mean' for key in NOTE_TYPES},
    decay_param=0.01,
).dropna()

# =========================
# SHARED DEATH + MET DATASET
# =========================
_write_death_met_outputs(
    base_vte_data_sub=base_vte_data_sub,
    pooled_embedding_df=pooled_embedding_df,
)


# =========================
# ICD-10 LEVEL 3 DATASETS (first instance & first post-treatment)
# =========================
icd_data_base = split_ehr_icd_subset.copy()
icd_data_base['ICD10_LEVEL_3_CD'] = icd_data_base['DIAGNOSIS_ICD10_CD'].map(_to_icd10_level_3)
icd_data_base = icd_data_base.dropna(subset=['ICD10_LEVEL_3_CD']).copy()
icd_data_base = icd_data_base.loc[~icd_data_base['ICD10_LEVEL_3_CD'].map(_is_excluded_icd10)].copy()
icd_data_base['START_DT'] = pd.to_datetime(icd_data_base['START_DT'], errors='coerce')
icds_to_analyze = _dedupe_in_order(icd_data_base['ICD10_LEVEL_3_CD'].tolist())

# Pre-filter: compute union of codes passing min_events across both variants
cohort_mrns = set(base_vte_data_sub['DFCI_MRN'])
_shared_icd3_codes: set[str] = set()
for _label, _tf in [('first_instance', None), ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0)]:
    _icd_tmp = icd_data_base.copy()
    if _tf is not None:
        _icd_tmp = _icd_tmp.loc[_tf(_icd_tmp)].copy()
    _icd_tmp = _icd_tmp.sort_values(['DFCI_MRN', 'ICD10_LEVEL_3_CD', 'START_DT']).drop_duplicates(subset=['DFCI_MRN', 'ICD10_LEVEL_3_CD'], keep='first')
    _passed = _prefilter_events_by_min_count(_icd_tmp, cohort_mrns, 'ICD10_LEVEL_3_CD', 'TIME_TO_ICD', icds_to_analyze, min_events=100)
    _shared_icd3_codes.update(_passed)
shared_icd3_list = [c for c in icds_to_analyze if c in _shared_icd3_codes]
print(f"  Shared level-3 ICD code list: {len(shared_icd3_list)} codes (union across variants)")

for variant, time_filter, surv_fn, emb_fn in [
    ('first_instance', None, 'level_3_ICD_surv_df.csv', 'level_3_ICD_embedding_prediction_df.csv'),
    ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0, 'level_3_ICD_post_surv_df.csv', 'level_3_ICD_post_embedding_prediction_df.csv'),
]:
    vte_data_sub = base_vte_data_sub.copy()
    icd_data = icd_data_base.copy()
    if time_filter is not None:
        icd_data = icd_data.loc[time_filter(icd_data)].copy()
    icd_data = (
        icd_data
        .sort_values(['DFCI_MRN', 'ICD10_LEVEL_3_CD', 'START_DT'])
        .drop_duplicates(subset=['DFCI_MRN', 'ICD10_LEVEL_3_CD'], keep='first')
    )
    icd_event_cols = _map_events_to_columns(
        vte_data_sub=vte_data_sub,
        event_data=icd_data,
        events_to_analyze=shared_icd3_list,
        event_col='ICD10_LEVEL_3_CD',
        time_col='TIME_TO_ICD',
        progress_desc=f'Generating level-3 ICD events ({variant})',
    )
    vte_data_sub = pd.concat([vte_data_sub, icd_event_cols], axis=1)
    _finalize_base_covariates(vte_data_sub)
    kept = _filter_endpoint_events_by_min_post_baseline_count(vte_data_sub, shared_icd3_list, min_events=100)
    _write_outputs(
        vte_data_sub=vte_data_sub,
        endpoint_events=kept,
        surv_filename=surv_fn,
        embedding_filename=emb_fn,
        pooled_embedding_df=pooled_embedding_df,
    )


# =========================
# ICD-10 LEVEL 4 DATASETS (first instance & first post-treatment)
# =========================
icd_data_base = split_ehr_icd_subset.copy()
icd_data_base['ICD10_LEVEL_4_CD'] = icd_data_base['DIAGNOSIS_ICD10_CD'].map(_to_icd10_level_4)
icd_data_base = icd_data_base.dropna(subset=['ICD10_LEVEL_4_CD']).copy()
icd_data_base = icd_data_base.loc[~icd_data_base['ICD10_LEVEL_4_CD'].map(_is_excluded_icd10)].copy()
icd_data_base['START_DT'] = pd.to_datetime(icd_data_base['START_DT'], errors='coerce')
icds_to_analyze = _dedupe_in_order(icd_data_base['ICD10_LEVEL_4_CD'].tolist())

# Pre-filter: compute union of codes passing min_events across both variants
cohort_mrns = set(base_vte_data_sub['DFCI_MRN'])
_shared_icd4_codes: set[str] = set()
for _label, _tf in [('first_instance', None), ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0)]:
    _icd_tmp = icd_data_base.copy()
    if _tf is not None:
        _icd_tmp = _icd_tmp.loc[_tf(_icd_tmp)].copy()
    _icd_tmp = _icd_tmp.sort_values(['DFCI_MRN', 'ICD10_LEVEL_4_CD', 'START_DT']).drop_duplicates(subset=['DFCI_MRN', 'ICD10_LEVEL_4_CD'], keep='first')
    _passed = _prefilter_events_by_min_count(_icd_tmp, cohort_mrns, 'ICD10_LEVEL_4_CD', 'TIME_TO_ICD', icds_to_analyze, min_events=100)
    _shared_icd4_codes.update(_passed)
shared_icd4_list = [c for c in icds_to_analyze if c in _shared_icd4_codes]
print(f"  Shared level-4 ICD code list: {len(shared_icd4_list)} codes (union across variants)")

for variant, time_filter, surv_fn, emb_fn in [
    ('first_instance', None, 'level_4_ICD_surv_df.csv', 'level_4_ICD_embedding_prediction_df.csv'),
    ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0, 'level_4_ICD_post_surv_df.csv', 'level_4_ICD_post_embedding_prediction_df.csv'),
]:
    vte_data_sub = base_vte_data_sub.copy()
    icd_data = icd_data_base.copy()
    if time_filter is not None:
        icd_data = icd_data.loc[time_filter(icd_data)].copy()
    icd_data = (
        icd_data
        .sort_values(['DFCI_MRN', 'ICD10_LEVEL_4_CD', 'START_DT'])
        .drop_duplicates(subset=['DFCI_MRN', 'ICD10_LEVEL_4_CD'], keep='first')
    )
    icd_event_cols = _map_events_to_columns(
        vte_data_sub=vte_data_sub,
        event_data=icd_data,
        events_to_analyze=shared_icd4_list,
        event_col='ICD10_LEVEL_4_CD',
        time_col='TIME_TO_ICD',
        progress_desc=f'Generating level-4 ICD events ({variant})',
    )
    vte_data_sub = pd.concat([vte_data_sub, icd_event_cols], axis=1)
    _finalize_base_covariates(vte_data_sub)
    kept = _filter_endpoint_events_by_min_post_baseline_count(vte_data_sub, shared_icd4_list, min_events=100)
    _write_outputs(
        vte_data_sub=vte_data_sub,
        endpoint_events=kept,
        surv_filename=surv_fn,
        embedding_filename=emb_fn,
        pooled_embedding_df=pooled_embedding_df,
    )


# =========================
# PHECODE DATASETS (first instance & first post-treatment)
# =========================
mapping_file = os.path.join(CODE_PATH, 'icd10_to_phecode_mapping.csv')
mapping_df = pd.read_csv(mapping_file)
mapping_icd_col = _resolve_column(mapping_df, 'icd10_code')
mapping_phecode_col = _resolve_column(mapping_df, 'phecode')
mapping_df['ICD10_NORM'] = mapping_df[mapping_icd_col].map(_normalize_icd10_undotted)
mapping_df['PHECODE'] = mapping_df[mapping_phecode_col].map(_normalize_phecode)
mapping_df = mapping_df.dropna(subset=['ICD10_NORM', 'PHECODE']).drop_duplicates(subset=['ICD10_NORM', 'PHECODE'])

phecode_data_base = split_ehr_icd_subset.copy()
phecode_data_base['ICD10_NORM'] = phecode_data_base['DIAGNOSIS_ICD10_CD'].map(_normalize_icd10_undotted)
phecode_data_base = phecode_data_base.dropna(subset=['ICD10_NORM'])
phecode_data_base = phecode_data_base.merge(
    mapping_df[['ICD10_NORM', 'PHECODE']],
    on='ICD10_NORM',
    how='inner',
)
phecode_data_base['START_DT'] = pd.to_datetime(phecode_data_base['START_DT'], errors='coerce')
phecodes_to_analyze = _dedupe_in_order(phecode_data_base['PHECODE'].dropna().tolist())
phecodes_to_analyze = [p for p in phecodes_to_analyze if not _is_excluded_phecode(p)]
phecode_data_base = phecode_data_base.loc[~phecode_data_base['PHECODE'].map(_is_excluded_phecode)].copy()

# Pre-filter: compute union of codes passing min_events across both variants
cohort_mrns = set(base_vte_data_sub['DFCI_MRN'])
_shared_phecodes: set[str] = set()
for _label, _tf in [('first_instance', None), ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0)]:
    _phe_tmp = phecode_data_base.copy()
    if _tf is not None:
        _phe_tmp = _phe_tmp.loc[_tf(_phe_tmp)].copy()
    _phe_tmp = _phe_tmp.sort_values(['DFCI_MRN', 'PHECODE', 'START_DT']).drop_duplicates(subset=['DFCI_MRN', 'PHECODE'], keep='first')
    _passed = _prefilter_events_by_min_count(_phe_tmp, cohort_mrns, 'PHECODE', 'TIME_TO_ICD', phecodes_to_analyze, min_events=100)
    _shared_phecodes.update(_passed)
shared_phecode_list = [c for c in phecodes_to_analyze if c in _shared_phecodes]
print(f"  Shared phecode list: {len(shared_phecode_list)} codes (union across variants)")

for variant, time_filter, surv_fn, emb_fn in [
    ('first_instance', None, 'phecode_surv_df.csv', 'phecode_embedding_prediction_df.csv'),
    ('first_post_treatment', lambda df: df['TIME_TO_ICD'] > 0, 'phecode_post_surv_df.csv', 'phecode_post_embedding_prediction_df.csv'),
]:
    vte_data_sub = base_vte_data_sub.copy()
    phecode_data = phecode_data_base.copy()
    if time_filter is not None:
        phecode_data = phecode_data.loc[time_filter(phecode_data)].copy()
    phecode_data = (
        phecode_data
        .sort_values(['DFCI_MRN', 'PHECODE', 'START_DT'])
        .drop_duplicates(subset=['DFCI_MRN', 'PHECODE'], keep='first')
    )
    phecode_event_cols = _map_events_to_columns(
        vte_data_sub=vte_data_sub,
        event_data=phecode_data,
        events_to_analyze=shared_phecode_list,
        event_col='PHECODE',
        time_col='TIME_TO_ICD',
        progress_desc=f'Generating phecode events ({variant})',
    )
    vte_data_sub = pd.concat([vte_data_sub, phecode_event_cols], axis=1)
    _finalize_base_covariates(vte_data_sub)
    kept = _filter_endpoint_events_by_min_post_baseline_count(vte_data_sub, shared_phecode_list, min_events=100)
    _write_outputs(
        vte_data_sub=vte_data_sub,
        endpoint_events=kept,
        surv_filename=surv_fn,
        embedding_filename=emb_fn,
        pooled_embedding_df=pooled_embedding_df,
    )
