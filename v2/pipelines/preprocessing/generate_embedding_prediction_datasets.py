"""Generate Embedding Prediction Datasets script for data preprocessing workflows.

Supports `--anchor {treatment,sequencing}` (see anchors.py). The sequencing
arm re-anchors and excludes (shift `tt`, drop patients not eligible at the
new t=0) rather than expressing delayed entry, which sksurv's
CoxnetSurvivalAnalysis cannot represent — see V2_PIPELINE_PLAN.md. Endpoint
times are re-derived from the anchor date (not the pre-baked, treatment-anchor
`TIME_TO_ICD`/`TIME_TO_MET` columns) so no feature or event is observed
relative to the wrong t=0.
"""

import argparse
import io
import os
import re
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
import zstandard as zstd
from tqdm import tqdm

from anchors import DEFAULT_ANCHOR, anchor_suffix, date_col, ensure_anchor, note_time_col
from config import CODE_PATH, NOTES_PATH, PROCESSED_DATA_PATH, SURV_PATH
from shared.icd10 import normalize_icd10_undotted, to_icd10_level_3, to_icd10_level_4
from survival import generate_survival_embedding_df, map_time_to_event

# Shared columns/config
BASE_INPUT_COLS = [
    'DFCI_MRN',
    'AGE_AT_TREATMENTSTART',
    'AGE_AT_SEQUENCING',
    'GENDER',
    'first_treatment_date',
    'sequencing_date',
    'death_date',
    'last_contact_date',
    'tt_death',
    'tt_death_treatment',
    'tt_death_sequencing',
    'death',
    'eligible_treatment',
    'eligible_sequencing',
]
BASE_OUTPUT_COLS = [
    'DFCI_MRN', 'first_treatment_date', 'sequencing_date', 'last_contact_date',
    'AGE_AT_TREATMENTSTART', 'AGE_AT_SEQUENCING', 'GENDER',
]
CORE_EVENT_COLS = ['death', 'tt_death']
NOTE_TYPES = ['Clinician', 'Imaging', 'Pathology']
MET_SITES = ['brain', 'bone', 'adrenal', 'liver', 'lung', 'node', 'peritoneal']


# Promoted to shared/icd10.py so pipelines.preprocessing.generate_all_non_text_covariates
# can reuse them without importing this heavier, later-stage module.
_normalize_icd10_undotted = normalize_icd10_undotted
_to_icd10_level_3 = to_icd10_level_3
_to_icd10_level_4 = to_icd10_level_4


def _normalize_phecode(code: str) -> Optional[str]:
    if code is None:
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
        left = left.lstrip('0') or '0'
        right = right.rstrip('0')
        return left if right == '' else f"{left}.{right}"
    return code.lstrip('0') or '0'


def _resolve_column(df: pl.DataFrame, expected: str) -> str:
    col_map = {col.strip().lower(): col for col in df.columns}
    if expected not in col_map:
        raise ValueError(f"Expected column '{expected}' not found. Available columns: {list(df.columns)}")
    return col_map[expected]


def _suffixed(filename: str, anchor: str) -> str:
    """Insert `anchor_suffix()` before the extension; `treatment` reproduces
    `filename` exactly, e.g. 'death_met_embedding_prediction_df.parquet' ->
    'death_met_embedding_prediction_df__sequencing.parquet'."""
    suffix = anchor_suffix(anchor)
    if not suffix:
        return filename
    stem, ext = os.path.splitext(filename)
    return f"{stem}{suffix}{ext}"


def _dedupe_in_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


# ICD-10 chapter exclusions: external causes (V-Y), pregnancy (O), neoplasms (C, D00-D49)
# C77-C79 (secondary/metastatic neoplasms) are excluded here as ICD endpoint
# codes, but are deliberately consumed elsewhere: see build_met_burden_df in
# generate_all_non_text_covariates.py, which reads timestamped_icd_info.parquet
# directly and never passes through _is_excluded_icd10.
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
    cohort_df: pl.DataFrame,
    endpoint_events: list[str],
    min_events: int = 100,
) -> list[str]:
    kept_events = []
    for event in endpoint_events:
        tt_col = f'tt_{event}'
        if event not in cohort_df.columns or tt_col not in cohort_df.columns:
            continue

        post_baseline_events = cohort_df.filter(
            (pl.col(event) == 1) & (pl.col(tt_col) > 0)
        ).height
        if post_baseline_events >= min_events:
            kept_events.append(event)
    return kept_events


def _load_shared_inputs(anchor: str = DEFAULT_ANCHOR) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray, pd.DataFrame]:
    """Load and anchor-restrict shared inputs. `base_cohort_df` is filtered to
    `eligible_<anchor>` rows (drops patients not at risk at the chosen t=0 —
    the "exclude" half of "re-anchor and exclude") and gets a `tt_death`
    column aliased from `tt_death_<anchor>` so downstream code, which is
    anchor-agnostic, keeps working unmodified. ICD event times
    (`split_ehr_icd_subset['TIME_TO_ICD']`) are re-derived here from the raw
    `START_DT` relative to the anchor's date column rather than trusting the
    treatment-anchor-locked value baked in by extract_ICD_times.py."""
    ensure_anchor(anchor)
    base_cohort_df = pl.read_parquet(os.path.join(SURV_PATH, 'cohort_df.parquet')).select(BASE_INPUT_COLS)

    eligible_col = f'eligible_{anchor}'
    base_cohort_df = base_cohort_df.filter(pl.col(eligible_col))
    base_cohort_df = base_cohort_df.with_columns(pl.col(f'tt_death_{anchor}').alias('tt_death'))

    anchor_date_col = date_col(anchor)
    mrn_anchor_dict = dict(
        zip(
            base_cohort_df['DFCI_MRN'].to_list(),
            pd.to_datetime(base_cohort_df[anchor_date_col].to_pandas(), errors='coerce'),
        )
    )

    split_ehr_icd_subset = pl.read_parquet(os.path.join(SURV_PATH, 'timestamped_icd_info.parquet'))
    cohort_mrn_set = set(base_cohort_df['DFCI_MRN'].to_list())
    split_ehr_icd_subset = split_ehr_icd_subset.filter(pl.col('DFCI_MRN').is_in(cohort_mrn_set))

    icd_start_dt = pd.to_datetime(split_ehr_icd_subset['START_DT'].to_pandas(), errors='coerce')
    icd_anchor_dt = split_ehr_icd_subset['DFCI_MRN'].to_pandas().map(mrn_anchor_dict)
    time_to_icd = (icd_start_dt - icd_anchor_dt).dt.days
    split_ehr_icd_subset = split_ehr_icd_subset.with_columns(
        pl.Series('TIME_TO_ICD', time_to_icd.to_numpy())
    )

    with open(os.path.join(NOTES_PATH, 'full_clinical_notes_embeddings_as_array.npy.zst'), 'rb') as f:
        embeddings_data = np.load(io.BytesIO(zstd.decompress(f.read())))
    embeddings_data = embeddings_data.astype(np.float32)
    # generate_survival_embedding_df / pool_embedding_series_vectorized (v2/survival) remain
    # pandas-only, so notes_meta is kept as pandas at this boundary.
    notes_meta = pd.read_parquet(os.path.join(NOTES_PATH, 'full_clinical_notes_embeddings_metadata.parquet'))
    return base_cohort_df, split_ehr_icd_subset, embeddings_data, notes_meta


def _prefilter_events_by_min_count(
    event_data: pl.DataFrame,
    cohort_mrns: set,
    event_col: str,
    time_col: str,
    events_to_analyze: list[str],
    min_events: int = 100,
) -> list[str]:
    """Fast pre-filter: keep only codes with >= min_events unique patients
    in the cohort who have a post-baseline (time > 0) occurrence."""
    # restrict to cohort and post-baseline
    filtered = event_data.filter(
        pl.col('DFCI_MRN').is_in(cohort_mrns) & (pl.col(time_col) > 0)
    ).unique(subset=['DFCI_MRN', event_col])
    counts_df = filtered[event_col].value_counts()
    counts = dict(zip(counts_df[event_col].to_list(), counts_df['count'].to_list())) if filtered.height else {}
    kept = [e for e in events_to_analyze if counts.get(e, 0) >= min_events]
    n_dropped = len(events_to_analyze) - len(kept)
    print(f"  Pre-filter: {len(kept)}/{len(events_to_analyze)} codes have >= {min_events} post-baseline events ({n_dropped} dropped)")
    return kept


def _prefilter_events_by_prevalence(
    event_data: pl.DataFrame,
    cohort_mrns: set,
    event_col: str,
    events_to_analyze: list[str],
    min_prevalence: float = 0.01,
) -> list[str]:
    """Keep only codes with >= min_prevalence unique-patient prevalence in the full cohort
    (any time, pre- or post-baseline)."""
    cohort_size = len(cohort_mrns)
    min_patients = max(1, int(min_prevalence * cohort_size))
    filtered = event_data.filter(pl.col('DFCI_MRN').is_in(cohort_mrns)).unique(subset=['DFCI_MRN', event_col])
    counts_df = filtered[event_col].value_counts()
    counts = dict(zip(counts_df[event_col].to_list(), counts_df['count'].to_list())) if filtered.height else {}
    kept = [e for e in events_to_analyze if counts.get(e, 0) >= min_patients]
    n_dropped = len(events_to_analyze) - len(kept)
    print(f"  Prevalence filter (>={min_prevalence:.1%}, n>={min_patients}): {len(kept)}/{len(events_to_analyze)} codes ({n_dropped} dropped)")
    return kept


def _map_events_to_columns(
    cohort_df: pl.DataFrame,
    event_data: pl.DataFrame,
    events_to_analyze: list[str],
    event_col: str,
    time_col: str,
    progress_desc: str,
) -> pl.DataFrame:
    """`map_time_to_event` (v2/survival, pandas-only) returns Series aligned
    positionally with the pandas view of `cohort_df` passed in (same row
    order, same length, no independent filter/sort) — so building result
    columns from those Series and attaching them back via `with_columns` on
    the same `cohort_df` is a safe positional operation, not an
    index-alignment join."""
    cohort_pd = cohort_df.to_pandas()
    new_cols = {}
    for event in tqdm(events_to_analyze, desc=progress_desc):
        event_data_sub = event_data.filter(pl.col(event_col) == event).to_pandas()
        tt_series, event_series = map_time_to_event(
            event_data_sub, cohort_pd, 'DFCI_MRN', event, time_col
        )
        new_cols[f'tt_{event}'] = tt_series.to_numpy()
        new_cols[event] = event_series.to_numpy()
    return cohort_df.with_columns([pl.Series(name, values) for name, values in new_cols.items()])


def _add_metastatic_events(cohort_df: pl.DataFrame, anchor: str = DEFAULT_ANCHOR) -> tuple[pl.DataFrame, list[str]]:
    dfs_to_concat = [
        pl.read_csv(os.path.join(PROCESSED_DATA_PATH, f'clinical_to_{site}_met.csv'))
        .filter(pl.col('event') == 1)
        .select(['dfci_mrn', 'date', 'type'])
        for site in MET_SITES
    ]
    met_date_df = pl.concat(dfs_to_concat, how='vertical')
    met_date_df = met_date_df.rename({'dfci_mrn': 'DFCI_MRN', 'date': 'MET_DATE', 'type': 'MET_LOCATION'})

    anchor_date_col = date_col(anchor)
    cohort_mrn_set = set(cohort_df['DFCI_MRN'].to_list())
    met_date_df = met_date_df.filter(pl.col('DFCI_MRN').is_in(cohort_mrn_set))

    mrn_anchor_dict = dict(
        zip(
            cohort_df['DFCI_MRN'].to_list(),
            pd.to_datetime(cohort_df[anchor_date_col].to_pandas(), errors='coerce'),
        )
    )
    met_date_pd = met_date_df.to_pandas()
    met_date_pd[anchor_date_col] = met_date_pd['DFCI_MRN'].map(mrn_anchor_dict)
    met_date_pd['MET_DATE'] = pd.to_datetime(met_date_pd['MET_DATE'].astype(str).str.split(' ').str[0], errors='coerce')
    met_date_pd['TIME_TO_MET'] = (met_date_pd['MET_DATE'] - met_date_pd[anchor_date_col]).dt.days
    met_date_pd = met_date_pd.dropna(subset=['TIME_TO_MET'])
    # Matches the ICD3/ICD4/phecode paths below (e.g. `icd_data_base['TIME_TO_ICD'] > 0`):
    # a met event recorded before the anchor date is not a valid post-baseline event.
    met_date_pd = met_date_pd.loc[met_date_pd['TIME_TO_MET'] > 0]
    met_date_df = pl.from_pandas(met_date_pd)

    met_events_added = []
    cohort_pd = cohort_df.to_pandas()
    new_cols = {}
    for met_loc in sorted(v for v in met_date_df['MET_LOCATION'].unique().to_list() if v is not None):
        # Emit the 'M' suffix at write time (e.g. 'brain' -> 'brainM') so the
        # event name matches what survival/preprocessing.py and
        # slurm_array_utils.py already expect, instead of patching it in
        # downstream consumers.
        event_name = f'{met_loc}M'
        cur_met_data_sub = met_date_df.filter(pl.col('MET_LOCATION') == met_loc).to_pandas()
        tt_series, event_series = map_time_to_event(
            cur_met_data_sub, cohort_pd, 'DFCI_MRN', met_loc, 'TIME_TO_MET'
        )
        new_cols[f'tt_{event_name}'] = tt_series.to_numpy()
        new_cols[event_name] = event_series.to_numpy()
        met_events_added.append(event_name)

    if new_cols:
        # Same positional safety as `_map_events_to_columns`: Series come from
        # `map_time_to_event(..., cohort_pd, ...)`, aligned row-for-row with
        # `cohort_df` as passed in (no independent reorder/filter in between).
        cohort_df = cohort_df.with_columns([pl.Series(name, values) for name, values in new_cols.items()])

    return cohort_df, met_events_added


def _write_outputs(
    cohort_df: pl.DataFrame,
    endpoint_events: list[str],
    surv_filename: str,
    embedding_filename: str,
    pooled_embedding_df: pl.DataFrame,
) -> None:
    event_cols = [event for event in endpoint_events if event in cohort_df.columns]
    event_cols = _dedupe_in_order(event_cols)
    tt_event_cols = [f'tt_{event}' for event in event_cols]

    # Always include core survival events (death, vte) so downstream scripts can use them
    core_cols = [c for c in CORE_EVENT_COLS if c in cohort_df.columns]
    events_data_sub = cohort_df.select(BASE_OUTPUT_COLS + core_cols + event_cols + tt_event_cols)
    events_data_sub.write_parquet(os.path.join(SURV_PATH, surv_filename))

    monthly_data = events_data_sub.join(pooled_embedding_df, on='DFCI_MRN', how='left')
    embedding_cols = [col for col in pooled_embedding_df.columns if col != 'DFCI_MRN']
    monthly_data = monthly_data.drop_nulls(subset=embedding_cols)
    monthly_data.write_parquet(os.path.join(SURV_PATH, embedding_filename))


def _write_death_met_outputs(
    base_cohort_df: pl.DataFrame,
    pooled_embedding_df: pl.DataFrame,
    surv_filename: str = 'death_met_surv_df.parquet',
    embedding_filename: str = 'death_met_embedding_prediction_df.parquet',
    min_events: int = 100,
    anchor: str = DEFAULT_ANCHOR,
) -> None:
    cohort_df = base_cohort_df.clone()
    cohort_df, met_events_added = _add_metastatic_events(cohort_df, anchor=anchor)

    met_events_added = _filter_endpoint_events_by_min_post_baseline_count(
        cohort_df, met_events_added, min_events=min_events
    )

    event_cols = ['death'] + [event for event in met_events_added if event in cohort_df.columns]
    event_cols = _dedupe_in_order(event_cols)
    tt_event_cols = [f'tt_{event}' for event in event_cols]

    core_cols = [c for c in CORE_EVENT_COLS if c in cohort_df.columns and c not in event_cols and c not in tt_event_cols]
    events_data_sub = cohort_df.select(BASE_OUTPUT_COLS + core_cols + event_cols + tt_event_cols)
    events_data_sub.write_parquet(os.path.join(SURV_PATH, surv_filename))

    monthly_data = events_data_sub.join(pooled_embedding_df, on='DFCI_MRN', how='left')
    embedding_cols = [col for col in pooled_embedding_df.columns if col != 'DFCI_MRN']
    monthly_data = monthly_data.drop_nulls(subset=embedding_cols)
    monthly_data.write_parquet(os.path.join(SURV_PATH, embedding_filename))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--anchor", choices=["treatment", "sequencing"], default=DEFAULT_ANCHOR,
        help="Time-zero anchor (see anchors.py). Default: treatment.",
    )
    args = parser.parse_args()
    anchor = ensure_anchor(args.anchor)

    # Shared input loading (once)
    base_cohort_df, split_ehr_icd_subset, embeddings_data, notes_meta = _load_shared_inputs(anchor=anchor)

    # Pool embeddings once; merge into each endpoint-specific survival table below.
    pooled_embedding_df = generate_survival_embedding_df(
        notes_meta=notes_meta,
        survival_df=None,
        embedding_array=embeddings_data,
        note_types=NOTE_TYPES,
        note_timing_col=note_time_col(anchor),
        continuous_window=False,
        pool_fx={key: 'time_decay_mean' for key in NOTE_TYPES},
        decay_param=0.01,
    )
    n_any_text = len(pooled_embedding_df)
    pooled_embedding_df = pooled_embedding_df.dropna()
    print(
        "  Complete-case text cohort (Clinician + Imaging + Pathology): "
        f"{len(pooled_embedding_df)}/{n_any_text} patients retained"
    )
    if pooled_embedding_df.empty:
        raise ValueError("No patients have complete pre-anchor embeddings for all note types.")
    pooled_embedding_df = pl.from_pandas(pooled_embedding_df)

    # =========================
    # SHARED DEATH + MET DATASET
    # =========================
    _write_death_met_outputs(
        base_cohort_df=base_cohort_df,
        pooled_embedding_df=pooled_embedding_df,
        surv_filename=_suffixed('death_met_surv_df.parquet', anchor),
        embedding_filename=_suffixed('death_met_embedding_prediction_df.parquet', anchor),
        anchor=anchor,
    )

    # =========================
    # SHARED ICD SETUP
    # =========================
    cohort_mrns = set(base_cohort_df['DFCI_MRN'].to_list())

    icd_data_base = split_ehr_icd_subset.clone()
    icd_data_base = icd_data_base.with_columns([
        pl.col('DIAGNOSIS_ICD10_CD').map_elements(_to_icd10_level_3, return_dtype=pl.Utf8).alias('ICD10_LEVEL_3_CD'),
        pl.col('DIAGNOSIS_ICD10_CD').map_elements(_to_icd10_level_4, return_dtype=pl.Utf8).alias('ICD10_LEVEL_4_CD'),
    ])
    icd_data_base = icd_data_base.drop_nulls(subset=['ICD10_LEVEL_3_CD'])
    icd_data_base = icd_data_base.filter(
        ~pl.col('ICD10_LEVEL_3_CD').map_elements(_is_excluded_icd10, return_dtype=pl.Boolean)
    )
    start_dt_pd = pd.to_datetime(icd_data_base['START_DT'].to_pandas(), errors='coerce')
    icd_data_base = icd_data_base.with_columns(pl.Series('START_DT', start_dt_pd))

    # =========================
    # ICD-10 LEVEL 3 DATASET (first post-treatment instance)
    # =========================
    icd3_codes_raw = _dedupe_in_order(icd_data_base['ICD10_LEVEL_3_CD'].to_list())
    icd3_codes = _prefilter_events_by_prevalence(
        icd_data_base, cohort_mrns, 'ICD10_LEVEL_3_CD', icd3_codes_raw, min_prevalence=0.01
    )
    print(f"  Level-3 ICD codes (>=1% prevalence): {len(icd3_codes)}")

    pl.DataFrame({'ICD10_LEVEL_3_CD': sorted(icd3_codes)}).write_csv(
        os.path.join(CODE_PATH, 'allowed_icd3_post_codes.csv')
    )

    cohort_df = base_cohort_df.clone()
    icd3_data = icd_data_base.filter(pl.col('TIME_TO_ICD') > 0)
    icd3_data = (
        icd3_data
        .sort(['DFCI_MRN', 'ICD10_LEVEL_3_CD', 'START_DT'])
        .unique(subset=['DFCI_MRN', 'ICD10_LEVEL_3_CD'], keep='first')
    )
    cohort_df = _map_events_to_columns(
        cohort_df=cohort_df,
        event_data=icd3_data,
        events_to_analyze=icd3_codes,
        event_col='ICD10_LEVEL_3_CD',
        time_col='TIME_TO_ICD',
        progress_desc='Generating level-3 ICD events (first_post_treatment)',
    )
    kept = _filter_endpoint_events_by_min_post_baseline_count(cohort_df, icd3_codes, min_events=100)
    _write_outputs(
        cohort_df=cohort_df,
        endpoint_events=kept,
        surv_filename=_suffixed('level_3_ICD_post_surv_df.parquet', anchor),
        embedding_filename=_suffixed('level_3_ICD_post_embedding_prediction_df.parquet', anchor),
        pooled_embedding_df=pooled_embedding_df,
    )

    # =========================
    # ICD-10 LEVEL 4 DATASET (first post-treatment instance)
    # =========================
    icd4_data_base = icd_data_base.drop_nulls(subset=['ICD10_LEVEL_4_CD'])
    icd4_codes_raw = _dedupe_in_order(icd4_data_base['ICD10_LEVEL_4_CD'].to_list())
    icd4_codes = _prefilter_events_by_prevalence(
        icd4_data_base, cohort_mrns, 'ICD10_LEVEL_4_CD', icd4_codes_raw, min_prevalence=0.01
    )
    print(f"  Level-4 ICD codes (>=1% prevalence): {len(icd4_codes)}")

    pl.DataFrame({'ICD10_LEVEL_4_CD': sorted(icd4_codes)}).write_csv(
        os.path.join(CODE_PATH, 'allowed_icd4_post_codes.csv')
    )

    cohort_df = base_cohort_df.clone()
    icd4_data = icd4_data_base.filter(pl.col('TIME_TO_ICD') > 0)
    icd4_data = (
        icd4_data
        .sort(['DFCI_MRN', 'ICD10_LEVEL_4_CD', 'START_DT'])
        .unique(subset=['DFCI_MRN', 'ICD10_LEVEL_4_CD'], keep='first')
    )
    cohort_df = _map_events_to_columns(
        cohort_df=cohort_df,
        event_data=icd4_data,
        events_to_analyze=icd4_codes,
        event_col='ICD10_LEVEL_4_CD',
        time_col='TIME_TO_ICD',
        progress_desc='Generating level-4 ICD events (first_post_treatment)',
    )
    kept = _filter_endpoint_events_by_min_post_baseline_count(cohort_df, icd4_codes, min_events=100)
    _write_outputs(
        cohort_df=cohort_df,
        endpoint_events=kept,
        surv_filename=_suffixed('level_4_ICD_post_surv_df.parquet', anchor),
        embedding_filename=_suffixed('level_4_ICD_post_embedding_prediction_df.parquet', anchor),
        pooled_embedding_df=pooled_embedding_df,
    )

    # =========================
    # PHECODE DATASET (first post-treatment instance)
    # =========================
    mapping_file = os.path.join(CODE_PATH, 'icd10_to_phecode_mapping.csv')
    mapping_df = pl.read_csv(mapping_file)
    mapping_icd_col = _resolve_column(mapping_df, 'icd10_code')
    mapping_phecode_col = _resolve_column(mapping_df, 'phecode')
    mapping_df = mapping_df.with_columns([
        pl.col(mapping_icd_col).map_elements(_normalize_icd10_undotted, return_dtype=pl.Utf8).alias('ICD10_NORM'),
        pl.col(mapping_phecode_col).map_elements(_normalize_phecode, return_dtype=pl.Utf8).alias('PHECODE'),
    ])
    mapping_df = mapping_df.drop_nulls(subset=['ICD10_NORM', 'PHECODE']).unique(subset=['ICD10_NORM', 'PHECODE'])

    phe_data = split_ehr_icd_subset.clone()
    phe_data = phe_data.with_columns(
        pl.col('DIAGNOSIS_ICD10_CD').map_elements(_normalize_icd10_undotted, return_dtype=pl.Utf8).alias('ICD10_NORM')
    )
    phe_data = phe_data.drop_nulls(subset=['ICD10_NORM'])
    phe_data = phe_data.join(mapping_df.select(['ICD10_NORM', 'PHECODE']), on='ICD10_NORM', how='inner')
    start_dt_pd = pd.to_datetime(phe_data['START_DT'].to_pandas(), errors='coerce')
    phe_data = phe_data.with_columns(pl.Series('START_DT', start_dt_pd))
    phe_data = phe_data.filter(~pl.col('PHECODE').map_elements(_is_excluded_phecode, return_dtype=pl.Boolean))

    phecode_codes_raw = _dedupe_in_order(phe_data['PHECODE'].drop_nulls().to_list())
    phecode_codes = _prefilter_events_by_prevalence(
        phe_data, cohort_mrns, 'PHECODE', phecode_codes_raw, min_prevalence=0.01
    )
    print(f"  Phecode codes (>=1% prevalence): {len(phecode_codes)}")

    pl.DataFrame({'PHECODE': sorted(phecode_codes)}).write_csv(
        os.path.join(CODE_PATH, 'allowed_phecode_post_codes.csv')
    )

    cohort_df = base_cohort_df.clone()
    phecode_data = phe_data.filter(pl.col('TIME_TO_ICD') > 0)
    phecode_data = (
        phecode_data
        .sort(['DFCI_MRN', 'PHECODE', 'START_DT'])
        .unique(subset=['DFCI_MRN', 'PHECODE'], keep='first')
    )
    cohort_df = _map_events_to_columns(
        cohort_df=cohort_df,
        event_data=phecode_data,
        events_to_analyze=phecode_codes,
        event_col='PHECODE',
        time_col='TIME_TO_ICD',
        progress_desc='Generating phecode events (first_post_treatment)',
    )
    kept = _filter_endpoint_events_by_min_post_baseline_count(cohort_df, phecode_codes, min_events=100)
    _write_outputs(
        cohort_df=cohort_df,
        endpoint_events=kept,
        surv_filename=_suffixed('phecode_post_surv_df.parquet', anchor),
        embedding_filename=_suffixed('phecode_post_embedding_prediction_df.parquet', anchor),
        pooled_embedding_df=pooled_embedding_df,
    )


if __name__ == "__main__":
    main()
