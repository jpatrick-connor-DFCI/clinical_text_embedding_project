"""Preprocessing utilities for the embed surv utils package."""

import re

try:
    import icd10
except ImportError:  # Keep non-ICD utilities importable in minimal environments.
    icd10 = None
import numpy as np
import polars as pl

def clean_text(text: str) -> str:
    """
    Clean input text by normalizing spaces and removing unwanted characters.

    - Collapses multiple spaces into one.
    - Removes non-alphanumeric characters, while preserving common punctuation.

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text.
    """
    # Replace multiple spaces/tabs/newlines with a single space
    text = re.sub(r'\s\s+', ' ', text)

    # Keep letters, numbers, and common punctuation
    return re.sub(r'[^A-Za-z0-9 .,?!()/]+', ' ', text)

def deduplicate_texts(text_list: list[str]) -> list[str]:
    """
    Remove duplicate text entries while preserving original order.

    Args:
        text_list (list[str]): List of text strings.

    Returns:
        list[str]: Deduplicated list of text strings.
    """
    seen = set()
    deduped = []

    for txt in text_list:
        if txt not in seen:
            seen.add(txt)       # Mark text as seen
            deduped.append(txt) # Keep first occurrence

    return deduped

def find_icd_code(code: str) -> str:
    """
    Return the ICD-10 description if it exists, otherwise return the code itself.

    Args:
        code (str): ICD-10 code string.

    Returns:
        str: Description of the ICD-10 code or the code itself.
    """
    if icd10 is None or not isinstance(code, str):
        return code

    raw_code = code.strip().upper()
    undotted = re.sub(r"[^A-Z0-9]", "", raw_code)
    if not undotted:
        return code

    # Endpoint generation normally emits level-4 codes as E11.6, but older
    # result directories contain undotted E116-style names.  icd10-cm expects
    # the canonical dot after the three-character category.
    canonical = (
        f"{undotted[:3]}.{undotted[3:]}" if len(undotted) > 3 else undotted
    )
    for candidate in dict.fromkeys((canonical, raw_code)):
        if icd10.exists(candidate):
            return icd10.find(candidate).description
    return code

def find_icd_block_description(code: str) -> str | None:
    """
    Return the ICD-10 block description or custom labels for special codes.

    Custom labels include metastasis locations and death/VTE.

    Args:
        code (str): ICD-10 code or custom code string.

    Returns:
        str | None: Block description or custom label. None if not found.
    """
    if icd10 is not None and icd10.exists(code):
        try:
            return icd10.find(code).block_description
        except Exception:
            return None

    # Custom labels for metastasis
    metastasis_codes = ['brainM', 'boneM', 'adrenalM', 'liverM', 'lungM', 'nodeM', 'peritonealM']
    if code in metastasis_codes:
        return 'Metastasis'

    # Custom labels for death or VTE
    if code in ['death', 'vte']:
        return code

    return None

def map_time_to_event(df_events: pl.DataFrame, df_all: pl.DataFrame, mrn_col: str, event_col: str,
                      time_col: str, censor_time_col: str = 'tt_death') -> tuple[pl.Series, pl.Series]:
    """
    Map patient MRNs to time-to-event and event indicators for survival analysis.

    Args:
        df_events (pl.DataFrame): DataFrame with observed events.
        df_all (pl.DataFrame): DataFrame with all patients (including censored).
        mrn_col (str): Column name for patient identifiers (MRN).
        event_col (str): Column name indicating the event.
        time_col (str): Column name for time-to-event.
        censor_time_col (str, optional): Column name for censoring time. Defaults to 'tt_death'.

    Returns:
        tuple[pl.Series, pl.Series]:
            - Series mapping MRNs to time-to-event.
            - Series mapping MRNs to event indicator (1=event, 0=censored).
    """
    # Patients with observed events
    mrns_with_event = set(df_events[mrn_col].unique().to_list())

    # Time to event for observed events
    tt_events = (
        df_events.group_by(mrn_col).agg(pl.col(time_col).min().alias('_tt'))
    )
    tt_dict = dict(zip(tt_events[mrn_col].to_list(), tt_events['_tt'].to_list()))
    event_dict = {mrn: 1 for mrn in mrns_with_event}

    # Patients without event (censored)
    mrns_without_event = list(set(df_all[mrn_col].unique().to_list()) - mrns_with_event)
    censored = df_all.filter(pl.col(mrn_col).is_in(mrns_without_event)).select([mrn_col, censor_time_col])
    tt_dict.update(dict(zip(censored[mrn_col].to_list(), censored[censor_time_col].to_list())))
    event_dict.update({mrn: 0 for mrn in mrns_without_event})

    # Map dictionaries back to Series aligned with df_all
    mrn_list = df_all[mrn_col].to_list()
    tt_series = pl.Series(time_col, [tt_dict.get(mrn) for mrn in mrn_list])
    event_series = pl.Series(event_col, [event_dict.get(mrn) for mrn in mrn_list])
    return tt_series, event_series

def find_continuous_records_to_analyze(notes_meta: pl.DataFrame, note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                       gap_threshold: int = 2 * 365, max_note_window: int = 0) -> pl.DataFrame:
    """
    Identify continuous sequences of patient notes within a specified time window. Assume that time 0 is the beginning of the prediction window.

    A continuous segment is defined as consecutive notes where gaps do not exceed `gap_threshold`.

    Args:
        notes_meta (pl.DataFrame): Metadata for notes, must include 'DFCI_MRN' and note timing column.
        note_timing_col (str, optional): Column representing time relative to treatment start. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        gap_threshold (int, optional): Maximum allowed gap (in days) between consecutive notes in a segment. Defaults to 2*365.

    Returns:
        pl.DataFrame: Subset of notes_meta containing only continuous records within allowed windows.
    """

    notes_within_window = notes_meta.filter(pl.col(note_timing_col) < max_note_window)
    notes_within_window = notes_within_window.with_columns(
        (pl.col(note_timing_col) - max_note_window).alias(note_timing_col)
    )
    notes_within_window = notes_within_window.with_row_index('_row_id')
    rows_to_include = []

    # Process each patient separately
    for mrn_val in notes_within_window['DFCI_MRN'].unique().to_list():
        mrn_notes = (
            notes_within_window.filter(pl.col('DFCI_MRN') == mrn_val)
            .sort(note_timing_col)
        )
        note_times = mrn_notes[note_timing_col].to_numpy()
        row_ids = mrn_notes['_row_id'].to_numpy()
        if len(note_times) == 0:
            continue

        # Build continuous segments
        idx_series = []
        lbound = note_times[0]
        for i in range(1, len(note_times)):
            if (note_times[i] - note_times[i - 1]) >= gap_threshold:
                idx_series.append((lbound, note_times[i - 1]))
                lbound = note_times[i]
        idx_series.append((lbound, note_times[-1]))

        # Choose the last valid window within bounds
        chosen_window = None
        for lbound, ubound in idx_series:
            if ubound <= 0:
                chosen_window = (lbound, ubound)

        if chosen_window is not None:
            lbound, ubound = chosen_window
            # Include all notes within the chosen segment
            mask = (note_times >= lbound) & (note_times <= ubound)
            rows_to_include.extend(row_ids[mask].tolist())

    notes_within_window = notes_within_window.filter(pl.col('_row_id').is_in(rows_to_include)).drop('_row_id')

    # Sanity check to ensure no notes exceed allowed window
    if not notes_within_window.is_empty():
        max_time = notes_within_window[note_timing_col].max()
        if max_time > 0:
            raise ValueError(
                f'Found note with time {max_time} > max_note_window {0}. '
                'Check your NOTE_TIME_REL_FIRST_TREATMENT_START values.'
            )

    return notes_within_window

def pool_embedding_series_vectorized(meta_df: pl.DataFrame, embedding_array: np.ndarray, note_types: list[str],
                                     note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                     pool_fx: dict[str, str] | None = None, decay_param: float | None = None,
                                     year_adj_cols: list[str] = ['Imaging', 'Pathology']) -> pl.DataFrame:
    """
    Pool embeddings for each patient and note type using specified strategies.

    Supports pooling strategies:
        - 'recent': most recent note
        - 'mean': unweighted mean
        - 'time_decay_mean': weighted mean by time proximity

    Also optionally computes year-based adjustments for specified note types.

    Args:
        meta_df (pl.DataFrame): Metadata for notes, including embedding indices.
        embedding_array (np.ndarray): Array of embeddings (rows correspond to embeddings in meta_df).
        note_types (list[str]): Note types to include.
        note_timing_col (str, optional): Column representing note timing. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        pool_fx (dict[str, str] | None, optional): Pooling strategy per note type. Defaults to None (mean).
        decay_param (float | None, optional): Decay parameter for time-decayed pooling. Required if using decay strategies. Defaults to None.
        year_adj_cols (list[str], optional): Note types for which year-based adjustments are computed. Defaults to ['Imaging', 'Pathology'].

    Returns:
        pl.DataFrame: DataFrame with one row per patient and pooled embeddings per note type.
    """
    unique_mrns = meta_df['DFCI_MRN'].unique(maintain_order=True).to_list()
    embed_dim = embedding_array.shape[1]

    # Add year column for year-based adjustments
    meta_df = meta_df.with_columns(
        pl.col('NOTE_DATETIME').cast(pl.Utf8).str.split('-').list.get(0).cast(pl.Int64).alias('NOTE_YEAR')
    )

    # Initialize pooled embeddings and year adjustments
    pooled_embeddings = {nt: np.full((len(unique_mrns), embed_dim), np.nan) for nt in note_types}
    year_adjustments = {nt: np.full((len(unique_mrns), 1), np.nan) for nt in note_types if nt in year_adj_cols}

    # Map MRN to row index for efficient assignment
    mrn_to_idx = {mrn: i for i, mrn in enumerate(unique_mrns)}

    # Group by patient and note type
    grouped = meta_df.group_by(['DFCI_MRN', 'NOTE_TYPE'])
    for (mrn, note_type), group in grouped:
        if note_type not in note_types:
            continue
        idx = mrn_to_idx[mrn]
        embed_indices = group['EMBEDDING_INDEX'].to_numpy()
        note_times = group[note_timing_col].to_numpy()
        embeddings = embedding_array[embed_indices, :]

        strategy = pool_fx.get(note_type, 'mean') if pool_fx else 'mean'

        # Validate pooling strategy
        valid_strategies = {'recent', 'time_decay_mean', 'mean'}
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}.")

        # Apply pooling strategy
        if strategy == 'recent':
            recent_idx = np.argmax(note_times)
            pooled_embeddings[note_type][idx, :] = embeddings[recent_idx, :]
        elif strategy == 'time_decay_mean':
            if decay_param is None:
                raise ValueError("decay_param must be provided for 'time_decay_mean'")
            weights = np.exp(-decay_param * np.abs(note_times)).reshape(-1, 1)
            weights /= np.sum(weights) if np.sum(weights) > 0 else 1
            pooled_embeddings[note_type][idx, :] = np.nansum(weights * embeddings, axis=0)
        elif strategy == 'mean':
            pooled_embeddings[note_type][idx, :] = np.nanmean(embeddings, axis=0)

        # Year-based adjustment
        if note_type in year_adj_cols:
            year_adjustments[note_type][idx] = (group['NOTE_YEAR'] < 2015).to_numpy().mean()

    # Build the frame column-by-column so MRNs do not force the numeric arrays
    # through a single NumPy object matrix.  The old concatenate path inferred
    # every column as Polars Object when DFCI_MRN was object-typed, which then
    # made otherwise-numeric embedding columns impossible to cast or validate.
    output_columns: dict[str, object] = {'DFCI_MRN': unique_mrns}
    for nt in year_adj_cols:
        if nt in year_adjustments:
            output_columns[f'PERCENT_{nt.upper()}_NOTES_PRE_2015'] = year_adjustments[nt][:, 0]
    for nt in note_types:
        for i in range(embed_dim):
            output_columns[f'{nt.upper()}_EMBEDDING_{i}'] = pooled_embeddings[nt][:, i]

    return pl.DataFrame(output_columns)

def generate_survival_embedding_df(notes_meta: pl.DataFrame, survival_df: pl.DataFrame | None, embedding_array: np.ndarray,
                                   note_types: list[str], note_timing_col: str = 'NOTE_TIME_REL_FIRST_TREATMENT_START',
                                   max_note_window: int = 0, pool_fx: dict[str, str] | None = None, decay_param: float | None = None,
                                   continuous_window: bool = True) -> pl.DataFrame:
    """
    Generate a DataFrame of pooled embeddings optionally merged with survival outcomes.

    Args:
        notes_meta (pl.DataFrame): Note metadata including embedding indices and timing.
        survival_df (pl.DataFrame | None): Survival outcome data. Must include 'DFCI_MRN' or 'PATIENT_ID'. Can be None.
        embedding_array (np.ndarray): Array of note embeddings.
        note_types (list[str]): Note types to include.
        note_timing_col (str, optional): Column for note timing relative to treatment. Defaults to 'NOTE_TIME_REL_FIRST_TREATMENT_START'.
        pool_fx (dict[str, str] | None, optional): Pooling strategy per note type. Defaults to None.
        decay_param (float | None, optional): Decay parameter for time-decayed pooling. Defaults to None.
        max_note_window (int, optional): Maximum note timing. Defaults to 0.
        continuous_window (bool, optional): If True, include only continuous note sequences. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with pooled embeddings, optionally merged with survival data.
    """
    # Select notes based on window strategy
    if continuous_window:
        notes_to_include = find_continuous_records_to_analyze(
            notes_meta, note_timing_col=note_timing_col, max_note_window=max_note_window)
    else:
        notes_to_include = notes_meta.filter(pl.col(note_timing_col) < max_note_window)
        notes_to_include = notes_to_include.with_columns(
            (pl.col(note_timing_col) - max_note_window).alias(note_timing_col)
        )

    assert(notes_to_include[note_timing_col].max() <= 0)

    # Pool embeddings for selected notes
    pooled_embedding_df = pool_embedding_series_vectorized(notes_to_include, embedding_array, note_types,
                                                           note_timing_col=note_timing_col, pool_fx=pool_fx,
                                                           decay_param=decay_param)

    if survival_df is not None:
        # Standardize patient ID column
        if 'PATIENT_ID' in survival_df.columns:
            survival_df = survival_df.rename({'PATIENT_ID': 'DFCI_MRN'})

        if max_note_window != 0:
            tt_event_cols = [col for col in survival_df.columns if col.startswith('tt_')]
            survival_df = survival_df.with_columns(
                [(pl.col(c) - max_note_window).alias(c) for c in tt_event_cols]
            )

        # Merge pooled embeddings with survival outcomes
        merged_df = survival_df.join(pooled_embedding_df, on='DFCI_MRN', how='left')

        return merged_df
    else:
        return pooled_embedding_df
