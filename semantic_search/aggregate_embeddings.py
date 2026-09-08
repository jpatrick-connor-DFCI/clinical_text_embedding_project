"""Stage 1: pool note embeddings into one feature vector per patient.

Writes 5 spaces x 2 windows = 10 parquet files under SEMANTIC_SEARCH_PATH/features/:

    clinician / imaging / pathology   768   one note type only
    concat                           2304   the three per-type means side by side
    merged                            768   every note pooled, NOTE_TYPE ignored

x

    alltime        every note, no anchor
    pretreatment   notes strictly before first_treatment_date

Pooling is a plain unweighted mean (not the production `time_decay_mean`): this
arm asks what a patient's notes say on average, not what they said most recently.

`merged` is a SECOND pooling pass over notes whose NOTE_TYPE has been overwritten
with a single literal -- NOT the average of the three per-type means.  The two
differ whenever a patient's note counts are unequal across types, which is almost
always: the second pass is note-weighted (a patient with 200 imaging and 5
pathology notes is imaging-dominated), the average-of-means would be
type-weighted.  Note-weighted is the intended "a note is a note" reading.

Run:
    python -m semantic_search.aggregate_embeddings [--windows alltime pretreatment]
                                                   [--overwrite] [--limit-mrns N]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from data.schema import assert_schema
from pipelines.biomarkers.biomarker_common import load_note_embeddings
from semantic_search.common import (
    MERGED_NOTE_TYPE,
    NOTE_TIMING_COL,
    NOTE_TYPES,
    PATIENT_KEY,
    SINGLE_TYPE_SPACES,
    SPACES,
    WINDOWS,
    ensure_dirs,
    feature_path,
    write_result,
)
from shared.polars_utils import filter_finite_rows
from survival import pool_embedding_series_vectorized

FEATURE_SUMMARY_COLUMNS = [
    "space", "window", "n_patients", "n_features",
    "median_notes_per_patient", "mean_notes_per_patient",
]

# pool_embedding_series_vectorized computes a per-note-type "% of notes before
# 2015" covariate by default.  That is a documentation-era confounder used by the
# supervised arm; it is not a semantic feature, so it is excluded from every
# space here to keep the feature blocks purely embedding dimensions.
NO_YEAR_ADJUSTMENT: list[str] = []


def _pretreatment_notes(notes_meta: pl.DataFrame) -> pl.DataFrame:
    """Notes strictly before the treatment anchor.

    Reproduces `generate_survival_embedding_df(..., continuous_window=False,
    max_note_window=0)`: filter to negative relative time, then re-centre (a
    no-op at max_note_window=0, kept explicit so the two stay comparable).
    Rows with a null timing value -- patients with no anchor date -- drop out
    of the filter, which is the intended behaviour.
    """
    return notes_meta.filter(pl.col(NOTE_TIMING_COL) < 0)


def _select_notes(notes_meta: pl.DataFrame, window: str) -> pl.DataFrame:
    if window == "alltime":
        return notes_meta
    if window == "pretreatment":
        return _pretreatment_notes(notes_meta)
    raise ValueError(f"Unknown window {window!r}")


def _pool_by_type(notes: pl.DataFrame, embeddings: np.ndarray) -> pl.DataFrame:
    """One row per patient, {TYPE}_EMBEDDING_{i} for all three note types.

    Called directly rather than through `generate_survival_embedding_df` because
    that wrapper asserts `max(timing) <= 0`, which the alltime window violates by
    construction.  With pool_fx='mean' the timing column is never read -- see
    pool_embedding_series_vectorized's 'mean' branch -- so the same call serves
    both windows and the two are guaranteed to differ only by row selection.
    """
    return pool_embedding_series_vectorized(
        notes,
        embeddings,
        note_types=NOTE_TYPES,
        note_timing_col=NOTE_TIMING_COL,
        pool_fx={nt: "mean" for nt in NOTE_TYPES},
        year_adj_cols=NO_YEAR_ADJUSTMENT,
    )


def _pool_merged(notes: pl.DataFrame, embeddings: np.ndarray) -> pl.DataFrame:
    """One row per patient, EMBEDDING_{i} pooled over every note type at once."""
    flattened = notes.with_columns(pl.lit(MERGED_NOTE_TYPE).alias("NOTE_TYPE"))
    pooled = pool_embedding_series_vectorized(
        flattened,
        embeddings,
        note_types=[MERGED_NOTE_TYPE],
        note_timing_col=NOTE_TIMING_COL,
        pool_fx={MERGED_NOTE_TYPE: "mean"},
        year_adj_cols=NO_YEAR_ADJUSTMENT,
    )
    prefix = f"{MERGED_NOTE_TYPE.upper()}_EMBEDDING_"
    return pooled.rename({
        c: c.replace(prefix, "EMBEDDING_") for c in pooled.columns if c.startswith(prefix)
    })


def _single_type_space(pooled: pl.DataFrame, note_type: str) -> pl.DataFrame:
    """Extract one note type's block and strip the type prefix from the names."""
    prefix = f"{note_type.upper()}_EMBEDDING_"
    cols = [c for c in pooled.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(
            f"No {prefix}* columns in the pooled frame. Have: {pooled.columns[:8]}..."
        )
    return pooled.select([PATIENT_KEY] + cols).rename({
        c: c.replace(prefix, "EMBEDDING_") for c in cols
    })


def _concat_space(pooled: pl.DataFrame) -> pl.DataFrame:
    """All three type blocks side by side, names left prefixed so the blocks
    stay distinguishable downstream."""
    cols = [c for c in pooled.columns if "_EMBEDDING_" in c]
    return pooled.select([PATIENT_KEY] + cols)


def build_spaces(notes: pl.DataFrame, embeddings: np.ndarray) -> dict[str, pl.DataFrame]:
    """All five feature spaces for one already-windowed note selection.

    Each space is independently complete-cased: a patient missing a note type is
    dropped from that type's space and from `concat`, but still appears in
    `merged` and in the spaces for the types they do have.
    """
    pooled = _pool_by_type(notes, embeddings)

    spaces: dict[str, pl.DataFrame] = {}
    for note_type in NOTE_TYPES:
        spaces[note_type.lower()] = _single_type_space(pooled, note_type)
    spaces["concat"] = _concat_space(pooled)
    spaces["merged"] = _pool_merged(notes, embeddings)

    for name, df in spaces.items():
        cols = [c for c in df.columns if c != PATIENT_KEY]
        spaces[name] = filter_finite_rows(df, cols)
    return spaces


def _note_counts(notes: pl.DataFrame, space: str) -> pl.DataFrame:
    """Notes per patient contributing to a given space."""
    if space in SINGLE_TYPE_SPACES:
        relevant = notes.filter(pl.col("NOTE_TYPE").str.to_lowercase() == space)
    else:
        relevant = notes
    return relevant.group_by(PATIENT_KEY).len(name="n_notes")


def _summarize(space: str, window: str, df: pl.DataFrame, notes: pl.DataFrame) -> dict:
    counts = _note_counts(notes, space).join(
        df.select(PATIENT_KEY), on=PATIENT_KEY, how="inner"
    )
    n_notes = counts.get_column("n_notes") if counts.height else None
    return {
        "space": space,
        "window": window,
        "n_patients": df.height,
        "n_features": len([c for c in df.columns if c != PATIENT_KEY]),
        "median_notes_per_patient": float(n_notes.median()) if n_notes is not None else float("nan"),
        "mean_notes_per_patient": float(n_notes.mean()) if n_notes is not None else float("nan"),
    }


def run(windows: list[str], overwrite: bool = False, limit_mrns: int | None = None) -> pl.DataFrame:
    ensure_dirs()

    wanted = [(s, w) for w in windows for s in SPACES]
    if not overwrite:
        missing = [(s, w) for s, w in wanted if not os.path.exists(feature_path(s, w))]
        if not missing:
            print("All requested feature files already exist; nothing to do "
                  "(pass --overwrite to rebuild).", flush=True)
            return pl.DataFrame(schema={c: pl.Utf8 for c in FEATURE_SUMMARY_COLUMNS})
        windows = sorted({w for _, w in missing}, key=WINDOWS.index)

    print("Loading note embeddings...", flush=True)
    notes_meta, embeddings = load_note_embeddings()
    print(f"  {notes_meta.height:,} notes, embedding array {embeddings.shape}", flush=True)

    if limit_mrns is not None:
        keep = notes_meta.get_column(PATIENT_KEY).unique(maintain_order=True).head(limit_mrns)
        notes_meta = notes_meta.filter(pl.col(PATIENT_KEY).is_in(keep))
        print(f"  --limit-mrns {limit_mrns}: {notes_meta.height:,} notes retained", flush=True)

    summary_rows: list[dict] = []
    for window in windows:
        notes = _select_notes(notes_meta, window)
        print(f"\n[{window}] {notes.height:,} notes, "
              f"{notes.get_column(PATIENT_KEY).n_unique():,} patients", flush=True)
        if notes.is_empty():
            print(f"  no notes in window {window!r}; skipping", flush=True)
            continue

        spaces = build_spaces(notes, embeddings)
        for space, df in spaces.items():
            path = feature_path(space, window)
            if os.path.exists(path) and not overwrite:
                print(f"  {space}: exists, skipping", flush=True)
            else:
                cols = [c for c in df.columns if c != PATIENT_KEY]
                assert_schema(df, f"{space}_{window}", [PATIENT_KEY] + cols,
                              key_col=PATIENT_KEY)
                df.write_parquet(path)
                print(f"  {space}: {df.height:,} patients x {len(cols)} features -> {path}",
                      flush=True)
            summary_rows.append(_summarize(space, window, df, notes))

    summary = pl.DataFrame(summary_rows).select(FEATURE_SUMMARY_COLUMNS) if summary_rows \
        else pl.DataFrame(schema={c: pl.Utf8 for c in FEATURE_SUMMARY_COLUMNS})
    write_result(summary, "feature_summary")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=WINDOWS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild feature files that already exist.")
    parser.add_argument("--limit-mrns", type=int, default=None,
                        help="Debug: pool only the first N patients.")
    args = parser.parse_args()
    run(windows=args.windows, overwrite=args.overwrite, limit_mrns=args.limit_mrns)


if __name__ == "__main__":
    main()
