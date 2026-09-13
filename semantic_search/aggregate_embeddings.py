"""Stage 1: pool note embeddings into one 3-block feature vector per patient.

Writes one parquet per requested window under SEMANTIC_SEARCH_PATH/features/:

    concat   2304   clinician, imaging, and pathology means side by side

for:

    alltime        every note, no anchor
    pretreatment   notes strictly before first_treatment_date

Pooling is a plain unweighted mean (not the production `time_decay_mean`): this
arm asks what a patient's notes say on average, not what they said most recently.

A patient must have at least one finite embedding in every note-type block. The
three blocks remain separately named and are concatenated without averaging
across note types.

Run:
    python -m semantic_search.aggregate_embeddings [--windows alltime pretreatment]
                                                   [--overwrite] [--limit-mrns N]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from data.schema import assert_schema
from pipelines.biomarkers.biomarker_common import load_note_embeddings
from semantic_search.common import (
    DEFAULT_WINDOWS,
    NOTE_TIMING_COL,
    NOTE_TYPES,
    PATIENT_KEY,
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


def _pool_by_type(
    notes: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> pl.DataFrame:
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
        show_progress=progress_desc is not None,
        progress_desc=progress_desc,
    )


def _concat_space(pooled: pl.DataFrame) -> pl.DataFrame:
    """All three equal-width type blocks in the fixed ``NOTE_TYPES`` order."""
    blocks = []
    for note_type in NOTE_TYPES:
        prefix = f"{note_type.upper()}_EMBEDDING_"
        block = [c for c in pooled.columns if c.startswith(prefix)]
        block.sort(key=lambda column: int(column.rsplit("_", 1)[1]))
        if not block:
            raise ValueError(f"Pooled frame has no {prefix}* feature columns")
        blocks.append(block)
    widths = {len(block) for block in blocks}
    if len(widths) != 1:
        raise ValueError(f"Embedding blocks have unequal widths: {[len(b) for b in blocks]}")
    cols = [column for block in blocks for column in block]
    return pooled.select([PATIENT_KEY] + cols)


def build_spaces(
    notes: pl.DataFrame,
    embeddings: np.ndarray,
    *,
    progress_desc: str | None = None,
) -> dict[str, pl.DataFrame]:
    """The 3-block concatenated space for one already-windowed note selection.

    A patient missing any note type is complete-cased out because every output
    row must contain all three embedding-block means.
    """
    pooled = (
        _pool_by_type(notes, embeddings)
        if progress_desc is None
        else _pool_by_type(notes, embeddings, progress_desc=progress_desc)
    )
    concatenated = _concat_space(pooled)
    cols = [c for c in concatenated.columns if c != PATIENT_KEY]
    return {"concat": filter_finite_rows(concatenated, cols)}


def _note_counts(notes: pl.DataFrame, space: str) -> pl.DataFrame:
    """Notes per patient contributing to a given space."""
    if space != "concat":
        raise ValueError(f"Unknown feature space: {space}")
    return notes.group_by(PATIENT_KEY).len(name="n_notes")


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
    for window in tqdm(windows, desc="Embedding windows", unit="window"):
        notes = _select_notes(notes_meta, window)
        print(f"\n[{window}] {notes.height:,} notes, "
              f"{notes.get_column(PATIENT_KEY).n_unique():,} patients", flush=True)
        if notes.is_empty():
            print(f"  no notes in window {window!r}; skipping", flush=True)
            continue

        spaces = build_spaces(
            notes,
            embeddings,
            progress_desc=f"{window}: pooling patient/type groups",
        )
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
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    parser.add_argument("--overwrite", action="store_true",
                        help="Rebuild feature files that already exist.")
    parser.add_argument("--limit-mrns", type=int, default=None,
                        help="Debug: pool only the first N patients.")
    args = parser.parse_args()
    run(windows=args.windows, overwrite=args.overwrite, limit_mrns=args.limit_mrns)


if __name__ == "__main__":
    main()
