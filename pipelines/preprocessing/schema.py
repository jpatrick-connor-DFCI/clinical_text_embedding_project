"""Small Polars schema checks used by preprocessing entry points."""

from __future__ import annotations

import polars as pl


def assert_schema(
    df: pl.DataFrame,
    name: str,
    required_cols: list[str],
    key_col: str | None = None,
    unique_key: bool = True,
) -> None:
    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}. Have: {df.columns}")
    if key_col is None:
        return
    values = df.get_column(key_col)
    if values.null_count():
        raise ValueError(f"{name}: key column '{key_col}' has {values.null_count()} null value(s).")
    if unique_key:
        duplicates = df.height - values.n_unique()
        if duplicates:
            raise ValueError(f"{name}: key column '{key_col}' has {duplicates} duplicate value(s).")
