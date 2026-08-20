"""Polars helpers for numeric missing-value semantics."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def filter_finite_rows(df: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Keep rows whose selected columns cast to finite Float64 values."""
    if not columns:
        return df
    return df.filter(
        pl.all_horizontal([
            pl.col(c).cast(pl.Float64, strict=False).is_finite()
            for c in columns
        ])
    )


def finite_or_zero(column: str) -> pl.Expr:
    """Cast a numeric-like column and map null, NaN, and infinities to zero."""
    value = pl.col(column).cast(pl.Float64, strict=False)
    return pl.when(value.is_finite()).then(value).otherwise(0.0)


def to_pandas_via_numpy(df: pl.DataFrame):
    """polars -> pandas without going through Arrow.

    `DataFrame.to_pandas()` converts via Arrow record batches and hard-requires
    pyarrow, which is not installed in the cluster environment. Inside a marker
    screen that raised `ModuleNotFoundError: No module named 'pyarrow'` for every
    marker, `_safe_fit` swallowed it, and the screen wrote a header-only CSV.

    Per-column `Series.to_numpy()` keeps each column's own dtype; a single
    `df.to_numpy()` would upcast the whole frame to one common dtype (object,
    once a boolean or integer sits beside a float), which lifelines then chokes
    on. pandas is imported lazily so this module stays importable without it.
    """
    import pandas as pd

    return pd.DataFrame({name: series.to_numpy()
                         for name, series in zip(df.columns, df.get_columns())})
