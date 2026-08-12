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
