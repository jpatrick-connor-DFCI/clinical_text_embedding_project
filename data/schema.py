"""Build-time schema validation for pipeline stage outputs.

There is otherwise no build-time schema validation anywhere in the pipeline — the only
existing checks are `validate_cox_inputs` at train time. `assert_schema` is
meant to be called at the end of each preprocessing stage script, right
before (or after) writing its output file, so an upstream column rename or
missing key is caught at build time rather than several stages downstream.
"""

import polars as pl


def assert_schema(
    df: pl.DataFrame,
    name: str,
    required_cols: list[str],
    key_col: str | None = None,
    unique_key: bool = True,
) -> None:
    """Assert that `df` has all `required_cols` and, optionally, a unique key column.

    Args:
        df: The dataframe to validate.
        name: Human-readable name of the dataset, used in error messages.
        required_cols: Column names that must be present.
        key_col: If given, must be present and non-null in every row.
        unique_key: If True (default) and `key_col` is given, `key_col` must
            be unique across rows.

    Raises:
        ValueError: If any check fails.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}. Have: {list(df.columns)}")

    if key_col is not None:
        if key_col not in df.columns:
            raise ValueError(f"{name}: key column '{key_col}' not present.")
        if df.get_column(key_col).null_count() > 0:
            n_null = df.get_column(key_col).null_count()
            raise ValueError(f"{name}: key column '{key_col}' has {n_null} null value(s).")
        if unique_key:
            n_dupes = df.height - df.get_column(key_col).n_unique()
            if n_dupes:
                raise ValueError(f"{name}: key column '{key_col}' has {n_dupes} duplicate value(s).")
