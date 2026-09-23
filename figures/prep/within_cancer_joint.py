"""Refit the Figure 3 joint Cox model within each selected cancer type.

Unlike figures.prep.within_cancer, which evaluates the pooled models inside each
cancer type, this refits the joint model: per (scheme, event, cancer type), the
held-out modality risk scores are standardized within the cancer type and entered
together into a lifelines Cox model, with the same inputs, eligibility rules
(n >= 20, >= 5 events, >= 5 non-events, >= 2 non-constant modalities), variants
(unpenalized; ridge_0.01) and pathological-fit guard as fig3_joint_betas.csv.
Only SELECTED_CANCER_TYPES (shared/palette.json) are fitted.

Writes to FIGURE_DATA_DIR:
- fig3_within_cancer_joint_betas.csv  cancer_type + fig3_joint_betas.csv columns
- fig3_within_cancer_joint_fits.csv   scheme, event, cancer_type, n, n_events,
                                      n_modalities, status (fitted / not_fitted /
                                      no_patients); one row per attempted stratum
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from config import FEATURE_PATH
from figures.io import save_figure_data
from figures.prep.figure3 import (
    JOINT_BETA_COLUMNS, _fit_joint_cox, _joint_event_frames, _map_schemes,
)
from figures.prep.within_cancer import _load_cancer_types
from shared.palette import SELECTED_CANCER_TYPES

WITHIN_CANCER_JOINT_BETA_COLUMNS = ["cancer_type", *JOINT_BETA_COLUMNS]
FIT_SCHEMA = {
    "scheme": pl.String, "event": pl.String, "cancer_type": pl.String,
    "n": pl.Int64, "n_events": pl.Int64, "n_modalities": pl.Int64, "status": pl.String,
}
BETA_SCHEMA = {
    "cancer_type": pl.String, "scheme": pl.String, "event": pl.String,
    "fit_variant": pl.String, "modality": pl.String, "beta": pl.Float64,
    "se": pl.Float64, "hr": pl.Float64, "p_value": pl.Float64,
    "n": pl.Int64, "n_events": pl.Int64,
}


def _selected_cancer_labels() -> pl.DataFrame:
    """DFCI_MRN -> uppercase CANCER_GROUP code, restricted to the selected types."""
    cancer = _load_cancer_types(Path(FEATURE_PATH) / "cancer_type_df.csv.gz")
    return cancer.select(
        "DFCI_MRN", pl.col("cancer_type").str.to_uppercase().alias("cancer_type"),
    ).filter(pl.col("cancer_type").is_in(list(SELECTED_CANCER_TYPES)))


def within_cancer_joint_betas(
    scheme: str, cancer: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Joint Cox betas and per-stratum fit status for one scheme."""
    betas: list[dict[str, object]] = []
    fits: list[dict[str, object]] = []
    for event, merged, risk_cols in _joint_event_frames(scheme):
        merged = merged.with_columns(pl.col("DFCI_MRN").cast(pl.String).str.strip_chars())
        labelled = merged.join(cancer, on="DFCI_MRN", how="inner", validate="1:1")
        for cancer_type in SELECTED_CANCER_TYPES:
            stratum = labelled.filter(pl.col("cancer_type") == cancer_type)
            n_events = int(stratum[event].sum()) if stratum.height else 0
            rows = (_fit_joint_cox(stratum, risk_cols, scheme=scheme, event=event, tag=cancer_type)
                    if stratum.height else [])
            betas.extend({"cancer_type": cancer_type, **row} for row in rows)
            fits.append({
                "scheme": scheme, "event": event, "cancer_type": cancer_type,
                "n": stratum.height, "n_events": n_events,
                "n_modalities": len({row["modality"] for row in rows}),
                "status": ("fitted" if rows else "not_fitted") if stratum.height else "no_patients",
            })
    return (pl.DataFrame(betas, schema=BETA_SCHEMA).select(WITHIN_CANCER_JOINT_BETA_COLUMNS),
            pl.DataFrame(fits, schema=FIT_SCHEMA))


def prepare_within_cancer_joint() -> tuple[pl.DataFrame, pl.DataFrame]:
    cancer = _selected_cancer_labels()
    results = _map_schemes(lambda scheme: within_cancer_joint_betas(scheme, cancer),
                           "within-cancer betas")
    betas = [b for b, _ in results if not b.is_empty()]
    fits = [f for _, f in results if not f.is_empty()]
    return (pl.concat(betas) if betas else pl.DataFrame(schema=BETA_SCHEMA),
            pl.concat(fits) if fits else pl.DataFrame(schema=FIT_SCHEMA))


def main() -> None:
    betas, fits = prepare_within_cancer_joint()
    save_figure_data(betas, "fig3_within_cancer_joint_betas.csv")
    save_figure_data(fits, "fig3_within_cancer_joint_fits.csv")


if __name__ == "__main__":
    main()
