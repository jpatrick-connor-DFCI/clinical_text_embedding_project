"""Figure 2c/d (overall survival by stage and by text-risk quartile) per selected cancer type.

Same cohort definition as fig2_km_stage_vs_risk.csv (patients with a known major
stage and a held-out death text risk score from the full-cohort model), split by
SELECTED_CANCER_TYPES (shared/palette.json). Risk quartiles are recomputed within
each cancer type so every panel has four equal-frequency groups. Cancer types
below the thresholds (default 20 patients, 5 deaths) are omitted from the KM
table and recorded with a status in the C-index table.

Writes to FIGURE_DATA_DIR:
- fig2_km_stage_vs_risk_by_cancer.csv       cancer_type, DFCI_MRN, tt_death, death,
                                            text_risk_score, stage_group, stage_ordinal,
                                            risk_quartile
- fig2_stage_vs_risk_cindex_by_cancer.csv   cancer_type, predictor, cindex, n, n_events, status
"""

from __future__ import annotations

import argparse
import os

import polars as pl

from config import SURV_PATH
from figures.io import save_figure_data
from figures.prep.figure2 import (
    RISK_QUARTILE_LABELS, _safe_quantiles, _stage_vs_risk, _stage_vs_risk_cindex,
)
from figures.prep.within_cancer_joint import _selected_cancer_labels
from shared.palette import SELECTED_CANCER_TYPES

KM_SCHEMA = {
    "cancer_type": pl.String, "DFCI_MRN": pl.String, "tt_death": pl.Float64,
    "death": pl.Float64, "text_risk_score": pl.Float64, "stage_group": pl.String,
    "stage_ordinal": pl.Int64, "risk_quartile": pl.String,
}
CINDEX_SCHEMA = {
    "cancer_type": pl.String, "predictor": pl.String, "cindex": pl.Float64,
    "n": pl.Int64, "n_events": pl.Int64, "status": pl.String,
}


def stage_vs_risk_by_cancer(
    pooled: pl.DataFrame, cancer: pl.DataFrame, *, min_patients: int = 20, min_events: int = 5,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the pooled known-stage table by cancer type and re-cut risk quartiles."""
    pooled = pooled.with_columns(pl.col("DFCI_MRN").cast(pl.String).str.strip_chars())
    labelled = pooled.join(cancer, on="DFCI_MRN", how="inner", validate="1:1")
    km: list[pl.DataFrame] = []
    cindex: list[pl.DataFrame] = []
    for cancer_type in SELECTED_CANCER_TYPES:
        stratum = labelled.filter(pl.col("cancer_type") == cancer_type)
        n_events = int(stratum["death"].sum()) if stratum.height else 0
        if stratum.height < min_patients or n_events < min_events:
            status = "no_patients" if not stratum.height else "below_threshold"
            cindex.append(pl.DataFrame(
                [{"cancer_type": cancer_type, "predictor": p, "cindex": None,
                  "n": stratum.height, "n_events": n_events, "status": status}
                 for p in ("stage", "text_risk")], schema=CINDEX_SCHEMA))
            continue
        stratum = stratum.with_columns(
            _safe_quantiles(stratum["text_risk_score"], 4, RISK_QUARTILE_LABELS,
                            f"text_risk/{cancer_type}").alias("risk_quartile")
        )
        km.append(stratum.select(list(KM_SCHEMA)).cast(KM_SCHEMA))
        cindex.append(_stage_vs_risk_cindex(stratum).with_columns(
            pl.lit(cancer_type).alias("cancer_type"), pl.lit(n_events).alias("n_events"),
            pl.lit("ok").alias("status"),
        ).select(list(CINDEX_SCHEMA)).cast(CINDEX_SCHEMA))
    return (pl.concat(km) if km else pl.DataFrame(schema=KM_SCHEMA),
            pl.concat(cindex) if cindex else pl.DataFrame(schema=CINDEX_SCHEMA))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-patients", type=int, default=20)
    parser.add_argument("--min-events", type=int, default=5)
    args = parser.parse_args()
    surv_df = pl.read_parquet(os.path.join(SURV_PATH, "death_met_surv_df.parquet"))
    km, cindex = stage_vs_risk_by_cancer(
        _stage_vs_risk(surv_df), _selected_cancer_labels(),
        min_patients=args.min_patients, min_events=args.min_events,
    )
    save_figure_data(km, "fig2_km_stage_vs_risk_by_cancer.csv")
    save_figure_data(cindex, "fig2_stage_vs_risk_cindex_by_cancer.csv")


if __name__ == "__main__":
    main()
