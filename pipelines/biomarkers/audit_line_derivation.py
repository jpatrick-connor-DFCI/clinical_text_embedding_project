"""Audit the PROFILE_DATA line-of-therapy derivation against ALL_MEDICATION_LINES.csv.

Read-only. Run once on the cluster before trusting the migrated biomarker
pipeline. This is the only remaining consumer of `config.MED_LINES_FILE`; no
pipeline stage reads it.

Gate for accepting the migration:
  - line-1 start dates agree within +/-7 days for the large majority of shared MRNs
  - HAS_ICI concordance kappa > 0.9
Below that, reconcile `profile_lines.ICI_DRUGS` against the discordant drug
names this report prints before running anything downstream.

Usage:
  python -m pipelines.biomarkers.audit_line_derivation
"""

import os

import polars as pl

from config import MED_CLASSES_FILE, MED_LINES_FILE, SURV_PATH
from pipelines.biomarkers.profile_lines import (
    LINE_WINDOW_DAYS,
    derive_lines_of_therapy,
    ici_concordance_report,
)
from pipelines.preprocessing import profile_sources as ps

WINDOW_SENSITIVITY = (14, LINE_WINDOW_DAYS, 60)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _cohens_kappa(both: pl.DataFrame, col_a: str, col_b: str) -> float:
    """Cohen's kappa for two binary columns over the same rows."""
    n = both.height
    if n == 0:
        return float("nan")
    agree = both.filter(pl.col(col_a) == pl.col(col_b)).height / n
    p_a = both.get_column(col_a).mean()
    p_b = both.get_column(col_b).mean()
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected >= 1.0:
        return float("nan")
    return (agree - expected) / (1 - expected)


def main() -> None:
    _section("Inputs")
    med_long = ps.unpivot_medications_summary()
    print(f"MEDICATIONS_SUMMARY (long): {med_long.height} drug rows, "
          f"{med_long['DFCI_MRN'].n_unique()} patients")

    reference = pl.read_csv(MED_LINES_FILE).rename({"MRN": "DFCI_MRN"})
    print(f"{MED_LINES_FILE}: {reference.height} rows, "
          f"{reference['DFCI_MRN'].n_unique()} patients")

    surv_df = pl.read_parquet(os.path.join(SURV_PATH, "death_met_surv_df.parquet"))
    cohort_mrns = surv_df["DFCI_MRN"].unique().to_list()
    print(f"death_met_surv_df cohort: {len(cohort_mrns)} patients")

    # ------------------------------------------------------------------
    _section("ICI definition concordance")
    med_classes = pl.read_csv(MED_CLASSES_FILE) if os.path.exists(MED_CLASSES_FILE) else None
    if med_classes is None:
        print(f"  (MED_CLASSES_FILE not found at {MED_CLASSES_FILE}; skipping MOA cross-tab)")
    ici_concordance_report(med_long, med_classes)

    # ------------------------------------------------------------------
    _section(f"Line-count sensitivity to window_days {WINDOW_SENSITIVITY}")
    for window in WINDOW_SENSITIVITY:
        lines = derive_lines_of_therapy(med_long, window_days=window)
        per_patient = lines.group_by("DFCI_MRN").agg(pl.len().alias("n_lines"))
        marker = "  <- default" if window == LINE_WINDOW_DAYS else ""
        print(f"  window={window:>3}d: {lines.height} lines, "
              f"median {per_patient['n_lines'].median()} / "
              f"mean {per_patient['n_lines'].mean():.2f} / "
              f"max {per_patient['n_lines'].max()} lines per patient{marker}")

    derived = derive_lines_of_therapy(med_long)

    # ------------------------------------------------------------------
    _section("Line 1 agreement with the reference file")
    derived_l1 = (
        derived.filter(pl.col("LINE") == 1)
        .select(
            "DFCI_MRN",
            pl.col("treatment_start_date").cast(pl.Date).alias("derived_start"),
            pl.col("HAS_ICI").cast(pl.Int64).alias("derived_ici"),
        )
    )
    reference_l1 = (
        reference.filter(pl.col("LINE") == 1)
        .with_columns(pl.col("MED_START_DT").str.to_date(strict=False).alias("ref_start"))
        .drop_nulls("ref_start")
        .group_by("DFCI_MRN")
        .agg(
            pl.col("ref_start").min(),
            pl.col("HAS_ICI").max().cast(pl.Int64).alias("ref_ici"),
        )
    )

    both = derived_l1.join(reference_l1, on="DFCI_MRN", how="inner")
    print(f"  MRNs with a line 1 in both sources: {both.height}")
    print(f"  Derived only: {derived_l1.height - both.height}, "
          f"reference only: {reference_l1.height - both.height}")

    if both.is_empty():
        print("  No shared MRNs — cannot audit. Check that both sources use DFCI_MRN.")
        return

    both = both.with_columns(
        (pl.col("derived_start") - pl.col("ref_start")).dt.total_days().abs().alias("abs_days")
    )
    for tol in (0, 7, 30):
        n = both.filter(pl.col("abs_days") <= tol).height
        label = "exact" if tol == 0 else f"+/-{tol}d"
        print(f"  Line-1 start agreement {label:>7}: {n}/{both.height} ({100 * n / both.height:.1f}%)")
    print(f"  |delta| days: median={both['abs_days'].median()}, "
          f"p90={both['abs_days'].quantile(0.9)}, max={both['abs_days'].max()}")

    # ------------------------------------------------------------------
    _section("HAS_ICI concordance at line 1")
    print(
        both.group_by(["derived_ici", "ref_ici"]).agg(pl.len().alias("n")).sort(
            ["derived_ici", "ref_ici"]
        )
    )
    kappa = _cohens_kappa(both, "derived_ici", "ref_ici")
    print(f"\n  Cohen's kappa: {kappa:.4f}   (gate: > 0.9)")
    print(f"  Derived ICI rate: {both['derived_ici'].mean():.4f}, "
          f"reference ICI rate: {both['ref_ici'].mean():.4f}")

    # ------------------------------------------------------------------
    _section("Discordant line-1 patients: what drugs are involved")
    for derived_flag, ref_flag, label in ((1, 0, "derived ICI, reference not"),
                                          (0, 1, "reference ICI, derived not")):
        discordant = both.filter(
            (pl.col("derived_ici") == derived_flag) & (pl.col("ref_ici") == ref_flag)
        )
        print(f"\n  {label}: {discordant.height} patients")
        if discordant.is_empty():
            continue
        drugs = (
            derived.filter(pl.col("LINE") == 1)
            .join(discordant.select("DFCI_MRN"), on="DFCI_MRN", how="inner")
            .get_column("drugs")
            .value_counts(sort=True)
            .head(20)
        )
        print(drugs)

    # ------------------------------------------------------------------
    _section("Cohort-restricted view (death_met_surv_df patients only)")
    in_cohort = both.filter(pl.col("DFCI_MRN").is_in(cohort_mrns))
    print(f"  Shared line-1 MRNs inside the cohort: {in_cohort.height}")
    if not in_cohort.is_empty():
        n7 = in_cohort.filter(pl.col("abs_days") <= 7).height
        print(f"  Start agreement +/-7d: {n7}/{in_cohort.height} "
              f"({100 * n7 / in_cohort.height:.1f}%)")
        print(f"  HAS_ICI kappa: {_cohens_kappa(in_cohort, 'derived_ici', 'ref_ici'):.4f}")


if __name__ == "__main__":
    main()
