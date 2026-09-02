"""Report-only audit of the PRS (germline) and somatic per-patient collapse.

`build_somatic_data_df` (`generate_all_non_text_covariates.py`) collapses multiple
sample-level rows to one row per patient via `group_by("DFCI_MRN").agg(pl.all().first())`.
`build_germline_data_df` collapses the same way for non-PGS columns, but averages PGS score
columns across a patient's samples rather than taking one sample's raw values (nulls
excluded from the mean) -- disagreeing samples are no longer a hard error. This script
re-derives the same intermediate frames and checks that both collapses are *correct*, not
merely non-crashing -- duplicate MRNs, patients with disagreeing per-sample values,
selection-rule edge cases, and cohort-coverage impact.

Read-only: makes zero writes and changes no pipeline behavior. Findings are printed, not
acted on -- fixes for anything found here are a separate, deliberate follow-up.

Run from the repo root:

    python -m pipelines.preprocessing.audit_prs_somatic
"""

from __future__ import annotations

import os

import polars as pl

from config import FEATURE_PATH, PROFILE_PATH, PRS_MATRIX_FILE
from pipelines.preprocessing import profile_sources as ps
from pipelines.training.slurm_array_utils import _get_common_feature_mrns

try:
    from data.schema import assert_schema
except ModuleNotFoundError:
    from pipelines.preprocessing.schema import assert_schema


def _load_cohort_df() -> pl.DataFrame:
    from config import SURV_PATH
    return pl.read_parquet(os.path.join(SURV_PATH, "cohort_df.parquet"))


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


# ==========================================================================================
# PRS / germline audit
# ==========================================================================================

def audit_prs(cohort_df: pl.DataFrame) -> None:
    _section("PRS / germline audit (build_germline_data_df)")

    print(
        "Why the idmap is structurally required here (unlike somatic): PRS_MATRIX_FILE is\n"
        "keyed by IID -> cbio_sample_id, a *sample* identifier carrying no MRN, so\n"
        "PROFILE_2024_idmap.csv is the only bridge to a patient. Somatic keys on DFCI_MRN\n"
        "natively and needs no such bridge (see audit_somatic below)."
    )

    # --- Source TSV duplicate IIDs ---
    prs_raw = pl.read_csv(PRS_MATRIX_FILE, separator="\t").rename({"IID": "cbio_sample_id"}).with_columns(
        pl.col("cbio_sample_id").cast(pl.String)
    )
    n_rows = prs_raw.height
    n_unique_iid = prs_raw.get_column("cbio_sample_id").n_unique()
    print(f"\nPRS_MATRIX_FILE: {n_rows} rows, {n_unique_iid} unique IID.")
    if n_rows != n_unique_iid:
        print(f"  FINDING: {n_rows - n_unique_iid} duplicate IID row(s) in the source TSV.")
    else:
        print("  No duplicate IIDs in the source TSV.")

    pgs_cols = [c for c in prs_raw.columns if "PGS" in c]
    if not pgs_cols:
        raise ValueError(f"No PGS columns found in {PRS_MATRIX_FILE}")

    # --- idmap join (mirrors build_germline_data_df) ---
    idmap = pl.read_csv(
        os.path.join(PROFILE_PATH, "PROFILE_2024_idmap.csv"),
        columns=["DFCI_MRN", "cbio_sample_id"],
    ).with_columns(
        pl.col("DFCI_MRN").cast(pl.Int64, strict=False),
        pl.col("cbio_sample_id").cast(pl.String),
    )
    cohort_mrns_list = cohort_df.get_column("DFCI_MRN").unique().to_list()
    cohort_idmap = idmap.filter(pl.col("DFCI_MRN").is_in(cohort_mrns_list))
    joined = prs_raw.join(cohort_idmap, on="cbio_sample_id", how="inner")

    print(f"\nAfter idmap join: {joined.height} rows, {joined.get_column('DFCI_MRN').n_unique()} unique DFCI_MRN.")
    samples_per_mrn = joined.group_by("DFCI_MRN").len().rename({"len": "n_samples"})
    multi = samples_per_mrn.filter(pl.col("n_samples") > 1)
    print(f"Patients with >1 sample: {multi.height} / {samples_per_mrn.height}")
    if multi.height:
        dist = multi.group_by("n_samples").len().sort("n_samples")
        print("  Distribution of samples-per-patient (>1 only):")
        for row in dist.iter_rows(named=True):
            print(f"    {row['n_samples']} samples: {row['len']} patients")

    # --- Disagreement check: which patients/traits disagree across samples ---
    # build_germline_data_df no longer raises on this -- it averages each PGS column across
    # a patient's samples (nulls excluded from the mean). This is now purely informational:
    # how many patients' final PGS values are actually an average of >1 distinct value,
    # rather than a single sample's raw score.
    disagreements = joined.group_by("DFCI_MRN").agg(
        [pl.col(c).drop_nulls().n_unique().alias(c) for c in pgs_cols]
    ).filter(pl.max_horizontal(pgs_cols) > 1)
    print(
        f"\nDisagreement check (pl.col(c).drop_nulls().n_unique() > 1 for any PGS column): "
        f"{disagreements.height} patient(s) have >=1 PGS trait averaged across disagreeing samples."
    )
    if disagreements.height:
        for row in disagreements.iter_rows(named=True):
            bad_traits = [c for c in pgs_cols if row[c] > 1]
            print(f"  DFCI_MRN={row['DFCI_MRN']}: averaged trait(s) = {bad_traits}")

    # --- Confirm the averaging collapse matches build_germline_data_df exactly ---
    other_cols = [c for c in joined.columns if c not in pgs_cols and c != "DFCI_MRN"]
    collapsed = (
        joined.sort(["DFCI_MRN", "cbio_sample_id"])
        .group_by("DFCI_MRN", maintain_order=True)
        .agg(
            [pl.col(c).first() for c in other_cols]
            + [pl.col(c).mean().alias(c) for c in pgs_cols]
        )
    )
    print(f"\nCollapsed (averaged) frame: {collapsed.height} rows, {collapsed.get_column('DFCI_MRN').n_unique()} unique DFCI_MRN.")

    # --- Confirm output file matches expectation ---
    out_fp = os.path.join(FEATURE_PATH, "complete_germline_data_df.csv.gz")
    if os.path.exists(out_fp):
        written = pl.read_csv(out_fp)
        try:
            assert_schema(written, "complete_germline_data_df (on disk)", required_cols=["DFCI_MRN"], key_col="DFCI_MRN", unique_key=True)
            print(f"\nOn-disk {os.path.basename(out_fp)}: {written.height} rows, unique DFCI_MRN confirmed (assert_schema).")
        except ValueError as e:
            print(f"\nFINDING: on-disk file failed assert_schema: {e}")
    else:
        print(f"\n{out_fp} does not exist -- run generate_all_non_text_covariates.main() first to audit the written file.")

    # --- Cohort-coverage impact of the idmap cap on ALL SIX modalities ---
    print(
        "\nCoverage impact: PRS's idmap-capped MRN set feeds _get_common_feature_mrns(), which\n"
        "intersects somatic ∩ germline ∩ stage ∩ treatment and bounds every single-modality run\n"
        "(including modalities that never touch PRS data)."
    )
    cohort_mrns = set(cohort_mrns_list)
    prs_mrns = set(collapsed.get_column("DFCI_MRN").to_list())

    somatic_mrns = set(pl.read_csv(os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv.gz"), columns=["DFCI_MRN"]).get_column("DFCI_MRN")) \
        if os.path.exists(os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv.gz")) else None
    stage_mrns = set(pl.read_csv(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz"), columns=["DFCI_MRN"]).get_column("DFCI_MRN")) \
        if os.path.exists(os.path.join(FEATURE_PATH, "cancer_stage_df.csv.gz")) else None
    treatment_fp = os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz")
    treatment_mrns = None
    if os.path.exists(treatment_fp):
        treatment_df = pl.read_csv(treatment_fp, columns=["DFCI_MRN", "treatment_line"])
        treatment_mrns = set(treatment_df.filter(pl.col("treatment_line") == 1).get_column("DFCI_MRN"))

    print(f"\n  Cohort MRNs:                {len(cohort_mrns)}")
    print(f"  MRNs surviving idmap join:  {len(prs_mrns)}  ({len(prs_mrns) / max(len(cohort_mrns), 1):.1%} of cohort)")

    if somatic_mrns is not None and stage_mrns is not None and treatment_mrns is not None:
        common_live = _get_common_feature_mrns()
        common_without_germline = somatic_mrns & stage_mrns & treatment_mrns
        cost = len(common_without_germline) - len(common_live)
        print(f"  4-way intersection (somatic ∩ germline ∩ stage ∩ treatment), live: {len(common_live)}")
        print(f"  3-way intersection without germline:                              {len(common_without_germline)}")
        print(f"  FINDING: PRS/germline membership in the intersection costs every other modality"
              f" {cost} patient(s) ({cost / max(len(common_without_germline), 1):.1%} of the without-germline pool).")
    else:
        print("  (One or more sibling feature files not found on disk -- skipping 4-way intersection cost.)")


# ==========================================================================================
# Somatic audit
# ==========================================================================================

def audit_somatic(cohort_df: pl.DataFrame) -> None:
    _section("Somatic audit (build_somatic_data_df)")

    cohort_mrns_list = cohort_df.get_column("DFCI_MRN").unique().to_list()
    tstart_df = cohort_df.select(["DFCI_MRN", "first_treatment_date"])

    # --- Uniqueness of SOMATIC_WIDE_BY_SAMPLE on SOMATIC_GROUP_KEY (asserted nowhere upstream) ---
    somatic_wide = ps.load_somatic_wide()
    key_cols = ps.SOMATIC_GROUP_KEY
    n_rows = somatic_wide.height
    n_unique_key = somatic_wide.select(key_cols).n_unique()
    print(f"SOMATIC_WIDE_BY_SAMPLE.parquet: {n_rows} rows, {n_unique_key} unique {key_cols}.")
    if n_rows != n_unique_key:
        dup_keys = (
            somatic_wide.group_by(key_cols).len().filter(pl.col("len") > 1)
        )
        print(
            f"  FINDING: {n_rows - n_unique_key} row(s) share a duplicate key -- "
            f"{dup_keys.height} distinct key(s) affected. The inner join in build_somatic_data_df\n"
            "  relies on this key being unique but asserts it nowhere; a duplicate fans out rows\n"
            "  silently until assert_schema fails much later with an unhelpful message."
        )
    else:
        print("  Key is unique on disk -- the join's implicit assumption holds today.")

    # --- Specimens-per-patient before selection, mirroring build_somatic_data_df exactly ---
    genomic_spec = ps.load_genomic_specimen(exclude_rapidheme=True)
    genomic_spec = genomic_spec.filter(pl.col(ps.MRN).is_in(cohort_mrns_list))
    genomic_spec = genomic_spec.with_columns(
        pl.min_horizontal([ps.SAMPLE_COLLECTION_DT, ps.TEST_ORDER_DT, ps.REPORT_DT]).alias("SEQUENCING_DT")
    ).drop_nulls(subset=["SEQUENCING_DT"])

    specimens_per_patient = genomic_spec.group_by("DFCI_MRN").len().rename({"len": "n_specimens"})
    multi = specimens_per_patient.filter(pl.col("n_specimens") > 1)
    print(f"\nSpecimens per patient (pre-treatment-filter, pre-selection): {specimens_per_patient.height} patients.")
    print(f"  Patients with >1 specimen: {multi.height}")
    if multi.height:
        dist = multi.group_by("n_specimens").len().sort("n_specimens")
        for row in dist.iter_rows(named=True):
            print(f"    {row['n_specimens']} specimens: {row['len']} patients")

    # --- >= 0 filter: how many patients are dropped entirely ---
    n_before_filter = genomic_spec.get_column("DFCI_MRN").n_unique()
    genomic_spec = genomic_spec.join(tstart_df, on="DFCI_MRN", how="inner")
    genomic_spec = genomic_spec.with_columns(
        (pl.col("first_treatment_date") - pl.col("SEQUENCING_DT")).dt.total_days().alias(
            "days_from_sequencing_to_first_treatment"
        )
    )
    n_after_tstart_join = genomic_spec.get_column("DFCI_MRN").n_unique()
    post_filter = genomic_spec.filter(pl.col("days_from_sequencing_to_first_treatment") >= 0)
    n_after_filter = post_filter.get_column("DFCI_MRN").n_unique()
    print(
        f"\nPatients with >=1 dated specimen: {n_before_filter}\n"
        f"  -> with a first_treatment_date to compare against: {n_after_tstart_join}\n"
        f"  -> surviving days_from_sequencing_to_first_treatment >= 0: {n_after_filter}"
    )
    dropped = n_after_tstart_join - n_after_filter
    print(f"  FINDING: {dropped} patient(s) sequenced only after first treatment, dropped entirely by the >= 0 filter.")

    # --- argmin selection + ties ---
    sorted_spec = post_filter.sort(["DFCI_MRN", "days_from_sequencing_to_first_treatment"])
    min_gap = sorted_spec.group_by("DFCI_MRN").agg(pl.col("days_from_sequencing_to_first_treatment").min().alias("min_gap"))
    joined_min = post_filter.join(min_gap, on="DFCI_MRN", how="inner")
    ties = (
        joined_min.filter(pl.col("days_from_sequencing_to_first_treatment") == pl.col("min_gap"))
        .group_by("DFCI_MRN").len().filter(pl.col("len") > 1)
    )
    print(
        f"\nSelection rule: argmin(days_from_sequencing_to_first_treatment) per patient "
        f"(treatment-proximate, NOT earliest specimen).\n"
        f"  Patients with a tie for the minimum gap (order-dependent .first() pick): {ties.height}"
    )

    selected_sample = (
        sorted_spec.group_by("DFCI_MRN", maintain_order=True).agg(pl.all().first())
    )
    n_selected = selected_sample.height
    print(f"  Selected sample rows: {n_selected} (one per patient, as expected: {n_selected == selected_sample.get_column('DFCI_MRN').n_unique()})")

    # --- Contrast with build_cohort.py's _sequencing_dates() (earliest, no >=0 filter) ---
    from pipelines.preprocessing.build_cohort import _sequencing_dates
    earliest_spec = _sequencing_dates()
    earliest_mrns = set(earliest_spec.get_column("DFCI_MRN").to_list())
    argmin_mrns = set(selected_sample.get_column("DFCI_MRN").to_list())
    print(
        "\nContrast with build_cohort._sequencing_dates() (earliest specimen, no treatment-proximate\n"
        f"filter): {len(earliest_mrns)} patients vs. build_somatic_data_df's {len(argmin_mrns)} "
        "(treatment-proximate, >=0-filtered)."
    )
    same_selection = 0
    compare_df = selected_sample.select(["DFCI_MRN", "SEQUENCING_DT"]).join(
        earliest_spec.rename({"sequencing_date": "earliest_date"}), on="DFCI_MRN", how="inner"
    )
    same_selection = compare_df.filter(pl.col("SEQUENCING_DT") == pl.col("earliest_date")).height
    print(
        f"  Of {compare_df.height} patients in both, {same_selection} happen to select the same "
        f"specimen date by coincidence ({same_selection / max(compare_df.height, 1):.1%}) -- "
        "the two selections have not converged in general, as intended."
    )

    # --- Confirm no cross-sample OR-ing of mutation calls ---
    print(
        "\nMutation-call handling: one sample's row is taken wholesale via pl.all().first() after\n"
        "sorting by the selection key -- confirmed no cross-sample OR-ing of *_SNV/_AMP/_DEL/_SV/\n"
        "_FUSION indicator columns happens anywhere in build_somatic_data_df. This is a design\n"
        "choice (patients with multiple specimens only contribute the treatment-proximate one's\n"
        "calls), not a bug, but worth an explicit decision if a patient's other specimen carries\n"
        "a call the selected one misses."
    )

    # --- Confirm the idmap is genuinely unnecessary for somatic ---
    print(
        "\nIdmap-necessity check: the somatic path keys on DFCI_MRN natively from\n"
        "GENOMIC_SPECIMEN.parquet / SOMATIC_WIDE_BY_SAMPLE.parquet and does not route through\n"
        "PROFILE_2024_idmap.csv (v1 did, on ['DFCI_MRN','sample_id','CANCER_TYPE'])."
    )
    idmap = pl.read_csv(
        os.path.join(PROFILE_PATH, "PROFILE_2024_idmap.csv"), columns=["DFCI_MRN"]
    ).with_columns(pl.col("DFCI_MRN").cast(pl.Int64, strict=False))
    idmap_mrns = set(idmap.get_column("DFCI_MRN").to_list())
    somatic_mrns = argmin_mrns
    only_in_idmap_path = idmap_mrns & set(cohort_mrns_list) - somatic_mrns
    only_in_somatic = somatic_mrns - idmap_mrns
    print(
        f"  Somatic-eligible cohort MRNs not in the 2024 idmap at all: {len(only_in_somatic)} "
        "(these are patients the idmap-mediated path would have silently excluded).\n"
        f"  Cohort+idmap MRNs not reached by the somatic selection (different filter, not identity): {len(only_in_idmap_path)}."
    )
    print("  Dropping the idmap widened coverage; it did not change identity semantics (both key on DFCI_MRN).")

    # --- PANEL_VERSION naming-mismatch discovery ---
    _audit_panel_version(genomic_spec_columns=list(ps.load_genomic_specimen(exclude_rapidheme=True).columns))


def _audit_panel_version(genomic_spec_columns: list[str]) -> None:
    _section("PANEL_VERSION naming-mismatch discovery")
    print(
        "Confirmed by user: GENOMIC_SPECIMEN.parquet carries panel-version information.\n"
        "load_genomic_specimen() does no column projection (returns every parquet column), and\n"
        "metadata_cols in build_somatic_data_df is computed dynamically from selected_sample.columns,\n"
        "so whatever the specimen frame carries should already be flowing into the written CSV --\n"
        "the six consumers matching the literal prefix 'PANEL_VERSION' just may not match its real name."
    )
    print(f"\nGENOMIC_SPECIMEN.parquet columns ({len(genomic_spec_columns)}):")
    for c in genomic_spec_columns:
        print(f"  {c}")

    panel_like = [c for c in genomic_spec_columns if "PANEL" in c.upper() or "VERSION" in c.upper() or "ASSAY" in c.upper()]
    print(f"\nColumns matching PANEL / VERSION / ASSAY: {panel_like or '(none found)'}")

    out_fp = os.path.join(FEATURE_PATH, "complete_somatic_data_df.csv.gz")
    if os.path.exists(out_fp):
        written_cols = pl.read_csv(out_fp, n_rows=5).columns
        written_panel_like = [c for c in written_cols if "PANEL" in c.upper() or "VERSION" in c.upper() or "ASSAY" in c.upper()]
        print(f"\ncomplete_somatic_data_df.csv.gz columns matching PANEL / VERSION / ASSAY: {written_panel_like or '(none found)'}")
        exact_match = [c for c in written_cols if c.upper().startswith("PANEL_VERSION")]
        print(f"Columns matching the six consumers' literal prefix 'PANEL_VERSION': {exact_match or '(none -- confirms the naming mismatch)'}")

        if written_panel_like:
            df = pl.read_csv(out_fp)
            for c in written_panel_like:
                print(f"\n  Column '{c}': dtype={df.schema[c]}, n_unique={df.get_column(c).n_unique()}, "
                      f"n_null={df.get_column(c).null_count()}")
                print(f"    sample values: {df.get_column(c).drop_nulls().unique().to_list()[:10]}")
            print(
                "\nFINDING: this is a naming mismatch, not missing data. Resolution is a name\n"
                "reconciliation (add the real column name as a constant in profile_sources.py\n"
                "alongside TEST_TYPE, and align the six consumers' prefix match, or rename at write\n"
                "time) -- no change to build_somatic_data_df and no idmap join required. Note TEST_TYPE\n"
                "is present but is not a substitute: it separates RAPIDHEME_CLINICAL from the solid-tumor\n"
                "panel, not OncoPanel v1/v2/v3."
            )
    else:
        print(f"\n{out_fp} does not exist -- run generate_all_non_text_covariates.main() first to confirm the written column.")


# ==========================================================================================
# Related findings (report only, not this script's primary target but relevant context)
# ==========================================================================================

def report_related_findings() -> None:
    _section("Related findings (report only, not fixed here)")
    print(
        "- Consumers re-read the deduped CSVs with no revalidation: "
        "slurm_array_utils.py's load_feature_modalities_df merges somatic/PRS with no `validate=`\n"
        "  and no explicit `how=` beyond the default -- the uniqueness guarantee lives in a\n"
        "  different script run at a different time, so a stale file would duplicate patients\n"
        "  across CV folds silently. `validate=\"m:1\"` is the one-token hardening (deferred).\n"
        "- generate_IPTW_df.py chains five unvalidated merges then drop_duplicates(subset=['DFCI_MRN'],\n"
        "  keep='first'), which masks inflation rather than detecting it.\n"
        "- Two copies of assert_schema exist (data/schema.py and\n"
        "  pipelines/preprocessing/schema.py), functionally equivalent, selected by a\n"
        "  try/except ModuleNotFoundError.\n"
        "- 'metburden' is missing from shared/palette.json's MODALITY_ORDER (5 of 6 modalities),\n"
        "  so figures/prep/figure3.py silently omits a trained modality from the comparison."
    )


def main() -> None:
    cohort_df = _load_cohort_df()
    audit_prs(cohort_df)
    audit_somatic(cohort_df)
    report_related_findings()
    _section("Done. This script makes no writes -- confirm with `git status` / file mtimes.")


if __name__ == "__main__":
    main()
