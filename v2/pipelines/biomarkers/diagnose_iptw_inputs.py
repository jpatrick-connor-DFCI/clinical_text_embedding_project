"""Explain why an IPTW run produced empty result files, without re-running it.

Every Track 1/2 fit starts with

    df_fit_pl = filter_finite_rows(df.select(cols), cols)

where `cols` is `[tt_death, death, (PX_on_ICI,)] + base_vars + [marker]`.
`filter_finite_rows` casts each column to Float64 with `strict=False`, so a
single *non-numeric* column anywhere in `base_vars` casts to all-null, fails
`is_finite()`, and drops every row — for every marker, in every screen. The
fit then raises, `_safe_fit` catches it, `results` is empty, and the screen
writes a header-only CSV. `compile_IPTW_results` reads those happily and
reports 0 significant hits.

This script reproduces that column selection on the saved IPTW_df files and
reports where the chain breaks. Read-only; takes seconds.

Usage:
  python -m pipelines.biomarkers.diagnose_iptw_inputs
"""

import gzip
import os
import traceback

import polars as pl

from config import BIOMARKER_PATH
from pipelines.biomarkers.run_IPTW_analysis import (
    COHORTS,
    MIN_CANCER_TYPE_TOTAL,
    MIN_MARKER_NEG_PER_ARM,
    MIN_MARKER_POS_PER_ARM,
    MIN_EVENTS_PER_MARKER_GROUP,
    PS_MODELS,
    _fit_track1_marker,
    _fit_track2_marker,
    marker_has_ici_only_support,
    marker_has_within_arm_support,
    merge_rare_cancer_types_into_other,
)

MUTATION_TAGS = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
NUMERIC = (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _report_run_dir(spec: str) -> None:
    """Row counts of what the last run actually wrote, not just byte sizes:
    a gzipped header-only CSV is ~100 bytes, not 0, so `ls -l` looks plausible."""
    run_path = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{spec}/')
    if not os.path.isdir(run_path):
        print(f"  {run_path}: not present")
        return
    for root, _dirs, files in os.walk(run_path):
        for name in sorted(f for f in files if f.endswith('.csv.gz')):
            path = os.path.join(root, name)
            try:
                with gzip.open(path, 'rt') as handle:
                    n_rows = max(sum(1 for _ in handle) - 1, 0)
            except OSError as exc:
                print(f"  {os.path.relpath(path, run_path):<60} UNREADABLE: {exc}")
                continue
            flag = "  <- EMPTY (header only)" if n_rows == 0 else ""
            print(f"  {os.path.relpath(path, run_path):<60} {n_rows:>6} rows{flag}")


def _diagnose_spec(spec: str) -> None:
    _section(f"{spec}")

    input_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{spec}.csv.gz')
    if not os.path.isfile(input_file):
        print(f"  MISSING INPUT: {input_file}")
        return
    full_df = pl.read_csv(input_file)
    print(f"  IPTW_df: {full_df.height} patients x {full_df.width} columns")
    if full_df.is_empty():
        print("  >>> INPUT IS EMPTY. The break is upstream, in generate_IPTW_df.py.")
        return
    n_ici = int(full_df['PX_on_ICI'].sum())
    print(f"  PX_on_ICI: {n_ici} ICI / {full_df.height - n_ici} control")
    print(f"  deaths: {int(full_df['death'].sum())}")

    # --- Column groups, exactly as run_IPTW_analysis.main() builds them ---
    required_vars = ['DFCI_MRN', 'tt_death', 'death']
    base_covars = ['GENDER', 'AGE_AT_TREATMENTSTART']
    line_cols = sorted([c for c in full_df.columns if c.startswith('LINE_')])
    panel_cols = [c for c in full_df.columns if c.upper().startswith('PANEL_VERSION_')]
    cancer_type_cols = [c for c in full_df.columns if c.startswith('CANCER_TYPE_')]
    excluded = (required_vars + base_covars + line_cols + panel_cols +
                cancer_type_cols + ['PX_on_ICI', 'ICI_prediction'])
    biomarker_cols = [c for c in full_df.columns
                      if c not in excluded and any(t in c.upper() for t in MUTATION_TAGS)]
    print(f"  line dummies={len(line_cols)}, PANEL_VERSION_*={len(panel_cols)}, "
          f"CANCER_TYPE_*={len(cancer_type_cols)}, markers={len(biomarker_cols)}")

    # --- The actual trap: non-numeric columns that reach a model ---
    _section(f"{spec}: non-numeric columns")
    schema = full_df.schema
    non_numeric = [(c, schema[c]) for c in full_df.columns if schema[c] not in NUMERIC]
    if not non_numeric:
        print("  none")
    for col, dtype in non_numeric:
        print(f"  {col:<45} {dtype}")

    # pan_cancer base_vars, verbatim from main()
    type_df, _kept, _merged = merge_rare_cancer_types_into_other(
        full_df.clone(), min_total=MIN_CANCER_TYPE_TOTAL)
    panel_cols_fit = [c for c in type_df.columns if 'PANEL' in c]
    ct_cols_fit = [c for c in type_df.columns if 'CANCER_TYPE' in c]
    base_vars = base_covars + line_cols + panel_cols_fit + ct_cols_fit

    missing = [c for c in base_vars if c not in type_df.columns]
    if missing:
        print(f"\n  >>> base_vars names absent from the frame: {missing}")
    offenders = [(c, type_df.schema[c]) for c in base_vars
                 if c in type_df.schema and type_df.schema[c] not in NUMERIC]
    print(f"\n  pan_cancer base_vars: {len(base_vars)} columns "
          f"({len(panel_cols_fit)} matched by the loose 'PANEL' in c test)")
    if offenders:
        print("  >>> NON-NUMERIC base_vars — these empty every model frame:")
        for col, dtype in offenders:
            n_null = int(type_df[col].cast(pl.Float64, strict=False).is_null().sum())
            print(f"        {col:<41} {dtype}  ({n_null}/{type_df.height} cast to null)")
    else:
        print("  all base_vars are numeric")

    # Rows surviving the finite filter on base_vars alone, before any marker
    finite_mask = pl.all_horizontal([
        pl.col(c).cast(pl.Float64, strict=False).is_finite()
        for c in ['tt_death', 'death', 'PX_on_ICI'] + base_vars if c in type_df.columns
    ])
    n_surviving = type_df.filter(finite_mask).height
    print(f"\n  Rows surviving filter_finite_rows on base_vars: "
          f"{n_surviving}/{type_df.height}")
    if n_surviving == 0:
        print("  >>> THIS IS THE BUG: every model frame is empty before a marker is added.")

    # Per-column attribution, so the culprit is named even if several are numeric-but-null
    print("\n  Rows lost per base_var (dropped in isolation):")
    for col in ['tt_death', 'death', 'PX_on_ICI'] + base_vars:
        if col not in type_df.columns:
            continue
        kept = type_df.filter(
            pl.col(col).cast(pl.Float64, strict=False).is_finite()).height
        if kept < type_df.height:
            print(f"        {col:<41} keeps {kept}/{type_df.height}")

    # --- Marker support ---
    _section(f"{spec}: marker support (pan_cancer)")
    t2 = [m for m in biomarker_cols
          if marker_has_within_arm_support(type_df, m,
                                           min_pos_per_arm=MIN_MARKER_POS_PER_ARM,
                                           min_neg_per_arm=MIN_MARKER_NEG_PER_ARM,
                                           min_events_per_group=MIN_EVENTS_PER_MARKER_GROUP)]
    t1 = [m for m in biomarker_cols
          if marker_has_ici_only_support(type_df, m, min_pos=5,
                                         min_events=MIN_EVENTS_PER_MARKER_GROUP)]
    print(f"  Track 2 markers with support: {len(t2)}/{len(biomarker_cols)}")
    print(f"  Track 1 markers with support: {len(t1)}/{len(biomarker_cols)}")

    # --- One real fit, with the traceback _safe_fit would have swallowed ---
    _section(f"{spec}: trial fits (uncaught)")
    for label, markers, fit_fn in (("Track 2", t2, _fit_track2_marker),
                                   ("Track 1", t1, _fit_track1_marker)):
        if not markers:
            print(f"  {label}: no markers with support — nothing to fit")
            continue
        marker = markers[0]
        try:
            fit_fn(type_df, marker, base_vars, None)
            print(f"  {label}: {marker} fitted OK")
        except Exception:
            print(f"  {label}: {marker} RAISED — this is what _safe_fit hid:")
            print(traceback.format_exc())

    _section(f"{spec}: what the last run wrote")
    _report_run_dir(spec)


def main() -> None:
    for cohort in COHORTS:
        for ps_model in PS_MODELS:
            _diagnose_spec(f'{cohort}_{ps_model}')


if __name__ == "__main__":
    main()
