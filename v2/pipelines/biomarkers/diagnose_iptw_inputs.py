"""Explain why an IPTW run produced empty result files, without re-running it.

Every Track 1/2 fit starts with

    df_fit_pl = filter_finite_rows(df.select(cols), cols)

where `cols` is `[tt_death, death, (PX_on_ICI,)] + base_vars + [marker]`.
`filter_finite_rows` casts each column to Float64 with `strict=False`, so a
single *non-numeric* column anywhere in `base_vars` casts to all-null, fails
`is_finite()`, and drops every row — for every marker, in every screen. The
fit then raises, `_safe_fit` catches it, `results` is empty, and the screen
writes a zero-row parquet. `compile_IPTW_results` reads those happily and
reports 0 significant hits.

Any other per-marker exception ends the same way — a missing pyarrow behind
`DataFrame.to_pandas()` did exactly this, in every screen including the
per-cancer-type ones. That is why the trial fits below run uncaught.

This script reproduces that column selection on the saved IPTW_df files and
reports where the chain breaks. Read-only; takes seconds.

Usage:
  python -m pipelines.biomarkers.diagnose_iptw_inputs

  # cap the per-marker support scans (the slow part) for a fast structural check;
  # same env vars and seeded sampling as run_IPTW_analysis, so this inspects the
  # same subset a smoke run screened
  IPTW_MAX_MARKERS=50 python -m pipelines.biomarkers.diagnose_iptw_inputs
"""

import os
import random
import traceback

import numpy as np
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
    resolve_marker_subset,
)

MUTATION_TAGS = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
NUMERIC = (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
           pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def _section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _report_run_dir(spec: str) -> None:
    """Row counts of what the last run actually wrote, not just byte sizes:
    a zero-row parquet still carries a schema and a footer, so it is a couple of
    kB on disk and `ls -l` looks plausible. Row counts come from the parquet
    metadata, so this stays cheap on the large result files."""
    run_path = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{spec}/')
    if not os.path.isdir(run_path):
        print(f"  {run_path}: not present")
        return
    for name in sorted(f for f in os.listdir(run_path) if f.endswith('.parquet')):
        path = os.path.join(run_path, name)
        try:
            n_rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        except (OSError, pl.exceptions.PolarsError) as exc:
            print(f"  {name:<60} UNREADABLE: {exc}")
            continue
        flag = "  <- EMPTY (no rows)" if n_rows == 0 else ""
        print(f"  {name:<60} {n_rows:>6} rows{flag}")


def _diagnose_spec(spec: str) -> None:
    _section(f"{spec}")

    input_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{spec}.parquet')
    if not os.path.isfile(input_file):
        print(f"  MISSING INPUT: {input_file}")
        return
    full_df = pl.read_parquet(input_file)
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
    n_markers_total = len(biomarker_cols)

    # The support scans below call marker_has_*_support once per marker, each a
    # full numpy pass over the frame -- ~1500 markers x 2 tracks x 4 specs, and
    # the slow part of this script. None of the structural checks (schema, rank,
    # empty model frame) depend on seeing every marker. IPTW_MAX_MARKERS /
    # IPTW_MARKER_FRACTION cap it, sharing the analysis script's env vars and its
    # seeded sampling, so the diagnosis inspects the same subset a smoke run
    # screened. Unset => every marker, exactly as before.
    biomarker_cols = resolve_marker_subset(biomarker_cols)
    subset_note = ("" if len(biomarker_cols) == n_markers_total
                   else f" (SUBSET: {len(biomarker_cols)} of {n_markers_total} -- "
                        f"support counts below are out of the subset)")
    print(f"  line dummies={len(line_cols)}, PANEL_VERSION_*={len(panel_cols)}, "
          f"CANCER_TYPE_*={len(cancer_type_cols)}, markers={n_markers_total}{subset_note}")

    # --- The actual trap: non-numeric columns that reach a model ---
    _section(f"{spec}: non-numeric columns")
    schema = full_df.schema
    non_numeric = [(c, schema[c]) for c in full_df.columns if schema[c] not in NUMERIC]
    if not non_numeric:
        print("  none")
    for col, dtype in non_numeric:
        print(f"  {col:<45} {dtype}")

    # pan_cancer base_vars, matching main(). The cancer-type dummies MUST come
    # from the merge's own returned list, not a fresh startswith() scan of the
    # frame: merge_rare_cancer_types_into_other leaves the reference level
    # (CANCER_TYPE_OTHER, all-zero once the rare types are folded in) *in* the
    # returned frame but *out* of its column list. Re-scanning puts the complete
    # dummy partition back, which is singular by construction -- so the
    # diagnostic reported a rank deficiency and failed every trial fit that
    # main() itself never hits. Fixing this is what the pipeline's own comment
    # at the ct_cols_fit assignment in main() warns about.
    type_df, pan_ct_cols, _merged = merge_rare_cancer_types_into_other(
        full_df.clone(), min_total=MIN_CANCER_TYPE_TOTAL)
    panel_cols_fit = [c for c in type_df.columns if c.upper().startswith('PANEL_VERSION_')]
    ct_cols_fit = pan_ct_cols
    base_vars = base_covars + line_cols + panel_cols_fit + ct_cols_fit

    # What the superseded substring filters would have added. Reported so a frame
    # still carrying the raw labels is visible even though they no longer reach a
    # model — and so this script does not keep flagging a bug that is fixed.
    swept = sorted(({c for c in type_df.columns if 'CANCER_TYPE' in c or 'PANEL' in c}
                    - set(panel_cols_fit) - set(ct_cols_fit)))
    if swept:
        print(f"\n  Raw label columns present but correctly excluded from base_vars: "
              f"{', '.join(swept)}")
        print("  (the superseded `'CANCER_TYPE' in c` / `'PANEL' in c` filters swept these in, "
              "which emptied every pan-cancer model frame)")

    missing = [c for c in base_vars if c not in type_df.columns]
    if missing:
        print(f"\n  >>> base_vars names absent from the frame: {missing}")
    offenders = [(c, type_df.schema[c]) for c in base_vars
                 if c in type_df.schema and type_df.schema[c] not in NUMERIC]
    print(f"\n  pan_cancer base_vars: {len(base_vars)} columns "
          f"({len(panel_cols_fit)} panel dummies, {len(ct_cols_fit)} cancer-type dummies)")
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

    # --- Design-matrix rank: the "matrix is singular" failure ---
    # An empty model frame is not the only way every fit in a screen can fail.
    # If the base covariates are linearly dependent, the Cox partial-likelihood
    # Hessian is singular and lifelines raises ConvergenceError identically for
    # every marker. The raw matrix can be full-rank while the Hessian is not, so
    # report both the rank deficiency and the near-constant / duplicated columns
    # that usually cause it.
    _section(f"{spec}: design-matrix rank on base_vars")
    fit_cols = ['PX_on_ICI'] + [c for c in base_vars if c in type_df.columns]
    finite_df = type_df.select(fit_cols).filter(
        pl.all_horizontal([pl.col(c).cast(pl.Float64, strict=False).is_finite()
                           for c in fit_cols]))
    X = finite_df.to_numpy().astype(float)
    if X.shape[0] == 0:
        print("  Model frame is empty; rank is undefined (see the section above).")
    else:
        rank = np.linalg.matrix_rank(X)
        print(f"  Design matrix: {X.shape[0]} rows x {X.shape[1]} cols, rank={rank}")
        if rank < X.shape[1]:
            print(f"  >>> RANK DEFICIENT by {X.shape[1] - rank}: "
                  f"the base covariates are linearly dependent.")
        else:
            print("  Full column rank (a singular Hessian can still come from "
                  "separation or near-collinearity below).")

        # Constant columns contribute nothing and can break the Hessian outright.
        const = [c for i, c in enumerate(fit_cols) if np.ptp(X[:, i]) == 0]
        if const:
            print(f"  CONSTANT columns ({len(const)}): {', '.join(const)}")

        # Exactly duplicated / complementary column pairs.
        dupes = []
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if np.array_equal(X[:, i], X[:, j]):
                    dupes.append(f"{fit_cols[i]} == {fit_cols[j]}")
                elif np.array_equal(X[:, i], 1 - X[:, j]):
                    dupes.append(f"{fit_cols[i]} == 1 - {fit_cols[j]}")
        if dupes:
            print(f"  DUPLICATED columns: {'; '.join(dupes)}")

        # Groups of dummies summing to a constant vector (a complete partition).
        dummy_like = [c for i, c in enumerate(fit_cols)
                      if set(np.unique(X[:, i])).issubset({0.0, 1.0})]
        for prefix in ('CANCER_TYPE_', 'PANEL_VERSION_', 'LINE_'):
            grp = [c for c in dummy_like if c.startswith(prefix)]
            if len(grp) < 2:
                continue
            sums = X[:, [fit_cols.index(c) for c in grp]].sum(axis=1)
            if np.ptp(sums) == 0:
                print(f"  >>> COMPLETE PARTITION: the {len(grp)} {prefix}* columns "
                      f"sum to {sums[0]:g} on every row — no reference level.")

        # Near-collinearity, which a plain rank check rounds away.
        with np.errstate(all='ignore'):
            sv = np.linalg.svd(X, compute_uv=False)
        if sv.size and sv[-1] > 0:
            print(f"  Condition number: {sv[0] / sv[-1]:.3e}"
                  f"{'   <- severe' if sv[0] / sv[-1] > 1e10 else ''}")
        tiny = [f"{fit_cols[i]}" for i in range(X.shape[1])
                if 0 < X[:, i].sum() < 5 and set(np.unique(X[:, i])).issubset({0.0, 1.0})]
        if tiny:
            print(f"  Near-empty dummies (<5 positives): {', '.join(tiny)}")

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
    scanned = ("" if len(biomarker_cols) == n_markers_total
               else f" scanned (of {n_markers_total} total)")
    print(f"  Track 2 markers with support: {len(t2)}/{len(biomarker_cols)}{scanned}")
    print(f"  Track 1 markers with support: {len(t1)}/{len(biomarker_cols)}{scanned}")

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
    # Same seed as run_IPTW_analysis.main(). resolve_marker_subset draws from the
    # global `random` state, so without this the marker cap here would select a
    # DIFFERENT subset than the smoke run being diagnosed -- and the whole point
    # of sharing IPTW_MAX_MARKERS is to inspect the same markers that ran.
    random.seed(42)
    for cohort in COHORTS:
        for ps_model in PS_MODELS:
            _diagnose_spec(f'{cohort}_{ps_model}')


if __name__ == "__main__":
    main()
