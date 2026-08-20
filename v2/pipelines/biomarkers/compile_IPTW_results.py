"""Compile IPTW biomarker results across all cohorts and PS models.

Reads per-scheme parquets from IPTW_runs_*/ directories — one
`{cancer_type}_results.parquet` and one `{cancer_type}_diagnostics.parquet` per
cancer type — and produces:
  - track1_all_significant_hits.csv: all FDR-significant Track 1 (ICI-only) results
  - track2_all_significant_hits.csv: all FDR-significant Track 2 (interaction) results
  - cohort_patient_counts.parquet: n_ICI and n_control per cancer type in each cohort
  - scheme_diagnostics_summary.parquet: PS AUC, ESS, SMD summaries per scheme

Each specification (cohort x ps_model x weighting x cancer_type) is reported
separately. Downstream analysis should evaluate which specifications produce
consistent and interpretable results.

Notebook-ready: no argparse, output directory set via variable.
"""

import os
import re

import polars as pl

from config import BIOMARKER_PATH, MATCHED_COHORT_PATH
from pipelines.biomarkers.run_IPTW_analysis import read_diagnostic_section

# ============================================================
# Configuration
# ============================================================
OUTPUT_DIR = os.path.join(BIOMARKER_PATH, 'compiled_results/')

COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']

# Track 1: ICI-only (ATE generalizability-weighted + unweighted)
TRACK1_WEIGHTS = ['unweighted', 'ATE']
# Track 2: full-cohort interaction (ATE + unweighted)
TRACK2_WEIGHTS = ['ATE', 'noIPTW']


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover cancer types from result filenames
    _cancer_types = set()
    for _c in COHORTS:
        for _p in PS_MODELS:
            _run = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{_c}_{_p}/')
            if not os.path.isdir(_run):
                continue
            for fname in os.listdir(_run):
                match = re.match(r'(.+)_results\.parquet$', fname)
                if match:
                    _cancer_types.add(match.group(1))
    CANCER_TYPES = sorted(_cancer_types)

    # ================================================
    # 1. Compile all significant hits
    # ================================================
    all_t1 = []
    all_t2 = []
    diag_rows = []

    for cohort in COHORTS:
        for ps_model in PS_MODELS:
            spec = f'{cohort}_{ps_model}'
            run_path = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{spec}/')

            if not os.path.isdir(run_path):
                print(f"  Skipping {spec}: directory not found")
                continue

            for cancer_type in CANCER_TYPES:
                # Diagnostics: one long-format file per cancer type; the cohort
                # section carries what effective_sample_sizes.parquet used to.
                diag_file = os.path.join(run_path, f'{cancer_type}_diagnostics.parquet')
                if os.path.isfile(diag_file):
                    ess = read_diagnostic_section(diag_file, 'cohort')
                    if not ess.is_empty():
                        diag_rows.append(ess.drop('key').with_columns(
                            pl.lit(cancer_type).alias('cancer_type'),
                            pl.lit(cohort).alias('cohort'),
                            pl.lit(ps_model).alias('ps_model'),
                        ))

                # Results: both tracks and all weightings share one file.
                results_file = os.path.join(run_path, f'{cancer_type}_results.parquet')
                if not os.path.isfile(results_file):
                    continue
                results = pl.read_parquet(results_file).with_columns(
                    pl.lit(cohort).alias('cohort'),
                    pl.lit(ps_model).alias('ps_model'),
                )

                # Track 1: ICI-only. `significant_marker` is null on track 2 rows,
                # and `== True` drops nulls, so the track filter is belt-and-braces.
                for weight in TRACK1_WEIGHTS:
                    rows = results.filter(
                        (pl.col('track') == 1) & (pl.col('weight_type') == weight))
                    if 'significant_marker' in rows.columns:
                        all_t1.append(rows.filter(pl.col('significant_marker') == True))

                # Track 2: full-cohort interaction.
                for weight in TRACK2_WEIGHTS:
                    rows = results.filter(
                        (pl.col('track') == 2) & (pl.col('weight_type') == weight))
                    if 'significant_predictive' in rows.columns:
                        all_t2.append(rows.filter(pl.col('significant_predictive') == True))

    t1 = pl.concat(all_t1, how='diagonal_relaxed') if all_t1 else pl.DataFrame()
    t2 = pl.concat(all_t2, how='diagonal_relaxed') if all_t2 else pl.DataFrame()

    print(f"Track 1 (ICI-only): {len(t1)} significant hits")
    print(f"Track 2 (interaction): {len(t2)} significant hits")

    t1.write_csv(os.path.join(OUTPUT_DIR, 'track1_all_significant_hits.csv'))
    t2.write_csv(os.path.join(OUTPUT_DIR, 'track2_all_significant_hits.csv'))

    # ================================================
    # 2. Cohort patient counts by cancer type
    # ================================================
    cohort_counts_rows = []

    for cohort in COHORTS:
        # --- Original matched cohort (before any filtering) ---
        cohort_file = os.path.join(MATCHED_COHORT_PATH, f'matched_cohort_{cohort}.parquet')
        if not os.path.isfile(cohort_file):
            print(f"  Cohort file not found: {cohort_file}")
            continue
        cdf = pl.read_parquet(cohort_file)
        ct_col = 'cancer_type' if 'cancer_type' in cdf.columns else 'CANCER_TYPE'
        for ct, grp in cdf.group_by(ct_col):
            ct = ct[0]
            n_ici = int(grp['PX_on_ICI'].sum())
            n_ctrl = len(grp) - n_ici
            cohort_counts_rows.append({
                'cancer_type': ct,
                'cohort': cohort,
                'ps_model': 'original_cohort',
                'n_ICI': n_ici,
                'n_control': n_ctrl,
                'n_total': len(grp),
            })

        # --- IPTW df counts per PS model (after common_source_df filtering) ---
        for ps_model in PS_MODELS:
            iptw_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{cohort}_{ps_model}.parquet')
            if not os.path.isfile(iptw_file):
                print(f"  IPTW file not found: {iptw_file}")
                continue
            idf = pl.read_parquet(iptw_file)
            cancer_type_cols = [c for c in idf.columns if c.startswith('CANCER_TYPE_')]
            for ct_c in cancer_type_cols:
                ct_name = ct_c.replace('CANCER_TYPE_', '')
                grp = idf.filter(pl.col(ct_c).cast(pl.Boolean))
                n_ici = int(grp['PX_on_ICI'].sum())
                n_ctrl = len(grp) - n_ici
                cohort_counts_rows.append({
                    'cancer_type': ct_name,
                    'cohort': cohort,
                    'ps_model': ps_model,
                    'n_ICI': n_ici,
                    'n_control': n_ctrl,
                    'n_total': len(grp),
                })
            # Also add pan-cancer totals
            n_ici = int(idf['PX_on_ICI'].sum())
            n_ctrl = len(idf) - n_ici
            cohort_counts_rows.append({
                'cancer_type': 'pan_cancer',
                'cohort': cohort,
                'ps_model': ps_model,
                'n_ICI': n_ici,
                'n_control': n_ctrl,
                'n_total': len(idf),
            })

        # --- Post-common-support-trimming counts from diagnostics ---
        for ps_model in PS_MODELS:
            spec = f'{cohort}_{ps_model}'
            run_path = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{spec}/')
            if not os.path.isdir(run_path):
                continue
            for cancer_type in CANCER_TYPES:
                diag_file = os.path.join(run_path, f'{cancer_type}_diagnostics.parquet')
                if not os.path.isfile(diag_file):
                    continue
                ess = read_diagnostic_section(diag_file, 'cohort')
                if ess.is_empty():
                    continue
                row = ess.row(0, named=True)
                cohort_counts_rows.append({
                    'cancer_type': cancer_type,
                    'cohort': cohort,
                    'ps_model': f'{ps_model}_trimmed',
                    'n_ICI': int(row['N_treated']),
                    'n_control': int(row['N_control']),
                    'n_total': int(row['N_treated'] + row['N_control']),
                })

    cohort_counts = pl.DataFrame(cohort_counts_rows)
    cohort_counts.write_parquet(os.path.join(OUTPUT_DIR, 'cohort_patient_counts.parquet'))
    print(f"\nCohort patient counts ({len(cohort_counts)} rows):")
    with pl.Config(tbl_rows=-1):
        print(cohort_counts)

    # ================================================
    # 3. Diagnostics summary
    # ================================================
    if diag_rows:
        diag_df = pl.concat(diag_rows, how='diagonal_relaxed')
        diag_df.write_parquet(os.path.join(OUTPUT_DIR, 'scheme_diagnostics_summary.parquet'))
        print(f"\nDiagnostics summary saved ({len(diag_df)} rows)")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
