"""Compile IPTW biomarker results across all cohorts and PS models.

Reads per-scheme CSVs from IPTW_runs_*/ directories and produces:
  - track1_all_significant_hits.csv: all FDR-significant Track 1 (ICI-only) results
  - track2_all_significant_hits.csv: all FDR-significant Track 2 (interaction) results
  - cohort_patient_counts.csv: n_ICI and n_control per cancer type in each cohort
  - scheme_diagnostics_summary.csv: PS AUC, ESS, SMD summaries per scheme

Each specification (cohort x ps_model x weighting x cancer_type) is reported
separately. Downstream analysis should evaluate which specifications produce
consistent and interpretable results.

Notebook-ready: no argparse, output directory set via variable.
"""

import os
import re

import polars as pl

from config import BIOMARKER_PATH, MATCHED_COHORT_PATH

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
                match = re.match(r'(.+)_track[12]_', fname)
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
                # Diagnostics
                diag_path = os.path.join(run_path, f'{cancer_type}_diagnostics/')
                ess_file = os.path.join(diag_path, 'effective_sample_sizes.csv.gz')
                if os.path.isfile(ess_file):
                    ess = pl.read_csv(ess_file)
                    ess = ess.with_columns(pl.lit(cohort).alias('cohort'), pl.lit(ps_model).alias('ps_model'))
                    diag_rows.append(ess)

                # Track 1
                for weight in TRACK1_WEIGHTS:
                    fname = f'{cancer_type}_track1_{weight}_ICI_only.csv.gz'
                    fpath = os.path.join(run_path, fname)
                    if os.path.isfile(fpath):
                        df = pl.read_csv(fpath)
                        if 'significant_marker' not in df.columns:
                            sig = df.clear()
                        else:
                            sig = df.filter(pl.col('significant_marker') == True)
                        sig = sig.with_columns(
                            pl.lit(cohort).alias('cohort'),
                            pl.lit(ps_model).alias('ps_model'),
                            pl.lit(weight).alias('weight_type'),
                            pl.lit(cancer_type).alias('cancer_type'),
                        )
                        all_t1.append(sig)

                # Track 2
                for weight in TRACK2_WEIGHTS:
                    fname = f'{cancer_type}_track2_{weight}_interaction.csv.gz'
                    fpath = os.path.join(run_path, fname)
                    if os.path.isfile(fpath):
                        df = pl.read_csv(fpath)
                        if 'significant_predictive' not in df.columns:
                            sig = df.clear()
                        else:
                            sig = df.filter(pl.col('significant_predictive') == True)
                        sig = sig.with_columns(
                            pl.lit(cohort).alias('cohort'),
                            pl.lit(ps_model).alias('ps_model'),
                            pl.lit(weight).alias('weight_type'),
                            pl.lit(cancer_type).alias('cancer_type'),
                        )
                        all_t2.append(sig)

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
        cohort_file = os.path.join(MATCHED_COHORT_PATH, f'matched_cohort_{cohort}.csv.gz')
        if not os.path.isfile(cohort_file):
            print(f"  Cohort file not found: {cohort_file}")
            continue
        cdf = pl.read_csv(cohort_file)
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
            iptw_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{cohort}_{ps_model}.csv.gz')
            if not os.path.isfile(iptw_file):
                print(f"  IPTW file not found: {iptw_file}")
                continue
            idf = pl.read_csv(iptw_file)
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
                ess_file = os.path.join(run_path, f'{cancer_type}_diagnostics/effective_sample_sizes.csv.gz')
                if not os.path.isfile(ess_file):
                    continue
                ess = pl.read_csv(ess_file)
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
    cohort_counts.write_csv(os.path.join(OUTPUT_DIR, 'cohort_patient_counts.csv.gz'), compression='gzip')
    print(f"\nCohort patient counts ({len(cohort_counts)} rows):")
    with pl.Config(tbl_rows=-1):
        print(cohort_counts)

    # ================================================
    # 3. Diagnostics summary
    # ================================================
    if diag_rows:
        diag_df = pl.concat(diag_rows, how='diagonal_relaxed')
        diag_df.write_csv(os.path.join(OUTPUT_DIR, 'scheme_diagnostics_summary.csv.gz'), compression='gzip')
        print(f"\nDiagnostics summary saved ({len(diag_df)} rows)")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
