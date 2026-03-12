"""Compile IPTW biomarker results across all cohorts and PS models.

Reads per-scheme CSVs from IPTW_runs_*/ directories and produces:
  - track1_all_significant_hits.csv: all FDR-significant Track 1 (ICI-only) results
  - track2_all_significant_hits.csv: all FDR-significant Track 2 (interaction) results
  - cohort_patient_counts.csv: n_ICI and n_control per cancer type in each cohort
  - scheme_diagnostics_summary.csv: PS AUC, ESS, SMD summaries per scheme

Usage:
  python compile_IPTW_results.py --output_dir /path/to/output
"""

import os
import re
import argparse
import pandas as pd
from biomarker_common import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Directory for compiled output')
args = parser.parse_args()

MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
os.makedirs(args.output_dir, exist_ok=True)

COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']

# Discover cancer types from result filenames
_cancer_types = set()
for _c in COHORTS:
    for _p in PS_MODELS:
        _run = os.path.join(MARKER_PATH, f'IPTW_runs_{_c}_{_p}/')
        if not os.path.isdir(_run):
            continue
        for fname in os.listdir(_run):
            match = re.match(r'(.+)_track[12]_', fname)
            if match:
                _cancer_types.add(match.group(1))
CANCER_TYPES = sorted(_cancer_types)

# Track 1: ICI-only (ATE generalizability-weighted + unweighted)
TRACK1_WEIGHTS = ['unweighted', 'ATE']
# Track 2: full-cohort interaction (ATE + unweighted)
TRACK2_WEIGHTS = ['ATE', 'noIPTW']

# ================================================
# 1. Compile all significant hits
# ================================================
all_t1 = []
all_t2 = []
diag_rows = []

for cohort in COHORTS:
    for ps_model in PS_MODELS:
        spec = f'{cohort}_{ps_model}'
        run_path = os.path.join(MARKER_PATH, f'IPTW_runs_{spec}/')

        if not os.path.isdir(run_path):
            print(f"  Skipping {spec}: directory not found")
            continue

        for cancer_type in CANCER_TYPES:
            # Diagnostics
            diag_path = os.path.join(run_path, f'{cancer_type}_diagnostics/')
            ess_file = os.path.join(diag_path, 'effective_sample_sizes.csv')
            if os.path.isfile(ess_file):
                ess = pd.read_csv(ess_file)
                ess['cohort'] = cohort
                ess['ps_model'] = ps_model
                diag_rows.append(ess)

            # Track 1
            for weight in TRACK1_WEIGHTS:
                fname = f'{cancer_type}_track1_{weight}_ICI_only.csv'
                fpath = os.path.join(run_path, fname)
                if os.path.isfile(fpath):
                    df = pd.read_csv(fpath)
                    sig = df[df.get('significant_marker', pd.Series(dtype=bool)) == True].copy()
                    sig['cohort'] = cohort
                    sig['ps_model'] = ps_model
                    sig['weight_type'] = weight
                    sig['cancer_type'] = cancer_type
                    all_t1.append(sig)

            # Track 2
            for weight in TRACK2_WEIGHTS:
                fname = f'{cancer_type}_track2_{weight}_interaction.csv'
                fpath = os.path.join(run_path, fname)
                if os.path.isfile(fpath):
                    df = pd.read_csv(fpath)
                    sig = df[df.get('significant_predictive', pd.Series(dtype=bool)) == True].copy()
                    sig['cohort'] = cohort
                    sig['ps_model'] = ps_model
                    sig['weight_type'] = weight
                    sig['cancer_type'] = cancer_type
                    all_t2.append(sig)

t1 = pd.concat(all_t1, ignore_index=True) if all_t1 else pd.DataFrame()
t2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()

print(f"Track 1 (ICI-only): {len(t1)} significant hits")
print(f"Track 2 (interaction): {len(t2)} significant hits")

t1.to_csv(os.path.join(args.output_dir, 'track1_all_significant_hits.csv'), index=False)
t2.to_csv(os.path.join(args.output_dir, 'track2_all_significant_hits.csv'), index=False)

# ================================================
# 2. Cohort patient counts by cancer type
# ================================================
COHORT_PATH = os.path.join(MARKER_PATH, 'matched_cohorts/')
cohort_counts_rows = []

for cohort in COHORTS:
    cohort_file = os.path.join(COHORT_PATH, f'matched_cohort_{cohort}.csv')
    if not os.path.isfile(cohort_file):
        print(f"  Cohort file not found: {cohort_file}")
        continue
    cdf = pd.read_csv(cohort_file)
    # Handle both 'cancer_type' and 'CANCER_TYPE' column names
    ct_col = 'cancer_type' if 'cancer_type' in cdf.columns else 'CANCER_TYPE'
    for ct, grp in cdf.groupby(ct_col):
        n_ici = int(grp['PX_on_ICI'].sum())
        n_ctrl = len(grp) - n_ici
        cohort_counts_rows.append({
            'cancer_type': ct,
            'cohort': cohort,
            'n_ICI': n_ici,
            'n_control': n_ctrl,
            'n_total': len(grp),
        })

cohort_counts = pd.DataFrame(cohort_counts_rows)
cohort_counts.to_csv(os.path.join(args.output_dir, 'cohort_patient_counts.csv'), index=False)
print(f"\nCohort patient counts ({len(cohort_counts)} rows):")
print(cohort_counts.to_string(index=False))

# ================================================
# 3. Diagnostics summary
# ================================================
if diag_rows:
    diag_df = pd.concat(diag_rows, ignore_index=True)
    diag_df.to_csv(os.path.join(args.output_dir, 'scheme_diagnostics_summary.csv'), index=False)
    print(f"\nDiagnostics summary saved ({len(diag_df)} rows)")

print(f"\nAll outputs saved to {args.output_dir}")
