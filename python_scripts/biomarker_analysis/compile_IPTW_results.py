"""Compile IPTW biomarker results across all schemes with cross-scheme filtering.

Reads per-scheme CSVs from IPTW_runs_*/ directories and produces:
  - track1_all_significant_hits.csv: all FDR-significant Track 1 results
  - track2_all_significant_hits.csv: all FDR-significant Track 2 results
  - track1_cross_scheme_robust.csv: markers significant in >= min_schemes with consistent direction
  - track2_cross_scheme_robust.csv: markers significant in >= min_schemes with consistent direction
  - scheme_diagnostics_summary.csv: PS AUC, ESS, SMD summaries per scheme

Usage:
  python compile_IPTW_results.py --output_dir /path/to/output
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from biomarker_common import DATA_PATH

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', required=True, help='Directory for compiled output')
parser.add_argument('--min_schemes_track1', type=int, default=2,
                    help='Min schemes for Track 1 cross-scheme robustness (default: 2)')
parser.add_argument('--min_schemes_track2', type=int, default=2,
                    help='Min schemes for Track 2 cross-scheme robustness (default: 2)')
args = parser.parse_args()

MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
os.makedirs(args.output_dir, exist_ok=True)

MATCHINGS = ['1to1', '1tok']
PS_MODELS = ['embeddings_only', 'all_covariates']
# Discover cancer types from result filenames across all run directories
_cancer_types = set()
for _m in MATCHINGS:
    for _p in PS_MODELS:
        _run = os.path.join(MARKER_PATH, f'IPTW_runs_{_m}_{_p}/')
        if not os.path.isdir(_run):
            continue
        for fname in os.listdir(_run):
            match = re.match(r'(.+)_track[12]_', fname)
            if match:
                _cancer_types.add(match.group(1))
CANCER_TYPES = sorted(_cancer_types)
TRACK2_WEIGHTS = ['ATE', 'ATT', 'noIPTW']
TRACK1_WEIGHTS = ['weighted', 'unweighted']

# ================================================
# 1. Compile all results
# ================================================
all_t1 = []
all_t2 = []
diag_rows = []

for matching in MATCHINGS:
    for ps_model in PS_MODELS:
        spec = f'{matching}_{ps_model}'
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
                ess['matching'] = matching
                ess['ps_model'] = ps_model
                diag_rows.append(ess)

            # Track 1
            for weight in TRACK1_WEIGHTS:
                fname = f'{cancer_type}_track1_{weight}_ICI_only.csv'
                fpath = os.path.join(run_path, fname)
                if os.path.isfile(fpath):
                    df = pd.read_csv(fpath)
                    sig = df[df.get('significant_marker', pd.Series(dtype=bool)) == True].copy()
                    sig['matching'] = matching
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
                    sig['matching'] = matching
                    sig['ps_model'] = ps_model
                    sig['weight_type'] = weight
                    sig['cancer_type'] = cancer_type
                    all_t2.append(sig)

# Concatenate
t1 = pd.concat(all_t1, ignore_index=True) if all_t1 else pd.DataFrame()
t2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()

print(f"Track 1: {len(t1)} total significant hits")
print(f"Track 2: {len(t2)} total significant hits")

# Save all hits
t1.to_csv(os.path.join(args.output_dir, 'track1_all_significant_hits.csv'), index=False)
t2.to_csv(os.path.join(args.output_dir, 'track2_all_significant_hits.csv'), index=False)

# ================================================
# 2. Cross-scheme robustness filtering
# ================================================

def cross_scheme_filter_track1(df, min_schemes=2):
    """Keep markers significant in >= min_schemes with consistent HR direction."""
    if df.empty:
        return df
    df['scheme'] = df['matching'] + '|' + df['ps_model'] + '|' + df['weight_type']
    grouped = df.groupby(['marker', 'cancer_type'])

    robust = []
    for (marker, ct), grp in grouped:
        n_schemes = grp['scheme'].nunique()
        if n_schemes < min_schemes:
            continue
        all_risk = (grp['HR_marker'] > 1).all()
        all_prot = (grp['HR_marker'] < 1).all()
        if not (all_risk or all_prot):
            continue  # inconsistent direction
        # Check no extreme HRs
        if 'extreme_hr_flag' in grp.columns and grp['extreme_hr_flag'].any():
            continue
        robust.append({
            'marker': marker,
            'cancer_type': ct,
            'n_schemes': n_schemes,
            'direction': 'risk' if all_risk else 'protective',
            'HR_median': grp['HR_marker'].median(),
            'HR_min': grp['HR_marker'].min(),
            'HR_max': grp['HR_marker'].max(),
            'FDR_min': grp['FDR_marker'].min(),
            'FDR_max': grp['FDR_marker'].max(),
            'mutation_type': grp['mutation_type'].iloc[0],
        })
    return pd.DataFrame(robust)


def cross_scheme_filter_track2(df, min_schemes=2):
    """Keep markers significant in >= min_schemes with consistent classifier."""
    if df.empty:
        return df
    df['scheme'] = df['matching'] + '|' + df['ps_model'] + '|' + df['weight_type']
    grouped = df.groupby(['marker', 'cancer_type'])

    robust = []
    for (marker, ct), grp in grouped:
        n_schemes = grp['scheme'].nunique()
        if n_schemes < min_schemes:
            continue
        classifiers = grp['classifier'].unique()
        if len(classifiers) > 1:
            continue  # inconsistent direction
        # Check no extreme/inf HRs
        if 'extreme_hr_flag' in grp.columns and grp['extreme_hr_flag'].any():
            continue
        has_inf = (~np.isfinite(grp['HR_markerxICI'])).any()
        if has_inf:
            continue

        sig_ici_count = (grp.get('significant_in_ICI', pd.Series(dtype=bool)) == True).sum()

        row = {
            'marker': marker,
            'cancer_type': ct,
            'n_schemes': n_schemes,
            'classifier': classifiers[0],
            'FDR_min': grp['FDR_markerxICI'].min(),
            'FDR_max': grp['FDR_markerxICI'].max(),
            'HR_ICI_median': grp['HR_marker_ICI'].median(),
            'HR_nonICI_median': grp['HR_marker_nonICI'].median(),
            'sig_in_ICI_count': sig_ici_count,
            'mutation_type': grp['mutation_type'].iloc[0],
        }

        # Include event counts if available
        for ec in ['n_ICI_pos', 'events_ICI_pos', 'n_nonICI_pos', 'events_nonICI_pos']:
            if ec in grp.columns:
                row[ec + '_median'] = grp[ec].median()

        robust.append(row)
    return pd.DataFrame(robust)


t1_robust = cross_scheme_filter_track1(t1, min_schemes=args.min_schemes_track1)
t2_robust = cross_scheme_filter_track2(t2, min_schemes=args.min_schemes_track2)

print(f"\nTrack 1 cross-scheme robust markers (>={args.min_schemes_track1} schemes): {len(t1_robust)}")
print(f"Track 2 cross-scheme robust markers (>={args.min_schemes_track2} schemes): {len(t2_robust)}")

t1_robust.to_csv(os.path.join(args.output_dir, 'track1_cross_scheme_robust.csv'), index=False)
t2_robust.to_csv(os.path.join(args.output_dir, 'track2_cross_scheme_robust.csv'), index=False)

# ================================================
# 3. Diagnostics summary
# ================================================
if diag_rows:
    diag_df = pd.concat(diag_rows, ignore_index=True)
    diag_df.to_csv(os.path.join(args.output_dir, 'scheme_diagnostics_summary.csv'), index=False)
    print(f"\nDiagnostics summary saved ({len(diag_df)} rows)")

print(f"\nAll outputs saved to {args.output_dir}")
