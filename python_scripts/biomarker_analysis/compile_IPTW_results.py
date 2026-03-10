"""Compile IPTW biomarker results across all schemes with cross-scheme filtering.

Reads per-scheme CSVs from IPTW_runs_*/ directories and produces:
  - track1a_all_significant_hits.csv: all FDR-significant Track 1a (standard) results
  - track1b_all_significant_hits.csv: all FDR-significant Track 1b (prognostic-adjusted) results
  - track2_all_significant_hits.csv: all FDR-significant Track 2 results
  - track1a_cross_scheme_robust.csv: Track 1a markers robust across weighting schemes
  - track1b_cross_scheme_robust.csv: Track 1b markers robust across weighting schemes
  - track1_confounding_robust.csv: markers robust in both 1a AND 1b (survives prognostic adjustment)
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
parser.add_argument('--min_schemes_track1a', type=int, default=3,
                    help='Min schemes for Track 1a cross-scheme robustness (default: 3, out of 4)')
parser.add_argument('--min_schemes_track1b', type=int, default=2,
                    help='Min schemes for Track 1b cross-scheme robustness (default: 2, out of 2)')
parser.add_argument('--min_schemes_track2', type=int, default=3,
                    help='Min schemes for Track 2 cross-scheme robustness (default: 3, out of 4)')
args = parser.parse_args()

MARKER_PATH = os.path.join(DATA_PATH, 'biomarker_analysis/')
os.makedirs(args.output_dir, exist_ok=True)

MATCHINGS = ['1to1']
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

# Track 1a: standard (unweighted + overlap)
TRACK1A_WEIGHTS = ['unweighted', 'OVL']
# Track 1b: prognostic-score-adjusted (unweighted only)
TRACK1B_WEIGHTS = ['progAdj']
# Track 2: full-cohort interaction (overlap + unweighted)
TRACK2_WEIGHTS = ['OVL', 'noIPTW']

# ================================================
# 1. Compile all results
# ================================================
all_t1a = []
all_t1b = []
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

            # Track 1a (standard)
            for weight in TRACK1A_WEIGHTS:
                fname = f'{cancer_type}_track1_{weight}_ICI_only.csv'
                fpath = os.path.join(run_path, fname)
                if os.path.isfile(fpath):
                    df = pd.read_csv(fpath)
                    sig = df[df.get('significant_marker', pd.Series(dtype=bool)) == True].copy()
                    sig['matching'] = matching
                    sig['ps_model'] = ps_model
                    sig['weight_type'] = weight
                    sig['cancer_type'] = cancer_type
                    all_t1a.append(sig)

            # Track 1b (prognostic-adjusted)
            for weight in TRACK1B_WEIGHTS:
                fname = f'{cancer_type}_track1_{weight}_ICI_only.csv'
                fpath = os.path.join(run_path, fname)
                if os.path.isfile(fpath):
                    df = pd.read_csv(fpath)
                    sig = df[df.get('significant_marker', pd.Series(dtype=bool)) == True].copy()
                    sig['matching'] = matching
                    sig['ps_model'] = ps_model
                    sig['weight_type'] = weight
                    sig['cancer_type'] = cancer_type
                    all_t1b.append(sig)

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
t1a = pd.concat(all_t1a, ignore_index=True) if all_t1a else pd.DataFrame()
t1b = pd.concat(all_t1b, ignore_index=True) if all_t1b else pd.DataFrame()
t2 = pd.concat(all_t2, ignore_index=True) if all_t2 else pd.DataFrame()

print(f"Track 1a (standard): {len(t1a)} total significant hits")
print(f"Track 1b (prognostic-adjusted): {len(t1b)} total significant hits")
print(f"Track 2: {len(t2)} total significant hits")

# Save all hits
t1a.to_csv(os.path.join(args.output_dir, 'track1a_all_significant_hits.csv'), index=False)
t1b.to_csv(os.path.join(args.output_dir, 'track1b_all_significant_hits.csv'), index=False)
t2.to_csv(os.path.join(args.output_dir, 'track2_all_significant_hits.csv'), index=False)

# ================================================
# 2. Cross-scheme robustness filtering
# ================================================

def cross_scheme_filter_track1(df, min_schemes=2):
    """Keep markers significant in >= min_schemes with consistent HR direction."""
    if df.empty:
        return df
    df = df.copy()
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
    df = df.copy()
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


# --- Track 1a: weighting robustness (standard, no prognostic adjustment) ---
t1a_robust = cross_scheme_filter_track1(t1a, min_schemes=args.min_schemes_track1a)

# --- Track 1b: weighting robustness (prognostic-adjusted) ---
t1b_robust = cross_scheme_filter_track1(t1b, min_schemes=args.min_schemes_track1b)

# --- Track 1 confounding robustness: markers robust in BOTH 1a and 1b ---
# A marker passes this filter if it appears in both t1a_robust and t1b_robust
# with consistent direction across the two sub-tracks.
if not t1a_robust.empty and not t1b_robust.empty:
    merge_cols = ['marker', 'cancer_type']
    t1_combined = t1a_robust[merge_cols + ['direction', 'HR_median', 'HR_min', 'HR_max',
                                            'FDR_min', 'FDR_max', 'n_schemes', 'mutation_type']].merge(
        t1b_robust[merge_cols + ['direction', 'HR_median', 'HR_min', 'HR_max',
                                  'FDR_min', 'FDR_max', 'n_schemes']],
        on=merge_cols, suffixes=('_1a', '_1b'))
    # Require consistent direction across 1a and 1b
    t1_confounding_robust = t1_combined[
        t1_combined['direction_1a'] == t1_combined['direction_1b']
    ].copy()
    t1_confounding_robust.rename(columns={'direction_1a': 'direction'}, inplace=True)
    t1_confounding_robust.drop(columns=['direction_1b'], inplace=True)
else:
    t1_confounding_robust = pd.DataFrame()

# --- Track 2 ---
t2_robust = cross_scheme_filter_track2(t2, min_schemes=args.min_schemes_track2)

print(f"\nTrack 1a cross-scheme robust (>={args.min_schemes_track1a} weighting schemes): {len(t1a_robust)}")
print(f"Track 1b cross-scheme robust (>={args.min_schemes_track1b} weighting schemes): {len(t1b_robust)}")
print(f"Track 1 confounding-robust (in both 1a AND 1b, consistent direction): {len(t1_confounding_robust)}")
print(f"Track 2 cross-scheme robust (>={args.min_schemes_track2} schemes): {len(t2_robust)}")

t1a_robust.to_csv(os.path.join(args.output_dir, 'track1a_cross_scheme_robust.csv'), index=False)
t1b_robust.to_csv(os.path.join(args.output_dir, 'track1b_cross_scheme_robust.csv'), index=False)
t1_confounding_robust.to_csv(os.path.join(args.output_dir, 'track1_confounding_robust.csv'), index=False)
t2_robust.to_csv(os.path.join(args.output_dir, 'track2_cross_scheme_robust.csv'), index=False)

# ================================================
# 3. Diagnostics summary
# ================================================
if diag_rows:
    diag_df = pd.concat(diag_rows, ignore_index=True)
    diag_df.to_csv(os.path.join(args.output_dir, 'scheme_diagnostics_summary.csv'), index=False)
    print(f"\nDiagnostics summary saved ({len(diag_df)} rows)")

# ================================================
# 4. Patient counts per track
# ================================================
count_rows = []
for matching in MATCHINGS:
    for ps_model in PS_MODELS:
        spec = f'{matching}_{ps_model}'
        run_path = os.path.join(MARKER_PATH, f'IPTW_runs_{spec}/')
        if not os.path.isdir(run_path):
            continue
        for cancer_type in CANCER_TYPES:
            ess_file = os.path.join(run_path, f'{cancer_type}_diagnostics/',
                                    'effective_sample_sizes.csv')
            if not os.path.isfile(ess_file):
                continue
            ess = pd.read_csv(ess_file)
            row = ess.iloc[0] if len(ess) else None
            if row is None:
                continue

            n_ici = int(row.get('N_treated', 0))
            n_ctrl = int(row.get('N_control', 0))
            ev_ici = int(row.get('events_treated', 0))
            ev_ctrl = int(row.get('events_control', 0))

            # Track 1a/1b: ICI-only
            t1a_markers = len(t1a[
                (t1a['cancer_type'] == cancer_type) &
                (t1a['matching'] == matching) &
                (t1a['ps_model'] == ps_model)
            ]) if not t1a.empty else 0
            t1b_markers = len(t1b[
                (t1b['cancer_type'] == cancer_type) &
                (t1b['matching'] == matching) &
                (t1b['ps_model'] == ps_model)
            ]) if not t1b.empty else 0

            # Track 2: full cohort
            t2_markers = len(t2[
                (t2['cancer_type'] == cancer_type) &
                (t2['matching'] == matching) &
                (t2['ps_model'] == ps_model)
            ]) if not t2.empty else 0

            count_rows.append({
                'cancer_type': cancer_type,
                'matching': matching,
                'ps_model': ps_model,
                'n_ICI': n_ici,
                'n_nonICI': n_ctrl,
                'n_total': n_ici + n_ctrl,
                'events_ICI': ev_ici,
                'events_nonICI': ev_ctrl,
                'event_rate_ICI': ev_ici / max(n_ici, 1),
                'event_rate_nonICI': ev_ctrl / max(n_ctrl, 1),
                'track1a_sig_hits': t1a_markers,
                'track1b_sig_hits': t1b_markers,
                'track2_sig_hits': t2_markers,
            })

if count_rows:
    counts_df = pd.DataFrame(count_rows)
    counts_df.to_csv(os.path.join(args.output_dir, 'patient_counts_by_track.csv'), index=False)
    print(f"\nPatient counts saved ({len(counts_df)} rows)")

print(f"\nAll outputs saved to {args.output_dir}")
