"""Run IPTW biomarker analysis with two tracks.

Track 1 (ICI-only, generalizability-weighted):
  S(t) ~ base_vars + line_dummies + marker
  Effect: marker coefficient

Track 2 (full cohort, IPTW-weighted):
  S(t) ~ base_vars + line_dummies + marker + PX_on_ICI + marker x ICI
  Effect: interaction coefficient

Notebook-ready: loops over all cohort x ps_model combinations automatically.
"""

import gzip
import logging
import os
import random
import warnings

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceWarning
from scipy import stats
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from config import BIOMARKER_PATH

from pipelines.biomarkers.biomarker_common import get_mutation_type

logger = logging.getLogger(__name__)

N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", "-1"))

# ============================================================
# Configuration — edit these to control which combinations to run
# ============================================================
COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']

# === Constants ===
MIN_CANCER_TYPE_TOTAL = 30   # for merging rare types into OTHER in pan-cancer
MIN_CANCER_TYPE_N = 100      # minimum total patients to run cancer-type-specific analysis
MIN_PER_ARM_CANCER_TYPE = 25 # minimum patients in each treatment arm for cancer-type-specific
COMMON_SUPPORT_PCT = (0.5, 99.5)
IPTW_TRUNC_PCT = (1, 99)
MIN_MARKER_POS_PER_ARM = 5
MIN_MARKER_NEG_PER_ARM = 5
MIN_MARKER_POS_ICI_ONLY = 5
MIN_EVENTS_PER_MARKER_GROUP = 5   # minimum deaths among marker+ patients
MIN_MARKERS_TO_TEST = 1
EXCLUDE_TYPES = {'OTHER', 'CUP'}
HR_EXTREME_THRESHOLD = 50


# =============================================
# Utility functions
# =============================================

def classify(row):
    if row['significant_predictive']:
        if row['HR_markerxICI'] < 1:
            return "predictive_ICI_benefit"
        else:
            return "predictive_ICI_harm"
    if row['significant_in_ICI'] and not row['significant_prognostic_nonICI']:
        return "ICI_specific_effect"
    if row['significant_prognostic_nonICI'] and not row['significant_in_ICI']:
        return "prognostic_nonICI"
    return "no_signal"


def merge_rare_cancer_types_into_other(df, min_total=30):
    out = df.clone()
    c_cols = [col for col in out.columns if col.startswith('CANCER_TYPE_')]
    if not c_cols:
        return out, [], []

    cancer_matrix = out.select([
        (pl.col(c).cast(pl.Float64, strict=False).fill_null(0) > 0).cast(pl.Int64).alias(c)
        for c in c_cols
    ])
    total_counts = {c: int(cancer_matrix[c].sum()) for c in c_cols}

    keep_cols = [c for c in c_cols if total_counts[c] >= min_total and c != 'CANCER_TYPE_OTHER']
    rare_cols = [c for c in c_cols if c not in keep_cols and c != 'CANCER_TYPE_OTHER']

    if 'CANCER_TYPE_OTHER' in cancer_matrix.columns:
        existing_other = cancer_matrix['CANCER_TYPE_OTHER']
    else:
        existing_other = pl.Series('CANCER_TYPE_OTHER', [0] * out.height, dtype=pl.Int64)

    merged_other = existing_other
    if rare_cols:
        rare_sum = cancer_matrix.select(pl.sum_horizontal(rare_cols)).to_series()
        merged_other = ((existing_other + rare_sum) > 0).cast(pl.Int64)

    out = out.drop(c_cols)
    out = out.with_columns([cancer_matrix[c].cast(pl.Int64).alias(c) for c in keep_cols])
    out = out.with_columns(merged_other.cast(pl.Int64).alias('CANCER_TYPE_OTHER'))

    kept = [c for c in (keep_cols + ['CANCER_TYPE_OTHER']) if int(out[c].sum()) > 0]
    return out, kept, rare_cols


def marker_has_within_arm_support(df, marker, treat_col='PX_on_ICI',
                                  min_pos_per_arm=10, min_neg_per_arm=10,
                                  min_events_per_group=5):
    marker_bin = (df[marker].cast(pl.Float64, strict=False).fill_null(0) > 0).cast(pl.Int64).to_numpy()
    treatment = df[treat_col].cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64).to_numpy()
    death = df['death'].to_numpy()
    for arm in (0, 1):
        arm_mask = treatment == arm
        if arm_mask.sum() == 0:
            return False
        arm_pos = int(marker_bin[arm_mask].sum())
        arm_neg = int(arm_mask.sum() - arm_pos)
        if arm_pos < min_pos_per_arm or arm_neg < min_neg_per_arm:
            return False
        events_pos = int(death[arm_mask & (marker_bin == 1)].sum())
        if events_pos < min_events_per_group:
            return False
    return True


def get_marker_event_counts(df, marker, treat_col='PX_on_ICI'):
    """Return event counts for a marker across treatment arms."""
    marker_bin = (df[marker].cast(pl.Float64, strict=False).fill_null(0) > 0).cast(pl.Int64).to_numpy()
    treatment = df[treat_col].cast(pl.Float64, strict=False).fill_null(0).cast(pl.Int64).to_numpy()
    death = df['death'].to_numpy()
    counts = {}
    for arm, arm_label in [(1, 'ICI'), (0, 'nonICI')]:
        arm_mask = treatment == arm
        pos_mask = arm_mask & (marker_bin == 1)
        neg_mask = arm_mask & (marker_bin == 0)
        counts[f'n_{arm_label}_pos'] = int(pos_mask.sum())
        counts[f'n_{arm_label}_neg'] = int(neg_mask.sum())
        counts[f'events_{arm_label}_pos'] = int(death[pos_mask].sum())
        counts[f'events_{arm_label}_neg'] = int(death[neg_mask].sum())
    return counts


def marker_has_ici_only_support(df, marker, min_pos=10, min_events=5):
    """Check if marker has enough positive cases and events within ICI patients."""
    ici_df = df.filter(pl.col('PX_on_ICI') == 1)
    marker_bin = (ici_df[marker].cast(pl.Float64, strict=False).fill_null(0) > 0).to_numpy()
    death = ici_df['death'].to_numpy()
    marker_pos = int(marker_bin.sum())
    marker_neg = len(ici_df) - marker_pos
    events_pos = int(death[marker_bin].sum())
    return marker_pos >= min_pos and marker_neg >= min_pos and events_pos >= min_events


def compute_smd(df, covariates, treat_col='PX_on_ICI', weights=None):
    t_mask = (df[treat_col] == 1).to_numpy()
    rows = []
    for cov in covariates:
        x = df[cov].cast(pl.Float64, strict=False).fill_null(0).to_numpy()
        x_t, x_c = x[t_mask], x[~t_mask]
        pooled_sd = np.sqrt((x_t.var() + x_c.var()) / 2)
        smd_raw = (x_t.mean() - x_c.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        smd_w = np.nan
        if weights is not None:
            w_t, w_c = weights[t_mask], weights[~t_mask]
            wm_t = np.average(x_t, weights=w_t) if w_t.sum() > 0 else x_t.mean()
            wm_c = np.average(x_c, weights=w_c) if w_c.sum() > 0 else x_c.mean()
            smd_w = (wm_t - wm_c) / pooled_sd if pooled_sd > 0 else 0.0
        rows.append({'covariate': cov, 'SMD_unweighted': smd_raw, 'SMD_weighted': smd_w})
    return pl.DataFrame(rows)


def recalibrate_propensity_within_subset(df, ps_col='ICI_prediction', treat_col='PX_on_ICI'):
    ps = df.select(ps_col).to_numpy()
    y = df[treat_col].to_numpy().astype(int)
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(ps, y)
    return lr.predict_proba(ps)[:, 1]


def fit_cph_log_warnings(cph, df_fit, duration_col, event_col,
                          weights_col=None, robust=True, marker_name=""):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fit_kwargs = dict(duration_col=duration_col, event_col=event_col, robust=robust)
        if weights_col is not None:
            fit_kwargs['weights_col'] = weights_col
        cph.fit(df_fit, **fit_kwargs)
    for w in caught:
        if issubclass(w.category, (ConvergenceWarning, LinAlgWarning)):
            logger.warning("marker=%s: %s: %s", marker_name, w.category.__name__, w.message)
    return cph


def _fdr_within_mutation_type(results_df, p_col, fdr_col, sig_col):
    fdr_values = [None] * results_df.height
    sig_values = [False] * results_df.height
    mut_types = results_df['mutation_type'].to_list()
    pvals_all = results_df[p_col].to_list()
    for mut_type in results_df['mutation_type'].unique().to_list():
        idxs = [i for i, m in enumerate(mut_types) if m == mut_type]
        pvals = [pvals_all[i] for i in idxs]
        if not pvals:
            continue
        rej, fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, f, r in zip(idxs, fdr, rej):
            fdr_values[i] = float(f)
            sig_values[i] = bool(r)
    return results_df.with_columns([
        pl.Series(fdr_col, fdr_values, dtype=pl.Float64),
        pl.Series(sig_col, sig_values, dtype=pl.Boolean),
    ])


# =============================================
# Track 2: Full-cohort interaction model
# =============================================

TRACK2_RESULT_COLS = [
    'marker', 'beta_markerxICI', 'HR_markerxICI',
    'CI95_markerxICI_low', 'CI95_markerxICI_high', 'p_markerxICI',
    'beta_marker_ICI', 'HR_marker_ICI', 'CI95_marker_ICI_low',
    'CI95_marker_ICI_high', 'p_marker_ICI',
    'beta_marker_nonICI', 'HR_marker_nonICI', 'CI95_marker_nonICI_low',
    'CI95_marker_nonICI_high', 'p_marker_nonICI',
    'beta_ICI_at_marker0', 'p_ICI_at_marker0',
    'n_ICI_pos', 'n_ICI_neg', 'events_ICI_pos', 'events_ICI_neg',
    'n_nonICI_pos', 'n_nonICI_neg', 'events_nonICI_pos', 'events_nonICI_neg',
]


def _fit_track2_marker(df, marker, base_vars, weights_col):
    """Track 2: interaction model on full cohort."""
    cols = ['tt_death', 'death', 'PX_on_ICI'] + base_vars + [marker]
    if weights_col is not None:
        cols.append(weights_col)
    df_fit_pl = df.select(cols).drop_nulls()

    # Compute event counts before fitting
    event_counts = get_marker_event_counts(df_fit_pl, marker)

    df_fit = df_fit_pl.to_pandas()
    mx = f"{marker}_x_ICI"
    df_fit[mx] = df_fit['PX_on_ICI'] * df_fit[marker]

    cph = CoxPHFitter(penalizer=0.01)
    cph = fit_cph_log_warnings(cph, df_fit, 'tt_death', 'death',
                                weights_col=weights_col, robust=True, marker_name=marker)

    summ = cph.summary.reset_index()
    V = cph.variance_matrix_
    b = cph.params_

    beta_m = float(b[marker])
    se_m = float(np.sqrt(V.loc[marker, marker]))
    p_m = float(summ.loc[summ['covariate'] == marker, 'p'].values[0])

    beta_mx = float(b[mx])
    se_mx = float(np.sqrt(V.loc[mx, mx]))
    p_mx = float(summ.loc[summ['covariate'] == mx, 'p'].values[0])
    ci_mx = (np.exp(beta_mx - 1.96 * se_mx), np.exp(beta_mx + 1.96 * se_mx))

    hr_nonici = np.exp(beta_m)
    ci_nonici = (np.exp(beta_m - 1.96 * se_m), np.exp(beta_m + 1.96 * se_m))

    cov_m_mx = float(V.loc[marker, mx])
    se_ici = np.sqrt(se_m**2 + se_mx**2 + 2 * cov_m_mx)
    beta_ici = beta_m + beta_mx
    hr_ici = np.exp(beta_ici)
    ci_ici = (np.exp(beta_ici - 1.96 * se_ici), np.exp(beta_ici + 1.96 * se_ici))
    z_ici = beta_ici / se_ici
    p_ici = 2 * (1 - stats.norm.cdf(abs(z_ici)))

    beta_IO0 = float(b['PX_on_ICI']) if 'PX_on_ICI' in b.index else np.nan
    p_IO0 = (float(summ.loc[summ['covariate'] == 'PX_on_ICI', 'p'].values[0])
             if 'PX_on_ICI' in summ['covariate'].values else np.nan)

    result = {
        "marker": marker,
        "beta_markerxICI": beta_mx, "HR_markerxICI": np.exp(beta_mx),
        "CI95_markerxICI_low": ci_mx[0], "CI95_markerxICI_high": ci_mx[1],
        "p_markerxICI": p_mx,
        "beta_marker_ICI": beta_ici, "HR_marker_ICI": hr_ici,
        "CI95_marker_ICI_low": ci_ici[0], "CI95_marker_ICI_high": ci_ici[1], "p_marker_ICI": p_ici,
        "beta_marker_nonICI": beta_m, "HR_marker_nonICI": hr_nonici,
        "CI95_marker_nonICI_low": ci_nonici[0], "CI95_marker_nonICI_high": ci_nonici[1],
        "p_marker_nonICI": p_m,
        "beta_ICI_at_marker0": beta_IO0, "p_ICI_at_marker0": p_IO0,
    }
    result.update(event_counts)
    return result


def add_track2_fdr_and_labels(results_df):
    if results_df.is_empty():
        return results_df.with_columns([
            pl.lit(None, dtype=pl.Utf8).alias('mutation_type'),
            pl.lit(None, dtype=pl.Utf8).alias('classifier'),
        ])

    results_df = results_df.with_columns(
        pl.col('marker').map_elements(get_mutation_type, return_dtype=pl.Utf8).alias('mutation_type')
    )
    results_df = _fdr_within_mutation_type(results_df, 'p_markerxICI', 'FDR_markerxICI', 'significant_predictive')
    results_df = _fdr_within_mutation_type(results_df, 'p_marker_ICI', 'FDR_marker_ICI', 'significant_in_ICI')
    results_df = _fdr_within_mutation_type(results_df, 'p_marker_nonICI', 'FDR_marker_nonICI',
                              'significant_prognostic_nonICI')
    classifier_values = [classify(row) for row in results_df.iter_rows(named=True)]
    results_df = results_df.with_columns(pl.Series('classifier', classifier_values, dtype=pl.Utf8))

    # Flag extreme HRs indicating possible model separation
    results_df = results_df.with_columns(
        (
            (~pl.col('HR_markerxICI').is_finite()) |
            (pl.col('HR_markerxICI') > HR_EXTREME_THRESHOLD) |
            (pl.col('HR_markerxICI') < 1.0 / HR_EXTREME_THRESHOLD)
        ).alias('extreme_hr_flag')
    )
    n_extreme = int(results_df['extreme_hr_flag'].sum())
    if n_extreme > 0:
        logger.warning(f"  {n_extreme} markers flagged with extreme interaction HRs (>{HR_EXTREME_THRESHOLD} or <{1/HR_EXTREME_THRESHOLD:.4f})")

    return results_df


def _run_track2_screen(type_df, cancer_type, base_vars, biomarker_cols, run_path, n_jobs):
    """Track 2: screen all markers with within-arm support via the interaction model."""
    track2_markers = [
        m for m in biomarker_cols
        if marker_has_within_arm_support(type_df, m, min_pos_per_arm=MIN_MARKER_POS_PER_ARM,
                                         min_neg_per_arm=MIN_MARKER_NEG_PER_ARM,
                                         min_events_per_group=MIN_EVENTS_PER_MARKER_GROUP)
    ]
    print(f"  Track 2 markers to test: {len(track2_markers)}")

    if len(track2_markers) < MIN_MARKERS_TO_TEST:
        print(f"  Skipping Track 2: fewer than {MIN_MARKERS_TO_TEST} markers with sufficient support.")
        return

    for spec_name, spec_weights in [('ATE', 'IPTW_ATE'), ('noIPTW', None)]:
        results, failed = _run_marker_screen(
            type_df, track2_markers, base_vars, spec_weights,
            _fit_track2_marker, n_jobs, label=f"T2 {cancer_type} {spec_name}")
        if failed:
            print(f"  Track 2 {spec_name} failures: {len(failed)}. "
                  f"First: {failed[0][0]} -> {failed[0][1]}")
        spec_df = pl.DataFrame(results, schema=TRACK2_RESULT_COLS) if results else pl.DataFrame(schema={c: pl.Float64 for c in TRACK2_RESULT_COLS})
        spec_df = add_track2_fdr_and_labels(spec_df)
        with gzip.open(os.path.join(run_path, f'{cancer_type}_track2_{spec_name}_interaction.csv.gz'), 'wb') as f:
            spec_df.write_csv(f)


# =============================================
# Track 1: ICI-only generalizability-weighted
# =============================================

TRACK1_RESULT_COLS = [
    'marker', 'beta_marker', 'HR_marker', 'CI95_marker_low', 'CI95_marker_high', 'p_marker',
    'n_marker_pos', 'n_marker_neg', 'events_marker_pos', 'events_marker_neg',
]


def _fit_track1_marker(df, marker, base_vars, weights_col):
    """Track 1: marker-only model on ICI patients."""
    cols = ['tt_death', 'death'] + base_vars + [marker]
    if weights_col is not None:
        cols.append(weights_col)
    df_fit_pl = df.select(cols).drop_nulls()

    # Compute event counts
    marker_bin = (df_fit_pl[marker].cast(pl.Float64, strict=False).fill_null(0) > 0).to_numpy()
    death = df_fit_pl['death'].to_numpy()
    n_pos = int(marker_bin.sum())
    n_neg = int((~marker_bin).sum())
    events_pos = int(death[marker_bin].sum())
    events_neg = int(death[~marker_bin].sum())

    df_fit = df_fit_pl.to_pandas()
    cph = CoxPHFitter(penalizer=0.01)
    cph = fit_cph_log_warnings(cph, df_fit, 'tt_death', 'death',
                                weights_col=weights_col, robust=True, marker_name=marker)

    summ = cph.summary.reset_index()
    b = cph.params_
    V = cph.variance_matrix_

    beta_m = float(b[marker])
    se_m = float(np.sqrt(V.loc[marker, marker]))
    p_m = float(summ.loc[summ['covariate'] == marker, 'p'].values[0])
    hr_m = np.exp(beta_m)
    ci_m = (np.exp(beta_m - 1.96 * se_m), np.exp(beta_m + 1.96 * se_m))

    return {
        "marker": marker,
        "beta_marker": beta_m, "HR_marker": hr_m,
        "CI95_marker_low": ci_m[0], "CI95_marker_high": ci_m[1],
        "p_marker": p_m,
        "n_marker_pos": n_pos, "n_marker_neg": n_neg,
        "events_marker_pos": events_pos, "events_marker_neg": events_neg,
    }


def add_track1_fdr(results_df):
    if results_df.is_empty():
        return results_df.with_columns(pl.lit(None, dtype=pl.Utf8).alias('mutation_type'))
    results_df = results_df.with_columns(
        pl.col('marker').map_elements(get_mutation_type, return_dtype=pl.Utf8).alias('mutation_type')
    )
    results_df = _fdr_within_mutation_type(results_df, 'p_marker', 'FDR_marker', 'significant_marker')

    results_df = results_df.with_columns(
        (
            (pl.col('HR_marker') > HR_EXTREME_THRESHOLD) |
            (pl.col('HR_marker') < 1.0 / HR_EXTREME_THRESHOLD)
        ).alias('extreme_hr_flag')
    )
    n_extreme = int(results_df['extreme_hr_flag'].sum())
    if n_extreme > 0:
        logger.warning(f"  {n_extreme} markers flagged with extreme HRs (>{HR_EXTREME_THRESHOLD} or <{1/HR_EXTREME_THRESHOLD:.4f})")

    return results_df


def _run_track1_screen(type_df, cancer_type, base_vars, biomarker_cols, run_path, n_jobs):
    """Track 1: screen markers with ICI-only support via the generalizability-weighted model."""
    eps = 1e-6
    ici_only_df = type_df.filter(pl.col('PX_on_ICI') == 1)

    # Generalizability weights for ICI subset: weight = 1 / ps
    # (reweight ICI patients to look like the full eligible population)
    ici_ps = ici_only_df['ICI_prediction'].clip(eps, 1 - eps).to_numpy()
    w_gen = 1.0 / ici_ps
    low_gen, high_gen = np.percentile(w_gen, IPTW_TRUNC_PCT)
    if np.isfinite(low_gen) and np.isfinite(high_gen):
        w_gen_trunc = np.clip(w_gen, low_gen, high_gen)
    else:
        w_gen_trunc = np.ones(len(ici_only_df))
    ici_only_df = ici_only_df.with_columns(pl.Series('IPTW_GEN', np.asarray(w_gen_trunc)))

    track1_markers = [
        m for m in biomarker_cols
        if marker_has_ici_only_support(type_df, m, min_pos=MIN_MARKER_POS_ICI_ONLY,
                                       min_events=MIN_EVENTS_PER_MARKER_GROUP)
    ]
    print(f"  Track 1 markers to test: {len(track1_markers)}")

    if len(track1_markers) < MIN_MARKERS_TO_TEST:
        print(f"  Skipping Track 1: fewer than {MIN_MARKERS_TO_TEST} markers with sufficient support.")
        return

    # ICI-only base vars (no cancer type interaction needed for type-specific,
    # include for pan-cancer)
    ici_base_vars = list(base_vars)

    for spec_name, spec_weights in [('ATE', 'IPTW_GEN'), ('unweighted', None)]:
        results, failed = _run_marker_screen(
            ici_only_df, track1_markers, ici_base_vars, spec_weights,
            _fit_track1_marker, n_jobs, label=f"T1 {cancer_type} {spec_name}")
        if failed:
            print(f"  Track 1 {spec_name} failures: {len(failed)}. "
                  f"First: {failed[0][0]} -> {failed[0][1]}")
        spec_df = pl.DataFrame(results, schema=TRACK1_RESULT_COLS) if results else pl.DataFrame(schema={c: pl.Float64 for c in TRACK1_RESULT_COLS})
        spec_df = add_track1_fdr(spec_df)
        with gzip.open(os.path.join(run_path, f'{cancer_type}_track1_{spec_name}_ICI_only.csv.gz'), 'wb') as f:
            spec_df.write_csv(f)


# =============================================
# Parallel runner
# =============================================

def _run_marker_screen(df, markers, base_vars, weights_col, fit_fn, n_jobs, label=""):
    def _safe_fit(marker):
        try:
            return fit_fn(df, marker, base_vars, weights_col), None
        except Exception as e:
            return None, (marker, str(e))

    raw = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_safe_fit)(m) for m in tqdm(markers, desc=label)
    )
    results = [r for r, _ in raw if r is not None]
    failed = [f for _, f in raw if f is not None]
    return results, failed


def main() -> None:
    random.seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # ============================================================
    # Main loop: iterate over cohort x ps_model combinations
    # ============================================================
    for COHORT in COHORTS:
        for PS_MODEL in PS_MODELS:
            SPEC_LABEL = f'{COHORT}_{PS_MODEL}'
            RUN_PATH = os.path.join(BIOMARKER_PATH, f'IPTW_runs_{SPEC_LABEL}/')
            os.makedirs(RUN_PATH, exist_ok=True)

            print(f"\n{'#'*60}")
            print(f"[run_IPTW_analysis] Spec: {SPEC_LABEL}")
            print(f"[run_IPTW_analysis] Output: {RUN_PATH}")
            print(f"{'#'*60}")

            # === Load data ===
            input_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{SPEC_LABEL}.csv.gz')
            full_df = pl.read_csv(input_file)

            # === Identify column groups ===
            required_vars = ['DFCI_MRN', 'tt_death', 'death']
            base_covars = ['GENDER', 'AGE_AT_TREATMENTSTART']
            line_cols = sorted([col for col in full_df.columns if col.startswith('LINE_')])
            panel_cols = [col for col in full_df.columns if col.upper().startswith('PANEL_VERSION_')]
            cancer_type_cols = [col for col in full_df.columns if col.startswith('CANCER_TYPE_')]
            excluded_cols = (required_vars + base_covars + line_cols + panel_cols +
                             cancer_type_cols + ['PX_on_ICI', 'ICI_prediction'])
            mutation_tags = ('_SNV', '_SV', '_FUSION', '_DEL', '_AMP')
            biomarker_cols = [
                col for col in full_df.columns
                if (col not in excluded_cols) and any(tag in col.upper() for tag in mutation_tags)
            ]

            # === Identify embedding columns for prognostic score ===
            embedding_cols = [col for col in full_df.columns
                              if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]
            print(f"  {len(embedding_cols)} embedding columns found for prognostic score")

            # === Cancer types to test ===
            types_to_test = ['pan_cancer'] + [
                col.replace('CANCER_TYPE_', '') for col in cancer_type_cols
                if col.replace('CANCER_TYPE_', '') not in EXCLUDE_TYPES
            ]

            for cancer_type in types_to_test:
                print(f"\n{'='*60}")
                print(f"Cancer type: {cancer_type}")
                print(f"{'='*60}")

                if cancer_type == 'pan_cancer':
                    type_df = full_df.clone()
                    type_df, pan_ct_cols, merged_rare = merge_rare_cancer_types_into_other(
                        type_df, min_total=MIN_CANCER_TYPE_TOTAL)
                    if merged_rare:
                        print(f"  Merged {len(merged_rare)} rare cancer types into OTHER: "
                              + ", ".join(sorted(merged_rare)))
                    panel_cols_fit = [c for c in type_df.columns if 'PANEL' in c]
                    ct_cols_fit = [c for c in type_df.columns if 'CANCER_TYPE' in c]
                    base_vars = base_covars + line_cols + panel_cols_fit + ct_cols_fit
                else:
                    ct_col = f'CANCER_TYPE_{cancer_type}'
                    if ct_col not in full_df.columns:
                        print(f"  Skipping: column {ct_col} not found.")
                        continue
                    type_df = full_df.filter(pl.col(ct_col).cast(pl.Boolean, strict=False))
                    if len(type_df) < MIN_CANCER_TYPE_N:
                        print(f"  Skipping: only {len(type_df)} patients (minimum {MIN_CANCER_TYPE_N}).")
                        continue
                    n_treated_ct = int(type_df['PX_on_ICI'].sum())
                    n_control_ct = len(type_df) - n_treated_ct
                    if n_treated_ct < MIN_PER_ARM_CANCER_TYPE or n_control_ct < MIN_PER_ARM_CANCER_TYPE:
                        print(f"  Skipping: insufficient per-arm counts "
                              f"(treated={n_treated_ct}, control={n_control_ct}, "
                              f"minimum={MIN_PER_ARM_CANCER_TYPE}).")
                        continue
                    # Recalibrate propensity within subset
                    if type_df['PX_on_ICI'].n_unique() >= 2 and len(type_df) > 10:
                        ps_before_mean = float(type_df['ICI_prediction'].mean())
                        recal_ps = recalibrate_propensity_within_subset(type_df)
                        type_df = type_df.with_columns(pl.Series('ICI_prediction', recal_ps))
                        print(f"  Recalibrated PS (mean {ps_before_mean:.3f} -> "
                              f"{float(type_df['ICI_prediction'].mean()):.3f})")
                    panel_cols_fit = [c for c in type_df.columns if 'PANEL' in c]
                    base_vars = base_covars + line_cols + panel_cols_fit

                if type_df.is_empty():
                    print(f"  Skipping: no rows.")
                    continue
                if type_df['PX_on_ICI'].n_unique() < 2:
                    print(f"  Skipping: only one treatment group.")
                    continue

                # --- Common support trimming ---
                eps = 1e-6
                ps_raw_series = type_df['ICI_prediction'].clip(eps, 1 - eps)
                treat_col_np = type_df['PX_on_ICI'].to_numpy()
                ps_raw = ps_raw_series.to_numpy()
                ps_t = ps_raw[treat_col_np == 1]
                ps_c = ps_raw[treat_col_np == 0]

                if len(ps_t) == 0 or len(ps_c) == 0:
                    print(f"  Skipping: missing treated or control for common support.")
                    continue

                lower = max(np.percentile(ps_t, COMMON_SUPPORT_PCT[0]),
                            np.percentile(ps_c, COMMON_SUPPORT_PCT[0]))
                upper = min(np.percentile(ps_t, COMMON_SUPPORT_PCT[1]),
                            np.percentile(ps_c, COMMON_SUPPORT_PCT[1]))

                if (lower is None or upper is None or np.isnan(lower) or np.isnan(upper)
                        or lower >= upper):
                    print(f"  Skipping: no propensity overlap.")
                    continue

                trim_mask = (ps_raw >= lower) & (ps_raw <= upper)
                type_df = type_df.filter(pl.Series(trim_mask))
                if type_df.is_empty() or type_df['PX_on_ICI'].n_unique() < 2:
                    print(f"  Skipping: no rows or one group after trimming.")
                    continue

                ps = type_df['ICI_prediction'].clip(eps, 1 - eps).to_numpy()
                treat_mask = (type_df['PX_on_ICI'] == 1).to_numpy()
                p_treated = float(type_df['PX_on_ICI'].mean())

                if p_treated <= 0 or p_treated >= 1:
                    print(f"  Skipping: invalid treated proportion ({p_treated:.4f}).")
                    continue

                # --- Stabilized ATE weights ---
                w_ate = np.where(treat_mask, p_treated / ps, (1 - p_treated) / (1 - ps))
                low, high = np.percentile(w_ate, IPTW_TRUNC_PCT)
                if not np.isfinite(low) or not np.isfinite(high):
                    print(f"  Skipping: non-finite ATE truncation bounds.")
                    continue
                w_ate_trunc = np.clip(w_ate, low, high)
                type_df = type_df.with_columns(pl.Series('IPTW_ATE', w_ate_trunc))

                # --- ESS ---
                w_t, w_c = w_ate_trunc[treat_mask], w_ate_trunc[~treat_mask]
                ess_t = w_t.sum() ** 2 / (w_t ** 2).sum()
                ess_c = w_c.sum() ** 2 / (w_c ** 2).sum()
                print(f"  ATE: N treated={treat_mask.sum()}, N control={(~treat_mask).sum()} | "
                      f"ESS treated={ess_t:.0f}, ESS control={ess_c:.0f}")

                # --- Diagnostics ---
                diag_path = os.path.join(RUN_PATH, f'{cancer_type}_diagnostics/')
                os.makedirs(diag_path, exist_ok=True)

                ps_diag = type_df.select(['PX_on_ICI', 'ICI_prediction']).with_columns(
                    pl.Series('IPTW_ATE', w_ate_trunc)
                )
                ps_summary = ps_diag.group_by('PX_on_ICI').agg([
                    pl.col('ICI_prediction').count().alias('count'),
                    pl.col('ICI_prediction').mean().alias('mean'),
                    pl.col('ICI_prediction').std().alias('std'),
                    pl.col('ICI_prediction').min().alias('min'),
                    pl.col('ICI_prediction').quantile(0.25).alias('25%'),
                    pl.col('ICI_prediction').median().alias('50%'),
                    pl.col('ICI_prediction').quantile(0.75).alias('75%'),
                    pl.col('ICI_prediction').max().alias('max'),
                ]).sort('PX_on_ICI')
                with gzip.open(os.path.join(diag_path, 'propensity_score_summary.csv.gz'), 'wb') as f:
                    ps_summary.write_csv(f)

                # PS AUC within this cancer type subset
                ps_auc = roc_auc_score(type_df['PX_on_ICI'].to_numpy(), type_df['ICI_prediction'].to_numpy())
                print(f"  PS AUC ({cancer_type}): {ps_auc:.4f}")

                # Cohort summary with event rates
                death_np = type_df['death'].to_numpy()
                n_treated = int(treat_mask.sum())
                n_control = int((~treat_mask).sum())
                events_treated = int(death_np[treat_mask].sum())
                events_control = int(death_np[~treat_mask].sum())
                print(f"  Cohort: {n_treated} ICI ({events_treated} deaths, {events_treated/max(n_treated,1)*100:.1f}%), "
                      f"{n_control} non-ICI ({events_control} deaths, {events_control/max(n_control,1)*100:.1f}%)")

                with gzip.open(os.path.join(diag_path, 'effective_sample_sizes.csv.gz'), 'wb') as f:
                    pl.DataFrame([{
                        'cancer_type': cancer_type,
                        'N_treated': n_treated, 'N_control': n_control,
                        'events_treated': events_treated, 'events_control': events_control,
                        'event_rate_treated': events_treated / max(n_treated, 1),
                        'event_rate_control': events_control / max(n_control, 1),
                        'PS_AUC': ps_auc,
                        'ESS_ATE_treated': ess_t, 'ESS_ATE_control': ess_c,
                    }]).write_csv(f)

                balance_covars = base_covars + line_cols + [
                    c for c in type_df.columns
                    if c.startswith('CANCER_TYPE_') or c.upper().startswith('PANEL_VERSION_')]
                smd_ate = compute_smd(type_df, balance_covars, weights=w_ate_trunc)
                with gzip.open(os.path.join(diag_path, 'covariate_balance_smd_ATE.csv.gz'), 'wb') as f:
                    smd_ate.write_csv(f)

                # Max SMD check (balance quality indicator)
                max_smd_ate = float(smd_ate['SMD_weighted'].abs().max())
                n_imbalanced_ate = int((smd_ate['SMD_weighted'].abs() > 0.1).sum())
                print(f"  ATE balance: max|SMD|={max_smd_ate:.4f}, {n_imbalanced_ate}/{len(smd_ate)} covariates with |SMD|>0.1")

                # === Track 2: full-cohort interaction ===
                _run_track2_screen(type_df, cancer_type, base_vars, biomarker_cols, RUN_PATH, N_JOBS)

                # === Track 1: ICI-only generalizability-weighted ===
                _run_track1_screen(type_df, cancer_type, base_vars, biomarker_cols, RUN_PATH, N_JOBS)

            print(f"\n[run_IPTW_analysis] Done with {SPEC_LABEL}. Results in {RUN_PATH}")


if __name__ == "__main__":
    main()
