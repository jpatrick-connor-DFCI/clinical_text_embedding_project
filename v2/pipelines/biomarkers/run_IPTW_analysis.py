"""Run IPTW biomarker analysis (full-cohort interaction model).

  S(t) ~ base_vars + line_dummies + marker + PX_on_ICI + marker x ICI
  Effect: interaction coefficient -- is the marker predictive of ICI benefit?

Notebook-ready: loops over all cohort x ps_model combinations automatically.
"""

import logging
import os
import random
import warnings
from collections import Counter

# === Thread-count caps: MUST be set before polars/numpy are imported ===
# The marker screens fan out over loky worker processes, and each worker is a
# fresh interpreter that re-imports polars and builds its own Rayon thread pool.
# Polars sizes that pool from the machine's core count, not from the SLURM
# allocation, so `n_jobs` workers x `cores` threads each (plus tokio, jemalloc
# background, and polars-ooc cleaner threads) overruns RLIMIT_NPROC / the
# cgroup pid limit and the run dies mid-screen with
#   PanicException: could not spawn threads: ... Os { code: 11, WouldBlock }
# Note that panic is a BaseException, so `_safe_fit`'s `except Exception` does
# not contain it — one exhausted worker kills the whole run.
#
# Each worker does one small select + filter and one Cox fit, so intra-worker
# parallelism buys nothing; all the parallelism that matters is across markers.
# loky workers inherit this environment at spawn, so setting it here caps them.
# setdefault, so an explicit shell setting still wins.
for _var in ("POLARS_MAX_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import polars as pl
from joblib import Parallel, delayed, parallel_config
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
from shared.polars_utils import filter_finite_rows, finite_or_zero, to_pandas_via_numpy

logger = logging.getLogger(__name__)

def _resolve_n_jobs() -> int:
    """Concrete worker count, never joblib's `-1`.

    `-1` expands to every core the machine reports, which on a shared node is
    far more than the job was allocated and is what makes the thread budget
    explode. Prefer the SLURM allocation; fall back to the visible core count
    capped at MAX_WORKERS_FALLBACK. `IPTW_N_JOBS` overrides both.
    """
    for var in ("IPTW_N_JOBS", "SLURM_CPUS_PER_TASK"):
        raw = os.getenv(var)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                logger.warning("%s=%r is not an integer; ignoring", var, raw)
                continue
            if value > 0:
                return value
    return min(os.cpu_count() or 1, MAX_WORKERS_FALLBACK)


MAX_WORKERS_FALLBACK = 16
N_JOBS = _resolve_n_jobs()

# ============================================================
# Configuration — edit these to control which combinations to run
# ============================================================
COHORTS = ['cohort1', 'cohort2']
PS_MODELS = ['covariates_only', 'covariates_plus_embeddings']

# === Constants ===
# Cancer types with fewer than this many patients are folded into
# CANCER_TYPE_OTHER (the reference level) rather than carrying their own dummy
# in the pan-cancer model. Aligned with MIN_CANCER_TYPE_N so the "big enough to
# model on its own" bar is the same whether the type gets its own screen or just
# its own dummy.
#
# This count is taken BEFORE common-support trimming, so it is a necessary but
# not sufficient guard against a dummy going constant: a type can clear the bar
# here and still be emptied entirely by the trim if none of its patients have
# propensity overlap (heme malignancies in a solid-tumor ICI cohort did exactly
# that -- LEUKEMIA and MYELOMA cleared 30 and were then trimmed to zero rows).
# The post-trim constant-column drop in main() is what actually closes that hole.
MIN_CANCER_TYPE_TOTAL = int(os.getenv("IPTW_MIN_CANCER_TYPE_TOTAL", "100"))
MIN_CANCER_TYPE_N = 100      # minimum total patients to run cancer-type-specific analysis
MIN_PER_ARM_CANCER_TYPE = 25 # minimum patients in each treatment arm for cancer-type-specific
COMMON_SUPPORT_PCT = (0.5, 99.5)
IPTW_TRUNC_PCT = (1, 99)
MIN_MARKER_POS_PER_ARM = 5
MIN_MARKER_NEG_PER_ARM = 5
MIN_EVENTS_PER_MARKER_GROUP = 5   # minimum deaths among marker+ patients
MIN_MARKERS_TO_TEST = 1
EXCLUDE_TYPES = {'OTHER', 'CUP'}
# Cancer-type-specific screens are restricted to these types. `pan_cancer` always
# runs and is unaffected — it still models every type via its CANCER_TYPE_* dummies,
# including the ones left out here. Set to None to screen every type that clears
# the MIN_CANCER_TYPE_N / MIN_PER_ARM_CANCER_TYPE gates instead.
INCLUDE_TYPES = {'KIDNEY', 'LUNG', 'SKIN'}
HR_EXTREME_THRESHOLD = 50

# === Smoke-test toggle: cap how many markers are screened ===
# A full pan-cancer screen is ~1500 markers x 4 screens and runs for hours, which
# is a slow way to find out that a path is broken. Set IPTW_MAX_MARKERS to screen
# at most that many markers, so a structural failure (singular design, missing
# dependency, unwritable output) surfaces in minutes.
#
# A count rather than a fraction because the cost that matters is the number of
# Cox fits, and that is what a count fixes directly: total fits are roughly
# markers x 2 tracks x 2 weightings x cancer types, regardless of how many
# markers the input happens to carry. IPTW_MARKER_FRACTION is still honoured as
# a fraction of the marker list; when both are set the smaller cap wins.
#
# Sampling is seeded off random.seed(42) in main(), so the same cap always picks
# the same markers -- a smoke run is reproducible and comparable to a previous
# one. Selection happens once per spec, before the cancer-type loop, so every
# cancer type screens the SAME marker subset; sampling inside the loop would make
# per-type results incomparable.
#
# RESULTS FROM A CAPPED RUN ARE NOT VALID. FDR is computed within mutation type
# over whatever is screened, so a subset changes every q-value -- it is not a
# subset of the full run's hits. Output goes to a _smoke suffixed directory so a
# test run can never overwrite real results (see RUN_PATH in main()).
def _read_positive_int(var):
    raw = os.getenv(var)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{var} must be a positive integer, got {raw!r}.") from None
    if value < 1:
        raise ValueError(f"{var} must be >= 1, got {value}.")
    return value


MAX_MARKERS = _read_positive_int("IPTW_MAX_MARKERS")

MARKER_FRACTION = float(os.getenv("IPTW_MARKER_FRACTION", "1.0"))
if not 0.0 < MARKER_FRACTION <= 1.0:
    raise ValueError(
        f"IPTW_MARKER_FRACTION must be in (0, 1], got {MARKER_FRACTION}. "
        f"Unset it (or set 1.0) to screen every marker."
    )

IS_SMOKE_RUN = MAX_MARKERS is not None or MARKER_FRACTION < 1.0


def resolve_marker_subset(markers):
    """Markers to screen, honouring IPTW_MAX_MARKERS / IPTW_MARKER_FRACTION.

    Returns the full list unchanged when neither is set. `sorted` before sampling
    makes the draw depend only on the seed and not on parquet column order, so
    the same cap picks the same markers across regenerated inputs.
    """
    if not IS_SMOKE_RUN or not markers:
        return markers
    n = len(markers)
    if MARKER_FRACTION < 1.0:
        n = max(1, round(len(markers) * MARKER_FRACTION))
    if MAX_MARKERS is not None:
        n = min(n, MAX_MARKERS)
    n = min(n, len(markers))
    return sorted(random.sample(sorted(markers), n))


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
        (finite_or_zero(c) > 0).cast(pl.Int64).alias(c)
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

    # Drop a reference level. Upstream build_cancer_type_df dummies with
    # drop_first=True, so what arrives here is already reference-dropped; folding
    # the rare types back in re-adds CANCER_TYPE_OTHER and restores the complete
    # partition. Every patient has exactly one cancer type, so a complete set sums
    # to the all-ones vector and makes the Cox partial-likelihood Hessian singular
    # -- lifelines then fails *every* fit with "matrix inversion problems", which
    # reads like a data problem rather than a design-matrix one. OTHER is the
    # reference because it is the heterogeneous residual category and the one
    # column this function is guaranteed to have created.
    reference = 'CANCER_TYPE_OTHER' if 'CANCER_TYPE_OTHER' in kept else (kept[0] if kept else None)
    kept_for_fit = [c for c in kept if c != reference]
    return out, kept_for_fit, rare_cols


NUMERIC_DTYPES = (pl.Boolean, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64)


def assert_base_design_is_identifiable(df, base_vars, context, weights_col=None,
                                       treat_col='PX_on_ICI'):
    """Fail fast on a rank-deficient or degenerate base design matrix.

    The all-fits-failed guard in `_run_marker_screen` catches this too, but only
    after every marker has been fitted -- an hour of compute on a pan-cancer
    screen to learn that the base covariates were never identifiable. The base
    design does not depend on the marker, so it can be checked once, up front.

    Only structural degeneracies are fatal here: a rank-deficient matrix, a
    constant column, or a set of dummies forming a complete partition (no
    reference level) all make the Cox partial-likelihood Hessian singular for
    every marker. Near-collinearity is merely warned about -- it can still
    converge, and judging it is the analyst's call.

    `treat_col` is part of the Track 2 design but not Track 1's, which fits the
    ICI arm alone -- there the column is constant by construction and must be
    left out, or this guard reports its own scaffolding as the defect. Pass
    treat_col=None for a screen that does not model treatment.
    """
    fit_cols = ([treat_col] if treat_col else []) + [c for c in base_vars if c in df.columns]
    fit_cols = [c for c in dict.fromkeys(fit_cols) if c in df.columns]
    if not fit_cols:
        return
    finite = df.select(fit_cols).filter(
        pl.all_horizontal([pl.col(c).cast(pl.Float64, strict=False).is_finite()
                           for c in fit_cols]))
    if finite.height == 0:
        return   # empty frame is assert_model_covariates_numeric's territory
    X = finite.to_numpy().astype(float)

    problems = []
    const = [c for i, c in enumerate(fit_cols) if np.ptp(X[:, i]) == 0]
    if const:
        problems.append(f"constant column(s): {', '.join(const)}")

    for prefix in ('CANCER_TYPE_', 'PANEL_VERSION_', 'LINE_'):
        grp = [c for c in fit_cols
               if c.startswith(prefix)
               and set(np.unique(X[:, fit_cols.index(c)])).issubset({0.0, 1.0})]
        if len(grp) < 2:
            continue
        sums = X[:, [fit_cols.index(c) for c in grp]].sum(axis=1)
        if np.ptp(sums) == 0:
            problems.append(
                f"the {len(grp)} {prefix}* dummies sum to {sums[0]:g} on every row "
                f"(complete partition -- no reference level)")

    rank = np.linalg.matrix_rank(X)
    if rank < X.shape[1]:
        problems.append(f"rank {rank} < {X.shape[1]} columns (linearly dependent)")

    if problems:
        raise ValueError(
            f"{context}: base design matrix is not identifiable, so every Cox fit "
            f"in this screen would fail with 'matrix inversion problems'. "
            + "; ".join(problems)
            + ". Run `python -m pipelines.biomarkers.diagnose_iptw_inputs` for the "
              "full rank report."
        )

    with np.errstate(all='ignore'):
        sv = np.linalg.svd(X, compute_uv=False)
    if sv.size and sv[-1] > 0 and sv[0] / sv[-1] > 1e10:
        logger.warning("%s: base design is near-collinear (condition number %.2e); "
                       "fits may converge poorly.", context, sv[0] / sv[-1])


def assert_model_covariates_numeric(df, cols, context):
    """Fail fast on a non-numeric model covariate.

    Every fit funnels through `filter_finite_rows`, which casts with
    strict=False: a string column becomes all-null, fails `is_finite()`, and
    silently drops every row of the model frame. The fit then raises for each
    marker in turn, `_safe_fit` swallows it, and the screen writes a zero-row
    parquet that is indistinguishable downstream from "no significant hits".
    Catching it here names the offending column instead.
    """
    offenders = [(c, df.schema[c]) for c in cols
                 if c in df.schema and df.schema[c] not in NUMERIC_DTYPES]
    if offenders:
        detail = ", ".join(f"{c} ({dtype})" for c, dtype in offenders)
        raise TypeError(
            f"{context}: non-numeric model covariates would empty every model "
            f"frame: {detail}. Model covariates must be numeric or dummy-coded."
        )


def marker_has_within_arm_support(df, marker, treat_col='PX_on_ICI',
                                  min_pos_per_arm=10, min_neg_per_arm=10,
                                  min_events_per_group=5):
    marker_bin = df.select((finite_or_zero(marker) > 0).cast(pl.Int64)).to_series().to_numpy()
    treatment = df.select(finite_or_zero(treat_col).cast(pl.Int64)).to_series().to_numpy()
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
    marker_bin = df.select((finite_or_zero(marker) > 0).cast(pl.Int64)).to_series().to_numpy()
    treatment = df.select(finite_or_zero(treat_col).cast(pl.Int64)).to_series().to_numpy()
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


def compute_smd(df, covariates, treat_col='PX_on_ICI', weights=None):
    t_mask = (df[treat_col] == 1).to_numpy()
    rows = []
    for cov in covariates:
        x = df.select(finite_or_zero(cov)).to_series().to_numpy()
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
    df_fit_pl = filter_finite_rows(df.select(cols), cols)

    # Compute event counts before fitting
    event_counts = get_marker_event_counts(df_fit_pl, marker)

    df_fit = to_pandas_via_numpy(df_fit_pl)
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


def _run_track2_screen(type_df, cancer_type, base_vars, biomarker_cols, n_jobs):
    """Track 2: screen all markers with within-arm support via the interaction model.

    Returns one tagged frame per weighting, for the caller to stack into the
    single per-cancer-type results file. FDR is applied inside each weighting
    before tagging, so the multiple-testing correction stays within its own
    screen and is not diluted by the other track sharing the file.
    """
    track2_markers = [
        m for m in biomarker_cols
        if marker_has_within_arm_support(type_df, m, min_pos_per_arm=MIN_MARKER_POS_PER_ARM,
                                         min_neg_per_arm=MIN_MARKER_NEG_PER_ARM,
                                         min_events_per_group=MIN_EVENTS_PER_MARKER_GROUP)
    ]
    print(f"  Track 2 markers to test: {len(track2_markers)}")

    if len(track2_markers) < MIN_MARKERS_TO_TEST:
        print(f"  Skipping Track 2: fewer than {MIN_MARKERS_TO_TEST} markers with sufficient support.")
        return []

    frames = []
    for spec_name, spec_weights in [('ATE', 'IPTW_ATE'), ('noIPTW', None)]:
        results, failed = _run_marker_screen(
            type_df, track2_markers, base_vars, spec_weights,
            _fit_track2_marker, n_jobs, label=f"T2 {cancer_type} {spec_name}")
        if failed:
            print(f"  Track 2 {spec_name} failures: {len(failed)}. "
                  f"First: {failed[0][0]} -> {failed[0][1]}")
        spec_df = pl.DataFrame(results, schema=TRACK2_RESULT_COLS) if results else pl.DataFrame(schema={c: pl.Float64 for c in TRACK2_RESULT_COLS})
        spec_df = add_track2_fdr_and_labels(spec_df)
        frames.append(spec_df.with_columns(
            pl.lit(2, dtype=pl.Int8).alias('track'),
            pl.lit(spec_name, dtype=pl.Utf8).alias('weight_type'),
        ))
    return frames


# =============================================
# Per-cancer-type diagnostics, one long-format file
# =============================================

DIAGNOSTIC_SECTIONS = ('propensity_score', 'cohort', 'balance_ATE')


def _melt_diagnostic(wide, section, key_col=None):
    """Melt a wide diagnostic frame into (section, key, metric, value) rows.

    The three diagnostics have incompatible widths — one row per treatment arm,
    one row for the cancer type, one row per covariate — so they share a file
    in long form rather than as a sparse column union. Values are cast to
    Float64; every diagnostic here is numeric.
    """
    metrics = [c for c in wide.columns if c != key_col]
    key_expr = (pl.col(key_col).cast(pl.Utf8) if key_col is not None
                else pl.lit(None, dtype=pl.Utf8))
    return wide.select(
        key_expr.alias('key'),
        *[pl.col(m).cast(pl.Float64) for m in metrics],
    ).unpivot(
        index='key', on=metrics, variable_name='metric', value_name='value',
    ).select(
        pl.lit(section, dtype=pl.Utf8).alias('section'), 'key', 'metric', 'value',
    )


def read_diagnostic_section(path, section):
    """Pivot one section of a `{cancer_type}_diagnostics.parquet` back to wide form.

    Consumers that want a specific diagnostic (compile_IPTW_results wants the
    cohort counts) should go through this rather than re-deriving the unpivot.
    Returns an empty frame if the section is absent.
    """
    long = pl.read_parquet(path).filter(pl.col('section') == section)
    if long.is_empty():
        return pl.DataFrame()
    return long.pivot(on='metric', index='key', values='value')


# =============================================
# Parallel runner
# =============================================

def _run_marker_screen(df, markers, base_vars, weights_col, fit_fn, n_jobs, label=""):
    def _safe_fit(marker):
        try:
            return fit_fn(df, marker, base_vars, weights_col), None
        except Exception as e:
            return None, (marker, str(e))

    # inner_max_num_threads is belt-and-braces alongside the module-level env
    # caps: it re-applies the BLAS/OpenMP limits inside each worker even if the
    # executor was created before those were set.
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        raw = Parallel()(
            delayed(_safe_fit)(m) for m in tqdm(markers, desc=label)
        )
    results = [r for r, _ in raw if r is not None]
    failed = [f for _, f in raw if f is not None]

    # A screen where every fit failed is a broken run, not a null result. Left
    # unchecked it writes a zero-row parquet that compile_IPTW_results reads
    # without complaint and reports as "0 significant hits" — the failure is
    # indistinguishable from a genuine finding of nothing. The usual cause is a
    # non-numeric column in `base_vars`: `filter_finite_rows` casts with
    # strict=False, so one string column casts to all-null and empties the model
    # frame for every marker. Fail loudly instead, and name the reasons.
    if markers and not results:
        tally = Counter(message for _marker, message in failed)
        detail = "; ".join(f"{count}x {message}" for message, count in tally.most_common(3))
        raise RuntimeError(
            f"{label}: all {len(markers)} marker fits failed, so this screen has no "
            f"results to write. Distinct errors: {detail}. Run "
            f"`python -m pipelines.biomarkers.diagnose_iptw_inputs` to see which "
            f"column empties the model frame."
        )
    if failed:
        tally = Counter(message for _marker, message in failed)
        logger.warning(
            "%s: %d/%d fits failed. Most common: %s",
            label, len(failed), len(markers),
            "; ".join(f"{count}x {message}" for message, count in tally.most_common(3)),
        )

    # A fit that returns non-finite coefficients is a failure that did not raise.
    # lifelines can converge onto a degenerate design (a covariate constant on the
    # frame actually being fit) and hand back NaN betas rather than erroring, so
    # `_safe_fit` records it as a success and the all-failed guard above never
    # trips. The screen then writes rows whose beta/p are null, which reads
    # downstream as "tested, not significant" -- the same silent-nothing failure
    # in a different disguise. Check the effect estimate the screen exists to
    # produce, and fail if none of them is usable.
    if results:
        beta_key = next((k for k in ('beta_marker', 'beta_markerxICI')
                         if k in results[0]), None)
        if beta_key is not None:
            n_null = sum(1 for r in results if not np.isfinite(r.get(beta_key, np.nan)))
            if n_null == len(results):
                raise RuntimeError(
                    f"{label}: all {len(results)} fits returned a non-finite "
                    f"{beta_key}, so this screen has no usable estimates. The fits "
                    f"did not raise -- the design is degenerate on the frame being "
                    f"fit (a covariate constant within this screen's subset). Run "
                    f"`python -m pipelines.biomarkers.diagnose_iptw_inputs`."
                )
            if n_null:
                logger.warning("%s: %d/%d fits returned a non-finite %s.",
                               label, n_null, len(results), beta_key)
    return results, failed


def main() -> None:
    random.seed(42)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger.info(
        "Parallelism: n_jobs=%d workers, POLARS_MAX_THREADS=%s, OMP_NUM_THREADS=%s "
        "(%d cores visible). Total thread budget is roughly n_jobs x threads-per-worker; "
        "override with IPTW_N_JOBS.",
        N_JOBS, os.environ.get("POLARS_MAX_THREADS"), os.environ.get("OMP_NUM_THREADS"),
        os.cpu_count() or 0,
    )

    # ============================================================
    # Main loop: iterate over cohort x ps_model combinations
    # ============================================================
    for COHORT in COHORTS:
        for PS_MODEL in PS_MODELS:
            SPEC_LABEL = f'{COHORT}_{PS_MODEL}'
            # A smoke run's numbers are not comparable to a real one's (see
            # MARKER_FRACTION), so it gets its own directory. Without this,
            # compile_IPTW_results would happily pick up subsampled parquets and
            # report them as the real screen -- the same silent-wrong-answer
            # failure mode the all-fits-failed guard exists to prevent.
            run_dir = f'IPTW_runs_{SPEC_LABEL}' + ('_smoke' if IS_SMOKE_RUN else '')
            RUN_PATH = os.path.join(BIOMARKER_PATH, run_dir + '/')
            os.makedirs(RUN_PATH, exist_ok=True)

            print(f"\n{'#'*60}")
            print(f"[run_IPTW_analysis] Spec: {SPEC_LABEL}")
            print(f"[run_IPTW_analysis] Output: {RUN_PATH}")
            print(f"{'#'*60}")

            # === Load data ===
            input_file = os.path.join(BIOMARKER_PATH, f'IPTW_df_{SPEC_LABEL}.parquet')
            full_df = pl.read_parquet(input_file)

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

            # Smoke-test subsample. Done once here, before the cancer-type loop,
            # so every cancer type screens the same markers.
            if IS_SMOKE_RUN:
                n_total = len(biomarker_cols)
                biomarker_cols = resolve_marker_subset(biomarker_cols)
                cap = (f"IPTW_MAX_MARKERS={MAX_MARKERS}" if MAX_MARKERS is not None
                       else f"IPTW_MARKER_FRACTION={MARKER_FRACTION}")
                print(f"  *** SMOKE RUN: {cap} -> screening "
                      f"{len(biomarker_cols)} of {n_total} markers. "
                      f"Results are NOT valid (FDR is computed over the subset); "
                      f"writing to a _smoke directory. ***")

            # === Identify embedding columns for prognostic score ===
            embedding_cols = [col for col in full_df.columns
                              if ('IMAGING' in col) or ('PATHOLOGY' in col) or ('CLINICIAN' in col)]
            print(f"  {len(embedding_cols)} embedding columns found for prognostic score")

            # === Cancer types to test ===
            available_types = {col.replace('CANCER_TYPE_', '') for col in cancer_type_cols}
            types_to_test = ['pan_cancer'] + sorted(
                t for t in available_types
                if t not in EXCLUDE_TYPES
                and (INCLUDE_TYPES is None or t in INCLUDE_TYPES)
            )
            if INCLUDE_TYPES is not None:
                # A label that matches no dummy column screens nothing, and the run
                # still finishes clean — the same silent-nothing failure mode the
                # all-fits-failed guard exists to prevent. Name it instead.
                missing = sorted(INCLUDE_TYPES - available_types)
                if missing:
                    raise ValueError(
                        f"INCLUDE_TYPES names cancer types with no CANCER_TYPE_* column: "
                        f"{', '.join(missing)}. Available: {', '.join(sorted(available_types))}."
                    )
                print(f"  Cancer-type-specific screens restricted to: "
                      f"{', '.join(sorted(INCLUDE_TYPES))}")

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
                    # Anchor on the dummy prefixes, not a loose substring test.
                    # build_cancer_type_df/build_somatic_data_df deliberately keep
                    # the raw string label ('CANCER_TYPE', 'PANEL_VERSION') beside
                    # their dummies; `'CANCER_TYPE' in c` swept that string column
                    # into base_vars, which emptied every pan-cancer model frame.
                    panel_cols_fit = [c for c in type_df.columns
                                      if c.upper().startswith('PANEL_VERSION_')]
                    # Use the merge's own column list, not a fresh scan of the frame:
                    # it has the reference level dropped and the all-zero columns
                    # removed. Re-deriving with startswith() here would put the
                    # complete dummy partition back into the model and re-create the
                    # singular Hessian.
                    ct_cols_fit = pan_ct_cols
                    base_vars = base_covars + line_cols + panel_cols_fit + ct_cols_fit
                    print(f"  Cancer-type dummies in model: {len(ct_cols_fit)} "
                          f"(reference level dropped)")
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
                    panel_cols_fit = [c for c in type_df.columns
                                      if c.upper().startswith('PANEL_VERSION_')]
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

                # Common-support trimming can empty a covariate that was well
                # populated when base_vars was assembled: a group with no
                # propensity overlap loses every row, and its dummy becomes an
                # all-zero column. That is a singular Cox Hessian, and lifelines
                # fails *every* marker fit in the screen with "matrix inversion
                # problems" -- an hour of compute to learn the design was never
                # identifiable. MIN_CANCER_TYPE_TOTAL only bounds the pre-trim
                # count, so it cannot prevent this; the columns must be rechecked
                # against the trimmed frame.
                #
                # Dropping these is bookkeeping, not a change of estimand: the
                # patients are already gone from the cohort, removed by the trim
                # rather than absorbed into the reference level. The cancer-type
                # reference (CANCER_TYPE_OTHER) was dropped upstream in
                # merge_rare_cancer_types_into_other, so removing constant
                # columns here cannot destroy it and what remains stays a proper
                # reference-dropped set. Applies to PANEL_VERSION_* and LINE_*
                # too, which are raw prefix scans with no post-trim recheck.
                constant_after_trim = [c for c in base_vars
                                       if type_df[c].n_unique() <= 1]
                if constant_after_trim:
                    print(f"  Dropping {len(constant_after_trim)} covariate(s) left "
                          f"constant by common-support trimming: "
                          f"{', '.join(constant_after_trim)}")
                    base_vars = [c for c in base_vars if c not in constant_after_trim]

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
                diag_frames = []

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
                diag_frames.append(_melt_diagnostic(
                    ps_summary, 'propensity_score', key_col='PX_on_ICI'))

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

                diag_frames.append(_melt_diagnostic(pl.DataFrame([{
                    'N_treated': n_treated, 'N_control': n_control,
                    'events_treated': events_treated, 'events_control': events_control,
                    'event_rate_treated': events_treated / max(n_treated, 1),
                    'event_rate_control': events_control / max(n_control, 1),
                    'PS_AUC': ps_auc,
                    'ESS_ATE_treated': ess_t, 'ESS_ATE_control': ess_c,
                }]), 'cohort'))

                balance_covars = base_covars + line_cols + [
                    c for c in type_df.columns
                    if c.startswith('CANCER_TYPE_') or c.upper().startswith('PANEL_VERSION_')]
                smd_ate = compute_smd(type_df, balance_covars, weights=w_ate_trunc)
                diag_frames.append(_melt_diagnostic(
                    smd_ate, 'balance_ATE', key_col='covariate'))

                # Max SMD check (balance quality indicator)
                max_smd_ate = float(smd_ate['SMD_weighted'].abs().max())
                n_imbalanced_ate = int((smd_ate['SMD_weighted'].abs() > 0.1).sum())
                print(f"  ATE balance: max|SMD|={max_smd_ate:.4f}, {n_imbalanced_ate}/{len(smd_ate)} covariates with |SMD|>0.1")

                pl.concat(diag_frames).with_columns(
                    pl.lit(cancer_type, dtype=pl.Utf8).alias('cancer_type')
                ).select('cancer_type', 'section', 'key', 'metric', 'value').write_parquet(
                    os.path.join(RUN_PATH, f'{cancer_type}_diagnostics.parquet'))

                assert_model_covariates_numeric(
                    type_df, base_vars + ['tt_death', 'death', 'PX_on_ICI'],
                    f"{SPEC_LABEL}/{cancer_type}")

                # Checked once here rather than discovered 1492 fits later: the
                # base design is marker-independent, so if it is singular every
                # fit in the screen is already doomed.
                assert_base_design_is_identifiable(
                    type_df, base_vars, f"{SPEC_LABEL}/{cancer_type}")

                # === Full-cohort interaction screen ===
                frames = _run_track2_screen(type_df, cancer_type, base_vars, biomarker_cols, N_JOBS)

                # One results file per cancer type. `weight_type` identifies which
                # weighting a row came from; `track` is retained as a constant 2 so
                # downstream readers and already-written parquets keep one schema.
                if frames:
                    results_df = pl.concat(frames, how='diagonal_relaxed')
                    results_df = results_df.select(
                        pl.lit(cancer_type, dtype=pl.Utf8).alias('cancer_type'),
                        'track', 'weight_type',
                        *[c for c in results_df.columns if c not in ('track', 'weight_type')],
                    )
                    results_file = os.path.join(RUN_PATH, f'{cancer_type}_results.parquet')
                    results_df.write_parquet(results_file)
                    print(f"  Wrote {results_df.height} result rows to {results_file}")
                else:
                    print(f"  No screen ran for {cancer_type}; no results file written.")

            print(f"\n[run_IPTW_analysis] Done with {SPEC_LABEL}. Results in {RUN_PATH}")


if __name__ == "__main__":
    main()
