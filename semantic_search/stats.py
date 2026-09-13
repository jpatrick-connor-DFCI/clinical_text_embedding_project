"""Association tests used by the semantic-search exploratory analyses.

Written fresh: the repo has no generic stats-helper module.  Its only prior
hypothesis testing is the Wald p-value and BH-FDR in
`pipelines/biomarkers/run_IPTW_analysis.py`, plus `kruskal_p`/`wilcoxon_vs0` on
the R side (`R/figure_utils.R`).  `scipy` and `statsmodels` are both already
dependencies (via scikit-survival / statsmodels in environment.yml).

Every function returns plain dicts or Polars frames and never raises on a
degenerate input -- a variable with one non-null level, or a comparison group with no
observations, yields a NaN statistic and a null p-value rather than aborting the
sweep.  A screen over hundreds of sparse somatic markers must not die on the
first all-zero column.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Expected-count threshold below which a 2xk table goes to Fisher rather than
# chi-square.  Cochran's rule of thumb; also what makes the sparse somatic
# markers testable at all.
MIN_EXPECTED_COUNT = 5.0
FDR_ALPHA = 0.05


def _finite_pairs(x, y) -> tuple[np.ndarray, np.ndarray]:
    """Cast two numeric arrays and retain rows finite in both."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def spearman_test(pc_values, clinical_values) -> dict:
    """Spearman association between one PC and one continuous characteristic."""
    x, y = _finite_pairs(pc_values, clinical_values)
    n = int(len(x))
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {
            "test": "spearman",
            "statistic": float("nan"),
            "effect": float("nan"),
            "effect_name": "spearman_rho",
            "p": None,
            "n": n,
            "n_levels": None,
        }
    rho, p = stats.spearmanr(x, y)
    return {
        "test": "spearman",
        "statistic": float(rho),
        "effect": float(rho),
        "effect_name": "spearman_rho",
        "p": float(p),
        "n": n,
        "n_levels": None,
    }


def categorical_pc_test(pc_values, levels) -> dict:
    """Kruskal-Wallis test of one PC across levels of a clinical variable.

    Epsilon-squared is reported as a scale-free omnibus effect size. It is
    unsigned because variables with more than two levels have no single
    clinically meaningful direction.
    """
    values = np.asarray(pc_values, dtype=float)
    levels = np.asarray(levels, dtype=object)
    mask = np.isfinite(values) & np.array([
        value is not None and not (isinstance(value, float) and math.isnan(value))
        for value in levels
    ])
    values, levels = values[mask], levels[mask]
    unique_levels = sorted(set(levels.tolist()), key=str)
    samples = [values[levels == level] for level in unique_levels]
    n = int(len(values))
    k = len(samples)
    if n < 3 or k < 2 or np.unique(values).size < 2:
        return {
            "test": "kruskal",
            "statistic": float("nan"),
            "effect": float("nan"),
            "effect_name": "epsilon_squared",
            "p": None,
            "n": n,
            "n_levels": k,
        }
    statistic, p = stats.kruskal(*samples)
    epsilon_squared = (
        min(1.0, max(0.0, float((statistic - k + 1) / (n - k))))
        if n > k
        else float("nan")
    )
    return {
        "test": "kruskal",
        "statistic": float(statistic),
        "effect": epsilon_squared,
        "effect_name": "epsilon_squared",
        "p": float(p),
        "n": n,
        "n_levels": k,
    }


def _clean_pairs(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is null/NaN."""
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    mask = ~np.isnan(values)
    return values[mask], groups[mask]


def kruskal_test(values, groups) -> dict:
    """Kruskal-Wallis across cluster labels for a continuous variable.

    Non-parametric by choice: age is skewed, N_MET_SITES is a small count, and
    PGS scores are standardized but not guaranteed normal within cluster.
    """
    values, groups = _clean_pairs(values, groups)
    samples = [values[groups == g] for g in np.unique(groups)]
    samples = [s for s in samples if len(s) > 0]
    if len(samples) < 2 or all(len(s) < 2 for s in samples):
        return {"test": "kruskal", "statistic": float("nan"), "p": None, "n": int(len(values))}
    # Constant input makes the H statistic undefined (zero variance in ranks).
    if np.unique(values).size < 2:
        return {"test": "kruskal", "statistic": float("nan"), "p": None, "n": int(len(values))}
    stat, p = stats.kruskal(*samples)
    return {"test": "kruskal", "statistic": float(stat), "p": float(p), "n": int(len(values))}


def _contingency(levels: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list, list]:
    """Counts table with rows = variable levels, cols = cluster labels."""
    row_vals = [v for v in np.unique(levels[levels != None]) if v is not None]  # noqa: E711
    col_vals = list(np.unique(groups))
    table = np.zeros((len(row_vals), len(col_vals)), dtype=int)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            table[i, j] = int(np.sum((levels == rv) & (groups == cv)))
    return table, list(row_vals), col_vals


def categorical_test(levels, groups) -> dict:
    """Chi-square across clusters for a categorical variable, falling back to
    Fisher's exact test when any expected cell is below MIN_EXPECTED_COUNT.

    Fisher is only exact for 2x2 in scipy; for larger sparse tables scipy has no
    exact routine, so those fall back to a Monte Carlo-free chi-square with the
    low-expected-count caveat recorded in the `test` field rather than silently
    reported as if the asymptotics held.
    """
    levels = np.asarray(levels, dtype=object)
    groups = np.asarray(groups)
    mask = np.array([v is not None and not (isinstance(v, float) and math.isnan(v))
                     for v in levels])
    levels, groups = levels[mask], groups[mask]

    n = int(len(levels))
    if n == 0:
        return {"test": "chi2", "statistic": float("nan"), "p": None, "n": 0}

    table, row_vals, col_vals = _contingency(levels, groups)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "chi2", "statistic": float("nan"), "p": None, "n": n}
    # A row or column of all zeros makes chi2_contingency raise.
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"test": "chi2", "statistic": float("nan"), "p": None, "n": n}

    chi2, p, _, expected = stats.chi2_contingency(table)
    sparse = bool((expected < MIN_EXPECTED_COUNT).any())
    if sparse and table.shape == (2, 2):
        odds, p_exact = stats.fisher_exact(table)
        return {"test": "fisher", "statistic": float(odds), "p": float(p_exact), "n": n}
    return {
        "test": "chi2_sparse" if sparse else "chi2",
        "statistic": float(chi2),
        "p": float(p),
        "n": n,
    }


def one_vs_rest_enrichment(levels, groups, level_value, cluster_value) -> dict:
    """2x2 enrichment of one variable level within one cluster vs all others.

    Returns the odds ratio with a Haldane-Anscombe 0.5 correction when a cell is
    empty (an infinite OR is unplottable and uninformative), and Fisher's p.
    """
    levels = np.asarray(levels, dtype=object)
    groups = np.asarray(groups)
    is_level = levels == level_value
    in_cluster = groups == cluster_value

    a = int(np.sum(is_level & in_cluster))
    b = int(np.sum(~is_level & in_cluster))
    c = int(np.sum(is_level & ~in_cluster))
    d = int(np.sum(~is_level & ~in_cluster))

    n_cluster = a + b
    n_other = c + d
    if n_cluster == 0 or n_other == 0:
        return {"n": a, "pct": float("nan"), "pct_overall": float("nan"),
                "odds_ratio": float("nan"), "p": None}

    if min(a, b, c, d) == 0:
        or_ = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    else:
        or_ = (a * d) / (b * c)
    _, p = stats.fisher_exact([[a, b], [c, d]])

    return {
        "n": a,
        "pct": 100.0 * a / n_cluster,
        "pct_overall": 100.0 * (a + c) / (n_cluster + n_other),
        "odds_ratio": float(or_),
        "p": float(p),
    }


def continuous_by_cluster(values, groups, cluster_value) -> dict:
    """Mean/median of a continuous variable inside one cluster vs the rest,
    with a Mann-Whitney p."""
    values, groups = _clean_pairs(values, groups)
    inside = values[groups == cluster_value]
    outside = values[groups != cluster_value]
    if len(inside) < 2 or len(outside) < 2:
        return {"n": int(len(inside)), "mean": float("nan"), "median": float("nan"),
                "mean_overall": float("nan"), "p": None}
    try:
        _, p = stats.mannwhitneyu(inside, outside, alternative="two-sided")
    except ValueError:
        p = None  # identical distributions with zero variance
    return {
        "n": int(len(inside)),
        "mean": float(np.mean(inside)),
        "median": float(np.median(inside)),
        "mean_overall": float(np.mean(values)),
        "p": float(p) if p is not None else None,
    }


def add_fdr_within(
    df: pl.DataFrame,
    group_cols: list[str],
    p_col: str = "p",
    fdr_col: str = "fdr",
    sig_col: str = "significant",
    alpha: float = FDR_ALPHA,
) -> pl.DataFrame:
    """Benjamini-Hochberg FDR applied independently within each group.

    Mirrors `_fdr_within_mutation_type` in run_IPTW_analysis.py, which stratifies
    BH within mutation type rather than pooling.  Here the strata are
    (space, window, family): the somatic and PRS families carry hundreds of
    tests each, and pooling them with the two demographic tests would bury the
    latter under a correction they did not earn.

    Rows with a null p (a test that could not be run) carry a null FDR and are
    excluded from the correction's denominator rather than counted as failures.
    """
    if df.height == 0:
        return df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias(fdr_col),
            pl.lit(None, dtype=pl.Boolean).alias(sig_col),
        ])

    out_frames = []
    for _, sub in df.group_by(group_cols, maintain_order=True):
        testable = sub.filter(pl.col(p_col).is_not_null())
        if testable.height == 0:
            out_frames.append(sub.with_columns([
                pl.lit(None, dtype=pl.Float64).alias(fdr_col),
                pl.lit(None, dtype=pl.Boolean).alias(sig_col),
            ]))
            continue
        pvals = testable.get_column(p_col).to_numpy().astype(float)
        rejected, fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        testable = testable.with_columns([
            pl.Series(fdr_col, fdr, dtype=pl.Float64),
            pl.Series(sig_col, rejected, dtype=pl.Boolean),
        ])
        untestable = sub.filter(pl.col(p_col).is_null()).with_columns([
            pl.lit(None, dtype=pl.Float64).alias(fdr_col),
            pl.lit(None, dtype=pl.Boolean).alias(sig_col),
        ])
        out_frames.append(pl.concat([testable, untestable], how="vertical_relaxed"))

    return pl.concat(out_frames, how="vertical_relaxed")
