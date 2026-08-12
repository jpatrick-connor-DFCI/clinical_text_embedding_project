"""Pre-compute inputs for Figure 4 (mortality-risk *dynamics*).

Patients are grouped by the SLOPE of their model-estimated mortality risk over
months 0..L, where L = SLOPE_LANDMARK_MONTHS. Each per-patient OLS slope uses
only that patient's observed landmarks (at least MIN_SLOPE_POINTS; no imputation).
The resulting landmark cohort is patients alive at L, rather than patients
selected for near-complete follow-up through a later landmark, so the figure
surfaces whose risk is *changing* (Falling / Stable / Rising) as clinical notes
accumulate over early follow-up.

Writes to FIGURE_DATA_DIR:
- fig4_trajectories_heatmap.csv        DFCI_MRN, cluster, <month columns kept>, downsampled to
                                       <= HEATMAP_ROWS_PER_CLUSTER per slope group and ordered by
                                       within-group mean risk (panel A reads this)
- fig4_km_data.csv                     DFCI_MRN, cluster, death, tt_death, stage,
                                       landmark_month  (stage = major stage I-IV for the
                                       within-stage supplement, NaN if unknown)
- fig4_cluster_severity.csv            cluster, mean_met_sites, rmst_months, pct_stage_iv,
                                       pct_ici, mean_slope, n_patients
- fig4_group_trajectories.csv          group, month, mean_risk, q25, q75  (per slope group +
                                       a "cohort" pseudo-group = cohort-wide average band)
- fig4_slope_by_stage.csv              stage, cluster, n_patients, mean_slope  (stage-matched
                                       breakdown: dynamics groups occur across every major stage)
- fig4_silhouette.csv                  k, silhouette  (slope-group-count justification, appendix)
"""

from __future__ import annotations

import argparse
import os
import re

# Some cluster nodes advertise more CPUs than the precompiled OpenBLAS build
# supports. Cap inherited/default thread counts before NumPy, lifelines, or
# scikit-learn initializes a BLAS runtime; otherwise OpenBLAS can segfault while
# allocating its thread metadata rather than raising a Python exception.
_BLAS_THREAD_LIMIT = 8
for _thread_var in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    try:
        _configured_threads = int(os.environ.get(_thread_var, _BLAS_THREAD_LIMIT))
    except ValueError:
        _configured_threads = _BLAS_THREAD_LIMIT + 1
    if not 1 <= _configured_threads <= _BLAS_THREAD_LIMIT:
        os.environ[_thread_var] = str(_BLAS_THREAD_LIMIT)
    else:
        os.environ.setdefault(_thread_var, str(_BLAS_THREAD_LIMIT))

import numpy as np
import polars as pl
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from config import FEATURE_PATH, RESULTS_PATH, SURV_PATH
from figures.io import save_figure_data
from shared.stages import STAGE_ORDER, is_stage_iv, load_stage_map, normalize_stage


N_SLOPE_GROUPS = 3  # Falling / Stable / Rising risk dynamics
SLOPE_LANDMARK_MONTHS = 12  # L; set to 24 for the longer-window variant
MIN_SLOPE_POINTS = 3
DEFAULT_DECAY = 0.1
HEATMAP_ROWS_PER_CLUSTER = 500

KM_COLUMNS = ["DFCI_MRN", "cluster", "death", "tt_death", "stage", "landmark_month"]
SEVERITY_COLUMNS = ["cluster", "mean_met_sites", "rmst_months",
                    "pct_stage_iv", "pct_ici", "mean_slope", "n_patients"]
GROUP_TRAJECTORY_COLUMNS = ["group", "month", "mean_risk", "q25", "q75"]
SLOPE_BY_STAGE_COLUMNS = ["stage", "cluster", "n_patients", "mean_slope"]
SILHOUETTE_COLUMNS = ["k", "silhouette"]
MET_EVENTS = ["brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"]
RMST_TAU_MONTHS = 120  # 10-year horizon from first treatment
SILHOUETTE_K_RANGE = range(2, 9)


def _major_stage_map() -> dict[int, str] | None:
    """MRN -> major stage (I-IV) from cancer_stage_df.csv.gz's raw CANCER_STAGE
    column. None on read failure."""
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  slope-by-stage will be empty")
        return None
    out: dict[int, str] = {}
    for mrn, raw in mrn_to_stage.items():
        stg = normalize_stage(raw)
        if stg is not None:
            out[mrn] = stg
    return out


def _stage_iv_mrns() -> set[int] | None:
    """MRNs with raw stage value normalizing to IV. None on read failure."""
    mrn_to_stage = load_stage_map()
    if mrn_to_stage is None:
        print("  pct_stage_iv will be NaN")
        return None
    return {mrn for mrn, raw in mrn_to_stage.items() if is_stage_iv(raw)}


def _find_ici_column(cols: list[str]) -> str | None:
    """Find the PX_on_<MOA_Category> column that corresponds to ICI treatment.
    The MOA category strings come from GPT_generated_med_classes.csv (free-form),
    so match on a normalized form rather than a literal column name."""
    def _norm(s: str) -> str:
        return re.sub(r"[\s_\-]+", " ", s).strip().lower()
    for c in cols:
        bare = _norm(c.replace("PX_on_", "", 1))
        if bare in {"immune checkpoint inhibitors", "immune checkpoint inhibitor",
                    "ici", "immunotherapy"}:
            return c
    return None


def _default_trajectory_input(decay: float) -> str:
    """Canonical output path of generate_mortality_trajectories.py."""
    return os.path.join(
        RESULTS_PATH, "death_met_results", "mortality_trajectories",
        f"survival_trajectories_w_decay_param_{decay}.csv",
    )


def _scaled_trajectory_matrix(input_path: str) -> tuple[pl.DataFrame, list[str]]:
    """Load the un-imputed trajectory window and retain sufficiently observed rows.

    Returns ``(traj_sub, months_to_keep)`` for landmarks in months 0..L. Patients
    must have at least ``MIN_SLOPE_POINTS`` observed values in the window. Despite
    the historical function name, standardization now happens only after the
    per-patient slope feature is computed.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Trajectory input not found: {input_path}\n"
            "Run pipelines/trajectories/generate_mortality_trajectories.py first, "
            "or pass --input to override."
        )
    traj_df = pl.read_csv(input_path)
    monthly_cols = [c for c in traj_df.columns if c != "DFCI_MRN"]
    n_rows = len(traj_df)
    missing_colwise = {c: int(traj_df[c].null_count()) for c in monthly_cols}
    months_to_keep = [c for c in monthly_cols if missing_colwise[c] < 0.9 * n_rows]
    month_nums = _month_nums(months_to_keep)
    months_to_keep = [col for col, month in zip(months_to_keep, month_nums)
                      if month <= SLOPE_LANDMARK_MONTHS]
    if months_to_keep:
        non_null_counts = traj_df.select(
            pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int64) for c in months_to_keep]).alias("_n")
        )["_n"].to_numpy()
    else:
        non_null_counts = np.zeros(n_rows, dtype=int)
    keep_mask = non_null_counts >= MIN_SLOPE_POINTS
    traj_sub = traj_df.filter(pl.Series(keep_mask)).select(["DFCI_MRN"] + months_to_keep)
    return traj_sub, months_to_keep


def _month_nums(month_cols: list[str]) -> np.ndarray:
    """Parse the numeric month from each 'plus_<n>_months_data' column name."""
    return np.array([int(re.search(r"\d+", c).group()) for c in month_cols], dtype=float)


def _ols_slopes(traj_sub: pl.DataFrame, months_to_keep: list[str]) -> np.ndarray:
    """Per-patient OLS risk slope using each row's observed landmarks in 0..L."""
    x = _month_nums(months_to_keep)
    values = traj_sub.select(months_to_keep).to_numpy().astype(float)
    slopes = np.full(len(traj_sub), np.nan, dtype=float)
    for i, y in enumerate(values):
        mask = ~np.isnan(y)
        if mask.sum() < MIN_SLOPE_POINTS:
            continue
        x_obs = x[mask]
        y_obs = y[mask]
        x_centered = x_obs - x_obs.mean()
        denom = np.sum(x_centered ** 2)
        if denom <= 0:
            continue
        slopes[i] = np.sum(x_centered * (y_obs - y_obs.mean())) / denom
    return slopes


def _cluster_trajectories(input_path: str) -> tuple[pl.DataFrame, list[str]]:
    """Group patients by risk-trajectory SLOPE (Falling / Stable / Rising).

    Returns (traj_sub with `cluster` and `slope` cols, months_to_keep). `cluster`
    is an integer 0..N-1 relabeled by ASCENDING mean slope
    (0 = most falling risk … N-1 = most rising), preserving the integer-cluster
    contract the KM / severity / heatmap code already keys on.
    """
    traj_sub, months_to_keep = _scaled_trajectory_matrix(input_path)
    if len(traj_sub) < N_SLOPE_GROUPS:
        print(f"  only {len(traj_sub)} patients survived filtering; need >= {N_SLOPE_GROUPS} "
              "for slope grouping. Emitting empty figure-data.")
        empty = pl.DataFrame(schema={c: pl.Float64 for c in ["DFCI_MRN", "cluster", "slope", *months_to_keep]})
        return empty, months_to_keep

    slopes = _ols_slopes(traj_sub, months_to_keep)
    finite = np.isfinite(slopes)
    if not finite.all():
        traj_sub = traj_sub.filter(pl.Series(finite))
        slopes = slopes[finite]
    if len(traj_sub) < N_SLOPE_GROUPS:
        print(f"  only {len(traj_sub)} patients have finite slopes; need >= {N_SLOPE_GROUPS} "
              "for slope grouping. Emitting empty figure-data.")
        empty = pl.DataFrame(schema={c: pl.Float64 for c in ["DFCI_MRN", "cluster", "slope", *months_to_keep]})
        return empty, months_to_keep
    traj_sub = traj_sub.with_columns(pl.Series("slope", slopes))

    # Cluster on the standardized 1-D slope so the Falling/Stable/Rising boundaries
    # are data-driven rather than arbitrary thresholds.
    slope_z = StandardScaler().fit_transform(slopes.reshape(-1, 1))
    raw_labels = KMeans(n_clusters=N_SLOPE_GROUPS, random_state=0, n_init=10).fit_predict(slope_z)

    # Relabel groups by ascending mean slope so cluster 0 is always most-falling.
    # Mirrors the previous ascending-mean-risk relabel; stabilizes labels across reruns.
    group_mean_slope = {k: slopes[raw_labels == k].mean() for k in np.unique(raw_labels)}
    label_order = sorted(group_mean_slope, key=group_mean_slope.get)
    relabel = {old: new for new, old in enumerate(label_order)}
    cluster_vals = [relabel[lbl] for lbl in raw_labels]
    traj_sub = traj_sub.with_columns(pl.Series("cluster", cluster_vals, dtype=pl.Int64))
    return traj_sub, months_to_keep


def _silhouette_scan(input_path: str) -> pl.DataFrame:
    """Silhouette score vs k on the standardized slope feature — justifies N_SLOPE_GROUPS."""
    traj_sub, months_to_keep = _scaled_trajectory_matrix(input_path)
    if len(traj_sub) <= max(SILHOUETTE_K_RANGE):
        return pl.DataFrame(schema={c: pl.Float64 for c in SILHOUETTE_COLUMNS})
    slopes = _ols_slopes(traj_sub, months_to_keep)
    slopes = slopes[np.isfinite(slopes)]
    if len(slopes) <= max(SILHOUETTE_K_RANGE):
        return pl.DataFrame(schema={c: pl.Float64 for c in SILHOUETTE_COLUMNS})
    slope_z = StandardScaler().fit_transform(slopes.reshape(-1, 1))
    rows = []
    for k in SILHOUETTE_K_RANGE:
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(slope_z)
        rows.append({"k": int(k), "silhouette": float(silhouette_score(slope_z, labels))})
    return pl.DataFrame(rows).select(SILHOUETTE_COLUMNS) if rows else pl.DataFrame(schema={c: pl.Float64 for c in SILHOUETTE_COLUMNS})


def _heatmap_downsample(traj_sub: pl.DataFrame, months_to_keep: list[str],
                         rows_per_cluster: int = HEATMAP_ROWS_PER_CLUSTER) -> pl.DataFrame:
    """Within each cluster, sample at most `rows_per_cluster` patients and order
    them by within-cluster mean trajectory so the heatmap shows clean cluster
    blocks rather than an unreadable smear of 30k+ rows.

    Saved values are column-wise z-scored — each month's distribution is centered
    on 0 with unit SD across the cohort. The raw mortality
    risk varies hugely across months (e.g. by 100× near event time), washing out
    cluster structure under a single linear color scale. Z-scoring makes each
    cluster's relative-to-cohort signal pop on a diverging scale centered at 0.
    """
    if traj_sub.is_empty():
        return traj_sub.clear()

    # Panel A reads only DFCI_MRN + cluster + month columns (it treats every other
    # column as a month), so drop the per-patient `slope` helper column here.
    keep = ["DFCI_MRN", "cluster"] + list(months_to_keep)
    scaled = traj_sub.select(keep)
    # Display-only forward fill keeps the raster readable. Grouping slopes above
    # always use the original, un-imputed observations.
    ffilled = scaled.select(months_to_keep).to_pandas().ffill(axis=1)
    display_values = ffilled.to_numpy(dtype=float)
    scaled_vals = StandardScaler().fit_transform(display_values)
    scaled = scaled.select(["DFCI_MRN", "cluster"]).with_columns([
        pl.Series(months_to_keep[i], scaled_vals[:, i]) for i in range(len(months_to_keep))
    ])

    rng = np.random.default_rng(0)
    chunks = []
    for (k,), g in scaled.group_by(["cluster"], maintain_order=True):
        if len(g) > rows_per_cluster:
            idx = rng.choice(len(g), size=rows_per_cluster, replace=False)
            g = g[idx.tolist()]
        row_mean = g.select(months_to_keep).mean_horizontal()
        g = g.with_columns(row_mean.alias("_mean")).sort("_mean").drop("_mean")
        chunks.append(g)
    return pl.concat(chunks, how="diagonal_relaxed") if chunks else scaled.clear()


def _km_data(traj_sub: pl.DataFrame) -> pl.DataFrame:
    if traj_sub.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in KM_COLUMNS})
    surv_df = pl.read_parquet(os.path.join(SURV_PATH, "death_met_surv_df.parquet"))
    out = (traj_sub.select(["DFCI_MRN", "cluster"])
           .join(surv_df.select(["DFCI_MRN", "death", "tt_death"]), on="DFCI_MRN", how="inner")
           .drop_nulls())
    out = out.filter(pl.col("tt_death") > 0)
    # Attach major stage (I-IV) so the Fig 4 supplement can build within-stage KM
    # curves (Stage I / Stage IV) stratified by trajectory cluster. NaN where the
    # pickle is unavailable or the MRN has no recognizable stage.
    stage_map = _major_stage_map()
    if stage_map is not None:
        stage_vals = [stage_map.get(mrn) for mrn in out["DFCI_MRN"].to_list()]
    else:
        stage_vals = [None] * out.height
    out = out.with_columns(pl.Series("stage", stage_vals, dtype=pl.Utf8))
    out = out.with_columns(pl.lit(SLOPE_LANDMARK_MONTHS).alias("landmark_month"))
    return out.select(KM_COLUMNS)


def _cluster_severity(traj_sub: pl.DataFrame, treatment_df: pl.DataFrame) -> pl.DataFrame:
    """Per-cluster disease-severity metrics for the Fig 4C characteristics panel.

    - mean_met_sites: mean number of distinct metastasis sites per patient (0-7).
    - rmst_months:    cluster restricted mean survival time over [0, RMST_TAU_MONTHS].
    - pct_stage_iv:   % of cluster patients whose raw stage normalizes to IV.
    - pct_ici:        % of cluster patients ever treated with ICI (any line).
    - mean_slope:     mean per-patient risk-trajectory slope (risk/month) — the
                      quantity the groups are defined on.

    pct_stage_iv and pct_ici are computed here rather than read from the
    cluster_composition_* CSVs because _composition() retains only the top-N
    treatments per cluster, and ICI usage typically falls outside the top 8 —
    so it gets folded into OTHER and disappears from the panel.
    """
    if traj_sub.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in SEVERITY_COLUMNS})
    surv_df = pl.read_parquet(os.path.join(SURV_PATH, "death_met_surv_df.parquet"))
    met_cols = [c for c in MET_EVENTS if c in surv_df.columns]
    keep_cols = ["DFCI_MRN", "death", "tt_death"] + met_cols
    traj_cols = ["DFCI_MRN", "cluster"] + (["slope"] if "slope" in traj_sub.columns else [])
    merged = traj_sub.select(traj_cols).join(
        surv_df.select(keep_cols), on="DFCI_MRN", how="inner",
    )
    if "slope" not in merged.columns:
        merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("slope"))
    if met_cols:
        # NOTE: this is a *post-index* outcome-derived count (any MET_EVENTS ever
        # observed, from death_met_surv_df), unrelated to the pre-index covariate
        # N_MET_SITES built by generate_all_non_text_covariates.build_met_burden_df
        # and used in the metburden modality. Different anatomy source, different
        # time window, different column casing — do not conflate the two.
        n_met_sites = pl.sum_horizontal([
            pl.col(c).fill_null(0).clip(upper_bound=1) for c in met_cols
        ]).alias("n_met_sites")
        merged = merged.with_columns(n_met_sites)
    else:
        merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("n_met_sites"))

    # Per-MRN flags for stage IV and ever-on-ICI
    stage_iv_set = _stage_iv_mrns()
    if stage_iv_set is not None:
        merged = merged.with_columns(
            pl.col("DFCI_MRN").is_in(stage_iv_set).cast(pl.Float64).alias("is_stage_iv")
        )
    else:
        merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("is_stage_iv"))

    ici_col = _find_ici_column([c for c in treatment_df.columns if c.startswith("PX_on_")])
    if ici_col is not None:
        ever_ici = (treatment_df.group_by("DFCI_MRN")
                    .agg(pl.col(ici_col).max().alias("ever_ici")))
        merged = merged.join(ever_ici, on="DFCI_MRN", how="left")
        merged = merged.with_columns(pl.col("ever_ici").fill_null(0).cast(pl.Float64))
    else:
        print(f"  no ICI column found among PX_on_* (looked for ICI/immune checkpoint inhibitors)")
        merged = merged.with_columns(pl.lit(None, dtype=pl.Float64).alias("ever_ici"))

    rows = []
    for (k,), sub in merged.group_by(["cluster"], maintain_order=True):
        valid = sub.drop_nulls(subset=["death", "tt_death"])
        valid = valid.filter(pl.col("tt_death") > 0)
        rmst = np.nan
        if not valid.is_empty():
            kmf = KaplanMeierFitter().fit(valid["tt_death"].to_numpy() / 30.44, valid["death"].to_numpy())
            try:
                rmst = float(restricted_mean_survival_time(kmf, t=RMST_TAU_MONTHS))
            except Exception as e:  # pragma: no cover - defensive
                print(f"  RMST failed for cluster {k}: {e}")
                rmst = np.nan
        rows.append({
            "cluster": int(k),
            "mean_met_sites": float(sub["n_met_sites"].mean()) if sub["n_met_sites"].mean() is not None else np.nan,
            "rmst_months": rmst,
            "pct_stage_iv": 100.0 * float(sub["is_stage_iv"].mean()) if sub["is_stage_iv"].mean() is not None else np.nan,
            "pct_ici":      100.0 * float(sub["ever_ici"].mean()) if sub["ever_ici"].mean() is not None else np.nan,
            "mean_slope":   float(sub["slope"].mean()) if sub["slope"].mean() is not None else np.nan,
            "n_patients": int(len(sub)),
        })
    return pl.DataFrame(rows).select(SEVERITY_COLUMNS) if rows else pl.DataFrame(schema={c: pl.Float64 for c in SEVERITY_COLUMNS})


def _group_trajectories(traj_sub: pl.DataFrame, months_to_keep: list[str]) -> pl.DataFrame:
    """Mean raw-risk trajectory (+ IQR) per slope group and for the whole cohort.

    Long format: group, month, mean_risk, q25, q75. Group values are 0..N-1 slope
    groups plus a "cohort" pseudo-group whose row at each month is the cohort-wide
    average band (the neutral reference the Fig 4D panel draws underneath).
    """
    if traj_sub.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in GROUP_TRAJECTORY_COLUMNS})
    month_nums = _month_nums(months_to_keep)

    def _summ(df: pl.DataFrame, group_val) -> list[dict]:
        vals = df.select(months_to_keep).to_numpy().astype(float)
        return [
            {"group": group_val, "month": float(month_nums[i]),
             "mean_risk": float(np.nanmean(vals[:, i])),
             "q25": float(np.nanpercentile(vals[:, i], 25)),
             "q75": float(np.nanpercentile(vals[:, i], 75))}
            for i in range(len(months_to_keep))
        ]

    rows: list[dict] = []
    for (k,), g in traj_sub.group_by(["cluster"], maintain_order=True):
        rows.extend(_summ(g, int(k)))
    rows.extend(_summ(traj_sub, "cohort"))
    return pl.DataFrame(rows).select(GROUP_TRAJECTORY_COLUMNS) if rows else pl.DataFrame(schema={c: pl.Float64 for c in GROUP_TRAJECTORY_COLUMNS})


def _slope_by_stage(traj_sub: pl.DataFrame) -> pl.DataFrame:
    """Slope-group composition within each major stage (I-IV).

    Shows that Falling/Stable/Rising dynamics occur across every stage — i.e. the
    trajectory signal is not merely a restatement of baseline stage severity.
    Long format: stage, cluster, n_patients, mean_slope.
    """
    if traj_sub.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in SLOPE_BY_STAGE_COLUMNS})
    stage_map = _major_stage_map()
    if stage_map is None:
        return pl.DataFrame(schema={c: pl.Float64 for c in SLOPE_BY_STAGE_COLUMNS})
    d = traj_sub.select(["DFCI_MRN", "cluster"])
    if "slope" in traj_sub.columns:
        d = d.with_columns(traj_sub["slope"])
    else:
        d = d.with_columns(pl.lit(None, dtype=pl.Float64).alias("slope"))
    stage_vals = [stage_map.get(mrn) for mrn in d["DFCI_MRN"].to_list()]
    d = d.with_columns(pl.Series("stage", stage_vals, dtype=pl.Utf8))
    d = d.filter(pl.col("stage").is_in(STAGE_ORDER))
    if d.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in SLOPE_BY_STAGE_COLUMNS})
    out = (d.group_by(["stage", "cluster"])
           .agg([
               pl.len().alias("n_patients"),
               pl.col("slope").mean().alias("mean_slope"),
           ]))
    out = out.with_columns([
        pl.col("cluster").cast(pl.Int64),
        pl.col("n_patients").cast(pl.Int64),
    ])
    return out.select(SLOPE_BY_STAGE_COLUMNS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decay", type=float, default=DEFAULT_DECAY,
                        help="Trajectory decay parameter — selects the input filename")
    parser.add_argument("--input", type=str, default=None,
                        help="Override trajectory input path entirely")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input or _default_trajectory_input(args.decay)
    print(f"  trajectory input: {input_path}")

    treatment_df = pl.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))

    traj_sub, months_to_keep = _cluster_trajectories(input_path)
    save_figure_data(_heatmap_downsample(traj_sub, months_to_keep),
                     "fig4_trajectories_heatmap.csv")
    save_figure_data(_km_data(traj_sub), "fig4_km_data.csv")
    save_figure_data(_cluster_severity(traj_sub, treatment_df), "fig4_cluster_severity.csv")
    save_figure_data(_group_trajectories(traj_sub, months_to_keep),
                     "fig4_group_trajectories.csv")
    save_figure_data(_slope_by_stage(traj_sub), "fig4_slope_by_stage.csv")
    save_figure_data(_silhouette_scan(input_path), "fig4_silhouette.csv")


if __name__ == "__main__":
    main()
