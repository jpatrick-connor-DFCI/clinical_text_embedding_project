"""Pre-compute inputs for Figure 4 (mortality-risk *dynamics*).

Patients are grouped by the SLOPE of their 0-60-month model-estimated mortality
risk (per-patient OLS slope of risk vs. month), not by their mean risk level, so
the figure surfaces whose risk is *changing* (Falling / Stable / Rising) as
clinical notes accumulate over follow-up.

Writes to FIGURE_DATA_DIR:
- fig4_trajectories_heatmap.csv        DFCI_MRN, cluster, <month columns kept>, downsampled to
                                       <= HEATMAP_ROWS_PER_CLUSTER per slope group and ordered by
                                       within-group mean risk (panel A reads this)
- fig4_km_data.csv                     DFCI_MRN, cluster, death, tt_death, stage  (stage = major
                                       stage I-IV for the within-stage supplement, NaN if unknown)
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
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _figure_utils import (
    FEATURE_PATH, RESULTS_PATH, SURV_PATH,
    save_figure_data,
)


N_SLOPE_GROUPS = 3  # Falling / Stable / Rising risk dynamics
DEFAULT_DECAY = 0.1
HEATMAP_ROWS_PER_CLUSTER = 500

KM_COLUMNS = ["DFCI_MRN", "cluster", "death", "tt_death", "stage"]
SEVERITY_COLUMNS = ["cluster", "mean_met_sites", "rmst_months",
                    "pct_stage_iv", "pct_ici", "mean_slope", "n_patients"]
GROUP_TRAJECTORY_COLUMNS = ["group", "month", "mean_risk", "q25", "q75"]
SLOPE_BY_STAGE_COLUMNS = ["stage", "cluster", "n_patients", "mean_slope"]
SILHOUETTE_COLUMNS = ["k", "silhouette"]
MET_EVENTS = ["brainM", "boneM", "adrenalM", "liverM", "lungM", "nodeM", "peritonealM"]
RMST_TAU_MONTHS = 120  # 5y past the 60-mo landmark entry requirement; avoids saturation at the entry cap
SILHOUETTE_K_RANGE = range(2, 9)

# Raw, complete stage labels — bypasses the drop_first issue in cancer_stage_df.csv.gz
# (Stage I is silently dropped). Same source as prep_figure_2._major_stage_labels().
STAGE_PATH = (
    "/data/gusev/PROFILE/CLINICAL/OncDRS/DERIVED_FROM_CLINICAL_TEXTS_2024_03/"
    "derived_files/cancer_stage/dfci_cancer_mrn_to_derived_cancer_stage.pkl"
)
_STAGE_IV_TOKEN = re.compile(r"^(IV|4(\.0+)?)[A-D]?$", re.IGNORECASE)
# General major-stage token (I/II/III/IV or 1-4, optional substage letter), same
# normalization used in prep_figure_1._normalize_stage / prep_figure_2.
_STAGE_TOKEN = re.compile(r"^(IV|III|II|I|4|3|2|1)[A-D]?$")
STAGE_ORDER = ["I", "II", "III", "IV"]


def _normalize_major_stage(raw) -> str | None:
    """Normalize a raw stage label to a major stage in {I, II, III, IV} (or None)."""
    if pd.isna(raw):
        return None
    s = str(raw).upper().strip().replace("STAGE", "").strip()
    s = re.sub(r"\.0+$", "", s)
    m = _STAGE_TOKEN.match(s)
    if not m:
        return None
    arabic_to_roman = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    token = m.group(1)
    return arabic_to_roman.get(token, token)


def _major_stage_map() -> dict[int, str] | None:
    """MRN -> major stage (I-IV) from the raw stage pickle. None on read failure."""
    try:
        with open(STAGE_PATH, "rb") as f:
            mrn_to_stage = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__}); slope-by-stage will be empty")
        return None
    out: dict[int, str] = {}
    for mrn, raw in mrn_to_stage.items():
        stg = _normalize_major_stage(raw)
        if stg is not None:
            out[mrn] = stg
    return out


def _stage_iv_mrns() -> set[int] | None:
    """MRNs with raw stage value normalizing to IV. None on read failure."""
    try:
        with open(STAGE_PATH, "rb") as f:
            mrn_to_stage = pickle.load(f)
    except (FileNotFoundError, OSError, pickle.UnpicklingError) as e:
        print(f"  stage pickle unavailable ({type(e).__name__}); pct_stage_iv will be NaN")
        return None
    out: set[int] = set()
    for mrn, raw in mrn_to_stage.items():
        if pd.isna(raw):
            continue
        s = str(raw).upper().strip().replace("STAGE", "").strip()
        if _STAGE_IV_TOKEN.match(s):
            out.add(mrn)
    return out


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


def _scaled_trajectory_matrix(input_path: str) -> tuple[pd.DataFrame, np.ndarray | None, list[str]]:
    """Load, filter, impute (ffill/bfill) and z-scale the trajectory matrix.

    Returns (traj_sub with imputed month columns, X_z or None if too few patients,
    months_to_keep). Shared by clustering and the silhouette scan so both operate on
    exactly the same feature matrix.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Trajectory input not found: {input_path}\n"
            "Run python_scripts/mortality_trajectories/generate_mortality_trajectories.py first, "
            "or pass --input to override."
        )
    traj_df = pd.read_csv(input_path)
    monthly_cols = [c for c in traj_df.columns if c != "DFCI_MRN"]
    missing_colwise = traj_df[monthly_cols].isna().sum(axis=0)
    months_to_keep = missing_colwise[missing_colwise < 0.9 * len(traj_df)].index.tolist()
    missing_rowwise = (~traj_df[monthly_cols].isna()).sum(axis=1)
    pxs_to_keep = missing_rowwise[missing_rowwise > 18].index
    traj_sub = traj_df.loc[pxs_to_keep, ["DFCI_MRN"] + months_to_keep].copy()

    if len(traj_sub) < N_SLOPE_GROUPS:
        return traj_sub, None, months_to_keep

    X = traj_sub[months_to_keep].values
    X_filled = pd.DataFrame(X).ffill(axis=1).bfill(axis=1).values
    X_z = StandardScaler().fit_transform(X_filled)
    # Replace original X with filled X so downstream uses the same values
    for i, col in enumerate(months_to_keep):
        traj_sub[col] = X_filled[:, i]
    return traj_sub, X_z, months_to_keep


def _month_nums(month_cols: list[str]) -> np.ndarray:
    """Parse the numeric month from each 'plus_<n>_months_data' column name."""
    return np.array([int(re.search(r"\d+", c).group()) for c in month_cols], dtype=float)


def _ols_slopes(traj_sub: pd.DataFrame, months_to_keep: list[str]) -> np.ndarray:
    """Per-patient OLS slope of (imputed, raw-scale) risk vs. month over 0-60 mo.

    Closed-form least-squares slope cov(x, y) / var(x) with x = month number,
    y = risk. Values are already ffill/bfill-imputed by _scaled_trajectory_matrix,
    so every row is complete over months_to_keep; x is identical across patients,
    which lets us compute all slopes as a single vectorized dot product.
    """
    x = _month_nums(months_to_keep)
    xc = x - x.mean()
    denom = np.sum(xc ** 2)
    Y = traj_sub[months_to_keep].to_numpy(dtype=float)      # (n_patients, n_months)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    return (Yc @ xc) / denom                                # (n_patients,) risk / month


def _cluster_trajectories(input_path: str) -> tuple[pd.DataFrame, list[str], np.ndarray | None]:
    """Group patients by risk-trajectory SLOPE (Falling / Stable / Rising).

    Returns (traj_sub with `cluster` and `slope` cols, months_to_keep, X_z aligned
    to traj_sub). `cluster` is an integer 0..N-1 relabeled by ASCENDING mean slope
    (0 = most falling risk … N-1 = most rising), preserving the integer-cluster
    contract the KM / severity / heatmap code already keys on.
    """
    traj_sub, X_z, months_to_keep = _scaled_trajectory_matrix(input_path)
    if X_z is None:
        print(f"  only {len(traj_sub)} patients survived filtering; need >= {N_SLOPE_GROUPS} "
              "for slope grouping. Emitting empty figure-data.")
        empty = pd.DataFrame(columns=["DFCI_MRN", "cluster", "slope", *months_to_keep])
        return empty, months_to_keep, None

    slopes = _ols_slopes(traj_sub, months_to_keep)
    traj_sub["slope"] = slopes

    # Cluster on the standardized 1-D slope so the Falling/Stable/Rising boundaries
    # are data-driven rather than arbitrary thresholds.
    slope_z = StandardScaler().fit_transform(slopes.reshape(-1, 1))
    raw_labels = KMeans(n_clusters=N_SLOPE_GROUPS, random_state=0, n_init=10).fit_predict(slope_z)

    # Relabel groups by ascending mean slope so cluster 0 is always most-falling.
    # Mirrors the previous ascending-mean-risk relabel; stabilizes labels across reruns.
    group_mean_slope = {k: slopes[raw_labels == k].mean() for k in np.unique(raw_labels)}
    label_order = sorted(group_mean_slope, key=group_mean_slope.get)
    relabel = {old: new for new, old in enumerate(label_order)}
    traj_sub["cluster"] = pd.Series(raw_labels, index=traj_sub.index).map(relabel)
    return traj_sub, months_to_keep, X_z


def _silhouette_scan(input_path: str) -> pd.DataFrame:
    """Silhouette score vs k on the standardized slope feature — justifies N_SLOPE_GROUPS."""
    traj_sub, X_z, months_to_keep = _scaled_trajectory_matrix(input_path)
    if X_z is None or len(traj_sub) <= max(SILHOUETTE_K_RANGE):
        return pd.DataFrame(columns=SILHOUETTE_COLUMNS)
    slope_z = StandardScaler().fit_transform(
        _ols_slopes(traj_sub, months_to_keep).reshape(-1, 1))
    rows = []
    for k in SILHOUETTE_K_RANGE:
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(slope_z)
        rows.append({"k": int(k), "silhouette": float(silhouette_score(slope_z, labels))})
    return pd.DataFrame(rows, columns=SILHOUETTE_COLUMNS)


def _heatmap_downsample(traj_sub: pd.DataFrame, months_to_keep: list[str],
                         X_z: np.ndarray | None = None,
                         rows_per_cluster: int = HEATMAP_ROWS_PER_CLUSTER) -> pd.DataFrame:
    """Within each cluster, sample at most `rows_per_cluster` patients and order
    them by within-cluster mean trajectory so the heatmap shows clean cluster
    blocks rather than an unreadable smear of 30k+ rows.

    When X_z is provided, the saved values are column-wise z-scored — each month's
    distribution is centered on 0 with unit SD across the cohort. The raw mortality
    risk varies hugely across months (e.g. by 100× near event time), washing out
    cluster structure under a single linear color scale. Z-scoring makes each
    cluster's relative-to-cohort signal pop on a diverging scale centered at 0.
    """
    if traj_sub.empty:
        return traj_sub.iloc[0:0].copy()

    # Panel A reads only DFCI_MRN + cluster + month columns (it treats every other
    # column as a month), so drop the per-patient `slope` helper column here.
    keep = ["DFCI_MRN", "cluster"] + list(months_to_keep)
    if X_z is not None and X_z.shape[0] == len(traj_sub):
        # X_z is aligned to traj_sub.index from _scaled_trajectory_matrix.
        # Swap raw month columns for their column-standardized counterparts.
        scaled = traj_sub[keep].copy()
        for i, col in enumerate(months_to_keep):
            scaled[col] = X_z[:, i]
    else:
        scaled = traj_sub[keep]

    rng = np.random.default_rng(0)
    chunks = []
    for k, g in scaled.groupby("cluster"):
        if len(g) > rows_per_cluster:
            idx = rng.choice(len(g), size=rows_per_cluster, replace=False)
            g = g.iloc[idx]
        g = g.assign(_mean=g[months_to_keep].mean(axis=1)).sort_values("_mean").drop(columns="_mean")
        chunks.append(g)
    return pd.concat(chunks, ignore_index=True)


def _km_data(traj_sub: pd.DataFrame) -> pd.DataFrame:
    if traj_sub.empty:
        return pd.DataFrame(columns=KM_COLUMNS)
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))
    out = (traj_sub[["DFCI_MRN", "cluster"]]
           .merge(surv_df[["DFCI_MRN", "death", "tt_death"]], on="DFCI_MRN", how="inner")
           .dropna())
    out = out[out["tt_death"] > 0]
    # Attach major stage (I-IV) so the Fig 4 supplement can build within-stage KM
    # curves (Stage I / Stage IV) stratified by trajectory cluster. NaN where the
    # pickle is unavailable or the MRN has no recognizable stage.
    stage_map = _major_stage_map()
    out["stage"] = out["DFCI_MRN"].map(stage_map) if stage_map is not None else np.nan
    return out


def _cluster_severity(traj_sub: pd.DataFrame, treatment_df: pd.DataFrame) -> pd.DataFrame:
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
    if traj_sub.empty:
        return pd.DataFrame(columns=SEVERITY_COLUMNS)
    surv_df = pd.read_csv(os.path.join(SURV_PATH, "death_met_surv_df.csv.gz"))
    met_cols = [c for c in MET_EVENTS if c in surv_df.columns]
    keep_cols = ["DFCI_MRN", "death", "tt_death"] + met_cols
    traj_cols = ["DFCI_MRN", "cluster"] + (["slope"] if "slope" in traj_sub.columns else [])
    merged = traj_sub[traj_cols].merge(
        surv_df[keep_cols], on="DFCI_MRN", how="inner",
    )
    if "slope" not in merged.columns:
        merged["slope"] = np.nan
    if met_cols:
        merged["n_met_sites"] = merged[met_cols].fillna(0).clip(upper=1).sum(axis=1)
    else:
        merged["n_met_sites"] = np.nan

    # Per-MRN flags for stage IV and ever-on-ICI
    stage_iv_set = _stage_iv_mrns()
    if stage_iv_set is not None:
        merged["is_stage_iv"] = merged["DFCI_MRN"].isin(stage_iv_set).astype(float)
    else:
        merged["is_stage_iv"] = np.nan

    ici_col = _find_ici_column([c for c in treatment_df.columns if c.startswith("PX_on_")])
    if ici_col is not None:
        ever_ici = (treatment_df.groupby("DFCI_MRN")[ici_col]
                    .max().rename("ever_ici").reset_index())
        merged = merged.merge(ever_ici, on="DFCI_MRN", how="left")
        merged["ever_ici"] = merged["ever_ici"].fillna(0).astype(float)
    else:
        print(f"  no ICI column found among PX_on_* (looked for ICI/immune checkpoint inhibitors)")
        merged["ever_ici"] = np.nan

    rows = []
    for k, sub in merged.groupby("cluster"):
        valid = sub.dropna(subset=["death", "tt_death"])
        valid = valid[valid["tt_death"] > 0]
        rmst = np.nan
        if not valid.empty:
            kmf = KaplanMeierFitter().fit(valid["tt_death"] / 30.44, valid["death"])
            try:
                rmst = float(restricted_mean_survival_time(kmf, t=RMST_TAU_MONTHS))
            except Exception as e:  # pragma: no cover - defensive
                print(f"  RMST failed for cluster {k}: {e}")
                rmst = np.nan
        rows.append({
            "cluster": int(k),
            "mean_met_sites": sub["n_met_sites"].mean(),
            "rmst_months": rmst,
            "pct_stage_iv": 100.0 * sub["is_stage_iv"].mean(),
            "pct_ici":      100.0 * sub["ever_ici"].mean(),
            "mean_slope":   sub["slope"].mean(),
            "n_patients": int(len(sub)),
        })
    return pd.DataFrame(rows, columns=SEVERITY_COLUMNS)


def _group_trajectories(traj_sub: pd.DataFrame, months_to_keep: list[str]) -> pd.DataFrame:
    """Mean raw-risk trajectory (+ IQR) per slope group and for the whole cohort.

    Long format: group, month, mean_risk, q25, q75. Group values are 0..N-1 slope
    groups plus a "cohort" pseudo-group whose row at each month is the cohort-wide
    average band (the neutral reference the Fig 4D panel draws underneath).
    """
    if traj_sub.empty:
        return pd.DataFrame(columns=GROUP_TRAJECTORY_COLUMNS)
    month_nums = _month_nums(months_to_keep)

    def _summ(df: pd.DataFrame, group_val) -> list[dict]:
        vals = df[months_to_keep].to_numpy(dtype=float)
        return [
            {"group": group_val, "month": float(month_nums[i]),
             "mean_risk": float(np.nanmean(vals[:, i])),
             "q25": float(np.nanpercentile(vals[:, i], 25)),
             "q75": float(np.nanpercentile(vals[:, i], 75))}
            for i in range(len(months_to_keep))
        ]

    rows: list[dict] = []
    for k, g in traj_sub.groupby("cluster"):
        rows.extend(_summ(g, int(k)))
    rows.extend(_summ(traj_sub, "cohort"))
    return pd.DataFrame(rows, columns=GROUP_TRAJECTORY_COLUMNS)


def _slope_by_stage(traj_sub: pd.DataFrame) -> pd.DataFrame:
    """Slope-group composition within each major stage (I-IV).

    Shows that Falling/Stable/Rising dynamics occur across every stage — i.e. the
    trajectory signal is not merely a restatement of baseline stage severity.
    Long format: stage, cluster, n_patients, mean_slope.
    """
    if traj_sub.empty:
        return pd.DataFrame(columns=SLOPE_BY_STAGE_COLUMNS)
    stage_map = _major_stage_map()
    if stage_map is None:
        return pd.DataFrame(columns=SLOPE_BY_STAGE_COLUMNS)
    d = traj_sub[["DFCI_MRN", "cluster"]].copy()
    if "slope" in traj_sub.columns:
        d["slope"] = traj_sub["slope"].to_numpy()
    else:
        d["slope"] = np.nan
    d["stage"] = d["DFCI_MRN"].map(stage_map)
    d = d[d["stage"].isin(STAGE_ORDER)]
    if d.empty:
        return pd.DataFrame(columns=SLOPE_BY_STAGE_COLUMNS)
    grp = d.groupby(["stage", "cluster"])
    out = (grp.agg(n_patients=("DFCI_MRN", "size"),
                   mean_slope=("slope", "mean"))
           .reset_index())
    out["cluster"] = out["cluster"].astype(int)
    out["n_patients"] = out["n_patients"].astype(int)
    return out[SLOPE_BY_STAGE_COLUMNS]


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

    treatment_df = pd.read_csv(os.path.join(FEATURE_PATH, "categorical_treatment_data_by_line.csv.gz"))

    traj_sub, months_to_keep, X_z = _cluster_trajectories(input_path)
    save_figure_data(_heatmap_downsample(traj_sub, months_to_keep, X_z),
                     "fig4_trajectories_heatmap.csv")
    save_figure_data(_km_data(traj_sub), "fig4_km_data.csv")
    save_figure_data(_cluster_severity(traj_sub, treatment_df), "fig4_cluster_severity.csv")
    save_figure_data(_group_trajectories(traj_sub, months_to_keep),
                     "fig4_group_trajectories.csv")
    save_figure_data(_slope_by_stage(traj_sub), "fig4_slope_by_stage.csv")
    save_figure_data(_silhouette_scan(input_path), "fig4_silhouette.csv")


if __name__ == "__main__":
    main()
