"""Stage 3: what do the clusters correspond to clinically?

For each (space, window) with cluster labels, tests every available clinical
variable against the partition, then describes each cluster's survival.

Writes to SEMANTIC_SEARCH_PATH/results/:
    cluster_sizes.csv         space, window, cluster, n_patients, pct
    cluster_join_coverage.csv how many clustered patients each family matched
    cluster_vs_clinical.csv   omnibus test per variable, BH-FDR within family
    cluster_enrichment.csv    per-cluster one-vs-rest for each variable/level
    cluster_survival.csv      per-cluster OS, logrank, crude and adjusted Cox
    cluster_note_volume.csv   the documentation-intensity confound check

The adjusted Cox is the load-bearing one: pathology notes name the tumor, so a
cluster/OS association that vanishes on adjustment for cancer type is a restatement
of the diagnosis rather than a new axis.

Run:
    python -m semantic_search.characterize_clusters [--spaces ...] [--windows ...]
"""

from __future__ import annotations

import argparse
import os

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

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from pipelines.biomarkers.biomarker_common import load_note_embeddings  # noqa: E402
from semantic_search import clinical_data  # noqa: E402
from semantic_search.common import (  # noqa: E402
    DEFAULT_WINDOWS,
    PATIENT_KEY,
    SPACES,
    WINDOWS,
    ensure_dirs,
    labels_path,
    load_labels,
    write_result,
)
from semantic_search.stats import (  # noqa: E402
    add_fdr_within,
    categorical_test,
    continuous_by_cluster,
    kruskal_test,
    one_vs_rest_enrichment,
)
from shared.polars_utils import to_pandas_via_numpy  # noqa: E402

DAYS_PER_MONTH = 30.44
RMST_TAU_MONTHS = 120  # 10-year horizon, matching figures/prep/figure4.py
# Below this share of clustered patients matched, a family's results are too
# thin to trust and the run says so loudly rather than reporting them silently.
MIN_JOIN_COVERAGE = 0.50

SIZE_COLUMNS = ["space", "window", "cluster", "n_patients", "pct"]
COVERAGE_COLUMNS = ["space", "window", "family", "n_clustered", "n_matched",
                    "coverage", "below_threshold"]
CLINICAL_COLUMNS = ["space", "window", "family", "variable", "kind", "test",
                    "statistic", "p", "n"]
ENRICHMENT_COLUMNS = ["space", "window", "cluster", "family", "variable", "level",
                      "n", "pct", "pct_overall", "odds_ratio", "p"]
SURVIVAL_COLUMNS = ["space", "window", "cluster", "n", "n_events",
                    "median_os_months", "rmst_months", "logrank_p",
                    "cox_hr", "cox_p", "cox_hr_adjusted", "cox_p_adjusted"]


def _cluster_sizes(labels: pl.DataFrame, space: str, window: str) -> list[dict]:
    total = labels.height
    counts = labels.group_by("cluster").len(name="n_patients").sort("cluster")
    return [{
        "space": space, "window": window,
        "cluster": int(r["cluster"]), "n_patients": int(r["n_patients"]),
        "pct": 100.0 * r["n_patients"] / total,
    } for r in counts.iter_rows(named=True)]


def _test_family(
    labels: pl.DataFrame, family: str, df: pl.DataFrame,
    continuous: list[str], categorical: list[str], space: str, window: str,
) -> tuple[list[dict], list[dict], dict]:
    """Omnibus tests plus per-cluster enrichment for one family."""
    merged = labels.join(df, on=PATIENT_KEY, how="left")
    n_clustered = labels.height

    if not continuous and not categorical:
        coverage = {"space": space, "window": window, "family": family,
                    "n_clustered": n_clustered, "n_matched": 0,
                    "coverage": 0.0, "below_threshold": True}
        return [], [], coverage

    # A patient counts as matched if any of the family's variables is non-null.
    probe = continuous + categorical
    n_matched = int(merged.select(
        pl.any_horizontal([pl.col(c).is_not_null() for c in probe]).sum()
    ).item())
    coverage_frac = n_matched / n_clustered if n_clustered else 0.0
    coverage = {
        "space": space, "window": window, "family": family,
        "n_clustered": n_clustered, "n_matched": n_matched,
        "coverage": coverage_frac,
        "below_threshold": bool(coverage_frac < MIN_JOIN_COVERAGE),
    }
    if coverage["below_threshold"]:
        print(f"    WARNING: {family} matched only {coverage_frac:.1%} of "
              f"clustered patients ({n_matched:,}/{n_clustered:,})", flush=True)

    groups = merged.get_column("cluster").to_numpy()
    clusters = sorted(set(int(c) for c in groups))
    tests: list[dict] = []
    enrichment: list[dict] = []

    for var in continuous:
        values = merged.get_column(var).cast(pl.Float64, strict=False).to_numpy()
        result = kruskal_test(values, groups)
        tests.append({"space": space, "window": window, "family": family,
                      "variable": var, "kind": "continuous", **result})
        for cl in clusters:
            summary = continuous_by_cluster(values, groups, cl)
            enrichment.append({
                "space": space, "window": window, "cluster": cl, "family": family,
                "variable": var, "level": "mean",
                "n": summary["n"], "pct": summary["mean"],
                "pct_overall": summary["mean_overall"],
                "odds_ratio": float("nan"), "p": summary["p"],
            })

    for var in categorical:
        levels = merged.get_column(var).to_numpy().astype(object)
        result = categorical_test(levels, groups)
        tests.append({"space": space, "window": window, "family": family,
                      "variable": var, "kind": "categorical", **result})
        present = [v for v in set(levels)
                   if v is not None and not (isinstance(v, float) and np.isnan(v))]
        # Binary indicators only need the positive level; a 0/1 column would
        # otherwise emit two mirror-image rows per cluster.
        if set(map(str, present)) <= {"0", "1", "0.0", "1.0", "True", "False"}:
            present = [v for v in present if str(v) in {"1", "1.0", "True"}]
        for cl in clusters:
            for level in sorted(present, key=str):
                summary = one_vs_rest_enrichment(levels, groups, level, cl)
                enrichment.append({
                    "space": space, "window": window, "cluster": cl,
                    "family": family, "variable": var, "level": str(level),
                    **summary,
                })

    return tests, enrichment, coverage


def _survival(labels: pl.DataFrame, surv: pl.DataFrame, cancer_type: pl.DataFrame,
              demographics: pl.DataFrame, space: str, window: str) -> list[dict]:
    """Per-cluster OS, global logrank, and crude vs cancer-type-adjusted Cox."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    from lifelines.utils import restricted_mean_survival_time

    df = labels.join(surv, on=PATIENT_KEY, how="inner").drop_nulls(["tt_death", "death"])
    df = df.filter(pl.col("tt_death") > 0)
    if df.height < 10 or df.get_column("cluster").n_unique() < 2:
        print("    survival: too few patients or clusters; skipping", flush=True)
        return []

    groups = df.get_column("cluster").to_numpy()
    times = df.get_column("tt_death").cast(pl.Float64).to_numpy() / DAYS_PER_MONTH
    events = df.get_column("death").cast(pl.Int64).to_numpy()

    try:
        logrank_p = float(multivariate_logrank_test(times, groups, events).p_value)
    except Exception as e:  # noqa: BLE001 - a degenerate stratum must not kill the sweep
        print(f"    logrank failed ({type(e).__name__}); reporting null", flush=True)
        logrank_p = None

    crude = _cox_hrs(df, cancer_type, demographics, adjusted=False)
    adjusted = _cox_hrs(df, cancer_type, demographics, adjusted=True)

    rows = []
    for cl in sorted(set(int(c) for c in groups)):
        mask = groups == cl
        kmf = KaplanMeierFitter().fit(times[mask], events[mask])
        try:
            rmst = float(restricted_mean_survival_time(kmf, t=RMST_TAU_MONTHS))
        except Exception:  # noqa: BLE001
            rmst = float("nan")
        median = kmf.median_survival_time_
        rows.append({
            "space": space, "window": window, "cluster": cl,
            "n": int(mask.sum()), "n_events": int(events[mask].sum()),
            "median_os_months": float(median) if np.isfinite(median) else float("nan"),
            "rmst_months": rmst,
            "logrank_p": logrank_p,
            "cox_hr": crude.get(cl, {}).get("hr", float("nan")),
            "cox_p": crude.get(cl, {}).get("p"),
            "cox_hr_adjusted": adjusted.get(cl, {}).get("hr", float("nan")),
            "cox_p_adjusted": adjusted.get(cl, {}).get("p"),
        })
    return rows


def _cox_hrs(df: pl.DataFrame, cancer_type: pl.DataFrame, demographics: pl.DataFrame,
             adjusted: bool) -> dict[int, dict]:
    """Cox HR per cluster vs the largest cluster (the reference).

    Cluster is entered as one-hot indicators against the largest cluster, so the
    reference is the same in the crude and adjusted models and the two HRs are
    directly comparable.  The adjusted model adds age, gender, and cancer type.
    """
    from lifelines import CoxPHFitter

    work = df
    if adjusted:
        work = work.join(demographics, on=PATIENT_KEY, how="left")
        work = work.join(cancer_type, on=PATIENT_KEY, how="left")

    sizes = work.group_by("cluster").len(name="n").sort("n", descending=True)
    if sizes.height < 2:
        return {}
    reference = int(sizes.row(0, named=True)["cluster"])
    others = [int(c) for c in sorted(work.get_column("cluster").unique()) if c != reference]

    design = work.select(["tt_death", "death"]).with_columns([
        (pl.col("tt_death").cast(pl.Float64) / DAYS_PER_MONTH).alias("tt_death"),
        pl.col("death").cast(pl.Int64),
    ])
    for cl in others:
        design = design.with_columns(
            (work.get_column("cluster") == cl).cast(pl.Int64).alias(f"cluster_{cl}")
        )

    if adjusted:
        for col in ("AGE_AT_TREATMENTSTART", "GENDER"):
            if col in work.columns:
                design = design.with_columns(
                    work.get_column(col).cast(pl.Float64, strict=False).alias(col)
                )
        if "CANCER_TYPE" in work.columns:
            types = work.get_column("CANCER_TYPE").fill_null("UNKNOWN")
            # Drop-first dummies, matching build_cancer_type_df's convention.
            for level in sorted(types.unique())[1:]:
                safe = "".join(ch if ch.isalnum() else "_" for ch in str(level))
                design = design.with_columns(
                    (types == level).cast(pl.Int64).alias(f"CT_{safe}")
                )

    design = design.drop_nulls()
    if design.height < 20:
        return {}
    # Constant columns make the Cox design singular.
    keep = [c for c in design.columns
            if c in ("tt_death", "death") or design.get_column(c).n_unique() > 1]
    design = design.select(keep)

    try:
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(to_pandas_via_numpy(design), duration_col="tt_death", event_col="death")
    except Exception as e:  # noqa: BLE001 - convergence failure is data-dependent
        print(f"    Cox ({'adjusted' if adjusted else 'crude'}) failed "
              f"({type(e).__name__}); reporting null", flush=True)
        return {}

    out: dict[int, dict] = {reference: {"hr": 1.0, "p": None}}
    for cl in others:
        name = f"cluster_{cl}"
        if name not in cph.params_.index:
            continue
        out[cl] = {
            "hr": float(np.exp(cph.params_[name])),
            "p": float(cph.summary.loc[name, "p"]),
        }
    return out


def run(spaces: list[str], windows: list[str]) -> None:
    ensure_dirs()

    print("Loading note metadata for the volume confound check...", flush=True)
    notes_meta, _ = load_note_embeddings()
    families = clinical_data.load_all(notes_meta)
    surv = clinical_data.load_survival()
    cancer_type_df = families["cancer_type"][0]
    demographics_df = families["demographics"][0]

    sizes, coverage, tests, enrichment, survival = [], [], [], [], []

    setups = [(space, window) for window in windows for space in spaces]
    for space, window in tqdm(setups, desc="Characterization setups", unit="setup"):
        if not os.path.exists(labels_path(space, window)):
            continue
        print(f"\n[{space}/{window}]", flush=True)
        labels = load_labels(space, window)
        sizes.extend(_cluster_sizes(labels, space, window))

        for family, (df, cont, cat) in tqdm(
            families.items(),
            total=len(families),
            desc=f"{space}/{window}: clinical families",
            unit="family",
            leave=False,
        ):
            fam_tests, fam_enrich, fam_cov = _test_family(
                labels, family, df, cont, cat, space, window)
            tests.extend(fam_tests)
            enrichment.extend(fam_enrich)
            coverage.append(fam_cov)

        if surv.height:
            survival.extend(_survival(labels, surv, cancer_type_df,
                                      demographics_df, space, window))

    write_result(_frame(sizes, SIZE_COLUMNS), "cluster_sizes")
    write_result(_frame(coverage, COVERAGE_COLUMNS), "cluster_join_coverage")

    tests_df = _frame(tests, CLINICAL_COLUMNS)
    if tests_df.height:
        tests_df = add_fdr_within(tests_df, ["space", "window", "family"])
    write_result(tests_df, "cluster_vs_clinical")

    enrich_df = _frame(enrichment, ENRICHMENT_COLUMNS)
    if enrich_df.height:
        enrich_df = add_fdr_within(enrich_df, ["space", "window", "family"])
    write_result(enrich_df, "cluster_enrichment")

    write_result(_frame(survival, SURVIVAL_COLUMNS), "cluster_survival")

    # The confound check is also written on its own so it is impossible to miss.
    if tests_df.height:
        write_result(tests_df.filter(pl.col("family") == "note_volume"),
                     "cluster_note_volume")


def _frame(rows: list[dict], columns: list[str]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
    return pl.DataFrame(rows, infer_schema_length=None).select(columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=SPACES)
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    args = parser.parse_args()
    run(spaces=args.spaces, windows=args.windows)


if __name__ == "__main__":
    main()
