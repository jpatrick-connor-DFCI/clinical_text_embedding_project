"""Stage 3: test embedding principal components against clinical variables.

Continuous characteristics use Spearman correlation. Categorical characteristics
use a Kruskal-Wallis omnibus test with epsilon-squared effect size. Overall
survival uses a univariate Cox model per standardized PC. Benjamini-Hochberg FDR
is applied across every PC-variable test within each (space, window, family).

Run:
    python -m semantic_search.correlate_pcs [--windows ...] [--max-pcs N]
"""

from __future__ import annotations

import argparse
import math
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

from semantic_search import clinical_data  # noqa: E402
from semantic_search.common import (  # noqa: E402
    DEFAULT_WINDOWS,
    PATIENT_KEY,
    SPACES,
    WINDOWS,
    ensure_dirs,
    load_pc_meta,
    load_pc_scores,
    pc_scores_path,
    result_path,
    write_result,
)
from semantic_search.stats import (  # noqa: E402
    add_fdr_within,
    categorical_pc_test,
    spearman_test,
)
from shared.polars_utils import to_pandas_via_numpy  # noqa: E402

MIN_JOIN_COVERAGE = 0.50
MIN_SURVIVAL_N = 20
MIN_SURVIVAL_EVENTS = 5

ASSOCIATION_COLUMNS = [
    "space", "window", "pc", "component", "explained_variance_ratio",
    "family", "variable", "kind", "test", "statistic", "effect",
    "effect_name", "p", "n", "n_levels",
]
COVERAGE_COLUMNS = [
    "space", "window", "family", "n_pc_patients", "n_matched", "coverage",
    "below_threshold",
]


def _pc_columns(scores: pl.DataFrame, max_pcs: int | None) -> list[str]:
    columns = [column for column in scores.columns if column.startswith("PC")]
    columns.sort(key=lambda column: int(column[2:]))
    return columns if max_pcs is None else columns[:max_pcs]


def _coverage(
    scores: pl.DataFrame,
    clinical: pl.DataFrame,
    variables: list[str],
    *,
    space: str,
    window: str,
    family: str,
) -> dict:
    merged = scores.select(PATIENT_KEY).join(clinical, on=PATIENT_KEY, how="left")
    n_pc_patients = scores.height
    n_matched = 0
    if variables:
        n_matched = int(merged.select(
            pl.any_horizontal([pl.col(variable).is_not_null() for variable in variables]).sum()
        ).item())
    fraction = n_matched / n_pc_patients if n_pc_patients else 0.0
    return {
        "space": space,
        "window": window,
        "family": family,
        "n_pc_patients": n_pc_patients,
        "n_matched": n_matched,
        "coverage": fraction,
        "below_threshold": fraction < MIN_JOIN_COVERAGE,
    }


def _family_associations(
    scores: pl.DataFrame,
    clinical: pl.DataFrame,
    continuous: list[str],
    categorical: list[str],
    *,
    pc_columns: list[str],
    variance: dict[str, float],
    space: str,
    window: str,
    family: str,
) -> tuple[list[dict], dict]:
    variables = continuous + categorical
    coverage = _coverage(
        scores,
        clinical,
        variables,
        space=space,
        window=window,
        family=family,
    )
    if coverage["below_threshold"]:
        print(
            f"    WARNING: {family} matched {coverage['coverage']:.1%} of PC patients "
            f"({coverage['n_matched']:,}/{coverage['n_pc_patients']:,})",
            flush=True,
        )
    if not variables:
        return [], coverage

    merged = scores.join(clinical, on=PATIENT_KEY, how="left")
    rows: list[dict] = []
    for pc in pc_columns:
        pc_values = merged.get_column(pc).to_numpy()
        base = {
            "space": space,
            "window": window,
            "pc": pc,
            "component": int(pc[2:]),
            "explained_variance_ratio": variance[pc],
            "family": family,
        }
        for variable in continuous:
            clinical_values = merged.get_column(variable).cast(
                pl.Float64, strict=False
            ).to_numpy()
            rows.append({
                **base,
                "variable": variable,
                "kind": "continuous",
                **spearman_test(pc_values, clinical_values),
            })
        for variable in categorical:
            rows.append({
                **base,
                "variable": variable,
                "kind": "categorical",
                **categorical_pc_test(pc_values, merged.get_column(variable).to_numpy()),
            })
    return rows, coverage


def _cox_pc_test(scores: pl.DataFrame, survival: pl.DataFrame, pc: str) -> dict:
    from lifelines import CoxPHFitter

    merged = scores.select(PATIENT_KEY, pc).join(
        survival, on=PATIENT_KEY, how="inner"
    ).drop_nulls([pc, "tt_death", "death"])
    merged = merged.filter(
        (pl.col("tt_death") > 0)
        & pl.col(pc).is_finite()
        & pl.col("death").is_in([0, 1])
    )
    n = merged.height
    n_events = int(merged.get_column("death").sum()) if n else 0
    empty = {
        "test": "cox_ph",
        "statistic": float("nan"),
        "effect": float("nan"),
        "effect_name": "hazard_ratio_per_sd",
        "p": None,
        "n": n,
        "n_levels": None,
    }
    if n < MIN_SURVIVAL_N or n_events < MIN_SURVIVAL_EVENTS:
        return empty

    values = merged.get_column(pc).cast(pl.Float64).to_numpy()
    sd = float(np.std(values, ddof=1))
    if not math.isfinite(sd) or sd == 0:
        return empty
    standardized = (values - float(np.mean(values))) / sd
    design = pl.DataFrame({
        "pc": standardized,
        "tt_death": merged.get_column("tt_death").cast(pl.Float64),
        "death": merged.get_column("death").cast(pl.Int64),
    })
    try:
        model = CoxPHFitter()
        model.fit(to_pandas_via_numpy(design), duration_col="tt_death", event_col="death")
        coefficient = float(model.params_["pc"])
        summary = model.summary.loc["pc"]
        return {
            "test": "cox_ph",
            "statistic": float(summary["z"]),
            "effect": float(np.exp(coefficient)),
            "effect_name": "hazard_ratio_per_sd",
            "p": float(summary["p"]),
            "n": n,
            "n_levels": None,
        }
    except Exception as error:  # noqa: BLE001 - convergence is data-dependent
        print(f"    {pc} survival Cox failed ({type(error).__name__})", flush=True)
        return empty


def _survival_associations(
    scores: pl.DataFrame,
    survival: pl.DataFrame,
    *,
    pc_columns: list[str],
    variance: dict[str, float],
    space: str,
    window: str,
) -> tuple[list[dict], dict]:
    coverage = _coverage(
        scores,
        survival,
        ["tt_death", "death"],
        space=space,
        window=window,
        family="survival",
    )
    rows = []
    for pc in tqdm(
        pc_columns,
        desc=f"{space}/{window}: survival PCs",
        unit="PC",
        leave=False,
    ):
        rows.append({
            "space": space,
            "window": window,
            "pc": pc,
            "component": int(pc[2:]),
            "explained_variance_ratio": variance[pc],
            "family": "survival",
            "variable": "overall_survival",
            "kind": "time_to_event",
            **_cox_pc_test(scores, survival, pc),
        })
    return rows, coverage


def _merge_setups(name: str, new: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    path = result_path(name)
    if new.height == 0:
        if os.path.exists(path):
            return pl.read_csv(path)
        return pl.DataFrame(schema={column: pl.Utf8 for column in columns})
    if os.path.exists(path):
        old = pl.read_csv(path)
        keys = new.select("space", "window").unique()
        old = old.join(keys, on=["space", "window"], how="anti")
        new = pl.concat([old, new], how="diagonal_relaxed")
    return new


def run(
    spaces: list[str],
    windows: list[str],
    *,
    max_pcs: int | None = None,
    avpc_nepc_labels_path: str | None = None,
) -> pl.DataFrame:
    ensure_dirs()
    note_metadata = clinical_data.load_note_metadata()
    families = clinical_data.load_all(
        note_metadata,
        avpc_nepc_labels_path=avpc_nepc_labels_path,
    )
    survival = clinical_data.load_survival()

    association_rows: list[dict] = []
    coverage_rows: list[dict] = []
    setups = [(space, window) for window in windows for space in spaces]
    for space, window in tqdm(setups, desc="PC-correlation setups", unit="setup"):
        if not os.path.exists(pc_scores_path(space, window)):
            print(f"  [{space}/{window}] no PC scores; skipping", flush=True)
            continue
        scores = load_pc_scores(space, window)
        meta = load_pc_meta(space, window)
        pc_columns = _pc_columns(scores, max_pcs)
        variance = {
            f"PC{index}": float(value)
            for index, value in enumerate(meta["explained_variance_ratio"], start=1)
        }
        print(
            f"\n[{space}/{window}] testing {len(pc_columns)} PCs across "
            f"{len(families)} clinical families",
            flush=True,
        )

        for family, (frame, continuous, categorical) in tqdm(
            families.items(),
            total=len(families),
            desc=f"{space}/{window}: clinical families",
            unit="family",
            leave=False,
        ):
            rows, coverage = _family_associations(
                scores,
                frame,
                continuous,
                categorical,
                pc_columns=pc_columns,
                variance=variance,
                space=space,
                window=window,
                family=family,
            )
            association_rows.extend(rows)
            coverage_rows.append(coverage)

        if survival.height:
            rows, coverage = _survival_associations(
                scores,
                survival,
                pc_columns=pc_columns,
                variance=variance,
                space=space,
                window=window,
            )
            association_rows.extend(rows)
            coverage_rows.append(coverage)

    if association_rows:
        associations = pl.DataFrame(
            association_rows, infer_schema_length=None
        ).select(ASSOCIATION_COLUMNS)
        associations = add_fdr_within(associations, ["space", "window", "family"])
        associations = associations.sort(["space", "window", "family", "component", "variable"])
    else:
        associations = pl.DataFrame(schema={column: pl.Utf8 for column in ASSOCIATION_COLUMNS})
    associations = _merge_setups(
        "pc_clinical_associations",
        associations,
        ASSOCIATION_COLUMNS + ["fdr", "significant"],
    )
    write_result(associations, "pc_clinical_associations")

    coverage = (
        pl.DataFrame(coverage_rows, infer_schema_length=None).select(COVERAGE_COLUMNS)
        if coverage_rows
        else pl.DataFrame(schema={column: pl.Utf8 for column in COVERAGE_COLUMNS})
    )
    coverage = _merge_setups("pc_join_coverage", coverage, COVERAGE_COLUMNS)
    write_result(coverage, "pc_join_coverage")
    return associations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=SPACES)
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    parser.add_argument(
        "--max-pcs",
        type=int,
        default=None,
        help="Test only PC1..PCN; default tests every retained component.",
    )
    parser.add_argument(
        "--avpc-nepc-labels",
        default=None,
        help="Override AVPC_NEPC_LABELS_PATH for a frozen LLM label run.",
    )
    args = parser.parse_args()
    if args.max_pcs is not None and args.max_pcs < 1:
        parser.error("--max-pcs must be >= 1")
    if "alltime" in args.windows:
        print(
            "NOTE: alltime PCs use the complete documented history; associations are "
            "retrospective and survival results are not prospective.",
            flush=True,
        )
    run(
        args.spaces,
        args.windows,
        max_pcs=args.max_pcs,
        avpc_nepc_labels_path=args.avpc_nepc_labels,
    )


if __name__ == "__main__":
    main()
