"""Pre-compute inputs for Figure 3 (modality comparison).

Writes to FIGURE_DATA_DIR:
- fig3_modality_cindex.csv        scheme, event, modality, cindex, auc  (both metrics side by side;
                                  auc is the sibling mean_auc(t) column from the same {mod}_test.csv)
- fig3_modality_avg_rank_cindex.csv   modality, mean_rank, sem_rank, n_events  (ranked by cindex)
- fig3_modality_avg_rank_auc.csv      modality, mean_rank, sem_rank, n_events  (ranked by mean AUC(t))
- fig3_modality_ranks_long_cindex.csv scheme, event, modality, rank  (per-endpoint, cindex-ranked)
- fig3_modality_ranks_long_auc.csv    scheme, event, modality, rank  (per-endpoint, AUC(t)-ranked)
                                  (the two ranks_long files let the R tier run a Friedman test across
                                  modalities for whichever metric is active)
- fig3_joint_betas.csv            scheme, event, modality, beta, se, hr, p_value, n, n_events
- fig3_risk_score_corr.csv        modality x modality correlation, plus n_patients in metadata row
"""

from __future__ import annotations

import logging
import os

# The joint Cox refits fan out over schemes (see _map_schemes), so several
# lifelines fits run concurrently in-process. Each would otherwise size its BLAS
# pool to the whole machine, and some cluster nodes advertise more CPUs than the
# precompiled OpenBLAS build supports -- which segfaults while allocating thread
# metadata rather than raising. Cap before NumPy/lifelines initialize a BLAS
# runtime. Kept in sync with the identical block in figure4.py.
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

from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.preprocessing import StandardScaler

from figures.io import save_figure_data
from pipelines.training.slurm_array_utils import get_events_from_df
from schemes import feature_held_out_dir, load_embedding_prediction_df, scheme_results_dir
from shared.palette import MODALITY_ORDER
from shared.polars_utils import filter_finite_rows

SCHEMES = ["death_met", "icd3_post", "icd4_post", "phecode_post"]
DEATH_SCHEME = "death_met"    # scheme for the correlation heatmap

logger = logging.getLogger(__name__)

MAX_WORKERS_FALLBACK = 4      # == len(SCHEMES); more workers than schemes buys nothing

# Fewest endpoints a modality set may rank over before it is not worth ranking.
# Selection (see _select_rankable_modalities) maximizes modalities subject to
# this floor, rather than thresholding each modality's coverage independently:
# what matters is the size of the *intersection*, and modalities often share the
# same trained endpoints (prs and text both cover the same ~575 of 1596), so
# admitting them together costs no endpoints beyond the first. Does not affect
# fig3_modality_cindex.csv, which stays the complete unfiltered record.
MIN_RANK_ENDPOINTS = 30

MODALITY_CINDEX_COLUMNS = ["scheme", "event", "modality", "cindex", "auc"]
MODALITY_AVG_RANK_COLUMNS = ["modality", "mean_rank", "sem_rank", "n_events"]
MODALITY_RANKS_LONG_COLUMNS = ["scheme", "event", "modality", "rank"]
JOINT_BETA_COLUMNS = [
    "scheme", "event", "modality",
    "beta", "se", "hr", "p_value", "n", "n_events",
]
RISK_SCORE_CORR_COLUMNS = ["modality", *MODALITY_ORDER, "n_patients"]


def _resolve_n_jobs() -> int:
    """Concrete worker count for the per-scheme fan-out.

    Mirrors run_IPTW_analysis._resolve_n_jobs: prefer the SLURM allocation over
    the machine's core count, which on a shared node is far more than the job
    was given. FIG3_N_JOBS overrides both. Capped at the scheme count.
    """
    for var in ("FIG3_N_JOBS", "SLURM_CPUS_PER_TASK"):
        raw = os.getenv(var)
        if raw:
            try:
                value = int(raw)
            except ValueError:
                logger.warning("%s=%r is not an integer; ignoring", var, raw)
                continue
            if value > 0:
                return min(value, MAX_WORKERS_FALLBACK)
    return min(os.cpu_count() or 1, MAX_WORKERS_FALLBACK)


def _map_schemes(fn, label: str) -> list[pl.DataFrame]:
    """Run `fn` over SCHEMES concurrently, preserving SCHEMES order in the result.

    Threads, not processes: the work is polars IO (read_parquet/read_csv) and
    lifelines fits, both of which release the GIL. Processes would re-import
    polars and re-read every parquet per worker for no gain. Ordered output
    keeps the concatenated CSVs byte-identical to the serial version.

    A scheme that raises is logged and contributes nothing, matching the
    serial behaviour of the per-scheme helpers, which already swallow their own
    expected failures — one bad scheme must not lose the other three.
    """
    n_workers = min(_resolve_n_jobs(), len(SCHEMES))
    if n_workers <= 1:
        results = []
        for scheme in SCHEMES:
            try:
                results.append(fn(scheme))
            except Exception:
                logger.exception("  [%s/%s] failed; skipping scheme", label, scheme)
        return results

    print(f"  [{label}] {len(SCHEMES)} scheme(s) across {n_workers} worker(s)")
    with ThreadPoolExecutor(max_workers=n_workers,
                            thread_name_prefix=f"fig3-{label}") as pool:
        futures = [(scheme, pool.submit(fn, scheme)) for scheme in SCHEMES]
        results = []
        for scheme, future in futures:
            try:
                results.append(future.result())
            except Exception:
                logger.exception("  [%s/%s] failed; skipping scheme", label, scheme)
    return results


def _modality_cindex(scheme: str) -> pl.DataFrame:
    root = os.path.join(scheme_results_dir(scheme), "feature_comps")
    events = sorted(os.listdir(root)) if os.path.isdir(root) else []
    rows = []
    n_failed = 0
    for ev in events:
        d = os.path.join(root, ev)
        for mod in MODALITY_ORDER:
            fp = os.path.join(d, f"{mod}_test.csv")
            if not os.path.exists(fp):
                continue
            try:
                _df = pl.read_csv(fp)
                if _df.is_empty():
                    raise IndexError("empty test metrics file")
                row = _df.row(0, named=True)
                rows.append({
                    "scheme": scheme, "event": ev, "modality": mod,
                    "cindex": row["mean_c_index"], "auc": row["mean_auc(t)"],
                })
            except (KeyError, IndexError, pl.exceptions.NoDataError, pl.exceptions.ComputeError) as e:
                print(f"  [{ev}:{mod}] failed to load test metrics — {type(e).__name__}: {e}")
                n_failed += 1
                continue
    if n_failed:
        print(f"  total skipped: {n_failed}")
    if rows:
        return pl.DataFrame(rows).select(MODALITY_CINDEX_COLUMNS)
    return pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_CINDEX_COLUMNS})


def _joint_betas(scheme: str) -> pl.DataFrame:
    """Refit the joint Cox model per (scheme, event) so we have p-values.

    The held-out modality risk scores are produced at training time by
    `run_feature_comp_task.py` (written under `<scheme>/held_out_risk_scores/`).
    sksurv's `CoxPHSurvivalAnalysis` does not expose SEs, so for Fig 3A we redo
    the joint fit here with `lifelines.CoxPHFitter` on the same merged inputs
    (held-out modality risk scores × event surv data), standardizing features.
    """
    risk_root = os.path.normpath(
        os.path.join(scheme_results_dir(scheme), "held_out_risk_scores")
    )
    if not os.path.isdir(risk_root):
        print(f"  missing {risk_root}; skipping joint refit")
        return pl.DataFrame(schema={c: pl.Float64 for c in JOINT_BETA_COLUMNS})
    try:
        tte_df = load_embedding_prediction_df(scheme)
    except FileNotFoundError as exc:
        print(f"  {scheme}: cannot load embedding prediction df ({exc})")
        return pl.DataFrame(schema={c: pl.Float64 for c in JOINT_BETA_COLUMNS})
    available_events = sorted(set(get_events_from_df(tte_df)))
    rows: list[dict[str, object]] = []
    for event in available_events:
        event_dir = os.path.join(risk_root, event)
        if not os.path.isdir(event_dir):
            continue
        risk_dfs = []
        for mod in MODALITY_ORDER:
            fp = os.path.join(event_dir, f"{mod}_risk_scores.csv")
            if not os.path.exists(fp):
                continue
            d = pl.read_csv(fp)
            risk_col = f"{mod}_risk_score"
            if risk_col not in d.columns and "risk_score" in d.columns:
                d = d.rename({"risk_score": risk_col})
            if risk_col not in d.columns:
                continue
            risk_dfs.append(d.select(["DFCI_MRN", risk_col]).unique(subset=["DFCI_MRN"]))
        if len(risk_dfs) < 2:
            continue
        merged = reduce(lambda l, r: l.join(r, on="DFCI_MRN", how="inner"), risk_dfs)
        keep = ["DFCI_MRN", event, f"tt_{event}"]
        if not set(keep).issubset(tte_df.columns):
            continue
        merged = merged.join(tte_df.select(keep), on="DFCI_MRN", how="inner")
        risk_cols = [c for c in merged.columns if c.endswith("_risk_score")]
        # Drop rows with NaN in *any* feature or in event/time. The per-modality
        # held-out risk-score files don't all cover the same patients (their
        # outer join can leave NaNs for partially-overlapping cohorts), and
        # lifelines refuses NaNs outright.
        merged = filter_finite_rows(
            merged, risk_cols + [event, f"tt_{event}"]
        ).filter(pl.col(f"tt_{event}") > 0)
        # Drop constant risk-score columns within this event slice; a feature
        # with zero variance breaks StandardScaler and makes the Cox design
        # matrix singular.
        constant_cols = [c for c in risk_cols if merged[c].n_unique() <= 1]
        if constant_cols:
            print(f"  [{scheme}/{event}] dropping constant risk columns: {constant_cols}")
            merged = merged.drop(constant_cols)
            risk_cols = [c for c in risk_cols if c not in constant_cols]
        if len(risk_cols) < 2:
            continue
        n = len(merged)
        n_events = int(merged[event].sum()) if n else 0
        if n < 20 or n_events < 5 or n - n_events < 5:
            continue
        try:
            X = StandardScaler().fit_transform(merged.select(risk_cols).cast(pl.Float64).to_numpy())
        except ValueError as exc:
            print(f"  [{scheme}/{event}] standardize failed: {exc}")
            continue
        if not np.isfinite(X).all():
            print(f"  [{scheme}/{event}] non-finite values after standardize; skipping")
            continue
        fit_df = pd.DataFrame(X, columns=risk_cols)
        fit_df[event] = merged[event].cast(pl.Int64).to_numpy()
        fit_df[f"tt_{event}"] = merged[f"tt_{event}"].cast(pl.Float64).to_numpy()
        # Small L2 ridge stabilizes the fit on highly-collinear modality risk
        # scores. Without it, near-rank-deficient designs produce huge
        # cancelling betas (|beta| > 1e4) and meaningless p-values. With a
        # penalty of 1e-2 on standardized features, well-identified coefficients
        # are essentially unchanged while pathological events stay finite.
        cph = CoxPHFitter(penalizer=1e-2, l1_ratio=0.0)
        # lifelines wraps several upstream failure modes (collinearity,
        # remaining NaNs slipping through, Newton-step singularities) in a mix
        # of ConvergenceError / TypeError / ValueError / LinAlgError. Catch all
        # of them so one pathological event can't kill the entire scheme.
        try:
            cph.fit(fit_df, duration_col=f"tt_{event}", event_col=event)
        except (ConvergenceError, TypeError, ValueError,
                np.linalg.LinAlgError) as exc:
            print(f"  [{scheme}/{event}] CoxPH refit failed: {type(exc).__name__}: {exc}")
            continue
        summary = cph.summary
        # Backstop: even with the ridge, drop the (scheme, event) entirely if
        # any coefficient is clearly numerical breakdown rather than biology.
        # On standardized features, |beta| > 5 (HR > ~150 per SD) is implausible
        # and almost always reflects residual separation / collinearity.
        if (summary["coef"].abs() > 5).any():
            offenders = summary.loc[summary["coef"].abs() > 5, "coef"].to_dict()
            print(f"  [{scheme}/{event}] dropping fit with pathological betas: {offenders}")
            continue
        for risk_col, srow in summary.iterrows():
            rows.append({
                "scheme": scheme,
                "event": event,
                "modality": str(risk_col).replace("_risk_score", ""),
                "beta": float(srow["coef"]),
                "se": float(srow["se(coef)"]),
                "hr": float(srow["exp(coef)"]),
                "p_value": float(srow["p"]),
                "n": n,
                "n_events": n_events,
            })
    if rows:
        return pl.DataFrame(rows).select(JOINT_BETA_COLUMNS)
    return pl.DataFrame(schema={c: pl.Float64 for c in JOINT_BETA_COLUMNS})


def _modality_cindex_all() -> pl.DataFrame:
    frames = [f for f in _map_schemes(_modality_cindex, "cindex") if not f.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_CINDEX_COLUMNS})


def _joint_betas_all() -> pl.DataFrame:
    frames = [f for f in _map_schemes(_joint_betas, "betas") if not f.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame(schema={c: pl.Float64 for c in JOINT_BETA_COLUMNS})


def _select_rankable_modalities(mat: pl.DataFrame, present: list[str],
                                value_col: str) -> list[str]:
    """Largest set of modalities that still share at least MIN_RANK_ENDPOINTS endpoints.

    Ranking is complete-case, so each added modality can only shrink the shared
    endpoint set. Thresholding each modality's coverage on its own gets this
    wrong: prs and text each cover only ~36% of endpoints and a 50% rule would
    drop both, even though they cover the *same* endpoints and so cost nothing
    once either is in.

    Greedy: while the intersection is below the floor, drop the modality whose
    removal recovers the most endpoints. That is exact when sparse modalities
    are nested or disjoint (the real case here: somatic ⊂ prs = text) and a
    reasonable approximation otherwise.
    """
    finite = mat.select([
        pl.col(m).cast(pl.Float64, strict=False).is_finite().fill_null(False).alias(m)
        for m in present
    ])
    n_endpoints = finite.height

    def n_shared(mods: list[str]) -> int:
        if not mods:
            return 0
        return int(finite.select(pl.all_horizontal([pl.col(m) for m in mods])
                                 .alias("_ok"))["_ok"].sum())

    coverage = {m: int(finite[m].sum()) for m in present}
    print(f"  [{value_col}] modality coverage over {n_endpoints} endpoints: "
          + ", ".join(f"{m} {coverage[m]}" for m in present))

    kept, dropped = list(present), []
    while kept and n_shared(kept) < MIN_RANK_ENDPOINTS:
        # Drop whichever modality is costing the most shared endpoints.
        worst = max(kept, key=lambda m: n_shared([k for k in kept if k != m]))
        if n_shared([k for k in kept if k != worst]) <= n_shared(kept):
            break   # removing anything helps nothing; the floor is unreachable
        kept.remove(worst)
        dropped.append(worst)

    shared = n_shared(kept)
    if not kept or shared < MIN_RANK_ENDPOINTS:
        print(f"  [{value_col}] no modality set reaches {MIN_RANK_ENDPOINTS} shared "
              f"endpoints (best {shared}); nothing to rank")
        return []
    if dropped:
        print(f"  [{value_col}] excluded to keep the shared endpoint set usable: "
              + ", ".join(f"{m} ({coverage[m]}/{n_endpoints})" for m in dropped))
    print(f"  [{value_col}] ranking {len(kept)} modalities over {shared} shared endpoints: "
          + ", ".join(kept))
    return [m for m in MODALITY_ORDER if m in kept]


def _complete_case_rank_matrix(metrics_df: pl.DataFrame, value_col: str):
    """Shared pivot+rank helper for _modality_avg_rank / _modality_ranks_long.

    Returns (scheme_event_df, modalities, ranks_2d) where ranks_2d is an
    (n_endpoints x n_modalities) numpy array of average ranks (1 = best),
    or (None, [], None) if there are no rankable endpoints.

    Two-step, because complete-case over all of MODALITY_ORDER collapses the
    panel to a single endpoint on real data (somatic is trained for 1 of 1596):

    1. `_select_rankable_modalities` picks the largest modality set still
       sharing a usable number of endpoints.
    2. Complete-case over that set, so every mean rank comes from the same
       endpoints.

    Step 2 is deliberately kept: ranking each endpoint over whatever it happens
    to have would score each modality on a different endpoint set and break the
    Friedman test's repeated-measures assumption. Step 1 is the price — a
    modality too sparse to share endpoints is reported and excluded rather than
    silently dragging the panel down to its own coverage.

    Selection is by measured coverage, not modality name, so an excluded
    modality rejoins automatically once its feature-comp tasks finish.
    """
    mat = metrics_df.pivot(on="modality", index=["scheme", "event"], values=value_col, aggregate_function="mean")
    present = [m for m in MODALITY_ORDER if m in mat.columns]
    absent = [m for m in MODALITY_ORDER if m not in mat.columns]
    if absent:
        print(f"  [{value_col}] no data at all for {absent}")
    if not present:
        return None, [], None

    modalities = _select_rankable_modalities(mat, present, value_col)
    if not modalities:
        return None, [], None

    key_df = mat.select(["scheme", "event"])
    mat = mat.select(modalities)
    # `is_finite()` is null where the pivot left a null, so fill to False: a
    # missing modality means the endpoint is not complete-case. Keeping the mask
    # a null-free Boolean Series also keeps it usable as a `filter` predicate.
    complete_mask = mat.select(
        pl.all_horizontal([
            pl.col(m).cast(pl.Float64, strict=False).is_finite().fill_null(False)
            for m in modalities
        ]).alias("_ok")
    )["_ok"]
    n_total, n_complete = mat.height, int(complete_mask.sum())
    if n_complete < n_total:
        # Name the modalities responsible, not just the count: with several
        # partially-covered modalities the endpoint loss is otherwise
        # untraceable back to which feature-comp tasks need re-running.
        blame = {
            m: int(mat.select(
                pl.col(m).cast(pl.Float64, strict=False).is_finite().fill_null(False)
            ).to_series().not_().sum())
            for m in modalities
        }
        blame = {m: k for m, k in sorted(blame.items(), key=lambda kv: -kv[1]) if k}
        print(f"  [{value_col}] complete-case: {n_complete}/{n_total} endpoints retained; "
              f"missing-by-modality: {blame}")
    if not complete_mask.any():
        print(f"  [{value_col}] no endpoint has every present modality; nothing to rank")
        return None, [], None
    key_df = key_df.filter(complete_mask)
    values = mat.filter(complete_mask).to_numpy()

    # Higher value -> better -> rank 1; ties share the average rank.
    order = np.argsort(-values, axis=1, kind="mergesort")
    ranks = np.empty_like(values, dtype=float)
    for i in range(values.shape[0]):
        row = values[i]
        row_order = order[i]
        sorted_vals = row[row_order]
        pos = np.arange(1, len(row) + 1, dtype=float)
        # average rank for ties
        avg_pos = pos.copy()
        j = 0
        while j < len(sorted_vals):
            k = j
            while k + 1 < len(sorted_vals) and sorted_vals[k + 1] == sorted_vals[j]:
                k += 1
            avg_pos[j:k + 1] = pos[j:k + 1].mean()
            j = k + 1
        ranks[i, row_order] = avg_pos
    return key_df, modalities, ranks


def _modality_avg_rank(metrics_df: pl.DataFrame, value_col: str = "cindex") -> pl.DataFrame:
    """Average performance rank per modality across endpoints.

    For each (scheme, event) endpoint, rank the modalities by `value_col` (cindex or
    auc; 1 = best); average those ranks per modality. Restricted to *complete-case*
    endpoints where every modality in MODALITY_ORDER has a value, so all averages
    rank the same set.
    """
    if metrics_df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_AVG_RANK_COLUMNS})

    _key_df, modalities, ranks = _complete_case_rank_matrix(metrics_df, value_col)
    if ranks is None:
        return pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_AVG_RANK_COLUMNS})

    n_events = ranks.shape[0]
    out = pl.DataFrame({
        "modality": modalities,
        "mean_rank": [float(ranks[:, i].mean()) for i in range(len(modalities))],
        "sem_rank": [float(ranks[:, i].std(ddof=1) / np.sqrt(n_events)) if n_events > 1 else 0.0
                     for i in range(len(modalities))],
        "n_events": [n_events] * len(modalities),
    })
    return out.select(MODALITY_AVG_RANK_COLUMNS)


def _modality_ranks_long(metrics_df: pl.DataFrame, value_col: str = "cindex") -> pl.DataFrame:
    """Per-endpoint modality ranks (1 = best) by `value_col`, complete-case only.

    Long-format companion to _modality_avg_rank's aggregated mean/SEM, so the
    R tier can run a Friedman test (repeated measures across modalities, one
    block per endpoint) for Fig 3C.
    """
    if metrics_df.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_RANKS_LONG_COLUMNS})

    key_df, modalities, ranks = _complete_case_rank_matrix(metrics_df, value_col)
    if ranks is None:
        return pl.DataFrame(schema={c: pl.Float64 for c in MODALITY_RANKS_LONG_COLUMNS})

    ranks_df = key_df.with_columns([
        pl.Series(m, ranks[:, i]) for i, m in enumerate(modalities)
    ])
    out = ranks_df.unpivot(
        index=["scheme", "event"], on=modalities, variable_name="modality", value_name="rank",
    )
    return out.select(MODALITY_RANKS_LONG_COLUMNS)


def _risk_score_corr(scheme: str, event: str = "death") -> pl.DataFrame:
    risk_dir = feature_held_out_dir(scheme, event)
    dfs = []
    for mod in MODALITY_ORDER:
        fp = os.path.join(risk_dir, f"{mod}_risk_scores.csv")
        if not os.path.exists(fp):
            print(f"  missing {fp}")
            continue
        d = pl.read_csv(fp)
        risk_col = f"{mod}_risk_score"
        if risk_col not in d.columns and "risk_score" in d.columns:
            d = d.rename({"risk_score": risk_col})
        dfs.append(d.select(["DFCI_MRN", risk_col]))
    if not dfs:
        return pl.DataFrame(schema={c: pl.Float64 for c in RISK_SCORE_CORR_COLUMNS})
    merged = reduce(lambda l, r: l.join(r, on="DFCI_MRN", how="inner"), dfs)
    cols = [c for c in merged.columns if c.endswith("_risk_score")]
    merged = filter_finite_rows(merged, cols)
    if merged.is_empty():
        return pl.DataFrame(schema={c: pl.Float64 for c in RISK_SCORE_CORR_COLUMNS})
    corr_mat = np.corrcoef(merged.select(cols).to_numpy(), rowvar=False)
    mod_names = [c.replace("_risk_score", "") for c in cols]
    out = pl.DataFrame({"modality": mod_names})
    out = out.with_columns([
        pl.Series(mod_names[i], corr_mat[:, i]) for i in range(len(mod_names))
    ])
    out = out.with_columns(pl.lit(len(merged)).alias("n_patients"))
    # Enforce stable schema: always emit all MODALITY_ORDER columns, NaN where
    # the modality risk-score file was missing for this scheme/event.
    missing = [c for c in RISK_SCORE_CORR_COLUMNS if c not in out.columns]
    if missing:
        out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in missing])
    return out.select(RISK_SCORE_CORR_COLUMNS)


def main() -> None:
    # The three top-level products are independent, and the joint refits dominate
    # the runtime, so start them first and do the cheap metric/rank work while
    # they run. Each already fans out over schemes internally; this only overlaps
    # the three phases with each other. Writes stay on the main thread in a fixed
    # order -- save_figure_data is not the thing being parallelized.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="fig3-main") as pool:
        betas_future = pool.submit(_joint_betas_all)
        corr_future = pool.submit(_risk_score_corr, DEATH_SCHEME, "death")
        metrics_all = _modality_cindex_all()

        save_figure_data(metrics_all, "fig3_modality_cindex.csv")
        for value_col, tag in [("cindex", "cindex"), ("auc", "auc")]:
            save_figure_data(_modality_avg_rank(metrics_all, value_col),
                              f"fig3_modality_avg_rank_{tag}.csv")
            save_figure_data(_modality_ranks_long(metrics_all, value_col),
                              f"fig3_modality_ranks_long_{tag}.csv")
        save_figure_data(betas_future.result(), "fig3_joint_betas.csv")
        save_figure_data(corr_future.result(), "fig3_risk_score_corr.csv")


if __name__ == "__main__":
    main()
