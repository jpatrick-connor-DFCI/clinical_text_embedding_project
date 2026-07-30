"""Reusable workflow for the mortality model-comparison notebooks.

The module deliberately keeps data loading separate from model fitting.  The
notebooks use the repository's existing ``death_met`` cohort builder, while
the functions below operate on an ordinary pandas DataFrame and are therefore
easy to test on synthetic data.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from tqdm.auto import tqdm
from xgboost import XGBRegressor


MODEL_NAMES = ("xgboost_cox", "elastic_net_cox", "deep_surv")
FEATURE_SET_NAMES = ("baseline", "baseline_text")

# Compact grids: 8 XGBoost, 12 elastic-net, and 6 DeepSurv candidates per feature set.
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "xgboost_cox": {
        "n_estimators": [250, 500],
        "max_depth": [2, 4],
        "learning_rate": [0.03, 0.10],
    },
    "elastic_net_cox": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
    "deep_surv": {
        "hidden_dims": [(128,), (256, 128)],
        "dropout": [0.0, 0.2, 0.4],
        "learning_rate": [1e-3],
        "epochs": [100],
    },
}


def expand_grid(grid: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand a mapping of parameter lists into candidate dictionaries."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def make_survival_array(event: Iterable[Any], duration: Iterable[Any]) -> np.ndarray:
    """Return a scikit-survival structured response array."""
    event_arr = np.asarray(event, dtype=bool)
    duration_arr = np.asarray(duration, dtype=float)
    y = np.empty(len(event_arr), dtype=[("event", "?"), ("time", "<f8")])
    y["event"] = event_arr
    y["time"] = duration_arr
    return y


def prepare_cohort(
    df: pd.DataFrame,
    baseline_cols: list[str],
    text_cols: list[str],
    *,
    id_col: str = "DFCI_MRN",
    event_col: str = "death",
    time_col: str = "tt_death",
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Create a common complete-case cohort for all six comparisons.

    Using one common cohort prevents apparent model differences from being
    driven by different missing-data exclusions.
    """
    required = [id_col, event_col, time_col] + baseline_cols + text_cols
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.loc[:, required].copy()
    out = out.dropna(subset=required)
    out = out.loc[out[time_col] > 0].drop_duplicates(subset=id_col, keep="first")
    if not set(out[event_col].unique()).issubset({0, 1, False, True}):
        raise ValueError(f"{event_col} must be binary.")

    feature_sets = {
        "baseline": list(baseline_cols),
        "baseline_text": list(baseline_cols) + list(text_cols),
    }
    # Drop constants once on the common cohort and update both feature sets.
    constants = {
        c for c in feature_sets["baseline_text"] if out[c].nunique(dropna=False) <= 1
    }
    if constants:
        warnings.warn(f"Dropping {len(constants)} constant columns.", stacklevel=2)
        feature_sets = {
            name: [c for c in cols if c not in constants]
            for name, cols in feature_sets.items()
        }
        out = out.drop(columns=sorted(constants))

    n_events = int(out[event_col].sum())
    n_censored = len(out) - n_events
    if min(n_events, n_censored) < 5:
        raise ValueError(
            f"Need at least five events and five censored records; got "
            f"{n_events} events and {n_censored} censored."
        )
    return out.reset_index(drop=True), feature_sets


def create_train_test_split(
    cohort: pd.DataFrame,
    *,
    id_col: str = "DFCI_MRN",
    event_col: str = "death",
    test_size: float = 0.2,
    random_state: int = 1234,
) -> pd.DataFrame:
    """Create the one fixed, event-stratified 80/20 split used by all models."""
    train_idx, test_idx = train_test_split(
        np.arange(len(cohort)),
        test_size=test_size,
        stratify=cohort[event_col].astype(int),
        random_state=random_state,
    )
    assignment = pd.DataFrame({id_col: cohort[id_col], "split": ""})
    assignment.loc[train_idx, "split"] = "train"
    assignment.loc[test_idx, "split"] = "test"
    return assignment


@dataclass
class InputPreprocessor:
    """Fold-fitted mean imputation and optional PCA for one feature group."""

    pca_indices: np.ndarray | None = None
    pca_components: int | None = None
    random_state: int = 1234
    imputer: SimpleImputer | None = None
    group_scaler: StandardScaler | None = None
    pca: PCA | None = None
    keep_indices: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        Xi = self.imputer.fit_transform(X).astype(np.float32, copy=False)
        if self.pca_indices is None or len(self.pca_indices) == 0:
            return Xi

        pca_idx = np.asarray(self.pca_indices, dtype=int)
        keep_mask = np.ones(Xi.shape[1], dtype=bool)
        keep_mask[pca_idx] = False
        self.keep_indices = np.flatnonzero(keep_mask)
        self.group_scaler = StandardScaler()
        group = self.group_scaler.fit_transform(Xi[:, pca_idx])
        n_components = min(
            int(self.pca_components or len(pca_idx)),
            len(pca_idx),
            len(Xi),
        )
        self.pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            iterated_power=2,
            random_state=self.random_state,
        )
        reduced = self.pca.fit_transform(group).astype(np.float32, copy=False)
        return np.concatenate([Xi[:, self.keep_indices], reduced], axis=1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.imputer is None:
            raise RuntimeError("InputPreprocessor must be fitted before transform.")
        Xi = self.imputer.transform(X).astype(np.float32, copy=False)
        if self.pca is None:
            return Xi
        group = self.group_scaler.transform(Xi[:, self.pca_indices])
        reduced = self.pca.transform(group).astype(np.float32, copy=False)
        return np.concatenate([Xi[:, self.keep_indices], reduced], axis=1)

    def map_original_indices(self, original_indices: Iterable[int]) -> np.ndarray:
        """Map retained original columns to positions after optional grouped PCA."""
        original = set(int(i) for i in original_indices)
        if self.pca is None:
            return np.asarray(sorted(original), dtype=int)
        return np.asarray(
            [new_i for new_i, old_i in enumerate(self.keep_indices) if old_i in original],
            dtype=int,
        )


@dataclass
class ColumnStandardizer:
    """Standardize selected columns while leaving binary indicators unchanged."""

    indices: np.ndarray
    scaler: StandardScaler | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=np.float32).copy()
        if len(self.indices):
            self.scaler = StandardScaler()
            Xt[:, self.indices] = self.scaler.fit_transform(Xt[:, self.indices])
        return Xt

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.asarray(X, dtype=np.float32).copy()
        if self.scaler is not None:
            Xt[:, self.indices] = self.scaler.transform(Xt[:, self.indices])
        return Xt


class _DeepSurvNetwork(torch.nn.Module):
    """Feed-forward network that estimates a Cox log-risk score."""

    def __init__(
        self,
        n_features: int,
        hidden_dims: Iterable[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = n_features
        for hidden_width in hidden_dims:
            layers.extend(
                [
                    torch.nn.Linear(width, int(hidden_width)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(dropout)),
                ]
            )
            width = int(hidden_width)
        layers.append(torch.nn.Linear(width, 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X).squeeze(-1)


def _cox_partial_likelihood_loss(
    log_risk: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Negative Breslow partial log likelihood with tied-time risk sets."""
    order = torch.argsort(times, descending=True, stable=True)
    ordered_risk = log_risk[order]
    ordered_times = times[order]
    ordered_events = events[order]
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)

    _, counts = torch.unique_consecutive(ordered_times, return_counts=True)
    group_ends = torch.cumsum(counts, dim=0) - 1
    tied_denominators = torch.repeat_interleave(log_risk_set[group_ends], counts)
    event_terms = ordered_risk[ordered_events] - tied_denominators[ordered_events]
    if event_terms.numel() == 0:
        raise ValueError("DeepSurv training data contain no events.")
    return -event_terms.mean()


@dataclass
class DeepSurvEstimator:
    """Small sklearn-like prediction wrapper around a trained PyTorch network."""

    network: _DeepSurvNetwork
    device: torch.device

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.network.eval()
        tensor = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            prediction = self.network(tensor)
        return prediction.detach().cpu().numpy()


@dataclass
class FittedSurvivalModel:
    kind: str
    estimator: Any
    scaler: StandardScaler | ColumnStandardizer | None
    train_y: np.ndarray
    train_risk: np.ndarray
    breslow_times: np.ndarray | None = None
    breslow_cumhaz: np.ndarray | None = None
    input_preprocessor: InputPreprocessor | None = None

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.input_preprocessor is not None:
            X = self.input_preprocessor.transform(X)
        return self.scaler.transform(X) if self.scaler is not None else X

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        Xt = self._transform(X)
        return np.asarray(self.estimator.predict(Xt), dtype=float).reshape(-1)

    def predict_survival(self, X: np.ndarray, times: np.ndarray) -> np.ndarray:
        Xt = self._transform(X)
        if self.kind == "xgboost_cox":
            risk = np.asarray(self.estimator.predict(Xt), dtype=float)
            idx = np.searchsorted(self.breslow_times, times, side="right") - 1
            base_hazard = np.where(
                idx >= 0,
                self.breslow_cumhaz[np.maximum(idx, 0)],
                0.0,
            )
            return np.exp(-np.outer(risk, base_hazard))

        if self.kind == "deep_surv":
            log_risk = np.asarray(self.estimator.predict(Xt), dtype=float)
            hazard_ratio = np.exp(np.clip(log_risk, -50.0, 50.0))
            idx = np.searchsorted(self.breslow_times, times, side="right") - 1
            base_hazard = np.where(
                idx >= 0,
                self.breslow_cumhaz[np.maximum(idx, 0)],
                0.0,
            )
            return np.exp(-np.outer(hazard_ratio, base_hazard))

        functions = self.estimator.predict_survival_function(Xt)
        return np.row_stack([fn(times) for fn in functions])


def _breslow_baseline(y: np.ndarray, hazard_ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a Breslow baseline cumulative hazard for XGBoost-Cox."""
    times = np.asarray(y["time"], dtype=float)
    events = np.asarray(y["event"], dtype=bool)
    hazard_ratio = np.asarray(hazard_ratio, dtype=float)
    order = np.argsort(times, kind="mergesort")
    sorted_times = times[order]
    sorted_hazard = hazard_ratio[order]
    risk_set_sums = np.cumsum(sorted_hazard[::-1])[::-1]

    event_times, deaths = np.unique(times[events], return_counts=True)
    risk_set_starts = np.searchsorted(sorted_times, event_times, side="left")
    increments = deaths / risk_set_sums[risk_set_starts]
    return event_times, np.cumsum(increments)


def fit_survival_model(
    kind: str,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int = 1234,
    n_jobs: int = 1,
    need_survival_function: bool = True,
    impute_features: bool = False,
    pca_indices: Iterable[int] | None = None,
    pca_components: int | None = None,
    continuous_indices: Iterable[int] | None = None,
    unpenalized_indices: Iterable[int] | None = None,
) -> FittedSurvivalModel:
    """Fit one model using a uniform prediction interface."""
    X = np.asarray(X, dtype=np.float32)
    use_preprocessor = impute_features or pca_indices is not None
    input_preprocessor = None
    if use_preprocessor:
        input_preprocessor = InputPreprocessor(
            pca_indices=(
                np.asarray(list(pca_indices), dtype=int)
                if pca_indices is not None
                else None
            ),
            pca_components=pca_components,
            random_state=random_state,
        )
        X = input_preprocessor.fit_transform(X)

    if kind == "elastic_net_cox":
        if continuous_indices is None:
            scaler: StandardScaler | ColumnStandardizer = StandardScaler()
        else:
            mapped_continuous = (
                input_preprocessor.map_original_indices(continuous_indices)
                if input_preprocessor is not None
                else np.asarray(list(continuous_indices), dtype=int)
            )
            scaler = ColumnStandardizer(mapped_continuous)
        Xt = scaler.fit_transform(X)
        penalty_factor = None
        if unpenalized_indices is not None:
            mapped_unpenalized = (
                input_preprocessor.map_original_indices(unpenalized_indices)
                if input_preprocessor is not None
                else np.asarray(list(unpenalized_indices), dtype=int)
            )
            penalty_factor = np.ones(Xt.shape[1], dtype=np.float32)
            penalty_factor[mapped_unpenalized] = 0.0
        estimator = CoxnetSurvivalAnalysis(
            alphas=[float(params["alpha"])],
            l1_ratio=float(params["l1_ratio"]),
            penalty_factor=penalty_factor,
            fit_baseline_model=need_survival_function,
            max_iter=100_000,
            tol=1e-7,
        )
        estimator.fit(Xt, y)
        train_risk = np.asarray(estimator.predict(Xt), dtype=float)
        return FittedSurvivalModel(
            kind, estimator, scaler, y, train_risk,
            input_preprocessor=input_preprocessor,
        )

    if kind == "xgboost_cox":
        estimator = XGBRegressor(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            tree_method="hist",
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
            **params,
        )
        # XGBoost's Cox objective represents right-censoring with negative times.
        signed_time = np.where(y["event"], y["time"], -y["time"])
        estimator.fit(X, signed_time, verbose=False)
        train_risk = np.asarray(estimator.predict(X), dtype=float)
        bt, bh = (
            _breslow_baseline(y, train_risk)
            if need_survival_function
            else (None, None)
        )
        return FittedSurvivalModel(
            kind, estimator, None, y, train_risk, bt, bh, input_preprocessor
        )

    if kind == "deep_surv":
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X).astype(np.float32, copy=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            torch.set_num_threads(max(1, int(n_jobs)))
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)

        estimator = DeepSurvEstimator(
            network=_DeepSurvNetwork(
                Xt.shape[1],
                params["hidden_dims"],
                float(params["dropout"]),
            ).to(device),
            device=device,
        )
        optimizer = torch.optim.Adam(
            estimator.network.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params.get("weight_decay", 0.0)),
        )
        X_tensor = torch.as_tensor(Xt, dtype=torch.float32, device=device)
        time_tensor = torch.as_tensor(
            np.asarray(y["time"], dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )
        event_tensor = torch.as_tensor(
            np.asarray(y["event"], dtype=bool),
            dtype=torch.bool,
            device=device,
        )
        estimator.network.train()
        for _ in range(int(params["epochs"])):
            optimizer.zero_grad(set_to_none=True)
            loss = _cox_partial_likelihood_loss(
                estimator.network(X_tensor),
                time_tensor,
                event_tensor,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("DeepSurv produced a non-finite training loss.")
            loss.backward()
            optimizer.step()

        train_risk = np.asarray(estimator.predict(Xt), dtype=float)
        bt, bh = (
            _breslow_baseline(
                y,
                np.exp(np.clip(train_risk, -50.0, 50.0)),
            )
            if need_survival_function
            else (None, None)
        )
        return FittedSurvivalModel(
            kind,
            estimator,
            scaler,
            y,
            train_risk,
            bt,
            bh,
            input_preprocessor=input_preprocessor,
        )

    raise ValueError(f"Unknown model kind: {kind}")


def _metric_times(y_train: np.ndarray, y_eval: np.ndarray, n_times: int = 50) -> np.ndarray:
    """Choose a common interior time grid valid for IPCW metrics."""
    train_times = np.asarray(y_train["time"], dtype=float)
    eval_times = np.asarray(y_eval["time"], dtype=float)
    event_times = train_times[y_train["event"]]
    reference = event_times if len(event_times) >= 2 else train_times
    lo, hi = np.percentile(reference, [10, 90])
    lo = max(float(lo), float(np.min(eval_times)))
    hi = min(
        float(hi),
        float(np.nextafter(np.max(train_times), -np.inf)),
        float(np.nextafter(np.max(eval_times), -np.inf)),
    )
    if not lo < hi:
        raise ValueError(f"No valid evaluation interval: lower={lo}, upper={hi}")
    return np.linspace(lo, hi, n_times)


def evaluate_model(
    fitted: FittedSurvivalModel,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> dict[str, float | int]:
    """Compute c-index on all records and IPCW AUC/IBS where support overlaps."""
    risk = fitted.predict_risk(X_eval)
    c_index = float(
        concordance_index_censored(y_eval["event"], y_eval["time"], risk)[0]
    )

    # IPCW estimators require evaluation follow-up to lie inside training support.
    supported = y_eval["time"] < np.max(fitted.train_y["time"])
    y_ipcw = y_eval[supported]
    X_ipcw = np.asarray(X_eval)[supported]
    if len(y_ipcw) < 2 or np.unique(y_ipcw["event"]).size < 2:
        raise ValueError("Insufficient test records within the training follow-up support.")
    times = _metric_times(fitted.train_y, y_ipcw)
    risk_ipcw = fitted.predict_risk(X_ipcw)
    _, mean_auc = cumulative_dynamic_auc(
        fitted.train_y, y_ipcw, risk_ipcw, times
    )
    survival = fitted.predict_survival(X_ipcw, times)
    ibs = integrated_brier_score(fitted.train_y, y_ipcw, survival, times)
    return {
        "mean_auc_t": float(mean_auc),
        "c_index": c_index,
        "integrated_brier_score": float(ibs),
        "n_eval": int(len(y_eval)),
        "n_ipcw_eval": int(len(y_ipcw)),
        "eval_time_min": float(times[0]),
        "eval_time_max": float(times[-1]),
    }


def evaluate_mean_auc(
    fitted: FittedSurvivalModel,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    """Compute only mean cumulative/dynamic AUC for efficient CV selection."""
    supported = y_eval["time"] < np.max(fitted.train_y["time"])
    y_ipcw = y_eval[supported]
    X_ipcw = np.asarray(X_eval)[supported]
    if len(y_ipcw) < 2 or np.unique(y_ipcw["event"]).size < 2:
        raise ValueError("Insufficient validation records within training support.")
    times = _metric_times(fitted.train_y, y_ipcw)
    risk = fitted.predict_risk(X_ipcw)
    _, mean_auc = cumulative_dynamic_auc(fitted.train_y, y_ipcw, risk, times)
    return float(mean_auc)


def tune_one_model(
    kind: str,
    candidates: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 1234,
    n_jobs: int = 1,
    preprocessing_params: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select hyperparameters by mean 5-fold cumulative/dynamic AUC."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(X, y["event"].astype(int)))
    rows: list[dict[str, Any]] = []
    preprocessing_params = preprocessing_params or {}

    with tqdm(
        total=len(candidates) * len(folds),
        desc=f"{kind} CV",
        unit="fit",
        leave=False,
        disable=not show_progress,
    ) as progress:
        for candidate_index, params in enumerate(candidates):
            progress.set_postfix(
                candidate=f"{candidate_index + 1}/{len(candidates)}",
                refresh=False,
            )
            fold_scores: list[float] = []
            fold_seconds: list[float] = []
            errors: list[str] = []
            for fold, (train_idx, val_idx) in enumerate(folds):
                started = time.perf_counter()
                try:
                    fitted = fit_survival_model(
                        kind,
                        params,
                        X[train_idx],
                        y[train_idx],
                        random_state=random_state + fold,
                        n_jobs=n_jobs,
                        need_survival_function=False,
                        **preprocessing_params,
                    )
                    fold_scores.append(evaluate_mean_auc(fitted, X[val_idx], y[val_idx]))
                    errors.append("")
                except Exception as exc:  # retain failures in the audit table
                    fold_scores.append(np.nan)
                    errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    fold_seconds.append(time.perf_counter() - started)
                    progress.update()

            rows.append(
                {
                    "candidate_index": candidate_index,
                    "params_json": json.dumps(params, sort_keys=True),
                    "mean_cv_auc_t": float(np.nanmean(fold_scores))
                    if np.isfinite(fold_scores).any()
                    else np.nan,
                    "sd_cv_auc_t": float(np.nanstd(fold_scores, ddof=1))
                    if np.isfinite(fold_scores).sum() > 1
                    else np.nan,
                    "valid_folds": int(np.isfinite(fold_scores).sum()),
                    "fold_auc_t_json": json.dumps(fold_scores),
                    "fold_seconds_json": json.dumps(fold_seconds),
                    "errors_json": json.dumps(errors),
                }
            )

    results = pd.DataFrame(rows).sort_values(
        ["mean_cv_auc_t", "candidate_index"], ascending=[False, True], na_position="last"
    )
    if results["mean_cv_auc_t"].notna().sum() == 0:
        raise RuntimeError(f"Every {kind} hyperparameter candidate failed.")
    best = json.loads(results.iloc[0]["params_json"])
    return best, results.reset_index(drop=True)


def _write_benchmark_checkpoint(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    best_params: dict[str, dict[str, Any]],
) -> None:
    """Atomically update the aggregate benchmark checkpoint files."""
    summary_path = output_dir / "test_performance_and_timing.csv"
    summary_tmp = output_dir / ".test_performance_and_timing.csv.tmp"
    params_path = output_dir / "best_hyperparameters.json"
    params_tmp = output_dir / ".best_hyperparameters.json.tmp"

    pd.DataFrame(summaries).to_csv(summary_tmp, index=False)
    with params_tmp.open("w") as handle:
        json.dump(best_params, handle, indent=2, sort_keys=True)
    summary_tmp.replace(summary_path)
    params_tmp.replace(params_path)


def _load_benchmark_checkpoint(
    output_dir: Path,
    comparisons: list[tuple[str, str]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    set[tuple[str, str]],
]:
    """Load completed model/feature pairs from aggregate checkpoint files."""
    summary_path = output_dir / "test_performance_and_timing.csv"
    params_path = output_dir / "best_hyperparameters.json"
    if not summary_path.exists() or not params_path.exists():
        return [], {}, set()

    summary_df = pd.read_csv(summary_path)
    required = {"model", "feature_set"}
    if not required.issubset(summary_df.columns):
        warnings.warn(
            f"Ignoring malformed benchmark checkpoint: {summary_path}",
            stacklevel=2,
        )
        return [], {}, set()
    with params_path.open() as handle:
        best_params = json.load(handle)

    requested = set(comparisons)
    # Retain only requested comparisons and the most recent duplicate. This
    # removes results from model definitions that have since been replaced.
    requested_rows = [
        (model, feature_set) in requested
        for model, feature_set in summary_df[["model", "feature_set"]].itertuples(
            index=False,
            name=None,
        )
    ]
    summary_df = summary_df.loc[requested_rows]
    summary_df = summary_df.drop_duplicates(
        subset=["model", "feature_set"],
        keep="last",
    )
    best_params = {
        kind: {
            feature_name: params
            for feature_name, params in feature_params.items()
            if (kind, feature_name) in requested
        }
        for kind, feature_params in best_params.items()
        if isinstance(feature_params, dict)
    }
    best_params = {
        kind: feature_params
        for kind, feature_params in best_params.items()
        if feature_params
    }
    summary_pairs = set(
        summary_df[["model", "feature_set"]].itertuples(index=False, name=None)
    )
    parameter_pairs = {
        (kind, feature_name)
        for kind, feature_params in best_params.items()
        if isinstance(feature_params, dict)
        for feature_name in feature_params
    }
    completed = summary_pairs & parameter_pairs & requested

    # A pair is resumable only when both aggregate files contain its result.
    incomplete = requested - completed
    if incomplete:
        keep = [
            (model, feature_set) not in incomplete
            for model, feature_set in summary_df[["model", "feature_set"]].itertuples(
                index=False,
                name=None,
            )
        ]
        summary_df = summary_df.loc[keep]
        for kind, feature_name in incomplete:
            if kind in best_params and isinstance(best_params[kind], dict):
                best_params[kind].pop(feature_name, None)
                if not best_params[kind]:
                    best_params.pop(kind)

    return summary_df.to_dict(orient="records"), best_params, completed


def run_train_test_benchmark(
    cohort: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    split_assignment: pd.DataFrame,
    output_dir: str | Path,
    *,
    param_grids: dict[str, dict[str, list[Any]]] = DEFAULT_PARAM_GRIDS,
    id_col: str = "DFCI_MRN",
    event_col: str = "death",
    time_col: str = "tt_death",
    random_state: int = 1234,
    n_jobs: int = 1,
    model_names: Iterable[str] = MODEL_NAMES,
    feature_set_names: Iterable[str] = FEATURE_SET_NAMES,
    preprocessing_by_feature: dict[str, dict[str, Any]] | None = None,
    show_progress: bool = True,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Tune, refit, evaluate, and persist model/feature comparisons.

    When ``resume`` is true, comparisons present in both aggregate checkpoint
    files are skipped. Use ``resume=False`` to intentionally rerun every
    requested comparison.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = split_assignment.set_index(id_col)["split"]
    labels = cohort[id_col].map(split_map)
    if labels.isna().any() or set(labels.unique()) != {"train", "test"}:
        raise ValueError("Split assignment does not exactly cover the cohort.")
    train_mask = labels.eq("train").to_numpy()
    test_mask = labels.eq("test").to_numpy()
    y = make_survival_array(cohort[event_col], cohort[time_col])

    preprocessing_by_feature = preprocessing_by_feature or {}
    model_names = tuple(model_names)
    feature_set_names = tuple(feature_set_names)
    comparisons = [
        (kind, feature_name)
        for kind in model_names
        for feature_name in feature_set_names
    ]
    if resume:
        summaries, best_params, completed = _load_benchmark_checkpoint(
            output_dir,
            comparisons,
        )
    else:
        summaries, best_params, completed = [], {}, set()

    remaining = [pair for pair in comparisons if pair not in completed]
    split_assignment.to_csv(output_dir / "train_test_split.csv", index=False)
    progress = tqdm(
        total=len(comparisons),
        initial=len(completed),
        desc="Tune/refit/test",
        unit="comparison",
        disable=not show_progress,
    )
    for kind, feature_name in remaining:
        progress.set_postfix(
            model=kind,
            features=feature_name,
            stage="tuning",
            refresh=True,
        )
        if kind not in best_params:
            best_params[kind] = {}
        cols = feature_sets[feature_name]
        preprocessing_params = preprocessing_by_feature.get(feature_name, {})
        X = cohort[cols].to_numpy(dtype=np.float32)
        total_started = time.perf_counter()
        tune_started = time.perf_counter()
        best, cv_results = tune_one_model(
            kind,
            expand_grid(param_grids[kind]),
            X[train_mask],
            y[train_mask],
            random_state=random_state,
            n_jobs=n_jobs,
            preprocessing_params=preprocessing_params,
            show_progress=show_progress,
        )
        tuning_seconds = time.perf_counter() - tune_started
        cv_results.insert(0, "feature_set", feature_name)
        cv_results.insert(0, "model", kind)
        cv_results.to_csv(
            output_dir / f"cv_results__{kind}__{feature_name}.csv", index=False
        )

        progress.set_postfix(
            model=kind,
            features=feature_name,
            stage="refit",
            refresh=True,
        )
        refit_started = time.perf_counter()
        fitted = fit_survival_model(
            kind,
            best,
            X[train_mask],
            y[train_mask],
            random_state=random_state,
            n_jobs=n_jobs,
            **preprocessing_params,
        )
        refit_seconds = time.perf_counter() - refit_started
        progress.set_postfix(
            model=kind,
            features=feature_name,
            stage="evaluation",
            refresh=True,
        )
        eval_started = time.perf_counter()
        metrics = evaluate_model(fitted, X[test_mask], y[test_mask])
        evaluation_seconds = time.perf_counter() - eval_started
        total_seconds = time.perf_counter() - total_started
        best_params[kind][feature_name] = best
        summaries.append(
            {
                "model": kind,
                "feature_set": feature_name,
                "n_features": len(cols),
                "best_params_json": json.dumps(best, sort_keys=True),
                **metrics,
                "tuning_seconds": tuning_seconds,
                "refit_seconds": refit_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": total_seconds,
            }
        )
        # Checkpoint aggregate outputs after every completed comparison so
        # earlier models remain available if a later model is interrupted.
        _write_benchmark_checkpoint(output_dir, summaries, best_params)
        progress.update()

    progress.close()
    summary_df = pd.DataFrame(summaries)
    return summary_df, best_params


def generate_oof_risk_scores(
    cohort: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    best_params: dict[str, dict[str, Any]],
    output_dir: str | Path,
    *,
    id_col: str = "DFCI_MRN",
    event_col: str = "death",
    time_col: str = "tt_death",
    n_splits: int = 5,
    random_state: int = 1234,
    n_jobs: int = 1,
    model_names: Iterable[str] = MODEL_NAMES,
    feature_set_names: Iterable[str] = FEATURE_SET_NAMES,
    preprocessing_by_feature: dict[str, dict[str, Any]] | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate 5-fold held-out risk scores for each requested comparison.

    Scores are calibrated using each fold's training-score mean and standard
    deviation.  This removes arbitrary fold-to-fold location/scale differences
    without using validation outcomes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y = make_survival_array(cohort[event_col], cohort[time_col])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(cv.split(cohort, cohort[event_col].astype(int)))

    scores = cohort[[id_col, event_col, time_col]].copy()
    fold_id = np.full(len(cohort), -1, dtype=int)
    timings: list[dict[str, Any]] = []
    preprocessing_by_feature = preprocessing_by_feature or {}
    model_names = tuple(model_names)
    feature_set_names = tuple(feature_set_names)
    total_fits = len(model_names) * len(feature_set_names) * len(folds)
    with tqdm(
        total=total_fits,
        desc="Generate OOF scores",
        unit="fit",
        disable=not show_progress,
    ) as progress:
        for kind in model_names:
            for feature_name in feature_set_names:
                progress.set_postfix(
                    model=kind,
                    features=feature_name,
                    refresh=False,
                )
                column = f"{kind}__{feature_name}"
                X = cohort[feature_sets[feature_name]].to_numpy(dtype=np.float32)
                preprocessing_params = preprocessing_by_feature.get(feature_name, {})
                oof = np.full(len(cohort), np.nan, dtype=float)
                started_all = time.perf_counter()
                for fold, (train_idx, val_idx) in enumerate(folds):
                    progress.set_postfix(
                        model=kind,
                        features=feature_name,
                        fold=f"{fold + 1}/{len(folds)}",
                        refresh=True,
                    )
                    started = time.perf_counter()
                    fitted = fit_survival_model(
                        kind,
                        best_params[kind][feature_name],
                        X[train_idx],
                        y[train_idx],
                        random_state=random_state + fold,
                        n_jobs=n_jobs,
                        need_survival_function=False,
                        **preprocessing_params,
                    )
                    raw = fitted.predict_risk(X[val_idx])
                    center = float(np.mean(fitted.train_risk))
                    scale = float(np.std(fitted.train_risk))
                    if not np.isfinite(scale) or scale <= 0:
                        raise RuntimeError(
                            f"{column} fold {fold} has constant training risks."
                        )
                    oof[val_idx] = (raw - center) / scale
                    fold_id[val_idx] = fold
                    timings.append(
                        {
                            "model": kind,
                            "feature_set": feature_name,
                            "fold": fold,
                            "fit_predict_seconds": time.perf_counter() - started,
                        }
                    )
                    progress.update()
                if np.isnan(oof).any():
                    raise RuntimeError(f"Incomplete OOF predictions for {column}.")
                scores[column] = oof
                timings.append(
                    {
                        "model": kind,
                        "feature_set": feature_name,
                        "fold": "all",
                        "fit_predict_seconds": time.perf_counter() - started_all,
                    }
                )
                scores[[id_col, event_col, time_col, column]].to_csv(
                    output_dir / f"oof_risk_scores__{kind}__{feature_name}.csv",
                    index=False,
                )

    scores.insert(1, "oof_fold", fold_id)
    timing_df = pd.DataFrame(timings)
    scores.to_csv(output_dir / "mortality_oof_risk_scores.csv", index=False)
    timing_df.to_csv(output_dir / "oof_fit_predict_timing.csv", index=False)
    return scores, timing_df
