"""Train XGBoost clinical-label classifiers from blockwise embedding PCs.

The default is a retrospective experiment on the single 3 x 768 concatenated
feature space using all available notes. Within every CV training split, the
three embedding blocks are independently compressed to 50 clinician/progress
PCs, 25 imaging PCs, and 25 pathology PCs. One nested-CV histogram XGBoost
classifier is then fit to the resulting 100 predictors with inverse-frequency
sample weights.

For each setup the script writes out-of-fold predictions, a final model tuned
on all available rows, and an auditable metadata JSON. Aggregate metrics and
class counts are written under ``semantic_search/results``.

Run:
    python -m semantic_search.train_prediction_models
    python -m semantic_search.train_prediction_models --windows alltime pretreatment
    python -m semantic_search.train_prediction_models --targets stage first_treatment
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

# Bound native libraries before importing NumPy/scikit-learn/XGBoost. GridSearchCV
# supplies the requested parallelism; nested native thread pools stay single-threaded.
for _thread_var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = "1"

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (  # noqa: E402
    LabelEncoder,
    Normalizer,
    StandardScaler,
    label_binarize,
)
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from semantic_search.common import (  # noqa: E402
    DEFAULT_WINDOWS,
    MODELS_DIR,
    PATIENT_KEY,
    PREDICTION_META_DIR,
    PREDICTIONS_DIR,
    RESULTS_DIR,
    SPACES,
    WINDOWS,
    embedding_cols,
    ensure_dirs,
    feature_path,
    load_features,
)
from semantic_search.prediction_targets import (  # noqa: E402
    TARGETS,
    collapse_rare_treatment_labels,
    load_target,
)

MODELS = ["xgboost"]
DEFAULT_SEED = 1234
PCA_COMPONENTS = {
    "CLINICIAN": 50,
    "IMAGING": 25,
    "PATHOLOGY": 25,
}

XGB_GRID = {
    "model__max_depth": [3, 6],
    "model__min_child_weight": [1, 5],
    "model__reg_lambda": [1.0, 10.0],
}

FOLD_METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "macro_ovr_auc",
    "macro_average_precision",
]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _software_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("numpy", "polars", "scikit-learn", "xgboost", "joblib"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unknown"
    return versions


def _atomic_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_joblib(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".joblib", dir=os.path.dirname(path))
    os.close(fd)
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _stem(target: str, space: str, window: str, model: str) -> str:
    return f"{target}__{space}__{window}__{model}"


def _artifact_paths(target: str, space: str, window: str, model: str) -> dict[str, str]:
    stem = _stem(target, space, window, model)
    return {
        "predictions": os.path.join(PREDICTIONS_DIR, f"{stem}.parquet"),
        "model": os.path.join(MODELS_DIR, f"{stem}.joblib"),
        "meta": os.path.join(PREDICTION_META_DIR, f"{stem}.json"),
    }


def _run_signature(
    data: pl.DataFrame,
    feature_cols: list[str],
    *,
    model: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    run_context: dict,
) -> dict:
    patient_labels = "\n".join(
        f"{mrn}\t{label}"
        for mrn, label in data.select(PATIENT_KEY, "label").sort(PATIENT_KEY).iter_rows()
    )
    return {
        "patient_label_sha256": hashlib.sha256(patient_labels.encode()).hexdigest(),
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_cols).encode()).hexdigest(),
        "hyperparameter_grid": XGB_GRID,
        "feature_transform": "blockwise_l2_standardize_pca",
        "pca_components": PCA_COMPONENTS,
        "n_transformed_features": sum(PCA_COMPONENTS.values()),
        "outer_folds_requested": outer_folds,
        "inner_folds_requested": inner_folds,
        "seed": seed,
        "software_versions": _software_versions(),
        **run_context,
    }


def _class_counts(labels: pl.DataFrame) -> dict[str, int]:
    rows = labels.group_by("label").len(name="n").sort("label").iter_rows(named=True)
    return {str(row["label"]): int(row["n"]) for row in rows}


def _validate_classes(y: np.ndarray, requested_outer_folds: int) -> int:
    counts = Counter(y.tolist())
    if len(counts) < 2:
        raise ValueError(f"Need at least two outcome classes; observed {dict(counts)}")
    minimum = min(counts.values())
    if minimum < 3:
        raise ValueError(
            "Nested stratified CV needs at least 3 patients in every class; "
            f"observed class counts {dict(counts)}"
        )
    return min(requested_outer_folds, minimum)


def _inner_splits(y_train: np.ndarray, requested: int) -> int:
    minimum = min(Counter(y_train.tolist()).values())
    folds = min(requested, minimum)
    if folds < 2:
        raise ValueError("An outer training fold has fewer than 2 rows in one class")
    return folds


def _feature_blocks(feature_cols: list[str]) -> dict[str, list[int]]:
    """Column indices for the three named embedding blocks."""
    blocks = {
        note_type: [
            index
            for index, column in enumerate(feature_cols)
            if column.startswith(f"{note_type}_EMBEDDING_")
        ]
        for note_type in PCA_COMPONENTS
    }
    missing = [note_type for note_type, indices in blocks.items() if not indices]
    if missing:
        raise ValueError(f"Missing embedding blocks required for blockwise PCA: {missing}")
    assigned = {index for indices in blocks.values() for index in indices}
    if len(assigned) != len(feature_cols):
        unexpected = [
            column for index, column in enumerate(feature_cols) if index not in assigned
        ]
        raise ValueError(f"Embedding columns outside the three PCA blocks: {unexpected[:5]}")
    too_narrow = {
        note_type: (len(indices), PCA_COMPONENTS[note_type])
        for note_type, indices in blocks.items()
        if len(indices) < PCA_COMPONENTS[note_type]
    }
    if too_narrow:
        raise ValueError(
            "Embedding blocks have fewer dimensions than their requested PCs: "
            f"{too_narrow}"
        )
    return blocks


def _validate_pca_sample_size(y: np.ndarray, inner_folds: int) -> None:
    """PCA component count must fit in every inner-training partition."""
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=0)
    smallest_train = min(
        len(train_index)
        for train_index, _ in splitter.split(np.zeros(len(y)), y)
    )
    required = max(PCA_COMPONENTS.values())
    if smallest_train < required:
        raise ValueError(
            f"Blockwise PCA requires at least {required} rows in every inner training "
            f"split; smallest has {smallest_train}"
        )


def _make_search(
    model: str,
    *,
    n_classes: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    feature_blocks: dict[str, list[int]],
    memory: str | None = None,
) -> GridSearchCV:
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    if model != "xgboost":
        raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")
    kwargs = {
        "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
        "n_estimators": 300,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "random_state": seed,
        "n_jobs": 1,
    }
    if n_classes > 2:
        kwargs["num_class"] = n_classes
    block_transformer = ColumnTransformer(
        [
            (
                note_type.lower(),
                Pipeline([
                    ("l2_normalize", Normalizer(norm="l2")),
                    ("standardize", StandardScaler()),
                    (
                        "pca",
                        PCA(
                            n_components=PCA_COMPONENTS[note_type],
                            random_state=seed,
                            svd_solver="randomized",
                        ),
                    ),
                ]),
                indices,
            )
            for note_type, indices in feature_blocks.items()
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    estimator = Pipeline(
        [
            ("block_pca", block_transformer),
            ("model", XGBClassifier(**kwargs)),
        ],
        memory=memory,
    )

    return GridSearchCV(
        estimator,
        param_grid=XGB_GRID,
        scoring="neg_log_loss",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )


def _fit_search(search: GridSearchCV, X: np.ndarray, y: np.ndarray) -> None:
    search.fit(
        X,
        y,
        model__sample_weight=compute_sample_weight("balanced", y),
    )


def _metric_bundle(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    probabilities = probabilities.astype(np.float64, copy=False)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    metrics = {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "log_loss": float(log_loss(y, probabilities, labels=np.arange(n_classes))),
    }
    try:
        if n_classes == 2:
            metrics["macro_ovr_auc"] = float(roc_auc_score(y, probabilities[:, 1]))
            metrics["macro_average_precision"] = float(
                average_precision_score(y, probabilities[:, 1])
            )
        else:
            binary = label_binarize(y, classes=np.arange(n_classes))
            metrics["macro_ovr_auc"] = float(
                roc_auc_score(binary, probabilities, average="macro", multi_class="ovr")
            )
            metrics["macro_average_precision"] = float(
                average_precision_score(binary, probabilities, average="macro")
            )
    except ValueError:
        # A small outer test fold can omit a class even though training remains
        # stratified. Accuracy/F1/log loss remain defined; aggregate OOF AUC/AP
        # are still computed over the complete cohort.
        metrics["macro_ovr_auc"] = float("nan")
        metrics["macro_average_precision"] = float("nan")
    return metrics


def _per_class_metrics(
    y: np.ndarray, probabilities: np.ndarray, classes: list[str]
) -> list[dict]:
    probabilities = probabilities.astype(np.float64, copy=False)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        predicted,
        labels=np.arange(len(classes)),
        zero_division=0,
    )
    rows = []
    for index, label in enumerate(classes):
        truth = (y == index).astype(int)
        row = {
            "class": label,
            "n": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "ovr_auc": float("nan"),
            "average_precision": float("nan"),
        }
        if len(np.unique(truth)) == 2:
            row["ovr_auc"] = float(roc_auc_score(truth, probabilities[:, index]))
            row["average_precision"] = float(
                average_precision_score(truth, probabilities[:, index])
            )
        rows.append(row)
    return rows


def _prepare_one_dataset(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    target: str,
    min_treatment_class_n: int,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    cols = embedding_cols(features)
    if not cols:
        raise ValueError("Feature frame has no embedding columns")
    joined = features.join(labels, on=PATIENT_KEY, how="inner").drop_nulls(["label"])
    collapsed: list[str] = []
    if target == "first_treatment":
        compact, collapsed = collapse_rare_treatment_labels(
            joined.select(PATIENT_KEY, "label"), min_treatment_class_n
        )
        joined = joined.drop("label").join(compact, on=PATIENT_KEY, how="inner")
    return joined.sort(PATIENT_KEY), cols, collapsed


def _shared_mrns(labels: pl.DataFrame, feature_frames: dict[str, pl.DataFrame]) -> pl.Series:
    shared = set(labels.get_column(PATIENT_KEY).to_list())
    for frame in feature_frames.values():
        shared.intersection_update(frame.get_column(PATIENT_KEY).to_list())
    return pl.Series(PATIENT_KEY, sorted(shared), dtype=pl.Int64)


def train_one(
    data: pl.DataFrame,
    feature_cols: list[str],
    *,
    target: str,
    space: str,
    window: str,
    model: str,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    overwrite: bool,
    run_context: dict | None = None,
) -> tuple[dict, list[dict], list[dict]]:
    paths = _artifact_paths(target, space, window, model)
    for path in paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
    signature = _run_signature(
        data,
        feature_cols,
        model=model,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        run_context=run_context or {},
    )
    if not overwrite and all(os.path.exists(path) for path in paths.values()):
        with open(paths["meta"]) as handle:
            meta = json.load(handle)
        if meta.get("run_signature") != _jsonable(signature):
            raise ValueError(
                f"Existing artifacts for {_stem(target, space, window, model)} were built "
                "with a different cohort or CV configuration; pass --overwrite to replace them."
            )
        print(f"    {model}: artifacts exist, reusing", flush=True)
        return meta, meta["fold_metrics"], meta["per_class_metrics"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(data.get_column("label").to_numpy())
    classes = [str(label) for label in encoder.classes_.tolist()]
    X = data.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    mrns = data.get_column(PATIENT_KEY).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError(f"{target}/{space}/{window} contains non-finite embedding values")
    feature_blocks = _feature_blocks(feature_cols)

    n_outer = _validate_classes(y, outer_folds)
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    probabilities = np.full((len(y), len(classes)), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int16)
    fold_metrics: list[dict] = []
    fold_tuning: list[dict] = []

    progress_label = f"{target}/{space}/{window}/{model}"
    with tqdm(total=n_outer, desc=progress_label, unit="outer fold") as progress:
        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            progress.set_postfix_str(f"outer fold {fold}/{n_outer}", refresh=True)
            n_inner = _inner_splits(y[train_idx], inner_folds)
            _validate_pca_sample_size(y[train_idx], n_inner)
            with tempfile.TemporaryDirectory(
                prefix=".block_pca_cache_",
                dir=PREDICTION_META_DIR,
            ) as cache_dir:
                search = _make_search(
                    model,
                    n_classes=len(classes),
                    inner_folds=n_inner,
                    seed=seed + fold,
                    n_jobs=n_jobs,
                    feature_blocks=feature_blocks,
                    memory=cache_dir,
                )
                _fit_search(search, X[train_idx], y[train_idx])
            probabilities[test_idx] = search.predict_proba(X[test_idx])
            fold_ids[test_idx] = fold
            metrics = _metric_bundle(y[test_idx], probabilities[test_idx])
            fold_metrics.append(
                {
                    "target": target,
                    "space": space,
                    "window": window,
                    "model": model,
                    "fold": fold,
                    "n": len(test_idx),
                    **metrics,
                }
            )
            fold_tuning.append(
                {
                    "fold": fold,
                    "inner_folds": n_inner,
                    "best_score_neg_log_loss": float(search.best_score_),
                    "best_params": search.best_params_,
                }
            )
            progress.set_postfix_str(
                f"fold {fold}: macro-F1={metrics['macro_f1']:.3f}, "
                f"bal-acc={metrics['balanced_accuracy']:.3f}",
                refresh=False,
            )
            progress.update()

    if np.isnan(probabilities).any() or (fold_ids < 0).any():
        raise RuntimeError("Outer CV did not produce exactly one prediction per patient")
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    pooled_metrics = _metric_bundle(y, probabilities)
    per_class = _per_class_metrics(y, probabilities, classes)
    for row in per_class:
        row.update({"target": target, "space": space, "window": window, "model": model})

    final_inner = _inner_splits(y, inner_folds)
    _validate_pca_sample_size(y, final_inner)
    with tempfile.TemporaryDirectory(
        prefix=".block_pca_cache_",
        dir=PREDICTION_META_DIR,
    ) as cache_dir:
        final_search = _make_search(
            model,
            n_classes=len(classes),
            inner_folds=final_inner,
            seed=seed,
            n_jobs=n_jobs,
            feature_blocks=feature_blocks,
            memory=cache_dir,
        )
        with tqdm(
            total=1,
            desc=f"{progress_label}: final refit",
            unit="search",
        ) as progress:
            _fit_search(final_search, X, y)
            progress.update()
    # The fitted transformer no longer needs its temporary training cache.
    final_search.best_estimator_.memory = None

    predicted = probabilities.argmax(axis=1)
    predictions = pl.DataFrame(
        {
            PATIENT_KEY: mrns,
            "fold": fold_ids,
            "true_label": [classes[i] for i in y],
            "predicted_label": [classes[i] for i in predicted],
            "predicted_probability": probabilities.max(axis=1),
            "class_probabilities": probabilities.tolist(),
        }
    ).sort(PATIENT_KEY)
    predictions.write_parquet(paths["predictions"])

    model_payload = {
        "estimator": final_search.best_estimator_,
        "classes": classes,
        "feature_columns": feature_cols,
        "transformed_feature_columns": [
            f"{note_type}_PC{component}"
            for note_type, count in PCA_COMPONENTS.items()
            for component in range(1, count + 1)
        ],
        "patient_key": PATIENT_KEY,
        "target": target,
        "space": space,
        "window": window,
    }
    _atomic_joblib(paths["model"], model_payload)

    meta = {
        "target": target,
        "space": space,
        "window": window,
        "model": model,
        "n_patients": len(y),
        "n_features": len(feature_cols),
        "n_transformed_features": sum(PCA_COMPONENTS.values()),
        "pca_components": PCA_COMPONENTS,
        "feature_transform": "per-block l2_normalize -> standard_scaler -> pca",
        "classes": classes,
        "class_counts": dict(Counter(classes[i] for i in y)),
        "seed": seed,
        "outer_folds": n_outer,
        "inner_folds_requested": inner_folds,
        "selection_metric": "neg_log_loss",
        "software_versions": _software_versions(),
        "run_signature": signature,
        "pooled_oof_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "per_class_metrics": per_class,
        "fold_tuning": fold_tuning,
        "final_best_score_neg_log_loss": float(final_search.best_score_),
        "final_best_params": final_search.best_params_,
        "artifacts": paths,
    }
    _atomic_json(paths["meta"], meta)
    return meta, fold_metrics, per_class


def _summary_row(meta: dict) -> dict:
    folds = meta["fold_metrics"]
    row = {
        "target": meta["target"],
        "space": meta["space"],
        "window": meta["window"],
        "model": meta["model"],
        "n_patients": meta["n_patients"],
        "n_features": meta["n_features"],
        "n_transformed_features": meta["n_transformed_features"],
        "n_classes": len(meta["classes"]),
    }
    for metric in FOLD_METRIC_NAMES:
        values = np.asarray(
            [item[metric] for item in folds if item.get(metric) is not None], dtype=float
        )
        values = values[np.isfinite(values)]
        row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
        row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        pooled = meta["pooled_oof_metrics"].get(metric)
        row[f"{metric}_pooled_oof"] = float(pooled) if pooled is not None else float("nan")
    return row


def _merge_write(path: str, rows: list[dict], key_cols: list[str]) -> None:
    """Replace matching rows, retaining other runs in the active feature space."""
    if not rows:
        print(f"  WARNING: no new rows for {os.path.basename(path)}", flush=True)
        return
    new = pl.DataFrame(rows)
    if os.path.exists(path):
        old = pl.read_csv(path)
        if "space" in old.columns:
            old = old.filter(pl.col("space").is_in(SPACES))
        if "model" in old.columns:
            old = old.filter(pl.col("model").is_in(MODELS))
        if all(column in old.columns for column in key_cols):
            keys = new.select(key_cols).unique()
            old = old.join(keys, on=key_cols, how="anti")
            new = pl.concat([old, new], how="diagonal_relaxed")
    new.write_csv(path)


def run(
    *,
    targets: list[str],
    spaces: list[str],
    windows: list[str],
    models: list[str],
    cohort_mode: str,
    min_treatment_class_n: int,
    treatment_granularity: str,
    avpc_nepc_labels_path: str | None,
    outer_folds: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    overwrite: bool,
) -> pl.DataFrame:
    ensure_dirs()
    if cohort_mode not in {"common", "per-space"}:
        raise ValueError("cohort_mode must be 'common' or 'per-space'")

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    class_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for target in targets:
        print(f"\n[{target}] loading labels", flush=True)
        try:
            labels = load_target(
                target,
                avpc_nepc_labels_path=avpc_nepc_labels_path,
                treatment_granularity=treatment_granularity,
            )
        except (FileNotFoundError, ValueError) as error:
            print(f"  unavailable: {error}", flush=True)
            continue
        print(f"  {labels.height:,} labeled patients; classes={_class_counts(labels)}", flush=True)

        for window in windows:
            feature_frames = {
                space: load_features(space, window)
                for space in spaces
                if os.path.exists(feature_path(space, window))
            }
            missing_spaces = [space for space in spaces if space not in feature_frames]
            if missing_spaces:
                print(f"  [{window}] missing feature spaces, skipping them: {missing_spaces}", flush=True)
            if not feature_frames:
                print(f"  [{window}] no requested feature files; skipping", flush=True)
                continue

            labels_for_window = labels
            if cohort_mode == "common":
                shared = _shared_mrns(labels, feature_frames)
                labels_for_window = labels.filter(pl.col(PATIENT_KEY).is_in(shared))
                print(
                    f"  [{window}] common cohort across {len(feature_frames)} spaces: "
                    f"{labels_for_window.height:,}",
                    flush=True,
                )

            for space, features in feature_frames.items():
                source_labels = labels_for_window if cohort_mode == "common" else labels
                data, cols, collapsed = _prepare_one_dataset(
                    features,
                    source_labels,
                    target=target,
                    min_treatment_class_n=min_treatment_class_n,
                )
                counts = _class_counts(data.select(PATIENT_KEY, "label"))
                print(
                    f"    [{space}/{window}] {data.height:,} patients x {len(cols)}; "
                    f"classes={counts}",
                    flush=True,
                )
                if collapsed:
                    print(f"      treatment labels collapsed to OTHER: {collapsed}", flush=True)
                for label, count in counts.items():
                    class_rows.append(
                        {
                            "target": target,
                            "space": space,
                            "window": window,
                            "cohort_mode": cohort_mode,
                            "class": label,
                            "n": count,
                        }
                    )

                try:
                    encoded = LabelEncoder().fit_transform(data.get_column("label").to_numpy())
                    _validate_classes(encoded, outer_folds)
                except ValueError as error:
                    print(f"      all models skipped ({error})", flush=True)
                    continue

                for model in models:
                    source_path = feature_path(space, window)
                    source_stat = os.stat(source_path)
                    run_context = {
                        "cohort_mode": cohort_mode,
                        "feature_artifact": source_path,
                        "feature_artifact_size": source_stat.st_size,
                        "feature_artifact_mtime_ns": source_stat.st_mtime_ns,
                    }
                    if target == "first_treatment":
                        run_context.update(
                            {
                                "treatment_granularity": treatment_granularity,
                                "min_treatment_class_n": min_treatment_class_n,
                                "collapsed_treatment_labels": collapsed,
                            }
                        )
                    meta, folds, per_class = train_one(
                        data,
                        cols,
                        target=target,
                        space=space,
                        window=window,
                        model=model,
                        outer_folds=outer_folds,
                        inner_folds=inner_folds,
                        seed=seed,
                        n_jobs=n_jobs,
                        overwrite=overwrite,
                        run_context=run_context,
                    )
                    fold_rows.extend(folds)
                    per_class_rows.extend(per_class)
                    summary_rows.append(_summary_row(meta))

    outputs = {
        "prediction_metrics_folds": (
            fold_rows,
            ["target", "space", "window", "model", "fold"],
        ),
        "prediction_metrics_summary": (
            summary_rows,
            ["target", "space", "window", "model"],
        ),
        "prediction_metrics_by_class": (
            per_class_rows,
            ["target", "space", "window", "model", "class"],
        ),
        "prediction_class_counts": (
            class_rows,
            ["target", "space", "window", "cohort_mode", "class"],
        ),
    }
    for name, (rows, key_cols) in outputs.items():
        path = os.path.join(RESULTS_DIR, f"{name}.csv")
        _merge_write(path, rows, key_cols)
    return pl.DataFrame(summary_rows) if summary_rows else pl.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=TARGETS)
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=SPACES)
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--cohort-mode",
        choices=["common", "per-space"],
        default="common",
        help="Use a shared patient intersection for fair space comparisons (default), "
        "or maximize sample size separately for each space.",
    )
    parser.add_argument("--min-treatment-class-n", type=int, default=25)
    parser.add_argument(
        "--treatment-granularity", choices=["category", "drug"], default="category"
    )
    parser.add_argument(
        "--avpc-nepc-labels",
        default=None,
        help="Override AVPC_NEPC_LABELS_PATH for a particular frozen LLM label run.",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel grid-search workers; -1 uses every available CPU (default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("--outer-folds and --inner-folds must both be >= 2")
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")

    if "alltime" in args.windows:
        print(
            "NOTE: alltime embeddings use the complete documented history. These runs measure "
            "retrospective clinical-label recovery, not prospective prediction.",
            flush=True,
        )

    run(
        targets=args.targets,
        spaces=args.spaces,
        windows=args.windows,
        models=args.models,
        cohort_mode=args.cohort_mode,
        min_treatment_class_n=args.min_treatment_class_n,
        treatment_granularity=args.treatment_granularity,
        avpc_nepc_labels_path=args.avpc_nepc_labels,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
