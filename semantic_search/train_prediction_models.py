"""Train XGBoost clinical-label classifiers from blockwise embedding PCs.

The default runs the single 3 x 768 concatenated feature space in both note
windows: ``alltime`` (every note, a retrospective label-recovery experiment) and
``pretreatment`` (notes pooled from before first_treatment_date only). The two
windows cover different patient sets, so their metrics are reported side by side
but are not paired.

Within every CV training split, the three embedding blocks are independently
compressed to 50 clinician/progress PCs, 25 imaging PCs, and 25 pathology PCs,
and one nested-CV histogram XGBoost classifier is fit to the resulting 100
predictors with inverse-frequency sample weights. Every transform is fit on the
training rows alone and applied to the held-out rows, at both the inner
(hyperparameter) and outer (evaluation) level, so no PCA sees validation or test
data.

For each setup the script writes out-of-fold predictions, a final model tuned
on all available rows, and an auditable metadata JSON. Aggregate metrics and
class counts are written under ``semantic_search/results``.

A non-text reference space, ``cancer_type_baseline``, is also selectable: its
features are one-hot cancer type and nothing else, trained through the same
nested CV so its AUC is directly comparable. It answers how much of a target is
explained by diagnosis alone -- the relevant question for the treatment targets,
where indication largely determines therapy. It skips PCA (a one-hot matrix is
already low-dimensional) and is refused for the ``cancer_type`` target, where its
features would be that target's own labels.

A second selectable space, ``concat_full``, reuses ``concat``'s feature file but
skips blockwise PCA entirely, so XGBoost trains on all 2304 raw embedding
dimensions instead of the compressed 100. It answers whether PCA compression
itself costs accuracy, by comparing directly against ``concat`` on the same
patients and folds.

Run:
    python -m semantic_search.train_prediction_models
    python -m semantic_search.train_prediction_models --windows pretreatment
    python -m semantic_search.train_prediction_models --targets stage first_treatment
    python -m semantic_search.train_prediction_models \
        --targets treatment_ici treatment_tki \
        --spaces concat cancer_type_baseline --cohort-mode common
    python -m semantic_search.train_prediction_models \
        --spaces concat concat_full --cohort-mode common
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

# Bound native libraries before importing NumPy/scikit-learn/XGBoost. Joblib
# parallelizes independent XGBoost fits; nested native thread pools stay single-threaded.
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
from sklearn.model_selection import ParameterGrid, StratifiedKFold  # noqa: E402
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

from config import MED_CLASSES_FILE  # noqa: E402
from semantic_search.common import (  # noqa: E402
    BASELINE_SPACE,
    BASELINE_SPACES,
    DEFAULT_WINDOWS,
    FULL_SPACE,
    FULL_SPACE_SOURCE,
    FULL_SPACES,
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
from semantic_search.drug_classes import (  # noqa: E402
    DRUG_CLASS_PATTERNS,
    DRUG_CLASS_SEX,
)
from semantic_search.prediction_targets import (  # noqa: E402
    DRUG_CLASS_TARGETS,
    TARGETS,
    collapse_rare_treatment_labels,
    load_n_lines_followup_stats,
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
    feature_blocks: dict[str, list[int]],
) -> dict:
    patient_labels = "\n".join(
        f"{mrn}\t{label}"
        for mrn, label in data.select(PATIENT_KEY, "label").sort(PATIENT_KEY).iter_rows()
    )
    # The baseline space fits no PCA, so recording the embedding PCA settings
    # for it would misdescribe the run and make its signature collide with an
    # embedding run's.
    uses_pca = bool(feature_blocks)
    return {
        "patient_label_sha256": hashlib.sha256(patient_labels.encode()).hexdigest(),
        "feature_columns_sha256": hashlib.sha256("\n".join(feature_cols).encode()).hexdigest(),
        "hyperparameter_grid": XGB_GRID,
        "feature_transform": (
            "blockwise_l2_standardize_pca" if uses_pca else "passthrough"
        ),
        "pca_components": PCA_COMPONENTS if uses_pca else None,
        "n_transformed_features": (
            sum(PCA_COMPONENTS.values()) if uses_pca else len(feature_cols)
        ),
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


# Prefix marking the one-hot cancer-type columns of the baseline space. Chosen
# so `embedding_cols` (which matches "EMBEDDING_") never picks them up.
BASELINE_COL_PREFIX = "CANCER_TYPE_"


def load_baseline_features() -> pl.DataFrame:
    """One-hot cancer type per patient: the non-text reference feature space.

    Built in memory from the same `cancer_type_df.csv.gz` the `cancer_type`
    target reads, so the baseline and that target cannot disagree about a
    patient's diagnosis.  Returns `DFCI_MRN` plus one 0/1 column per cancer type.

    Note this is the same source the `cancer_type` TARGET uses; running the
    baseline space against the `cancer_type` target would be circular (the
    features are the label one-hot encoded), so `run` refuses that pairing.
    """
    from semantic_search.clinical_data import load_cancer_type

    frame, _, _ = load_cancer_type()
    if "CANCER_TYPE" not in frame.columns:
        raise FileNotFoundError("Cancer-type baseline features are unavailable")
    frame = (
        frame.select(
            pl.col(PATIENT_KEY).cast(pl.Int64, strict=False),
            pl.col("CANCER_TYPE").cast(pl.String, strict=False).str.strip_chars(),
        )
        .drop_nulls([PATIENT_KEY, "CANCER_TYPE"])
        .filter(pl.col("CANCER_TYPE") != "")
        .unique(subset=PATIENT_KEY, keep="first")
        .sort(PATIENT_KEY)
    )
    if frame.is_empty():
        raise ValueError("Cancer-type baseline features are empty")
    types = sorted(frame.get_column("CANCER_TYPE").unique().to_list())
    # Explicit sorted indicators rather than `to_dummies`, so the column order
    # is deterministic across runs and the run signature stays stable.
    return frame.select(
        PATIENT_KEY,
        *[
            (pl.col("CANCER_TYPE") == value)
            .cast(pl.Int8)
            .alias(f"{BASELINE_COL_PREFIX}{re.sub(r'[^A-Za-z0-9]+', '_', value).strip('_').upper()}")
            for value in types
        ],
    )


def baseline_cols(df: pl.DataFrame) -> list[str]:
    """The feature columns of the baseline space, in file order."""
    return [c for c in df.columns if c.startswith(BASELINE_COL_PREFIX)]


def _feature_blocks(feature_cols: list[str], space: str) -> dict[str, list[int]]:
    """Column indices for the three named embedding blocks.

    The baseline space has no embedding blocks and skips PCA entirely, so it
    returns an empty mapping -- the sentinel `_make_block_transformer` reads as
    "pass the features through unchanged". `concat_full` uses the same sentinel:
    it wants the raw embedding columns with no PCA at all.
    """
    if space in BASELINE_SPACES or space in FULL_SPACES:
        return {}
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


def _validate_pca_sample_size(
    y: np.ndarray, inner_folds: int, feature_blocks: dict[str, list[int]]
) -> None:
    """PCA component count must fit in every inner-training partition.

    A space with no blocks (the baseline) fits no PCA, so there is nothing to
    bound and the check is skipped rather than applied to features it does not
    describe.
    """
    if not feature_blocks:
        return
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


def _make_block_transformer(
    feature_blocks: dict[str, list[int]], seed: int, n_features: int | None = None
) -> ColumnTransformer:
    """Build one leakage-safe transform for the three embedding blocks.

    With no blocks (the baseline space) the features are already low-dimensional
    indicators, so they pass through untouched: L2-normalizing and PCA-ing a
    one-hot matrix would destroy exactly the structure the baseline is meant to
    represent.
    """
    if not feature_blocks:
        if n_features is None:
            raise ValueError("A blockless transform needs n_features to select columns")
        # An explicit index list, not `slice(None)`: ColumnTransformer only
        # accepts integer/boolean selectors on a bare ndarray.
        return ColumnTransformer(
            [("passthrough", "passthrough", list(range(n_features)))],
            remainder="drop",
            sparse_threshold=0.0,
        )
    return ColumnTransformer(
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


def _make_xgboost(
    *, n_classes: int, seed: int, params: dict[str, object]
) -> XGBClassifier:
    """Build a single-threaded XGBoost fit for outer joblib parallelism."""
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
    kwargs.update(
        {
            key.removeprefix("model__"): value
            for key, value in params.items()
        }
    )
    return XGBClassifier(**kwargs)


def _fit_and_score_xgboost(
    candidate_index: int,
    inner_fold: int,
    params: dict[str, object],
    X_train_pc: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    X_valid_pc: np.ndarray,
    y_valid: np.ndarray,
    *,
    n_classes: int,
    seed: int,
) -> tuple[int, int, float]:
    estimator = _make_xgboost(
        n_classes=n_classes,
        seed=seed,
        params=params,
    )
    estimator.fit(
        X_train_pc,
        y_train,
        sample_weight=train_weights,
    )
    probabilities = estimator.predict_proba(X_valid_pc)
    loss = log_loss(y_valid, probabilities, labels=np.arange(n_classes))
    return candidate_index, inner_fold, float(loss)


def _fit_inner_pca(
    inner_fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray,
    *,
    feature_blocks: dict[str, list[int]],
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    transformer = _make_block_transformer(feature_blocks, seed, X.shape[1])
    X_train_pc = transformer.fit_transform(X[train_idx])
    X_valid_pc = transformer.transform(X[valid_idx])
    return (
        inner_fold,
        X_train_pc,
        y[train_idx],
        sample_weights[train_idx],
        X_valid_pc,
        y[valid_idx],
    )


def _tune_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    feature_blocks: dict[str, list[int]],
    progress_label: str,
) -> tuple[dict[str, object], float]:
    """Tune XGBoost while computing each inner-fold PCA only once in memory."""
    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=seed,
    )
    sample_weights = compute_sample_weight("balanced", y)
    split_indices = [
        (inner_fold, train_idx, valid_idx)
        for inner_fold, (train_idx, valid_idx) in enumerate(inner_cv.split(X, y))
    ]
    pca_parallel = joblib.Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        return_as="generator_unordered",
    )
    pca_result_stream = pca_parallel(
        joblib.delayed(_fit_inner_pca)(
            inner_fold,
            train_idx,
            valid_idx,
            X,
            y,
            sample_weights,
            feature_blocks=feature_blocks,
            seed=seed,
        )
        for inner_fold, train_idx, valid_idx in split_indices
    )
    fold_data = []
    with tqdm(
        total=inner_folds,
        desc=f"{progress_label}: PCA",
        unit="fold",
        leave=False,
    ) as progress:
        for result in pca_result_stream:
            fold_data.append(result)
            progress.update()

    candidates = list(ParameterGrid(XGB_GRID))
    tasks = [
        (candidate_index, params, fold)
        for candidate_index, params in enumerate(candidates)
        for fold in fold_data
    ]
    losses = np.full((len(candidates), inner_folds), np.nan, dtype=np.float64)
    parallel = joblib.Parallel(
        n_jobs=n_jobs,
        prefer="threads",
        return_as="generator_unordered",
    )
    result_stream = parallel(
        joblib.delayed(_fit_and_score_xgboost)(
            candidate_index,
            fold[0],
            params,
            fold[1],
            fold[2],
            fold[3],
            fold[4],
            fold[5],
            n_classes=n_classes,
            seed=seed,
        )
        for candidate_index, params, fold in tasks
    )
    with tqdm(
        total=len(tasks),
        desc=f"{progress_label}: XGBoost",
        unit="fit",
        leave=False,
    ) as progress:
        for candidate_index, inner_fold, loss in result_stream:
            losses[candidate_index, inner_fold] = loss
            progress.update()

    if np.isnan(losses).any():
        raise RuntimeError("Inner CV did not score every XGBoost candidate/fold")
    mean_losses = losses.mean(axis=1)
    best_index = int(np.argmin(mean_losses))
    return candidates[best_index], -float(mean_losses[best_index])


def _fit_final_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_classes: int,
    seed: int,
    feature_blocks: dict[str, list[int]],
    params: dict[str, object],
) -> Pipeline:
    """Fit the selected transform and classifier on an entire training partition."""
    transformer = _make_block_transformer(feature_blocks, seed, X.shape[1])
    X_pc = transformer.fit_transform(X)
    estimator = _make_xgboost(n_classes=n_classes, seed=seed, params=params)
    estimator.fit(
        X_pc,
        y,
        sample_weight=compute_sample_weight("balanced", y),
    )
    return Pipeline([("block_pca", transformer), ("model", estimator)])


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
    space: str,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    if space in BASELINE_SPACES:
        cols = baseline_cols(features)
        if not cols:
            raise ValueError("Baseline frame has no cancer-type columns")
    else:
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


def _shared_mrns(
    labels: pl.DataFrame, feature_frames: dict[str, pl.DataFrame]
) -> pl.DataFrame:
    shared = set(labels.get_column(PATIENT_KEY).to_list())
    for frame in feature_frames.values():
        shared.intersection_update(frame.get_column(PATIENT_KEY).to_list())
    return pl.DataFrame(
        {PATIENT_KEY: sorted(shared)},
        schema={PATIENT_KEY: pl.Int64},
    )


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
    feature_blocks = _feature_blocks(feature_cols, space)
    signature = _run_signature(
        data,
        feature_cols,
        model=model,
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        seed=seed,
        run_context=run_context or {},
        feature_blocks=feature_blocks,
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
            _validate_pca_sample_size(y[train_idx], n_inner, feature_blocks)
            if model != "xgboost":
                raise ValueError(f"Unknown model {model!r}; choose from {MODELS}")
            best_params, best_score = _tune_xgboost(
                X[train_idx],
                y[train_idx],
                n_classes=len(classes),
                inner_folds=n_inner,
                seed=seed + fold,
                n_jobs=n_jobs,
                feature_blocks=feature_blocks,
                progress_label=f"{progress_label} fold {fold}/{n_outer}",
            )
            outer_estimator = _fit_final_pipeline(
                X[train_idx],
                y[train_idx],
                n_classes=len(classes),
                seed=seed + fold,
                feature_blocks=feature_blocks,
                params=best_params,
            )
            probabilities[test_idx] = outer_estimator.predict_proba(X[test_idx])
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
                    "best_score_neg_log_loss": best_score,
                    "best_params": best_params,
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
    _validate_pca_sample_size(y, final_inner, feature_blocks)
    final_best_params, final_best_score = _tune_xgboost(
        X,
        y,
        n_classes=len(classes),
        inner_folds=final_inner,
        seed=seed,
        n_jobs=n_jobs,
        feature_blocks=feature_blocks,
        progress_label=f"{progress_label} final tuning",
    )
    with tqdm(
        total=1,
        desc=f"{progress_label}: final refit",
        unit="model",
    ) as progress:
        final_estimator = _fit_final_pipeline(
            X,
            y,
            n_classes=len(classes),
            seed=seed,
            feature_blocks=feature_blocks,
            params=final_best_params,
        )
        progress.update()

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
        "estimator": final_estimator,
        "classes": classes,
        "feature_columns": feature_cols,
        # A blockless space (baseline or concat_full) fits no PCA, so its
        # transformed columns are just its raw feature columns, unchanged.
        "transformed_feature_columns": (
            [
                f"{note_type}_PC{component}"
                for note_type, count in PCA_COMPONENTS.items()
                for component in range(1, count + 1)
            ]
            if feature_blocks
            else feature_cols
        ),
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
        # A blockless space (the baseline) fits no PCA, so its features reach
        # the classifier untransformed and at their original width.
        "n_transformed_features": (
            sum(PCA_COMPONENTS.values()) if feature_blocks else len(feature_cols)
        ),
        "pca_components": PCA_COMPONENTS if feature_blocks else None,
        "feature_transform": (
            "per-block l2_normalize -> standard_scaler -> pca"
            if feature_blocks
            else "passthrough"
        ),
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
        "final_best_score_neg_log_loss": final_best_score,
        "final_best_params": final_best_params,
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
            # SPACES + BASELINE_SPACES + FULL_SPACES, not SPACES: an
            # embedding-only run must not silently drop the baseline or
            # full-embedding rows a previous run wrote, since those are the
            # reference/comparison points those embedding rows are read against.
            old = old.filter(pl.col("space").is_in(SPACES + BASELINE_SPACES + FULL_SPACES))
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

        followup_stats = None
        if target == "n_lines":
            # Diagnostic only: a failure here must not block training, since the
            # labels themselves loaded fine.
            try:
                followup_stats = load_n_lines_followup_stats().to_dicts()
            except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
                print(f"  follow-up audit unavailable: {error}", flush=True)
            else:
                print("  follow-up by bin (censoring audit):", flush=True)
                for row in followup_stats:
                    median = row["median_follow_up_days"]
                    line = (
                        f"    {row['label']:>10}  n={row['n']:>6,}  "
                        f"median follow-up={median:.0f}d"
                        if median is not None
                        else f"    {row['label']:>10}  n={row['n']:>6,}  median follow-up=n/a"
                    )
                    if row["death_fraction"] is not None:
                        line += f"  deaths={row['death_fraction']:.1%}"
                    print(line, flush=True)

        # The baseline features ARE the cancer-type label one-hot encoded, so
        # pairing them with the cancer_type target would score a tautology.
        run_spaces = spaces
        if target == "cancer_type":
            dropped = [s for s in spaces if s in BASELINE_SPACES]
            if dropped:
                print(
                    f"  skipping {dropped} for cancer_type: the baseline features are "
                    "that target's own labels",
                    flush=True,
                )
            run_spaces = [s for s in spaces if s not in BASELINE_SPACES]
            if not run_spaces:
                continue

        for window in windows:
            feature_frames = {}
            for space in run_spaces:
                if space in BASELINE_SPACES:
                    # Not window-dependent: cancer type is a fixed attribute, so
                    # the same frame is reused in every window. It is still run
                    # per window so each window's comparison has its own
                    # like-for-like reference on the same patient set.
                    try:
                        feature_frames[space] = load_baseline_features()
                    except (FileNotFoundError, ValueError) as error:
                        print(f"  [{window}] {space} unavailable: {error}", flush=True)
                elif space in FULL_SPACES:
                    # No feature file of its own: reuses concat's, at full width.
                    if os.path.exists(feature_path(FULL_SPACE_SOURCE, window)):
                        feature_frames[space] = load_features(FULL_SPACE_SOURCE, window)
                elif os.path.exists(feature_path(space, window)):
                    feature_frames[space] = load_features(space, window)
            missing_spaces = [space for space in run_spaces if space not in feature_frames]
            if missing_spaces:
                print(f"  [{window}] missing feature spaces, skipping them: {missing_spaces}", flush=True)
            if not feature_frames:
                print(f"  [{window}] no requested feature files; skipping", flush=True)
                continue

            labels_for_window = labels
            if cohort_mode == "common":
                shared = _shared_mrns(labels, feature_frames)
                labels_for_window = labels.join(shared, on=PATIENT_KEY, how="inner")
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
                    space=space,
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
                    run_context = {"cohort_mode": cohort_mode}
                    if space in BASELINE_SPACES:
                        # Built in memory from the covariate file, so there is no
                        # feature artifact to stat; record what it is made of and
                        # which columns it produced instead.
                        run_context.update(
                            {
                                "feature_artifact": None,
                                "baseline_features": "cancer_type_one_hot",
                                "baseline_feature_columns": cols,
                            }
                        )
                    else:
                        # `concat_full` reuses `concat`'s file; stat that one.
                        source_space = FULL_SPACE_SOURCE if space in FULL_SPACES else space
                        source_path = feature_path(source_space, window)
                        source_stat = os.stat(source_path)
                        run_context.update(
                            {
                                "feature_artifact": source_path,
                                "feature_artifact_size": source_stat.st_size,
                                "feature_artifact_mtime_ns": source_stat.st_mtime_ns,
                            }
                        )
                    if target == "n_lines":
                        # Every patient is labeled regardless of follow-up, so
                        # low bins may partly reflect short observation rather
                        # than disease course.  Record follow-up per bin so the
                        # confound is measurable from the artifact itself.
                        run_context["n_lines_censoring"] = "all_patients_unadjusted"
                        if followup_stats is not None:
                            run_context["n_lines_followup_by_bin"] = followup_stats
                    if target == "first_treatment":
                        run_context.update(
                            {
                                "treatment_granularity": treatment_granularity,
                                "min_treatment_class_n": min_treatment_class_n,
                                "collapsed_treatment_labels": collapsed,
                            }
                        )
                        # At category granularity the label vocabulary is the
                        # GPT-generated MOA_Category table, so the run is only
                        # reproducible if we record which table produced it.
                        if treatment_granularity == "category":
                            run_context["med_classes_file"] = MED_CLASSES_FILE
                            if os.path.exists(MED_CLASSES_FILE):
                                med_stat = os.stat(MED_CLASSES_FILE)
                                run_context["med_classes_file_size"] = med_stat.st_size
                                run_context["med_classes_file_mtime_ns"] = med_stat.st_mtime_ns
                    if target in DRUG_CLASS_TARGETS:
                        drug_class = target.removeprefix("treatment_")
                        run_context.update(
                            {
                                "drug_class": drug_class,
                                # Exposure is "ever, at any line", so a positive
                                # label can be caused by a drug started after
                                # the notes the model reads. Neither window is
                                # leak-free against this target; see
                                # prediction_targets.load_drug_class_target.
                                "drug_class_exposure": "ever_any_line",
                                "drug_class_leakage": "exposure_may_postdate_notes",
                                "drug_class_definition": "gpt_moa_category_regex",
                                "drug_class_patterns": list(
                                    DRUG_CLASS_PATTERNS[drug_class]
                                ),
                                "drug_class_sex_restriction": DRUG_CLASS_SEX[drug_class],
                                "med_classes_file": MED_CLASSES_FILE,
                            }
                        )
                        if os.path.exists(MED_CLASSES_FILE):
                            med_stat = os.stat(MED_CLASSES_FILE)
                            run_context["med_classes_file_size"] = med_stat.st_size
                            run_context["med_classes_file_mtime_ns"] = med_stat.st_mtime_ns
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
    # The baseline is selectable but off by default: it is a reference run, and
    # under the default `common` cohort mode adding it shrinks the shared
    # patient intersection for every embedding space in the same call.
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=SPACES + BASELINE_SPACES + FULL_SPACES,
        default=SPACES,
        help=f"Feature spaces to train. {BASELINE_SPACE!r} is a non-text "
        f"reference: one-hot cancer type and nothing else. {FULL_SPACE!r} reuses "
        f"{FULL_SPACE_SOURCE!r}'s features at full width, with no PCA compression.",
    )
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
        help="Parallel XGBoost fits; -1 uses every available CPU (default).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.outer_folds < 2 or args.inner_folds < 2:
        parser.error("--outer-folds and --inner-folds must both be >= 2")
    if args.n_jobs == 0 or args.n_jobs < -1:
        parser.error("--n-jobs must be -1 or a positive integer")

    if "alltime" in args.windows:
        print(
            "NOTE: alltime embeddings use the complete documented history. Those runs measure "
            "retrospective clinical-label recovery, not prospective prediction.",
            flush=True,
        )
    if "pretreatment" in args.windows:
        print(
            "NOTE: pretreatment embeddings pool only notes before first_treatment_date. Those "
            "runs cover a different, smaller patient set than alltime (no anchor date, or no "
            "pre-anchor notes in some block, drops a patient), so the two windows' metrics are "
            "not paired. They are also not leak-free for the ever-exposure drug-class targets, "
            "whose labels can be set by a drug started after the notes end.",
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
