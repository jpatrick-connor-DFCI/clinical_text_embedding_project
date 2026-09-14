"""Stage 2: compute principal components of patient-level note embeddings.

The exploratory workflow fits PCA separately for clinician, imaging, and
pathology note embeddings (one 768-dimensional space per note type):

    L2-normalize rows -> StandardScaler -> PCA

For each requested note window this writes patient PC scores, long-form feature
loadings, the fitted transformer, and explained-variance metadata under
``SEMANTIC_SEARCH_PATH/pcs``.

Run:
    python -m semantic_search.compute_pcs [--windows ...] [--n-components 50]
                                          [--overwrite]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import Normalizer, StandardScaler  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from data.schema import assert_schema  # noqa: E402
from semantic_search.common import (  # noqa: E402
    DEFAULT_WINDOWS,
    PATIENT_KEY,
    PC_SPACES,
    WINDOWS,
    embedding_cols,
    ensure_dirs,
    feature_path,
    load_features,
    load_pc_meta,
    pc_loadings_path,
    pc_meta_path,
    pc_scores_path,
    pc_transformer_path,
    result_path,
    write_result,
)

N_COMPONENTS = 50
RANDOM_SEED = 0
VARIANCE_COLUMNS = [
    "space", "window", "pc", "component", "explained_variance_ratio",
    "cumulative_explained_variance_ratio", "n_patients", "n_features",
]


def _atomic_json(path: str, payload: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_joblib(path: str, payload) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".tmp_", suffix=".joblib", dir=os.path.dirname(path))
    os.close(fd)
    try:
        joblib.dump(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _feature_signature(path: str, columns: list[str]) -> dict:
    stat = os.stat(path)
    return {
        "path": path,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "columns_sha256": hashlib.sha256("\n".join(columns).encode()).hexdigest(),
    }


def _variance_rows(meta: dict) -> list[dict]:
    cumulative = 0.0
    rows = []
    for component, ratio in enumerate(meta["explained_variance_ratio"], start=1):
        cumulative += ratio
        rows.append({
            "space": meta["space"],
            "window": meta["window"],
            "pc": f"PC{component}",
            "component": component,
            "explained_variance_ratio": ratio,
            "cumulative_explained_variance_ratio": cumulative,
            "n_patients": meta["n_patients"],
            "n_features": meta["n_features"],
        })
    return rows


def _feature_parts(feature: str) -> tuple[str, int | None]:
    if "_EMBEDDING_" not in feature:
        return "unknown", None
    note_type, dimension = feature.rsplit("_EMBEDDING_", 1)
    try:
        parsed_dimension = int(dimension)
    except ValueError:
        parsed_dimension = None
    return note_type.title(), parsed_dimension


def fit_one(
    space: str,
    window: str,
    *,
    n_components: int,
    seed: int,
    overwrite: bool,
) -> tuple[dict, list[dict]]:
    source = feature_path(space, window)
    if not os.path.exists(source):
        raise FileNotFoundError(f"No feature file for {space}/{window}: {source}")

    artifacts = {
        "scores": pc_scores_path(space, window),
        "loadings": pc_loadings_path(space, window),
        "transformer": pc_transformer_path(space, window),
        "meta": pc_meta_path(space, window),
    }
    features = load_features(space, window)
    columns = embedding_cols(features)
    signature = _feature_signature(source, columns)

    if not overwrite and all(os.path.exists(path) for path in artifacts.values()):
        meta = load_pc_meta(space, window)
        if (
            meta.get("source_signature") != signature
            or meta.get("n_components_requested") != n_components
            or meta.get("seed") != seed
        ):
            raise ValueError(
                f"Existing PCs for {space}/{window} use a different feature artifact or "
                "configuration; pass --overwrite to replace them."
            )
        print(f"  [{space}/{window}] PC artifacts exist, reusing", flush=True)
        return meta, _variance_rows(meta)

    if not columns:
        raise ValueError(f"{space}/{window} has no embedding columns")
    X = features.select(columns).to_numpy().astype(np.float32, copy=False)
    if not np.isfinite(X).all():
        raise ValueError(f"{space}/{window} contains non-finite embeddings")
    retained = min(n_components, X.shape[0] - 1, X.shape[1])
    if retained < 1:
        raise ValueError(f"Need at least two patients for PCA; observed {X.shape[0]}")

    print(
        f"  [{space}/{window}] {X.shape[0]:,} patients x {X.shape[1]:,} features; "
        f"retaining {retained} PCs",
        flush=True,
    )
    transformer = Pipeline([
        ("l2_normalize", Normalizer(norm="l2")),
        ("standardize", StandardScaler()),
        ("pca", PCA(n_components=retained, random_state=seed)),
    ])
    scores_array = transformer.fit_transform(X)
    pca = transformer.named_steps["pca"]
    pc_names = [f"PC{index}" for index in range(1, retained + 1)]

    scores = pl.DataFrame({
        PATIENT_KEY: features.get_column(PATIENT_KEY),
        **{name: scores_array[:, index] for index, name in enumerate(pc_names)},
    })
    assert_schema(scores, f"{space}_{window}_pc_scores", [PATIENT_KEY] + pc_names,
                  key_col=PATIENT_KEY)
    scores.write_parquet(artifacts["scores"])

    feature_note_types, feature_dimensions = zip(*[_feature_parts(column) for column in columns])
    loadings_array = pca.components_.reshape(-1)
    loadings = pl.DataFrame({
        "space": [space] * loadings_array.size,
        "window": [window] * loadings_array.size,
        "pc": np.repeat(pc_names, len(columns)),
        "component": np.repeat(np.arange(1, retained + 1), len(columns)),
        "feature": np.tile(columns, retained),
        "note_type": np.tile(feature_note_types, retained),
        "embedding_dimension": np.tile(feature_dimensions, retained),
        "loading": loadings_array,
        "abs_loading": np.abs(loadings_array),
    })
    loadings.write_parquet(artifacts["loadings"])
    _atomic_joblib(artifacts["transformer"], transformer)

    ratios = [float(value) for value in pca.explained_variance_ratio_]
    meta = {
        "space": space,
        "window": window,
        "seed": seed,
        "n_patients": X.shape[0],
        "n_features": X.shape[1],
        "n_components_requested": n_components,
        "n_components_retained": retained,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": float(sum(ratios)),
        "preprocessing": "l2_normalize -> standard_scaler -> pca",
        "source_signature": signature,
        "artifacts": artifacts,
    }
    _atomic_json(artifacts["meta"], meta)
    print(
        f"    cumulative variance={meta['cumulative_explained_variance_ratio']:.1%} "
        f"-> {artifacts['scores']}",
        flush=True,
    )
    return meta, _variance_rows(meta)


def _merge_variance(rows: list[dict]) -> pl.DataFrame:
    new = pl.DataFrame(rows).select(VARIANCE_COLUMNS)
    path = result_path("pc_explained_variance")
    if os.path.exists(path):
        old = pl.read_csv(path)
        keys = new.select("space", "window").unique()
        old = old.join(keys, on=["space", "window"], how="anti")
        new = pl.concat([old, new], how="diagonal_relaxed")
    return new.sort(["space", "window", "component"])


def run(
    spaces: list[str],
    windows: list[str],
    *,
    n_components: int = N_COMPONENTS,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> pl.DataFrame:
    ensure_dirs()
    rows: list[dict] = []
    setups = [(space, window) for window in windows for space in spaces]
    for space, window in tqdm(setups, desc="PCA setups", unit="setup"):
        try:
            _, variance_rows = fit_one(
                space,
                window,
                n_components=n_components,
                seed=seed,
                overwrite=overwrite,
            )
        except FileNotFoundError as error:
            print(f"  {error}; skipping", flush=True)
            continue
        rows.extend(variance_rows)

    if not rows:
        existing = result_path("pc_explained_variance")
        variance = (
            pl.read_csv(existing)
            if os.path.exists(existing)
            else pl.DataFrame(schema={column: pl.Utf8 for column in VARIANCE_COLUMNS})
        )
    else:
        variance = _merge_variance(rows)
    write_result(variance, "pc_explained_variance")
    return variance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spaces", nargs="+", choices=PC_SPACES, default=PC_SPACES)
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    parser.add_argument("--n-components", type=int, default=N_COMPONENTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.n_components < 1:
        parser.error("--n-components must be >= 1")
    run(
        args.spaces,
        args.windows,
        n_components=args.n_components,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
