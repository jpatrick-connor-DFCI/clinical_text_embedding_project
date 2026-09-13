"""Stage 2: KMeans-cluster patients in the concatenated embedding space.

Preprocessing order is fixed and recorded in each run's meta JSON:

    L2-normalize rows -> StandardScaler -> PCA(N_COMPONENTS) -> KMeans(k)

L2 first because these are transformer embeddings, whose trained geometry is
cosine; on unit-norm rows Euclidean KMeans is monotone in cosine distance.
StandardScaler then prevents one of the three concatenated note-type blocks, or
a handful of high-variance dimensions, from dominating. PCA both denoises and
makes the silhouette scan affordable.

Cluster labels are relabeled by ascending cluster size so a rerun with the same
seed produces not just the same partition but the same integer names -- the
stability trick `figures/prep/figure4.py` applies with mean slope.

Run:
    python -m semantic_search.cluster_patients [--spaces ...] [--windows ...]
                                               [--k-min 2] [--k-max 12] [--overwrite]
"""

from __future__ import annotations

import argparse
import os

# Some cluster nodes advertise more CPUs than the precompiled OpenBLAS build
# supports.  Cap thread counts before NumPy/scikit-learn initializes a BLAS
# runtime, or OpenBLAS can segfault allocating thread metadata rather than
# raising.  Same guard, same limit, as figures/prep/figure4.py.
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
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler, normalize  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from data.schema import assert_schema  # noqa: E402
from semantic_search.common import (  # noqa: E402
    DEFAULT_WINDOWS,
    PATIENT_KEY,
    SPACES,
    WINDOWS,
    coords_path,
    embedding_cols,
    ensure_dirs,
    feature_path,
    labels_path,
    load_features,
    load_labels,
    write_cluster_meta,
    write_result,
)

RANDOM_SEED = 0
N_COMPONENTS = 50
K_MIN, K_MAX = 2, 12
N_INIT = 10
# silhouette_score is O(n^2) in memory and time; above this many rows it is
# estimated on a seeded subsample instead of refused.
SILHOUETTE_MAX_N = 20_000

SILHOUETTE_COLUMNS = ["space", "window", "k", "silhouette", "inertia", "n_patients"]
CONCORDANCE_COLUMNS = ["space_a", "window_a", "space_b", "window_b", "ari", "n_shared"]


def _prepare(X: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, float]:
    """L2 -> standardize -> PCA.  Returns the reduced matrix and the fraction of
    variance it retains."""
    Xn = normalize(X, norm="l2", axis=1)
    Xs = StandardScaler().fit_transform(Xn)
    n_components = min(n_components, Xs.shape[0], Xs.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    Xp = pca.fit_transform(Xs)
    return Xp, float(pca.explained_variance_ratio_.sum())


def _silhouette(X: np.ndarray, labels: np.ndarray, seed: int) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    if X.shape[0] > SILHOUETTE_MAX_N:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], SILHOUETTE_MAX_N, replace=False)
        X, labels = X[idx], labels[idx]
        if len(np.unique(labels)) < 2:
            return float("nan")
    return float(silhouette_score(X, labels))


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Rename clusters so label 0 is the smallest and k-1 the largest.

    KMeans label integers are an artifact of centroid initialization order and
    carry no meaning; without this a rerun can produce the same partition under
    permuted names, which would make every downstream per-cluster table
    incomparable across runs.
    """
    values, counts = np.unique(labels, return_counts=True)
    order = values[np.argsort(counts, kind="stable")]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[v] for v in labels], dtype=np.int64)


def scan_k(
    X: np.ndarray,
    k_values,
    seed: int,
    *,
    progress_desc: str | None = None,
) -> list[dict]:
    rows = []
    values = list(k_values)
    for k in tqdm(
        values,
        desc=progress_desc or "Silhouette scan",
        unit="k",
        leave=False,
        disable=progress_desc is None,
    ):
        if k >= X.shape[0]:
            continue
        km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)
        rows.append({
            "k": int(k),
            "silhouette": _silhouette(X, km.labels_, seed),
            "inertia": float(km.inertia_),
            "n_patients": int(X.shape[0]),
        })
    return rows


def cluster_one(
    space: str,
    window: str,
    k: int | None,
    k_values,
    seed: int,
    overwrite: bool,
) -> tuple[list[dict], dict | None]:
    """Scan k and fit the chosen k for one (space, window).

    Returns the scan rows and the run's meta dict (None if skipped).
    """
    path = feature_path(space, window)
    if not os.path.exists(path):
        print(f"  [{space}/{window}] no feature file; skipping", flush=True)
        return [], None
    if os.path.exists(labels_path(space, window)) and not overwrite:
        print(f"  [{space}/{window}] labels exist, skipping (use --overwrite)", flush=True)
        return [], None

    df = load_features(space, window)
    cols = embedding_cols(df)
    mrns = df.get_column(PATIENT_KEY)
    X = df.select(cols).to_numpy().astype(np.float32)
    print(f"  [{space}/{window}] {X.shape[0]:,} patients x {X.shape[1]} features", flush=True)

    Xp, pca_variance = _prepare(X, N_COMPONENTS, seed)
    print(f"    PCA -> {Xp.shape[1]} comps, {pca_variance:.1%} variance", flush=True)

    scan_rows = scan_k(
        Xp,
        k_values,
        seed,
        progress_desc=f"{space}/{window}: scanning k",
    )
    for row in scan_rows:
        row["space"], row["window"] = space, window

    if k is None:
        scored = [r for r in scan_rows if not np.isnan(r["silhouette"])]
        if not scored:
            print(f"    no scorable k for {space}/{window}; skipping fit", flush=True)
            return scan_rows, None
        k = max(scored, key=lambda r: r["silhouette"])["k"]
        print(f"    chose k={k} by max silhouette", flush=True)

    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(Xp)
    labels = _relabel_by_size(km.labels_)
    silhouette = _silhouette(Xp, labels, seed)

    labels_df = pl.DataFrame({PATIENT_KEY: mrns, "cluster": labels})
    assert_schema(labels_df, f"{space}_{window}_labels", [PATIENT_KEY, "cluster"],
                  key_col=PATIENT_KEY)
    labels_df.write_parquet(labels_path(space, window))

    # First two principal components double as the 2-D plotting coordinates:
    # umap-learn is not in environment.yml, and PCA keeps the scatter in the same
    # geometry the clustering actually used rather than a separate embedding.
    coords_df = pl.DataFrame({
        PATIENT_KEY: mrns,
        "dim1": Xp[:, 0].astype(float),
        "dim2": Xp[:, 1].astype(float) if Xp.shape[1] > 1 else np.zeros(Xp.shape[0]),
    })
    coords_df.write_parquet(coords_path(space, window))

    meta = {
        "space": space, "window": window, "k": int(k), "seed": seed,
        "n_patients": int(X.shape[0]), "n_features": int(X.shape[1]),
        "pca_components": int(Xp.shape[1]), "pca_variance": pca_variance,
        "silhouette": silhouette, "inertia": float(km.inertia_),
        "preprocessing": "l2_normalize -> standard_scaler -> pca",
        "coords_source": "pca",
        "cluster_sizes": {str(c): int(n) for c, n in
                          zip(*np.unique(labels, return_counts=True))},
    }
    write_cluster_meta(space, window, meta)
    print(f"    k={k}, silhouette={silhouette:.4f}, "
          f"sizes={list(meta['cluster_sizes'].values())}", flush=True)
    return scan_rows, meta


def concordance(pairs: list[tuple[str, str]]) -> pl.DataFrame:
    """Adjusted Rand index between every pair of runs, on their shared patients.

    With one feature space, this compares the all-time and pre-treatment
    partitions when both windows were run.
    """
    loaded = {}
    for space, window in pairs:
        if os.path.exists(labels_path(space, window)):
            loaded[(space, window)] = load_labels(space, window)

    keys = sorted(loaded)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            merged = loaded[a].join(loaded[b], on=PATIENT_KEY, how="inner",
                                    suffix="_b")
            if merged.height < 2:
                continue
            rows.append({
                "space_a": a[0], "window_a": a[1],
                "space_b": b[0], "window_b": b[1],
                "ari": float(adjusted_rand_score(
                    merged.get_column("cluster").to_numpy(),
                    merged.get_column("cluster_b").to_numpy())),
                "n_shared": int(merged.height),
            })
    return pl.DataFrame(rows).select(CONCORDANCE_COLUMNS) if rows else \
        pl.DataFrame(schema={c: pl.Utf8 for c in CONCORDANCE_COLUMNS})


def run(
    spaces: list[str],
    windows: list[str],
    k: int | None = None,
    k_min: int = K_MIN,
    k_max: int = K_MAX,
    seed: int = RANDOM_SEED,
    overwrite: bool = False,
) -> None:
    ensure_dirs()
    k_values = range(k_min, k_max + 1)

    scan_rows: list[dict] = []
    setups = [(space, window) for window in windows for space in spaces]
    for space, window in tqdm(setups, desc="Clustering setups", unit="setup"):
        print(f"\n[{window}]", flush=True)
        rows, _ = cluster_one(space, window, k, k_values, seed, overwrite)
        scan_rows.extend(rows)

    scan = pl.DataFrame(scan_rows).select(SILHOUETTE_COLUMNS) if scan_rows else \
        pl.DataFrame(schema={c: pl.Utf8 for c in SILHOUETTE_COLUMNS})
    write_result(scan, "silhouette_scan")

    print("\nCross-run concordance (ARI)...", flush=True)
    write_result(concordance([(s, w) for w in windows for s in spaces]),
                 "cluster_concordance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spaces", nargs="+", choices=SPACES, default=SPACES)
    parser.add_argument("--windows", nargs="+", choices=WINDOWS, default=DEFAULT_WINDOWS)
    parser.add_argument("--k", type=int, default=None,
                        help="Force this k for every space; default picks the silhouette argmax.")
    parser.add_argument("--k-min", type=int, default=K_MIN)
    parser.add_argument("--k-max", type=int, default=K_MAX)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(spaces=args.spaces, windows=args.windows, k=args.k, k_min=args.k_min,
        k_max=args.k_max, seed=args.seed, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
