import json

import numpy as np
import polars as pl
import pytest

from semantic_search.prediction_targets import (
    collapse_rare_treatment_labels,
    load_first_treatment_target,
    load_prostate_subtype_target,
)


def test_semantic_search_builds_only_the_three_block_concat_space(monkeypatch):
    from semantic_search import aggregate_embeddings, common

    pooled = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "CLINICIAN_EMBEDDING_0": [0.1, 0.2],
            "CLINICIAN_EMBEDDING_1": [0.3, 0.4],
            "IMAGING_EMBEDDING_0": [0.5, 0.6],
            "IMAGING_EMBEDDING_1": [0.7, 0.8],
            "PATHOLOGY_EMBEDDING_0": [0.9, float("nan")],
            "PATHOLOGY_EMBEDDING_1": [1.0, float("nan")],
        }
    )
    monkeypatch.setattr(aggregate_embeddings, "_pool_by_type", lambda notes, embeddings: pooled)

    spaces = aggregate_embeddings.build_spaces(pl.DataFrame(), np.empty((0, 0)))

    assert common.SPACES == ["concat"]
    assert common.DEFAULT_WINDOWS == ["alltime"]
    assert list(spaces) == ["concat"]
    assert spaces["concat"].shape == (1, 7)
    assert spaces["concat"].get_column("DFCI_MRN").to_list() == [1]
    assert spaces["concat"].columns[1:] == [
        "CLINICIAN_EMBEDDING_0",
        "CLINICIAN_EMBEDDING_1",
        "IMAGING_EMBEDDING_0",
        "IMAGING_EMBEDDING_1",
        "PATHOLOGY_EMBEDDING_0",
        "PATHOLOGY_EMBEDDING_1",
    ]


def test_prostate_subtype_is_cohort_bounded_and_nepc_takes_precedence(tmp_path):
    path = tmp_path / "avpc_nepc_labels.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [10, 20, 30, 40],
            "has_avpc": [0, 1, 0, 1],
            "has_nepc_timeline": [0, 0, 1, 1],
        }
    ).write_parquet(path)

    labels = load_prostate_subtype_target(labels_path=str(path))

    assert dict(labels.iter_rows()) == {
        10: "CONVENTIONAL",
        20: "AVPC",
        30: "NEPC",
        40: "NEPC",
    }


def test_first_treatment_uses_frozen_anchor_category_or_drug(tmp_path):
    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "ANCHOR_DRUG": ["Drug A", "Drug B"],
            "ANCHOR_DRUG_CATEG": ["Chemotherapy", "Targeted"],
        }
    ).write_parquet(path)

    category = load_first_treatment_target(cohort_path=str(path))
    drug = load_first_treatment_target(cohort_path=str(path), granularity="drug")

    assert category.get_column("label").to_list() == ["CHEMOTHERAPY", "TARGETED"]
    assert drug.get_column("label").to_list() == ["DRUG A", "DRUG B"]


def test_sparse_treatment_categories_collapse_to_other():
    labels = pl.DataFrame(
        {"DFCI_MRN": [1, 2, 3, 4, 5], "label": ["A", "A", "A", "B", "C"]}
    )
    collapsed, rare = collapse_rare_treatment_labels(labels, min_class_n=2)

    assert rare == ["B", "C"]
    assert collapsed.get_column("label").to_list() == ["A", "A", "A", "OTHER", "OTHER"]


def test_result_writer_replaces_matching_setup_and_preserves_other_runs(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    from semantic_search.train_prediction_models import _merge_write

    path = tmp_path / "metrics.csv"
    keys = ["target", "space"]
    _merge_write(
        str(path),
        [
            {"target": "stage", "space": "concat", "score": 0.5},
            {"target": "cancer_type", "space": "concat", "score": 0.6},
        ],
        keys,
    )
    _merge_write(
        str(path),
        [{"target": "stage", "space": "concat", "score": 0.9}],
        keys,
    )

    result = pl.read_csv(path).sort("target")
    assert result.select("target", "score").rows() == [
        ("cancer_type", 0.6),
        ("stage", 0.9),
    ]


@pytest.mark.parametrize("model", ["elastic_net", "xgboost"])
def test_nested_cv_writes_oof_predictions_and_refit_model(tmp_path, monkeypatch, model):
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    from semantic_search import train_prediction_models as training

    monkeypatch.setattr(training, "PREDICTIONS_DIR", str(tmp_path / "predictions"))
    monkeypatch.setattr(training, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(training, "PREDICTION_META_DIR", str(tmp_path / "meta"))
    monkeypatch.setattr(
        training,
        "LR_GRID",
        {"model__C": [1.0], "model__l1_ratio": [0.5]},
    )
    monkeypatch.setattr(
        training,
        "XGB_GRID",
        {"max_depth": [2], "min_child_weight": [1], "reg_lambda": [1.0]},
    )

    rng = np.random.default_rng(7)
    y = np.repeat(["A", "B", "C"], 6)
    signal = np.repeat(np.eye(3), 6, axis=0)
    X = np.column_stack([signal, rng.normal(scale=0.05, size=len(y))])
    data = pl.DataFrame(
        {
            "DFCI_MRN": np.arange(100, 100 + len(y)),
            "EMBEDDING_0": X[:, 0],
            "EMBEDDING_1": X[:, 1],
            "EMBEDDING_2": X[:, 2],
            "EMBEDDING_3": X[:, 3],
            "label": y,
        }
    )

    meta, folds, by_class = training.train_one(
        data,
        [f"EMBEDDING_{i}" for i in range(4)],
        target="stage",
        space="concat",
        window="pretreatment",
        model=model,
        outer_folds=3,
        inner_folds=2,
        seed=9,
        n_jobs=1,
        overwrite=False,
    )

    assert len(folds) == 3
    assert {row["class"] for row in by_class} == {"A", "B", "C"}
    assert meta["n_patients"] == 18
    predictions = pl.read_parquet(meta["artifacts"]["predictions"])
    assert predictions.height == 18
    assert predictions.get_column("DFCI_MRN").n_unique() == 18
    assert sorted(predictions.get_column("fold").unique().to_list()) == [1, 2, 3]
    assert all(len(values) == 3 for values in predictions["class_probabilities"])
    assert (tmp_path / "models" / f"stage__concat__pretreatment__{model}.joblib").exists()
    with open(meta["artifacts"]["meta"]) as handle:
        on_disk = json.load(handle)
    assert on_disk["classes"] == ["A", "B", "C"]

    reused, reused_folds, _ = training.train_one(
        data,
        [f"EMBEDDING_{i}" for i in range(4)],
        target="stage",
        space="concat",
        window="pretreatment",
        model=model,
        outer_folds=3,
        inner_folds=2,
        seed=9,
        n_jobs=1,
        overwrite=False,
    )
    assert reused["artifacts"] == meta["artifacts"]
    assert reused_folds == on_disk["fold_metrics"]

    with pytest.raises(ValueError, match="different cohort or CV configuration"):
        training.train_one(
            data,
            [f"EMBEDDING_{i}" for i in range(4)],
            target="stage",
            space="concat",
            window="pretreatment",
            model=model,
            outer_folds=3,
            inner_folds=2,
            seed=10,
            n_jobs=1,
            overwrite=False,
        )
