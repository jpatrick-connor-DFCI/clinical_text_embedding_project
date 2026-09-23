import json

import numpy as np
import polars as pl
import pytest

from semantic_search.prediction_targets import (
    _bin_line_count,
    collapse_rare_treatment_labels,
    load_first_treatment_target,
    load_n_lines_followup_stats,
    load_n_lines_target,
    load_prostate_subtype_target,
)


def test_semantic_search_builds_concat_and_isolated_note_type_spaces(monkeypatch):
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
    assert common.DEFAULT_WINDOWS == ["alltime", "pretreatment"]
    assert list(spaces) == ["concat", "clinician", "imaging", "pathology"]
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
    assert spaces["clinician"].shape == (2, 3)
    assert spaces["imaging"].shape == (2, 3)
    assert spaces["pathology"].shape == (1, 3)
    assert spaces["clinician"].columns[1:] == [
        "CLINICIAN_EMBEDDING_0", "CLINICIAN_EMBEDDING_1"
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
            "DFCI_MRN": [1, 2, 3],
            "ANCHOR_DRUG": ["Drug A", "Drug B", "Drug C"],
            # PROFILE's raw category must NOT be what the category label uses.
            "ANCHOR_DRUG_CATEG": ["Chemotherapy", "Targeted", "Chemotherapy"],
        }
    ).write_parquet(path)
    med_classes = tmp_path / "med_classes.csv"
    pl.DataFrame(
        {
            "MED_NAME": ["Drug A", "Drug B"],
            "MOA_Category": ["Taxane", "PARP Inhibitor"],
        }
    ).write_csv(med_classes)

    category = load_first_treatment_target(
        cohort_path=str(path), med_classes_path=str(med_classes)
    )
    drug = load_first_treatment_target(cohort_path=str(path), granularity="drug")

    # Condensed GPT MOA_Category, matching the PX_on_* covariate vocabulary;
    # Drug C is absent from the class table and falls back to OTHER.
    assert category.get_column("label").to_list() == [
        "TAXANE",
        "PARP INHIBITOR",
        "OTHER",
    ]
    assert drug.get_column("label").to_list() == ["DRUG A", "DRUG B", "DRUG C"]


def test_first_treatment_category_ignores_profile_raw_drug_categ(tmp_path):
    """The category label must come from the GPT class table, not ANCHOR_DRUG_CATEG."""
    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [1],
            "ANCHOR_DRUG": ["Drug A"],
            "ANCHOR_DRUG_CATEG": ["Chemotherapy"],
        }
    ).write_parquet(path)
    med_classes = tmp_path / "med_classes.csv"
    pl.DataFrame(
        {"MED_NAME": ["Drug A"], "MOA_Category": ["Androgen Receptor Inhibitor"]}
    ).write_csv(med_classes)

    labels = load_first_treatment_target(
        cohort_path=str(path), med_classes_path=str(med_classes)
    )

    assert labels.get_column("label").to_list() == ["ANDROGEN RECEPTOR INHIBITOR"]


def test_first_treatment_category_takes_last_duplicate_med_name(tmp_path):
    """Mirrors build_treatment_by_line_df's unique(MED_NAME, keep='last')."""
    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame({"DFCI_MRN": [1], "ANCHOR_DRUG": ["Drug A"]}).write_parquet(path)
    med_classes = tmp_path / "med_classes.csv"
    pl.DataFrame(
        {"MED_NAME": ["Drug A", "Drug A"], "MOA_Category": ["Stale", "Corrected"]}
    ).write_csv(med_classes)

    labels = load_first_treatment_target(
        cohort_path=str(path), med_classes_path=str(med_classes)
    )

    assert labels.get_column("label").to_list() == ["CORRECTED"]


def test_sparse_treatment_categories_collapse_to_other():
    labels = pl.DataFrame(
        {"DFCI_MRN": [1, 2, 3, 4, 5], "label": ["A", "A", "A", "B", "C"]}
    )
    collapsed, rare = collapse_rare_treatment_labels(labels, min_class_n=2)

    assert rare == ["B", "C"]
    assert collapsed.get_column("label").to_list() == ["A", "A", "A", "OTHER", "OTHER"]


def test_pc_stage_writes_scores_loadings_transformer_and_metadata(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    from semantic_search import compute_pcs

    feature_file = tmp_path / "clinician_alltime.parquet"
    scores_file = tmp_path / "clinician_alltime_scores.parquet"
    loadings_file = tmp_path / "clinician_alltime_loadings.parquet"
    transformer_file = tmp_path / "clinician_alltime_transformer.joblib"
    meta_file = tmp_path / "clinician_alltime_meta.json"
    rng = np.random.default_rng(11)
    feature_names = [f"CLINICIAN_EMBEDDING_{dimension}" for dimension in range(4)]
    features = pl.DataFrame({
        "DFCI_MRN": np.arange(1, 11),
        **{name: rng.normal(size=10) for name in feature_names},
    })
    features.write_parquet(feature_file)

    monkeypatch.setattr(compute_pcs, "feature_path", lambda space, window: str(feature_file))
    monkeypatch.setattr(compute_pcs, "load_features", lambda space, window: features)
    monkeypatch.setattr(compute_pcs, "pc_scores_path", lambda space, window: str(scores_file))
    monkeypatch.setattr(compute_pcs, "pc_loadings_path", lambda space, window: str(loadings_file))
    monkeypatch.setattr(
        compute_pcs, "pc_transformer_path", lambda space, window: str(transformer_file)
    )
    monkeypatch.setattr(compute_pcs, "pc_meta_path", lambda space, window: str(meta_file))

    meta, variance = compute_pcs.fit_one(
        "clinician", "alltime", n_components=3, seed=7, overwrite=False
    )

    assert meta["n_components_retained"] == 3
    assert len(variance) == 3
    assert pl.read_parquet(scores_file).columns == ["DFCI_MRN", "PC1", "PC2", "PC3"]
    loadings = pl.read_parquet(loadings_file)
    assert loadings.height == 3 * len(feature_names)
    assert set(loadings.get_column("note_type")) == {"Clinician"}
    assert transformer_file.exists()
    assert meta_file.exists()


def test_pc_clinical_tests_report_correlation_and_omnibus_effects():
    from semantic_search.stats import categorical_pc_test, spearman_test

    continuous = spearman_test([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])
    categorical = categorical_pc_test(
        [1, 2, 3, 10, 11, 12], ["A", "A", "A", "B", "B", "B"]
    )

    assert continuous["effect_name"] == "spearman_rho"
    assert continuous["effect"] == pytest.approx(1.0)
    assert continuous["p"] < 0.01
    assert categorical["effect_name"] == "epsilon_squared"
    assert categorical["n_levels"] == 2
    assert 0 <= categorical["effect"] <= 1


def test_pc_family_associations_cover_every_pc_and_variable():
    from semantic_search.correlate_pcs import _family_associations

    scores = pl.DataFrame({
        "DFCI_MRN": [1, 2, 3, 4, 5, 6],
        "PC1": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
        "PC2": [1.0, 0.0, -1.0, -1.0, 0.0, 1.0],
    })
    clinical = pl.DataFrame({
        "DFCI_MRN": [1, 2, 3, 4, 5, 6],
        "AGE": [10, 20, 30, 40, 50, 60],
        "TYPE": ["A", "A", "A", "B", "B", "B"],
    })

    rows, coverage = _family_associations(
        scores,
        clinical,
        ["AGE"],
        ["TYPE"],
        pc_columns=["PC1", "PC2"],
        variance={"PC1": 0.4, "PC2": 0.2},
        space="concat",
        window="alltime",
        family="toy",
    )

    assert len(rows) == 4
    assert {(row["pc"], row["variable"]) for row in rows} == {
        ("PC1", "AGE"), ("PC1", "TYPE"), ("PC2", "AGE"), ("PC2", "TYPE")
    }
    assert coverage["coverage"] == 1.0


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


def test_nested_cv_writes_oof_predictions_and_refit_xgboost(tmp_path, monkeypatch):
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    from semantic_search import train_prediction_models as training

    monkeypatch.setattr(training, "PREDICTIONS_DIR", str(tmp_path / "predictions"))
    monkeypatch.setattr(training, "MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(training, "PREDICTION_META_DIR", str(tmp_path / "meta"))
    monkeypatch.setattr(
        training,
        "PCA_COMPONENTS",
        {"CLINICIAN": 1, "IMAGING": 1, "PATHOLOGY": 1},
    )
    monkeypatch.setattr(
        training,
        "XGB_GRID",
        {
            "model__max_depth": [2],
            "model__min_child_weight": [1],
            "model__reg_lambda": [1.0],
        },
    )

    rng = np.random.default_rng(7)
    y = np.repeat(["A", "B", "C"], 6)
    signal = np.repeat(np.eye(3), 6, axis=0)
    X = np.column_stack([
        signal[:, 0], rng.normal(scale=0.05, size=len(y)),
        signal[:, 1], rng.normal(scale=0.05, size=len(y)),
        signal[:, 2], rng.normal(scale=0.05, size=len(y)),
    ])
    feature_names = [
        f"{note_type}_EMBEDDING_{dimension}"
        for note_type in ("CLINICIAN", "IMAGING", "PATHOLOGY")
        for dimension in range(2)
    ]
    data = pl.DataFrame(
        {
            "DFCI_MRN": np.arange(100, 100 + len(y)),
            **{name: X[:, index] for index, name in enumerate(feature_names)},
            "label": y,
        }
    )

    model = "xgboost"
    assert training.MODELS == [model]

    meta, folds, by_class = training.train_one(
        data,
        feature_names,
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
    assert meta["n_transformed_features"] == 3
    assert meta["pca_components"] == {
        "CLINICIAN": 1, "IMAGING": 1, "PATHOLOGY": 1,
    }
    predictions = pl.read_parquet(meta["artifacts"]["predictions"])
    assert predictions.height == 18
    assert predictions.get_column("DFCI_MRN").n_unique() == 18
    assert sorted(predictions.get_column("fold").unique().to_list()) == [1, 2, 3]
    assert all(len(values) == 3 for values in predictions["class_probabilities"])
    model_path = tmp_path / "models" / f"stage__concat__pretreatment__{model}.joblib"
    assert model_path.exists()
    payload = pytest.importorskip("joblib").load(model_path)
    assert payload["transformed_feature_columns"] == [
        "CLINICIAN_PC1", "IMAGING_PC1", "PATHOLOGY_PC1",
    ]
    transformed = payload["estimator"].named_steps["block_pca"].transform(X)
    assert transformed.shape == (18, 3)
    assert payload["estimator"].predict_proba(X).shape == (18, 3)
    with open(meta["artifacts"]["meta"]) as handle:
        on_disk = json.load(handle)
    assert on_disk["classes"] == ["A", "B", "C"]

    reused, reused_folds, _ = training.train_one(
        data,
        feature_names,
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
            feature_names,
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


# --- n_lines target ------------------------------------------------------

def _med_long(rows):
    """(DFCI_MRN, DRUG, START_DT) rows shaped like unpivot_medications_summary()."""
    return pl.DataFrame(
        {
            "DFCI_MRN": [r[0] for r in rows],
            "DRUG": [r[1] for r in rows],
            "START_DT": [r[2] for r in rows],
        }
    )


def _patch_meds(monkeypatch, long):
    from pipelines.preprocessing import profile_sources as ps

    monkeypatch.setattr(ps, "unpivot_medications_summary", lambda: long)


def test_bin_line_count_is_open_ended_at_the_top():
    assert _bin_line_count(1) == "1 line"
    assert _bin_line_count(2) == "2 lines"
    assert _bin_line_count(3) == "3 lines"
    # The 7-slot MEDICATIONS_SUMMARY ceiling must not create distinct classes.
    assert _bin_line_count(4) == "4+ lines"
    assert _bin_line_count(7) == "4+ lines"
    assert _bin_line_count(99) == "4+ lines"


def test_bin_labels_sort_in_clinical_order():
    """LabelEncoder orders classes lexicographically; that must match clinical order."""
    labels = sorted({_bin_line_count(n) for n in range(1, 10)})
    assert labels == ["1 line", "2 lines", "3 lines", "4+ lines"]


def test_n_lines_counts_distinct_lines_and_bins_them(tmp_path, monkeypatch):
    from datetime import date

    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame({"DFCI_MRN": [1, 2, 3]}).write_parquet(path)
    _patch_meds(
        monkeypatch,
        _med_long([
            # Patient 1: two drugs inside the 28-day window = one line.
            (1, "Drug A", date(2020, 1, 1)),
            (1, "Drug B", date(2020, 1, 10)),
            # Patient 2: three well-separated starts = three lines.
            (2, "Drug A", date(2020, 1, 1)),
            (2, "Drug B", date(2020, 6, 1)),
            (2, "Drug C", date(2021, 1, 1)),
            # Patient 3: five separated starts = 4+ bin.
            (3, "Drug A", date(2019, 1, 1)),
            (3, "Drug B", date(2019, 6, 1)),
            (3, "Drug C", date(2020, 1, 1)),
            (3, "Drug D", date(2020, 6, 1)),
            (3, "Drug E", date(2021, 1, 1)),
        ]),
    )

    labels = load_n_lines_target(cohort_path=str(path))

    assert dict(labels.iter_rows()) == {1: "1 line", 2: "3 lines", 3: "4+ lines"}


def test_n_lines_excludes_patients_outside_the_cohort(tmp_path, monkeypatch):
    from datetime import date

    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame({"DFCI_MRN": [1]}).write_parquet(path)
    _patch_meds(
        monkeypatch,
        _med_long([(1, "Drug A", date(2020, 1, 1)), (999, "Drug B", date(2020, 1, 1))]),
    )

    labels = load_n_lines_target(cohort_path=str(path))

    assert labels.get_column("DFCI_MRN").to_list() == [1]


def test_n_lines_omits_cohort_patients_with_no_medications(tmp_path, monkeypatch):
    """No medication row means an unknown count, not zero lines."""
    from datetime import date

    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame({"DFCI_MRN": [1, 2]}).write_parquet(path)
    _patch_meds(monkeypatch, _med_long([(1, "Drug A", date(2020, 1, 1))]))

    labels = load_n_lines_target(cohort_path=str(path))

    assert labels.get_column("DFCI_MRN").to_list() == [1]
    assert "0 lines" not in labels.get_column("label").to_list()


def test_n_lines_followup_stats_expose_the_censoring_confound(tmp_path, monkeypatch):
    """Short follow-up in the low bins is exactly the confound this must surface."""
    from datetime import date

    path = tmp_path / "cohort_df.parquet"
    pl.DataFrame(
        {
            "DFCI_MRN": [1, 2],
            "first_treatment_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "death_date": [None, date(2022, 1, 1)],
            "last_contact_date": [date(2020, 2, 1), None],
            "death": [0, 1],
        }
    ).write_parquet(path)
    _patch_meds(
        monkeypatch,
        _med_long([
            # Patient 1: one line, 31 days of follow-up.
            (1, "Drug A", date(2020, 1, 1)),
            # Patient 2: three lines, two years of follow-up.
            (2, "Drug A", date(2020, 1, 1)),
            (2, "Drug B", date(2020, 6, 1)),
            (2, "Drug C", date(2021, 1, 1)),
        ]),
    )

    stats = load_n_lines_followup_stats(cohort_path=str(path))
    by_label = {r["label"]: r for r in stats.to_dicts()}

    assert by_label["1 line"]["median_follow_up_days"] == 31
    assert by_label["3 lines"]["median_follow_up_days"] == 731
    assert by_label["1 line"]["death_fraction"] == 0.0
    assert by_label["3 lines"]["death_fraction"] == 1.0


# --- binary drug-class treatment targets ---------------------------------

def _med_classes(tmp_path, rows):
    """A (MED_NAME, MOA_Category) table shaped like MED_CLASSES_FILE."""
    path = tmp_path / "med_classes.csv"
    pl.DataFrame(
        {"MED_NAME": [r[0] for r in rows], "MOA_Category": [r[1] for r in rows]}
    ).write_csv(path)
    return str(path)


def _cohort(tmp_path, mrns, genders=None):
    path = tmp_path / "cohort_df.parquet"
    frame = {"DFCI_MRN": list(mrns)}
    if genders is not None:
        frame["GENDER"] = list(genders)
    pl.DataFrame(frame).write_parquet(path)
    return str(path)


def test_moa_categories_classify_into_the_six_drug_classes():
    from semantic_search.drug_classes import classify_moa_category

    # Free-form LLM spellings must all resolve; matching is normalized, not literal.
    assert classify_moa_category("Immune Checkpoint Inhibitor", "ici")
    assert classify_moa_category("PD-1 inhibitor", "ici")
    assert classify_moa_category("anti-PD-L1", "ici")
    assert classify_moa_category("Tyrosine Kinase Inhibitor", "tki")
    assert classify_moa_category("Aromatase Inhibitor", "estrogen")
    assert classify_moa_category("Selective Estrogen Receptor Degrader", "estrogen")
    assert classify_moa_category("Androgen Receptor Inhibitor", "androgen_axis")
    assert classify_moa_category("GnRH agonist", "androgen_axis")
    assert classify_moa_category("Antibody-Drug Conjugate", "adc")
    assert classify_moa_category("Monoclonal Antibody", "monoclonal_antibody")

    assert not classify_moa_category("Taxane", "ici")
    assert not classify_moa_category("Platinum Chemotherapy", "tki")
    # A null/unmapped category is negative, not null: one-vs-rest needs a decision.
    assert not classify_moa_category(None, "ici")
    assert not classify_moa_category("", "ici")


def test_adc_is_excluded_from_the_monoclonal_antibody_class():
    """'mAbs vs. all else' reads as naked antibodies; an ADC is its own class."""
    from semantic_search.drug_classes import classify_moa_category

    assert classify_moa_category("Antibody-Drug Conjugate", "adc")
    assert not classify_moa_category("Antibody-Drug Conjugate", "monoclonal_antibody")


def test_unknown_drug_class_is_rejected():
    from semantic_search.drug_classes import classify_moa_category

    with pytest.raises(ValueError, match="Unknown drug class"):
        classify_moa_category("Taxane", "chemotherapy")


def test_drug_class_target_is_ever_exposed_across_all_lines(tmp_path, monkeypatch):
    """Exposure is 'any drug, any line', not the anchor drug."""
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(
        monkeypatch,
        _med_long(
            [
                # Patient 1: ICI only at their SECOND drug -> still positive.
                (1, "Carboplatin", 0),
                (1, "Pembrolizumab", 100),
                # Patient 2: never an ICI -> negative.
                (2, "Carboplatin", 0),
                (2, "Paclitaxel", 100),
            ]
        ),
    )
    classes = _med_classes(
        tmp_path,
        [
            ("Pembrolizumab", "Immune Checkpoint Inhibitor"),
            ("Carboplatin", "Platinum Chemotherapy"),
            ("Paclitaxel", "Taxane"),
        ],
    )

    labels = load_drug_class_target(
        "ici",
        cohort_path=_cohort(tmp_path, [1, 2]),
        med_classes_path=classes,
    )

    assert dict(labels.iter_rows()) == {1: "ICI", 2: "NON_ICI"}


def test_drug_class_target_labels_unmapped_drugs_negative(tmp_path, monkeypatch):
    """A drug absent from the GPT class table is not evidence of exposure."""
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(monkeypatch, _med_long([(1, "Unlisted Drug", 0)]))
    classes = _med_classes(tmp_path, [("Pembrolizumab", "Immune Checkpoint Inhibitor")])

    labels = load_drug_class_target(
        "ici", cohort_path=_cohort(tmp_path, [1]), med_classes_path=classes
    )

    assert dict(labels.iter_rows()) == {1: "NON_ICI"}


def test_estrogen_target_is_restricted_to_female_patients(tmp_path, monkeypatch):
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(
        monkeypatch,
        _med_long([(1, "Anastrozole", 0), (2, "Anastrozole", 0), (3, "Docetaxel", 0)]),
    )
    classes = _med_classes(
        tmp_path,
        [("Anastrozole", "Aromatase Inhibitor"), ("Docetaxel", "Taxane")],
    )
    # GENDER is 0=MALE / 1=FEMALE per build_cohort; patient 4 has a null GENDER.
    cohort = _cohort(tmp_path, [1, 2, 3, 4], genders=[1, 0, 1, None])

    labels = load_drug_class_target(
        "estrogen", cohort_path=cohort, med_classes_path=classes
    )

    # Patient 2 is male and patient 4 has no recorded sex: both are excluded.
    assert dict(labels.iter_rows()) == {1: "ESTROGEN", 3: "NON_ESTROGEN"}


def test_androgen_axis_target_is_restricted_to_male_patients(tmp_path, monkeypatch):
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(
        monkeypatch, _med_long([(1, "Leuprolide", 0), (2, "Leuprolide", 0)])
    )
    classes = _med_classes(tmp_path, [("Leuprolide", "GnRH Agonist")])
    cohort = _cohort(tmp_path, [1, 2], genders=[0, 1])

    labels = load_drug_class_target(
        "androgen_axis", cohort_path=cohort, med_classes_path=classes
    )

    assert dict(labels.iter_rows()) == {1: "ANDROGEN_AXIS"}


def test_unrestricted_drug_class_target_needs_no_gender_column(tmp_path, monkeypatch):
    """ICI/TKI/mAb/ADC are asked of all patients, so GENDER is not required."""
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(monkeypatch, _med_long([(1, "Erlotinib", 0)]))
    classes = _med_classes(tmp_path, [("Erlotinib", "Tyrosine Kinase Inhibitor")])

    labels = load_drug_class_target(
        "tki", cohort_path=_cohort(tmp_path, [1]), med_classes_path=classes
    )

    assert dict(labels.iter_rows()) == {1: "TKI"}


def test_sex_restricted_target_requires_gender_column(tmp_path, monkeypatch):
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(monkeypatch, _med_long([(1, "Anastrozole", 0)]))
    classes = _med_classes(tmp_path, [("Anastrozole", "Aromatase Inhibitor")])

    with pytest.raises(ValueError, match="missing GENDER"):
        load_drug_class_target(
            "estrogen",
            cohort_path=_cohort(tmp_path, [1]),
            med_classes_path=classes,
        )


def test_drug_class_target_ignores_patients_outside_the_cohort(tmp_path, monkeypatch):
    from semantic_search.prediction_targets import load_drug_class_target

    _patch_meds(monkeypatch, _med_long([(1, "Erlotinib", 0), (999, "Erlotinib", 0)]))
    classes = _med_classes(tmp_path, [("Erlotinib", "Tyrosine Kinase Inhibitor")])

    labels = load_drug_class_target(
        "tki", cohort_path=_cohort(tmp_path, [1]), med_classes_path=classes
    )

    assert dict(labels.iter_rows()) == {1: "TKI"}


def test_all_six_drug_class_targets_are_registered():
    from semantic_search.prediction_targets import (
        DRUG_CLASS_TARGETS,
        TARGET_DISPLAY_NAMES,
        TARGETS,
    )

    assert DRUG_CLASS_TARGETS == [
        "treatment_estrogen",
        "treatment_androgen_axis",
        "treatment_ici",
        "treatment_tki",
        "treatment_monoclonal_antibody",
        "treatment_adc",
    ]
    for target in DRUG_CLASS_TARGETS:
        assert target in TARGETS
        assert target in TARGET_DISPLAY_NAMES


def test_drug_class_labels_are_binary_and_disjoint():
    from semantic_search.drug_classes import DRUG_CLASS_LABELS, DRUG_CLASSES

    assert set(DRUG_CLASS_LABELS) == set(DRUG_CLASSES)
    for positive, negative in DRUG_CLASS_LABELS.values():
        assert positive != negative
        assert negative == f"NON_{positive}"


def test_moa_coverage_audit_flags_unmatched_categories():
    """The audit is how the patterns get validated against the cluster-only table."""
    from semantic_search.drug_classes import audit_moa_coverage

    med_classes = pl.DataFrame(
        {
            "MED_NAME": ["A", "B", "C"],
            "MOA_Category": [
                "Immune Checkpoint Inhibitor",
                "Some Unrecognized Class",
                "Another Unrecognized Class",
            ],
        }
    )

    audit = audit_moa_coverage(med_classes)
    by_category = {
        row["MOA_Category"]: row for row in audit.iter_rows(named=True)
    }

    assert by_category["Immune Checkpoint Inhibitor"]["matched_classes"] == "ici"
    assert not by_category["Immune Checkpoint Inhibitor"]["unmatched"]
    assert by_category["Some Unrecognized Class"]["unmatched"]
    assert by_category["Some Unrecognized Class"]["n_matched_classes"] == 0


# --- cancer-type baseline space ------------------------------------------

def test_baseline_space_one_hot_encodes_cancer_type(monkeypatch):
    from semantic_search import clinical_data, train_prediction_models as tpm

    monkeypatch.setattr(
        clinical_data,
        "load_cancer_type",
        lambda: (
            pl.DataFrame(
                {"DFCI_MRN": [3, 1, 2], "CANCER_TYPE": ["Lung", "Breast", "Lung"]}
            ),
            [],
            ["CANCER_TYPE"],
        ),
    )

    features = tpm.load_baseline_features()

    assert features.get_column("DFCI_MRN").to_list() == [1, 2, 3]
    # Deterministic, sorted column order keeps the run signature stable.
    assert tpm.baseline_cols(features) == ["CANCER_TYPE_BREAST", "CANCER_TYPE_LUNG"]
    assert features.get_column("CANCER_TYPE_LUNG").to_list() == [0, 1, 1]
    assert features.get_column("CANCER_TYPE_BREAST").to_list() == [1, 0, 0]


def test_baseline_space_skips_pca_and_passes_features_through():
    from semantic_search import train_prediction_models as tpm
    from semantic_search.common import BASELINE_SPACE

    cols = ["CANCER_TYPE_LUNG", "CANCER_TYPE_BREAST"]
    blocks = tpm._feature_blocks(cols, BASELINE_SPACE)

    # No blocks is the sentinel for "no PCA"; L2-normalizing a one-hot matrix
    # would destroy exactly what the baseline is meant to represent.
    assert blocks == {}
    X = np.array([[1.0, 0.0], [0.0, 1.0]])
    transformer = tpm._make_block_transformer(blocks, 0, X.shape[1])
    assert np.array_equal(transformer.fit_transform(X), X)


def test_embedding_space_still_requires_its_three_blocks():
    """The baseline escape hatch must not weaken validation for embedding spaces."""
    from semantic_search import train_prediction_models as tpm

    with pytest.raises(ValueError, match="Missing embedding blocks"):
        tpm._feature_blocks(["CLINICIAN_EMBEDDING_0"], "concat")


def test_baseline_run_signature_records_passthrough_not_pca():
    from semantic_search import train_prediction_models as tpm

    data = pl.DataFrame({"DFCI_MRN": [1, 2], "label": ["ICI", "NON_ICI"]})
    cols = ["CANCER_TYPE_LUNG", "CANCER_TYPE_BREAST"]
    kwargs = dict(
        model="xgboost", outer_folds=5, inner_folds=3, seed=1, run_context={}
    )

    baseline = tpm._run_signature(data, cols, feature_blocks={}, **kwargs)
    embedding = tpm._run_signature(
        data, cols, feature_blocks={"CLINICIAN": [0, 1]}, **kwargs
    )

    assert baseline["feature_transform"] == "passthrough"
    assert baseline["pca_components"] is None
    assert baseline["n_transformed_features"] == len(cols)
    # A baseline run and an embedding run must never share a signature.
    assert embedding["feature_transform"] == "blockwise_l2_standardize_pca"
    assert baseline != embedding


def test_baseline_space_has_no_feature_artifact_path():
    from semantic_search.common import BASELINE_SPACE, feature_path

    with pytest.raises(ValueError, match="built in memory"):
        feature_path(BASELINE_SPACE, "alltime")


def test_baseline_space_trains_end_to_end(tmp_path, monkeypatch):
    """The blockless path must survive a real nested-CV fit, not just a transform."""
    pytest.importorskip("xgboost")
    from semantic_search import common, train_prediction_models as tpm

    for attr in ("PREDICTIONS_DIR", "MODELS_DIR", "PREDICTION_META_DIR"):
        monkeypatch.setattr(tpm, attr, str(tmp_path / attr.lower()))

    rng = np.random.default_rng(0)
    n = 90
    lung = rng.integers(0, 2, n)
    data = pl.DataFrame(
        {
            "DFCI_MRN": np.arange(n),
            "CANCER_TYPE_LUNG": lung,
            "CANCER_TYPE_BREAST": 1 - lung,
            # Signal, so the fit is a real one: lung patients mostly get the ICI.
            "label": np.where(
                rng.random(n) < np.where(lung == 1, 0.8, 0.2), "ICI", "NON_ICI"
            ),
        }
    )

    meta, folds, per_class = tpm.train_one(
        data,
        ["CANCER_TYPE_LUNG", "CANCER_TYPE_BREAST"],
        target="treatment_ici",
        space=common.BASELINE_SPACE,
        window="alltime",
        model="xgboost",
        outer_folds=3,
        inner_folds=2,
        seed=0,
        n_jobs=1,
        overwrite=True,
    )

    assert meta["run_signature"]["feature_transform"] == "passthrough"
    assert meta["n_transformed_features"] == 2
    assert sorted(meta["classes"]) == ["ICI", "NON_ICI"]
    assert len(folds) == 3
    assert {row["class"] for row in per_class} == {"ICI", "NON_ICI"}


def test_tki_class_excludes_non_tyrosine_kinase_inhibitors():
    """BRAF/MEK/mTOR/CDK are kinase inhibitors but not TYROSINE kinase inhibitors."""
    from semantic_search.drug_classes import classify_moa_category

    for category in (
        "Tyrosine Kinase Inhibitor",
        "VEGFR Tyrosine Kinase Inhibitor",
        "EGFR Inhibitor",
        "ALK Inhibitor",
        "BCR-ABL Inhibitor",
        "Multi-kinase inhibitor",
        "TKI",
    ):
        assert classify_moa_category(category, "tki"), category

    for category in (
        "Serine/Threonine Kinase Inhibitor",
        "BRAF kinase inhibitor",
        "MEK Inhibitor",
        "mTOR Inhibitor",
        "CDK4/6 Inhibitor",
        "PARP Inhibitor",
        "Proteasome Inhibitor",
    ):
        assert not classify_moa_category(category, "tki"), category


def test_estrogen_class_excludes_ambiguous_and_supportive_hormone_therapy():
    """A bare 'Hormone Therapy' does not say which axis; thyroid/GH are not oncologic."""
    from semantic_search.drug_classes import classify_moa_category

    for category in ("Hormone Therapy", "Thyroid Hormone Therapy", "Growth Hormone Therapy"):
        assert not classify_moa_category(category, "estrogen"), category
        assert not classify_moa_category(category, "androgen_axis"), category

    assert classify_moa_category("Endocrine Therapy", "estrogen")
    assert classify_moa_category("Aromatase Inhibitor", "estrogen")


def test_monoclonal_antibody_class_excludes_bispecifics_and_conjugates():
    from semantic_search.drug_classes import classify_moa_category

    assert classify_moa_category("Monoclonal Antibody", "monoclonal_antibody")
    assert classify_moa_category("Anti-CD20 Monoclonal Antibody", "monoclonal_antibody")

    for category in (
        "Bispecific Antibody",
        "Bispecific T-cell Engager",
        "Antibody-Drug Conjugate",
        "Radioimmunotherapy",
    ):
        assert not classify_moa_category(category, "monoclonal_antibody"), category


def test_ici_and_monoclonal_antibody_classes_may_both_fire():
    """The six targets are independent one-vs-rest questions, not a partition.

    Pembrolizumab genuinely is both an ICI and a monoclonal antibody, so a
    category naming both must count for both rather than being forced to pick.
    """
    from semantic_search.drug_classes import classify_moa_category

    assert classify_moa_category("Anti-PD-1 Monoclonal Antibody", "ici")
    assert classify_moa_category("Anti-PD-1 Monoclonal Antibody", "monoclonal_antibody")


def test_chemotherapy_categories_are_negative_for_all_six_classes():
    from semantic_search.drug_classes import DRUG_CLASSES, classify_moa_category

    for category in ("Taxane", "Platinum Chemotherapy", "Antimetabolite", "Alkylating Agent"):
        assert not any(classify_moa_category(category, c) for c in DRUG_CLASSES), category


def test_both_note_windows_run_by_default():
    """Every stage must build the pre-first-treatment arm without an extra flag.

    The windows are only comparable if each is actually produced; a default of
    alltime alone silently reports one arm.
    """
    from semantic_search import common

    assert common.DEFAULT_WINDOWS == ["alltime", "pretreatment"]
    assert set(common.DEFAULT_WINDOWS) == set(common.WINDOWS)


def test_windows_write_to_separate_artifacts():
    """The two windows must coexist on disk, not overwrite one another."""
    from semantic_search import train_prediction_models as tpm
    from semantic_search.common import feature_path

    assert feature_path("concat", "alltime") != feature_path("concat", "pretreatment")

    stems = {
        window: tpm._stem("treatment_ici", "concat", window, "xgboost")
        for window in ("alltime", "pretreatment")
    }
    assert stems["alltime"] != stems["pretreatment"]
    assert "pretreatment" in stems["pretreatment"]


def test_window_is_part_of_the_run_signature():
    """A window change must invalidate cached artifacts, not reuse them."""
    from semantic_search import train_prediction_models as tpm

    data = pl.DataFrame({"DFCI_MRN": [1, 2], "label": ["ICI", "NON_ICI"]})
    cols = ["CLINICIAN_EMBEDDING_0", "CLINICIAN_EMBEDDING_1"]
    kwargs = dict(
        model="xgboost",
        outer_folds=5,
        inner_folds=3,
        seed=1,
        feature_blocks={"CLINICIAN": [0, 1]},
    )

    alltime = tpm._run_signature(data, cols, run_context={"window": "alltime"}, **kwargs)
    pretreatment = tpm._run_signature(
        data, cols, run_context={"window": "pretreatment"}, **kwargs
    )

    assert alltime != pretreatment


def test_pretreatment_window_selects_only_pre_anchor_notes():
    """The pretreatment arm is strictly-before, and unanchored notes drop out."""
    from semantic_search import aggregate_embeddings as agg
    from semantic_search.common import NOTE_TIMING_COL

    notes = pl.DataFrame(
        {
            "DFCI_MRN": [1, 2, 3, 4, 5],
            NOTE_TIMING_COL: [-30.0, -1.0, 0.0, 12.0, None],
        }
    )

    assert agg._select_notes(notes, "alltime").height == 5
    pretreatment = agg._select_notes(notes, "pretreatment")
    # 0.0 is excluded (strictly before), as is the null-anchor patient.
    assert pretreatment.get_column("DFCI_MRN").to_list() == [1, 2]


def test_negative_class_labels_are_identifiable_by_prefix():
    """The figures tier finds the positive class by the absence of a NON_ prefix.

    5_figures.R computes drug-class prevalence with `!grepl("^NON_", class)`,
    so renaming a label such that the negative class loses the prefix -- or the
    positive class gains it -- would silently invert the reported prevalence.
    """
    from semantic_search.drug_classes import DRUG_CLASS_LABELS

    for drug_class, (positive, negative) in DRUG_CLASS_LABELS.items():
        assert not positive.startswith("NON_"), drug_class
        assert negative.startswith("NON_"), drug_class
        assert negative == f"NON_{positive}", drug_class


def test_figures_target_display_names_cover_every_target():
    """Every trained target needs a display name in both tiers.

    display_target() in 5_figures.R mirrors TARGET_DISPLAY_NAMES; a target
    missing from the Python map would also be missing from the figure legend.
    """
    from semantic_search.prediction_targets import TARGETS, TARGET_DISPLAY_NAMES

    assert set(TARGETS) <= set(TARGET_DISPLAY_NAMES)
