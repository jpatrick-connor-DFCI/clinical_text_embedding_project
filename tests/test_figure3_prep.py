import polars as pl

from figures.prep.figure3 import _complete_case_rank_matrix
from shared.palette import MODALITY_ORDER


def _metrics() -> pl.DataFrame:
    return pl.DataFrame([
        {
            "scheme": "death_met",
            "event": event,
            "modality": modality,
            "cindex": 0.9 - index / 100,
        }
        for event in ("event_a", "event_b")
        for index, modality in enumerate(MODALITY_ORDER)
    ])


def test_figure3_ranks_include_somatic() -> None:
    _, modalities, ranks = _complete_case_rank_matrix(_metrics(), "cindex")

    assert modalities == MODALITY_ORDER
    assert "somatic" in modalities
    assert ranks.shape == (2, len(MODALITY_ORDER))


def test_missing_somatic_endpoint_is_removed_not_somatic_modality() -> None:
    metrics = _metrics().filter(
        ~((pl.col("event") == "event_b") & (pl.col("modality") == "somatic"))
    )

    keys, modalities, ranks = _complete_case_rank_matrix(metrics, "cindex")

    assert modalities == MODALITY_ORDER
    assert keys["event"].to_list() == ["event_a"]
    assert ranks.shape == (1, len(MODALITY_ORDER))
