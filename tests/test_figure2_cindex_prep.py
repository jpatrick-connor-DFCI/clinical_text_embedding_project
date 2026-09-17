import polars as pl
import pytest

from figures.prep import figure2


@pytest.mark.parametrize("with_auc", [False, True])
def test_within_vs_pan_keeps_valid_cindex_without_auc(tmp_path, monkeypatch, with_auc):
    """AUC availability cannot exclude a valid C-index comparison."""
    directory = tmp_path / "pan_vs_within_cancer"
    directory.mkdir()
    data = {
        "CANCER_TYPE": ["Overall", "Lung", "Small", "Invalid"],
        "CINDEX_PAN": [0.6, 0.65, 0.7, float("nan")],
        "CINDEX_WITHIN": [0.7, 0.75, 0.8, 0.9],
        "N_HELDOUT": [20, 100, 29, 100],
    }
    if with_auc:
        data.update(AUC_PAN=[float("nan")] * 4, AUC_WITHIN=[float("nan")] * 4)
    pl.DataFrame(data).write_csv(directory / "metrics_by_cancer_type.csv")
    monkeypatch.setattr(figure2, "RESULTS_PATH", str(tmp_path))

    result = figure2._within_vs_pan("cancer")

    assert result["stratum"].to_list() == ["Overall", "Lung"]
    assert result["cindex_delta"].to_list() == pytest.approx([0.1, 0.1])
    assert result["is_overall"].to_list() == [True, False]
