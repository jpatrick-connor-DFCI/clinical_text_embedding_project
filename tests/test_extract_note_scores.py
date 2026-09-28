"""Regex extraction of clinician-documented published scores from notes."""

import datetime as dt

import polars as pl
import pytest

from pipelines.preprocessing import extract_note_scores as ens

# (text, mention, measure, expected first value); None = must not match.
PATTERN_CASES = [
    ("IPI score of 3, high-intermediate", "ipi", "points", "3"),
    ("R-IPI: 2", "ipi", "points", "2"),
    ("NCCN-IPI 5/8", "ipi", "points", "5"),
    ("started ipi 1 mg/kg + nivo 3 mg/kg", "ipi", "points", None),
    ("MELD-Na 18 today", "meld", "continuous", "18"),
    ("MELD 3.0 score of 22", "meld", "continuous", "22"),
    ("MELD 3", "meld", "continuous", None),
    ("ALBI grade 2", "albi", "grade", "2"),
    ("ALBI score -2.41", "albi", "continuous", "-2.41"),
    ("CAPRA score 4 (intermediate)", "capra", "points", "4"),
    ("mGPS 1", "mgps", "points", "1"),
    ("modified Glasgow prognostic score of 2", "mgps", "points", "2"),
    ("LIPI intermediate", "lipi", "group", "intermediate"),
    ("Royal Marsden score 2", "rmh", "points", "2"),
    ("IMDC intermediate risk", "imdc", "group", "intermediate"),
    ("poor risk per IMDC criteria", "imdc", "group", "poor"),
    ("IMDC score 2", "imdc", "points", "2"),
    ("Heng criteria: 3", "imdc", "points", "3"),
    ("discussed with Dr. Heng 2 weeks ago", "imdc", "points", None),
    ("MSKCC favourable risk", "mskcc", "group", "favorable"),
    ("seen at MSKCC last year for a second opinion", "mskcc", "group", None),
    ("ECOG PS 1", "ecog", "points", "1"),
    ("ECOG 1-2", "ecog", "points", "1"),
    ("performance status (ECOG): 0", "ecog", "points", "0"),
    ("enrolled on ECOG-ACRIN E1609", "ecog", "points", None),
    ("KPS 80%", "kps", "points", "80"),
]


def _parse(texts):
    notes = pl.DataFrame({"RPT_ID": [str(i) for i in range(len(texts))], "RPT_TEXT": texts})
    wide = notes.select("RPT_ID", *ens.match_columns(pl.col("RPT_TEXT"), snippets=True))
    return ens.parse_matches(wide, ["RPT_ID"])


@pytest.mark.parametrize(("text", "mention", "measure", "expected"), PATTERN_CASES)
def test_pattern(text, mention, measure, expected):
    got = _parse([text]).filter((pl.col("mention") == mention) & (pl.col("measure") == measure))["value"].to_list()
    if expected is None:
        assert got == []
    else:
        assert got[:1] == [expected]


def test_ecog_range_keeps_upper_end_and_snippet():
    row = _parse(["Pt is ECOG 1-2 today."]).filter(pl.col("mention") == "ecog").row(0, named=True)
    assert (row["value"], row["value_hi"]) == ("1", "2")
    assert "ECOG 1-2" in row["snippet"]


def test_no_matches_returns_empty_frame_with_columns():
    out = _parse(["nothing relevant here"])
    assert out.is_empty() and {"mention", "value", "value_num"} <= set(out.columns)


def _mentions(rows):
    """rows: (mrn, days_before_anchor, text)."""
    notes = pl.DataFrame({
        "DFCI_MRN": [r[0] for r in rows], "RPT_ID": [str(i) for i in range(len(rows))],
        "days_before_anchor": [r[1] for r in rows], "RPT_TEXT": [r[2] for r in rows],
    })
    wide = notes.select("DFCI_MRN", "RPT_ID", "days_before_anchor", *ens.match_columns(pl.col("RPT_TEXT")))
    return ens.parse_matches(wide, ["DFCI_MRN", "RPT_ID", "days_before_anchor"])


def test_note_score_frame_windows_latest_and_ties():
    mentions = _mentions([
        (1, 100, "IPI 5"),        # outside the 90-day window
        (1, 40, "IPI 1"),
        (1, 10, "IPI 2"),         # latest in window ...
        (1, 10, "R-IPI 3"),       # ... tie on the same day -> highest
        (1, -5, "IPI 4"),         # after anchor
        (2, 0, "NCCN-IPI 7"),     # NCCN-IPI dropped
    ])
    cohort = pl.DataFrame({"DFCI_MRN": [1, 2, 3]})
    frame = ens.note_score_frame(mentions, cohort, lookback_days=90)
    assert frame["DFCI_MRN"].to_list() == [1, 2, 3]
    assert frame["ipi__note_value"].to_list() == [3.0, None, None]
    assert frame["ipi__note_group"].to_list() == ["3 (high-intermediate)", None, None]
    assert frame["ipi__note_days_before_anchor"].to_list() == [10, None, None]
    assert frame["ipi__note_n_mentions"].to_list() == [3, None, None]
    assert frame["note_lookback_days"].unique().to_list() == [90]
    assert {f"{s}__note_value" for s in ens.NOTE_SCORES} <= set(frame.columns)


@pytest.mark.parametrize(("text", "score_id", "value", "group"), [
    ("IMDC poor risk", "imdc", 2.0, "poor"),
    ("IMDC score 2", "imdc", 1.0, "intermediate"),
    ("MSKCC risk factors: 0", "mskcc", 0.0, "favorable"),
    ("ALBI -1.2", "albi", 3.0, "grade 3"),
    ("ALBI grade 1", "albi", 1.0, "grade 1"),
    ("LIPI poor", "lipi", 2.0, "2 (poor)"),
    ("MELD-Na 22", "meld", 22.0, ">=20"),
    ("CAPRA 7", "capra_mod", 7.0, "high (6-10)"),
    ("CAPRA-S 7", "capra_mod", None, None),
    ("mGPS 2", "mgps", 2.0, "2"),
    ("GPS 2", "mgps", None, None),
])
def test_note_score_mapping(text, score_id, value, group):
    frame = ens.note_score_frame(_mentions([(1, 0, text)]), pl.DataFrame({"DFCI_MRN": [1]}))
    assert frame[f"{score_id}__note_value"].to_list() == [value]
    assert frame[f"{score_id}__note_group"].to_list() == [group]


def test_performance_status_ecog_range_kps_and_latest_day():
    mentions = _mentions([
        (1, 5, "ECOG 1-2"),        # range -> upper end
        (1, 30, "ECOG 0"),         # older
        (2, 3, "KPS 70%"),         # KPS 70 -> ECOG 2
        (3, 3, "KPS 80"),          # KPS 80 -> ECOG 1
        (4, 2, "ECOG 1"),          # same-day tie -> highest
        (4, 2, "KPS 60"),
        (5, 2, "ECOG 2"),          # same-day tie at equal value -> ECOG source
        (5, 2, "KPS 70"),
        (6, 120, "ECOG 3"),        # outside the lookback
    ])
    frame = ens.note_score_frame(mentions, pl.DataFrame({"DFCI_MRN": [1, 2, 3, 4, 5, 6]}), lookback_days=90)
    assert frame["ecog__note_value"].to_list() == [2.0, 2.0, 1.0, 2.0, 2.0, None]
    assert frame["ecog__note_source"].to_list() == ["ECOG", "KPS", "KPS", "KPS", "ECOG", None]
    assert frame["ecog__note_days_before_anchor"].to_list() == [5, 3, 3, 2, 2, None]
    assert frame["ecog__note_n_mentions"].to_list() == [2, 1, 1, 2, 2, None]
    # ECOG and KPS alone: each from its own mentions, KPS unconverted.
    assert frame["ecog_only__note_value"].to_list() == [2.0, None, None, 1.0, 2.0, None]
    assert frame["ecog_only__note_group"].to_list() == ["2", None, None, "1", "2", None]
    assert frame["ecog_only__note_n_mentions"].to_list() == [2, None, None, 1, 1, None]
    assert frame["kps_only__note_value"].to_list() == [None, 70.0, 80.0, 60.0, 70.0, None]
    assert frame["kps_only__note_group"].to_list() == [None, "60-70", "80-100", "60-70", "60-70", None]


@pytest.mark.parametrize("string_dates", [False, True])
def test_scan_note_mentions_end_to_end(tmp_path, string_dates):
    dates = [dt.datetime(2020, 5, 1, 15, tzinfo=dt.timezone.utc), dt.datetime(2020, 5, 20, tzinfo=dt.timezone.utc),
             dt.datetime(2020, 5, 20, tzinfo=dt.timezone.utc)]
    notes = pl.DataFrame({
        "DFCI_MRN": ["1", "2", "99"], "RPT_ID": [10, 11, 12], "EVENT_DATE": dates,
        "RPT_TEXT": ["IMDC intermediate risk. ECOG PS 1", "no scores documented", "IPI 3"],
    })
    if string_dates:
        notes = notes.with_columns(pl.col("EVENT_DATE").dt.to_string("%Y-%m-%dT%H:%M:%S%z"))
    notes.write_parquet(tmp_path / "PROGRESS_NOTES.parquet")
    cohort = pl.DataFrame({"DFCI_MRN": [1, 2], "_anchor_date": [dt.date(2020, 6, 1)] * 2})

    mentions = ens.scan_note_mentions(cohort, notes_path=str(tmp_path))
    assert set(mentions["DFCI_MRN"]) == {1}  # MRN 99 is not in the cohort
    assert set(mentions["mention"]) == {"imdc", "ecog"}
    assert mentions["days_before_anchor"].unique().to_list() == [31]
    frame = ens.note_score_frame(mentions, cohort)
    assert frame["imdc__note_value"].to_list() == [1.0, None]
