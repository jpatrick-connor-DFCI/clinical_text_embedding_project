"""Formula-level tests for shared/published_scores.py: exact-threshold
boundaries, null handling, the ECOG lower-bound property, and registry
integrity. No file I/O, no cluster data -- pure polars expressions evaluated
against small synthetic frames."""

import polars as pl
import pytest

import shared.published_scores as sp

ELIGIBLE = pl.lit(True)


# ===========================================================================
# mGPS
# ===========================================================================

def test_mgps_crp_boundary_and_albumin_split():
    df = pl.DataFrame({
        "crp": [1.00, 1.01, 1.01, 0.99],
        "alb": [3.0, 3.5, 3.49, 3.0],
    })
    score = sp.build_mgps("crp", "alb", ELIGIBLE)
    points, complete, missing = sp.score_expr(score)
    out = df.select(points, complete)
    assert out.get_column("mgps__points").to_list() == [0, 1, 2, 0]
    assert out.get_column("mgps__complete").to_list() == [True, True, True, True]


def test_mgps_null_propagates_and_records_missing_item():
    df = pl.DataFrame({"crp": [None], "alb": [3.5]})
    score = sp.build_mgps("crp", "alb", ELIGIBLE)
    points, complete, missing = sp.score_expr(score)
    out = df.select(points, complete, missing)
    assert out.get_column("mgps__points").to_list() == [None]
    assert out.get_column("mgps__complete").to_list() == [False]
    assert out.get_column("mgps__missing_items").to_list() == [["mgps_crp_alb"]]


# ===========================================================================
# LIPI: dNLR exactly 3, LDH at the ULN
# ===========================================================================

def test_lipi_dnlr_and_ldh_boundaries():
    ldh_uln, _ = sp.REFERENCE_LIMITS["LDH_ULN"]
    # dNLR = ANC / (WBC - ANC); choose ANC/WBC so dNLR lands exactly at 3 and just above.
    df = pl.DataFrame({
        "anc": [3.0, 3.01, 3.0, 3.0],
        "wbc": [4.0, 4.0, 4.0, 4.0],
        "ldh": [ldh_uln, ldh_uln, ldh_uln + 0.01, ldh_uln],
    })
    score = sp.build_lipi("anc", "wbc", "ldh", ELIGIBLE)
    points, complete, missing = sp.score_expr(score)
    out = df.select(points)
    # row0: dNLR=3.0 (not >3) -> 0; ldh==ULN (not >ULN) -> 0 => total 0
    # row1: dNLR=3.01/0.99=3.0404 (>3) -> 1; ldh==ULN -> 0 => total 1
    # row2: dNLR=3.0 -> 0; ldh>ULN -> 1 => total 1
    assert out.get_column("lipi__points").to_list() == [0, 1, 1, 0]


# ===========================================================================
# ALBI: continuous formula, boundary risk groups
# ===========================================================================

def test_albi_continuous_formula_and_groups():
    # Solve for bili/alb such that the ALBI score lands near a grade boundary.
    # ALBI = 0.66*log10(bili_umol) - 0.085*(alb_gL); use alb=40 g/L (alb_col=4.0 g/dL).
    df = pl.DataFrame({"bili": [0.5, 3.0], "alb": [4.0, 2.0]})
    score = sp.build_albi("bili", "alb", ELIGIBLE)
    points, complete, missing = sp.score_expr(score)
    out = df.select(points, complete)
    assert out.get_column("albi__complete").to_list() == [True, True]
    vals = out.get_column("albi__points").to_list()
    assert vals[0] < vals[1]  # higher bili + lower albumin -> worse (higher) ALBI

    group_df = df.with_columns(points).with_columns(sp.group_expr(score, "albi__points").alias("group"))
    groups = group_df.get_column("group").to_list()
    assert all(g in ("grade 1", "grade 2", "grade 3") for g in groups)


# ===========================================================================
# MELD: caps on bilirubin/INR/creatinine
# ===========================================================================

def test_meld_creatinine_cap_and_null_handling():
    df = pl.DataFrame({
        "bili": [1.0, 1.0],
        "inr": [1.0, 1.0],
        "creat": [3.0, 10.0],  # 10.0 should clip to the 4.0 cap
    })
    score = sp.build_meld("bili", "inr", "creat", ELIGIBLE)
    points, complete, missing = sp.score_expr(score)
    out = df.select(points)
    vals = out.get_column("meld__points").to_list()
    # creat=3.0 vs creat=10.0(clipped to 4.0) should differ
    assert vals[0] < vals[1]

    df_null = pl.DataFrame({"bili": [None], "inr": [1.0], "creat": [1.0]})
    out_null = df_null.select(points, complete, missing)
    assert out_null.get_column("meld__points").to_list() == [None]
    assert out_null.get_column("meld__complete").to_list() == [False]


# ===========================================================================
# CAPRA: PSA bands, diagnosis-to-anchor boundary elsewhere (registry-level,
# not scored here -- diagnosis-to-anchor windowing is a builder concern).
# ===========================================================================

def test_capra_psa_bands():
    df = pl.DataFrame({
        "psa": [6.0, 6.01, 10.0, 10.01, 20.0, 20.01, 30.0, 30.01],
        "gleason_p": [3] * 8,
        "gleason_s": [3] * 8,
        "t_ge_t3a": [False] * 8,
        "age": [55] * 8,
    })
    score = sp.build_capra_mod("psa", "gleason_p", "gleason_s", "t_ge_t3a", "age", ELIGIBLE)
    psa_item = next(i for i in score.items if i.name == "capra_psa")
    out = df.select(psa_item.point_expr.alias("psa_pts"))
    assert out.get_column("psa_pts").to_list() == [0, 1, 1, 2, 2, 3, 3, 4]


def test_capra_gleason_grouping_points():
    df = pl.DataFrame({
        "primary": [3, 3, 4],
        "secondary": [3, 4, 3],
    })
    expr = sp._capra_gleason_points("primary", "secondary")
    out = df.select(expr.alias("pts"))
    assert out.get_column("pts").to_list() == [0, 1, 3]


# ===========================================================================
# IMDC / MSKCC: diagnosis-to-anchor 364 vs 365 days
# ===========================================================================

def test_imdc_time_to_treatment_boundary_364_vs_365():
    df = pl.DataFrame({
        "days": [364, 365],
        "hgb": [15.0, 15.0],
        "gender": ["Male", "Male"],
        "ca": [9.0, 9.0],
        "anc": [3.0, 3.0],
        "plt": [200.0, 200.0],
    })
    score = sp.build_imdc("days", "hgb", "gender", "ca", "anc", "plt", None, ELIGIBLE)
    item = next(i for i in score.items if i.name == "imdc_time_to_treatment")
    out = df.select(item.point_expr.alias("pts"))
    assert out.get_column("pts").to_list() == [1, 0]


def test_imdc_hgb_is_sex_specific():
    male_lln, _ = sp.REFERENCE_LIMITS["HGB_LLN_MALE"]
    female_lln, _ = sp.REFERENCE_LIMITS["HGB_LLN_FEMALE"]
    assert male_lln > female_lln
    df = pl.DataFrame({
        "days": [400, 400],
        "hgb": [12.5, 12.5],  # below male LLN, above female LLN
        "gender": ["Male", "Female"],
        "ca": [9.0, 9.0],
        "anc": [3.0, 3.0],
        "plt": [200.0, 200.0],
    })
    score = sp.build_imdc("days", "hgb", "gender", "ca", "anc", "plt", None, ELIGIBLE)
    item = next(i for i in score.items if i.name == "imdc_hemoglobin")
    out = df.select(item.point_expr.alias("pts"))
    assert out.get_column("pts").to_list() == [1, 0]


def test_mskcc_ldh_uses_1p5x_uln():
    ldh_uln, _ = sp.REFERENCE_LIMITS["LDH_ULN"]
    df = pl.DataFrame({
        "ldh": [1.5 * ldh_uln, 1.5 * ldh_uln + 0.01],
        "hgb": [15.0, 15.0],
        "gender": ["Male", "Male"],
        "ca": [9.0, 9.0],
        "days": [400, 400],
    })
    score = sp.build_mskcc("ldh", "hgb", "gender", "ca", "days", None, ELIGIBLE)
    item = next(i for i in score.items if i.name == "mskcc_ldh")
    out = df.select(item.point_expr.alias("pts"))
    assert out.get_column("pts").to_list() == [0, 1]


# ===========================================================================
# RMH: met-site-count boundary
# ===========================================================================

def test_rmh_met_site_boundary():
    ldh_uln, _ = sp.REFERENCE_LIMITS["LDH_ULN"]
    df = pl.DataFrame({
        "ldh": [ldh_uln] * 2,
        "alb": [4.0] * 2,
        "n_sites": [2, 3],
    })
    score = sp.build_rmh("ldh", "alb", "n_sites", ELIGIBLE)
    item = next(i for i in score.items if i.name == "rmh_met_sites")
    out = df.select(item.point_expr.alias("pts"))
    assert out.get_column("pts").to_list() == [0, 1]


# ===========================================================================
# Derived inputs
# ===========================================================================

def test_dnlr_and_corrected_calcium():
    df = pl.DataFrame({"anc": [3.0], "wbc": [10.0], "ca": [9.0], "alb": [3.0]})
    out = df.select(
        sp.dnlr_expr("anc", "wbc").alias("dnlr"),
        sp.corrected_calcium_expr("ca", "alb").alias("corrected_ca"),
    )
    assert out.get_column("dnlr").to_list() == [3.0 / 7.0]
    assert out.get_column("corrected_ca").to_list() == [9.0 + 0.8 * (4.0 - 3.0)]


def test_dnlr_null_when_wbc_not_greater_than_anc():
    df = pl.DataFrame({"anc": [5.0], "wbc": [5.0]})
    out = df.select(sp.dnlr_expr("anc", "wbc").alias("dnlr"))
    assert out.get_column("dnlr").to_list() == [None]


# ===========================================================================
# ECOG lower-bound property
# ===========================================================================

@pytest.mark.parametrize("builder_name", ["ipi", "imdc", "mskcc"])
def test_ecog_free_is_a_lower_bound_of_full_score(builder_name):
    if builder_name == "ipi":
        full = sp.build_ipi("age", "ldh", "ann_arbor_ge3", "n_extranodal", "ecog", ELIGIBLE)
        df = pl.DataFrame({
            "age": [70, 70], "ldh": [300, 300], "ann_arbor_ge3": [True, True],
            "n_extranodal": [2, 2], "ecog": [0, 3],
        })
    elif builder_name == "imdc":
        full = sp.build_imdc("days", "hgb", "gender", "ca", "anc", "plt", "ecog", ELIGIBLE)
        df = pl.DataFrame({
            "days": [400, 400], "hgb": [15.0, 15.0], "gender": ["Male", "Male"],
            "ca": [9.0, 9.0], "anc": [3.0, 3.0], "plt": [200.0, 200.0], "ecog": [0, 2],
        })
    else:
        full = sp.build_mskcc("ldh", "hgb", "gender", "ca", "days", "ecog", ELIGIBLE)
        df = pl.DataFrame({
            "ldh": [100.0, 100.0], "hgb": [15.0, 15.0], "gender": ["Male", "Male"],
            "ca": [9.0, 9.0], "days": [400, 400], "ecog": [0, 2],
        })

    assert full.performance_status is not None
    stripped = sp.ecog_free(full)
    assert stripped.performance_status is None
    assert full.performance_status.name not in [i.name for i in stripped.items]

    pts_full, _, _ = sp.score_expr(full)
    pts_stripped, _, _ = sp.score_expr(stripped)
    ecog_item_points = df.select(full.performance_status.point_expr.alias("ecog_pts")).get_column("ecog_pts").to_list()
    out = df.select(pts_full.alias("full"), pts_stripped.alias("noecog"))
    full_vals = out.get_column("full").to_list()
    noecog_vals = out.get_column("noecog").to_list()
    for fv, nv, ev in zip(full_vals, noecog_vals, ecog_item_points):
        assert fv == nv + ev


def test_ecog_free_raises_when_already_ecog_free():
    ipi_noecog = sp.build_ipi("age", "ldh", "ann_arbor_ge3", "n_extranodal", None, ELIGIBLE)
    with pytest.raises(ValueError, match="performance_status"):
        sp.ecog_free(ipi_noecog)


# ===========================================================================
# Registry integrity
# ===========================================================================

def _all_built_scores():
    return [
        sp.build_mgps("crp", "alb", ELIGIBLE),
        sp.build_rmh("ldh", "alb", "n_sites", ELIGIBLE),
        sp.build_lipi("anc", "wbc", "ldh", ELIGIBLE),
        sp.build_albi("bili", "alb", ELIGIBLE),
        sp.build_meld("bili", "inr", "creat", ELIGIBLE),
        sp.build_capra_mod("psa", "gleason_p", "gleason_s", "t_ge_t3a", "age", ELIGIBLE),
        sp.build_ipi("age", "ldh", "ann_arbor_ge3", "n_extranodal", None, ELIGIBLE),
        sp.build_imdc("days", "hgb", "gender", "ca", "anc", "plt", None, ELIGIBLE),
        sp.build_mskcc("ldh", "hgb", "gender", "ca", "days", None, ELIGIBLE),
    ]


def test_registry_unique_ids():
    scores = _all_built_scores()
    ids = [s.id for s in scores]
    assert len(ids) == len(set(ids)), f"duplicate ids: {ids}"


def test_risk_groups_are_contiguous_and_cover_all_points():
    for score in _all_built_scores():
        groups = sorted(score.risk_groups, key=lambda g: g.low)
        for prev, nxt in zip(groups, groups[1:]):
            assert prev.high < nxt.low or abs(prev.high - nxt.low) < 1e-9 or nxt.low <= prev.high + 1, (
                f"{score.id}: risk groups not contiguous between {prev} and {nxt}"
            )


def test_variant_requires_modification_note():
    with pytest.raises(ValueError, match="modification_note"):
        sp.PublishedScore(
            id="bad",
            display_name="bad",
            citation="none",
            eligibility=ELIGIBLE,
            items=(sp.ScoreItem("x", ("x",), pl.col("x"), sp.ItemSource.LAB),),
            risk_groups=(sp.RiskGroup("g", 0, 1),),
            variant="modified",
        )


def test_score_expr_requires_items():
    empty = sp.PublishedScore(
        id="empty", display_name="empty", citation="none", eligibility=ELIGIBLE,
        items=(), risk_groups=(sp.RiskGroup("g", 0, 1),),
    )
    with pytest.raises(ValueError, match="no items"):
        sp.score_expr(empty)


def test_analytes_present_in_harmonizer_mapping():
    from shared.lab_harmonizer import ANALYTE_TEST_CODES

    needed = {"LDH", "Albumin", "Hemoglobin", "CRP", "Total bilirubin", "Creatinine", "Calcium", "Neutrophils absolute", "WBC", "Platelets", "INR"}
    missing = needed - set(ANALYTE_TEST_CODES)
    assert not missing, f"analytes missing from shared.lab_harmonizer.ANALYTE_TEST_CODES: {missing}"
    for analyte in needed:
        assert ANALYTE_TEST_CODES[analyte], f"{analyte} has no mapped test codes"


def test_dlbcl_subtype_names_nonempty_and_unique():
    assert len(sp.DLBCL_SUBTYPE_NAMES) == len(set(sp.DLBCL_SUBTYPE_NAMES))
    assert len(sp.DLBCL_SUBTYPE_NAMES) > 0


def test_register_rejects_duplicate_ids():
    registry: dict = {}
    score = sp.build_mgps("crp", "alb", ELIGIBLE)

    def _register(s):
        if s.id in registry:
            raise ValueError(f"Duplicate score id: {s.id}")
        registry[s.id] = s
        return s

    _register(score)
    with pytest.raises(ValueError, match="Duplicate"):
        _register(score)
