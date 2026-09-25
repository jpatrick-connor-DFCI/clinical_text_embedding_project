"""Self-contained lab-name/unit harmonizer for the ~10 analytes the published
prognostic scores need (LDH, albumin, CRP, total bilirubin, creatinine,
calcium, ANC, WBC, platelets, INR). Pure polars, no external repo or
sibling-checkout dependency.

Matches raw LABS.parquet rows to an analyte by `TEST_TYPE_CD` (exact,
case-insensitive), then converts `NUMERIC_RESULT` to the analyte's canonical
unit using `RESULT_UOM_NM`. A row whose unit isn't in `UNIT_CONVERSION` for
its analyte is dropped (unsupported unit), not zero-filled.

Canonical units match `shared/published_scores.py`'s expected inputs:
LDH in U/L; albumin and hemoglobin in g/dL; CRP, bilirubin, creatinine and
calcium in mg/dL; ANC, WBC and platelets in 10^3/uL; INR is unitless (ratio).

`TEST_TYPE_CD` codes below are the common DFCI/PROFILE codes for each
analyte; extend `ANALYTE_TEST_CODES` if the audit finds a cohort-specific
code not covered here.
"""

import polars as pl

# analyte name -> canonical unit (for documentation/validation only)
CANONICAL_UNIT = {
    "LDH": "u/l",
    "Albumin": "g/dl",
    "CRP": "mg/dl",
    "Total bilirubin": "mg/dl",
    "Creatinine": "mg/dl",
    "Calcium": "mg/dl",
    "Neutrophils absolute": "10^3/ul",
    "WBC": "10^3/ul",
    "Platelets": "10^3/ul",
    "INR": "ratio",
    "Hemoglobin": "g/dl",
}

# analyte name -> TEST_TYPE_CD codes observed for it
ANALYTE_TEST_CODES: dict[str, list[str]] = {
    "LDH": ["LDH"],
    "Albumin": ["ALB"],
    "CRP": ["CRP", "CRPT", "CRPRTN", "HSCRP"],
    "Total bilirubin": ["TBILI"],
    "Creatinine": ["CRE"],
    "Calcium": ["CA"],
    "Neutrophils absolute": ["ANEU", "ANEUT", "ANCAB", "ANEUTS", "DMANUT"],
    "WBC": ["WBC"],
    "Platelets": ["PLT"],
    "INR": ["PTINR", "PTI"],
    "Hemoglobin": ["HGB"],
}

# (analyte, normalized unit) -> multiplicative factor to the canonical unit.
# Normalization: lowercase, strip whitespace/brackets/parens, mu -> u.
_COUNT_1E3_PER_UL = {"k/ul": 1.0, "10*3/ul": 1.0, "10^3/ul": 1.0, "x103/ul": 1.0,
                      "k/mm3": 1.0, "th/cmm": 1.0, "thou/ul": 1.0}
_MASS_MG_DL = {"mg/dl": 1.0}
_MASS_G_DL = {"g/dl": 1.0, "gm/dl": 1.0}
_ACTIVITY_U_L = {"u/l": 1.0, "iu/l": 1.0}
_RATIO = {"ratio": 1.0, "": 1.0}

UNIT_CONVERSION: dict[tuple[str, str], float] = {}
for _analyte, _units in {
    "LDH": _ACTIVITY_U_L,
    "Albumin": _MASS_G_DL,
    "Hemoglobin": _MASS_G_DL,
    "CRP": _MASS_MG_DL,
    "Total bilirubin": _MASS_MG_DL,
    "Creatinine": _MASS_MG_DL,
    "Calcium": _MASS_MG_DL,
    "Neutrophils absolute": _COUNT_1E3_PER_UL,
    "WBC": _COUNT_1E3_PER_UL,
    "Platelets": _COUNT_1E3_PER_UL,
    "INR": _RATIO,
}.items():
    for _unit, _factor in _units.items():
        UNIT_CONVERSION[(_analyte, _unit)] = _factor


def _normalize_unit_expr(col: str) -> pl.Expr:
    return (
        pl.col(col).fill_null("").cast(pl.Utf8)
        .str.replace_all("μ", "u").str.replace_all("µ", "u")
        .str.replace_all(r"[\[\]() ]", "")
        .str.to_lowercase()
    )


def harmonize_labs(
    raw: pl.DataFrame,
    *,
    test_cd_col: str,
    result_col: str,
    uom_col: str,
) -> pl.DataFrame:
    """Map `raw` LABS rows to analyte + canonical-unit value.

    Returns a frame with the original columns plus `_analyte` (the
    `collapsed_measurement`-style name from `ANALYTE_TEST_CODES`) and
    `_harmonized_value` (Float64, in the analyte's canonical unit). Rows
    whose test code doesn't match a known analyte, whose result isn't
    numeric, or whose unit isn't in `UNIT_CONVERSION` for that analyte are
    dropped.
    """
    code_to_analyte = {
        code: analyte for analyte, codes in ANALYTE_TEST_CODES.items() for code in codes
    }
    mapping = pl.DataFrame(
        {"_test_cd_upper": list(code_to_analyte.keys()), "_analyte": list(code_to_analyte.values())}
    )

    conv_rows = [
        {"_analyte": analyte, "_unit_norm": unit, "_factor": factor}
        for (analyte, unit), factor in UNIT_CONVERSION.items()
    ]
    conv = pl.DataFrame(conv_rows)

    out = raw.with_columns(
        pl.col(test_cd_col).fill_null("").cast(pl.Utf8).str.to_uppercase().alias("_test_cd_upper"),
        pl.col(result_col).cast(pl.Float64, strict=False).alias("_numeric_value"),
        _normalize_unit_expr(uom_col).alias("_unit_norm"),
    ).join(mapping, on="_test_cd_upper", how="inner")

    out = out.join(conv, on=["_analyte", "_unit_norm"], how="inner")

    return out.filter(pl.col("_numeric_value").is_not_null()).with_columns(
        (pl.col("_numeric_value") * pl.col("_factor")).alias("_harmonized_value")
    ).drop("_test_cd_upper", "_unit_norm", "_factor", "_numeric_value")
