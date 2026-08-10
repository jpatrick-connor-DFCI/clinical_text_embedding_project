"""ICD-10-CM code normalization and metastatic-site grouping.

`normalize_icd10_undotted`/`to_icd10_level_3`/`to_icd10_level_4` were promoted
from `pipelines/preprocessing/generate_embedding_prediction_datasets.py`
(originally `_normalize_icd10_undotted` etc.) so the non-text covariate
builder can use them without importing that heavier, later-stage module.
Made pandas-free (`pd.isna` -> `is None`/self-inequality) so this module has
no pandas dependency; behavior for None, float('nan'), and pd.NA is
unchanged since all three fail equality with themselves except None, which
is handled explicitly.

`MET_SITE_GROUPS`/`met_site_group` classify the C77-C79 "secondary
malignant neoplasm" block into anatomic organ groups, for use as a
metastatic-burden covariate (see build_met_burden_df in
generate_all_non_text_covariates.py). Deliberately restricted to C77-C79
(the secondary/metastatic block) - C7B (secondary neuroendocrine tumors)
and C80 (disseminated/unspecified primary) are out of scope.
"""

import re
from typing import Optional


def normalize_icd10_undotted(code) -> Optional[str]:
    if code is None or code != code:  # noqa: PLR0124 (NaN check without pandas)
        return None
    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9.]", "", code)
    code = code.replace(".", "")
    return code if code else None


def to_icd10_level_3(code) -> Optional[str]:
    code = normalize_icd10_undotted(code)
    if code is None or len(code) < 3:
        return None
    return code[:3]


def to_icd10_level_4(code) -> Optional[str]:
    code = normalize_icd10_undotted(code)
    if code is None or len(code) < 3:
        return None
    if len(code) == 3:
        return code
    return f"{code[:3]}.{code[3]}"


# Same seven anatomic names as generate_embedding_prediction_datasets.MET_SITES,
# plus "other" for C77-C79 codes that don't map onto one of those seven sites.
# Keeping the names identical lets a consumer join site -> outcome event via
# f"{site}M" (e.g. "brain" -> "brainM").
MET_SITE_GROUPS = ["brain", "bone", "liver", "lung", "node", "adrenal", "peritoneal", "other"]

# "Secondary malignant neoplasm of unspecified site" carries no anatomic
# information - its presence proxies documentation intensity, not a distinct
# metastatic site, so it is dropped rather than folded into "other". Flip
# this set (empty it) for a sensitivity variant that keeps it as "other".
_DROP_UNSPECIFIED_SECONDARY = {"C799"}

# Ordered (undotted prefix, group) pairs, longest prefix first so 4-char
# rules (e.g. C7931 brain) beat the 3-char C79 fallback. C77's eight
# subcodes are all nodal by definition, so a 3-char rule is exhaustive there.
_MET_SITE_PREFIX_MAP: list[tuple[str, str]] = [
    # bone: secondary neoplasm of bone (C79.51) / bone marrow (C79.52)
    ("C7951", "bone"),
    ("C7952", "bone"),
    ("C795", "bone"),
    # brain: brain parenchyma (C79.31) / cerebral meninges (C79.32).
    # C79.40/C79.49 (unspecified/other nervous system - spinal cord, cranial
    # nerves) are deliberately left out of "brain" and fall through to
    # "other": they are CNS but not intracranial-parenchymal, and keeping
    # them separate avoids inflating concordance with the brainM outcome.
    ("C7931", "brain"),
    ("C7932", "brain"),
    # adrenal
    ("C7970", "adrenal"),
    ("C7971", "adrenal"),
    ("C7972", "adrenal"),
    ("C797", "adrenal"),
    # lung: parenchyma (C78.0x) and pleura (C78.2, judgement call - grouped
    # with lung rather than "other" since thoracic/pleural spread is
    # conventionally scored with lung in met-burden indices)
    ("C7800", "lung"),
    ("C7801", "lung"),
    ("C7802", "lung"),
    ("C780", "lung"),
    ("C782", "lung"),
    # liver (includes intrahepatic bile duct, per ICD-10-CM)
    ("C787", "liver"),
    # peritoneal: ICD-10-CM merges retroperitoneum and peritoneum into one
    # code (C78.6); "peritoneal" here really means retroperitoneal/peritoneal.
    ("C786", "peritoneal"),
    # node: secondary neoplasm of lymph nodes, all regions (C77.0-C77.9)
    ("C77", "node"),
]


def met_site_group(code) -> Optional[str]:
    """Map an ICD-10-CM code to a metastatic organ group in MET_SITE_GROUPS,
    or None if the code is not a qualifying C77-C79 secondary-neoplasm code.

    Returns None for: codes outside C77-C79; bare header codes with no 4th
    character (e.g. "C78", "C79"); and codes in _DROP_UNSPECIFIED_SECONDARY
    (unspecified-site secondaries). All other C77-C79 codes not matched by
    a specific prefix map to "other" (e.g. GI/GU/skin/ovary secondaries,
    C79.40/C79.49 nervous system, C78.30/C78.39 unspecified respiratory).
    """
    normalized = normalize_icd10_undotted(code)
    if normalized is None or len(normalized) < 3:
        return None
    category = normalized[:3]
    if category not in ("C77", "C78", "C79"):
        return None
    if len(normalized) < 4:
        return None
    if normalized in _DROP_UNSPECIFIED_SECONDARY:
        return None

    for prefix, group in _MET_SITE_PREFIX_MAP:
        if normalized.startswith(prefix):
            return group
    return "other"
