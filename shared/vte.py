"""Venous thromboembolism (VTE) ICD-10-CM and phecode definitions.

Pure code lists and classifiers, in the style of `shared/icd10.py` - no file
reading here. `pipelines/preprocessing/generate_embedding_prediction_datasets.py`
builds the `vte` scheme's endpoints from these.

The ICD-10-CM lists are Ang Li et al.'s VTE definition, ported verbatim from
Intae's `PROFILE_data_processing/intae_files/vte_data_process.py::
get_vte_icd_info`:
  - acute: acute PE, acute lower- and upper-extremity DVT -> the `vte` event;
  - history: chronic VTE (I82.5x/I82.7x/...) and personal history of VTE
    (Z86.71x) -> only used to exclude patients with prior VTE;
  - excluded: non-specific / splanchnic / renal / superficial vein codes ->
    never VTE.

Intae's script truncated every listed code to two decimal characters and
substring-matched, which conflates e.g. I82.890 (excluded) with I82.891
(history). Here a recorded code matches a listed code if it is that code or
a more specific child of it, or - for a less specific record with at least
two characters after the dot (e.g. I82.40) - if every listed code under it
falls in one class. An ambiguous parent (I82.89: excluded I82.890 and
history I82.891) is left unclassified.

`VTE_PHECODE_PREFIXES` is the phecode sensitivity endpoint (`vte_phecode`):
phecode v1.2 415.1x (pulmonary embolism) and 452.2x (DVT), deliberately
leaving out the rest of 452 (portal vein / Budd-Chiari / other venous
thrombosis). Confirm against `CODE_PATH/phecode_descriptions.csv` on the
cluster.
"""

from typing import Optional

from shared.icd10 import normalize_icd10_undotted

_ACUTE_DOTTED = (
    # Acute pulmonary embolism
    "I26.02", "I26.09", "I26.92", "I26.93", "I26.94", "I26.99",
    # Acute lower-extremity DVT
    "I80.10", "I80.11", "I80.12", "I80.13",
    "I80.201", "I80.202", "I80.203", "I80.209", "I80.211", "I80.212", "I80.213", "I80.219",
    "I80.221", "I80.222", "I80.223", "I80.229", "I80.231", "I80.232", "I80.233", "I80.239",
    "I80.241", "I80.242", "I80.243", "I80.249", "I80.251", "I80.252", "I80.253", "I80.259",
    "I80.291", "I80.292", "I80.293", "I80.299",
    "I82.220",
    "I82.401", "I82.402", "I82.403", "I82.409", "I82.411", "I82.412", "I82.413", "I82.419",
    "I82.421", "I82.422", "I82.423", "I82.429", "I82.431", "I82.432", "I82.433", "I82.439",
    "I82.441", "I82.442", "I82.443", "I82.449", "I82.451", "I82.452", "I82.453", "I82.459",
    "I82.461", "I82.462", "I82.463", "I82.469", "I82.491", "I82.492", "I82.493", "I82.499",
    "I82.4Y1", "I82.4Y2", "I82.4Y3", "I82.4Y9", "I82.4Z1", "I82.4Z2", "I82.4Z3", "I82.4Z9",
    # Acute upper-extremity DVT
    "I82.210", "I82.290", "I82.621", "I82.622", "I82.623", "I82.629",
    "I82.A11", "I82.A12", "I82.A13", "I82.A19", "I82.B11", "I82.B12", "I82.B13", "I82.B19",
    "I82.C11", "I82.C12", "I82.C13", "I82.C19", "I82.601", "I82.602", "I82.603", "I82.609",
)

_HISTORY_DOTTED = (
    # Chronic VTE
    "I27.82", "I82.211", "I82.221", "I82.291",
    "I82.501", "I82.502", "I82.503", "I82.509", "I82.511", "I82.512", "I82.513", "I82.519",
    "I82.521", "I82.522", "I82.523", "I82.529", "I82.531", "I82.532", "I82.533", "I82.539",
    "I82.541", "I82.542", "I82.543", "I82.549", "I82.551", "I82.552", "I82.553", "I82.559",
    "I82.561", "I82.562", "I82.563", "I82.569", "I82.591", "I82.592", "I82.593", "I82.599",
    "I82.5Y1", "I82.5Y2", "I82.5Y3", "I82.5Y9", "I82.5Z1", "I82.5Z2", "I82.5Z3", "I82.5Z9",
    "I82.701", "I82.702", "I82.703", "I82.709", "I82.721", "I82.722", "I82.723", "I82.729",
    "I82.891", "I82.91",
    "I82.A21", "I82.A22", "I82.A23", "I82.A29", "I82.B21", "I82.B22", "I82.B23", "I82.B29",
    "I82.C21", "I82.C22", "I82.C23", "I82.C29",
    # Personal history of VTE
    "Z86.711", "Z86.718",
)

_EXCLUDED_DOTTED = ("I82.0", "I82.1", "I82.3", "I82.890", "I82.90", "I80.3", "I80.8", "I80.9")

ACUTE = "acute"
HISTORY = "history"
EXCLUDED = "excluded"

VTE_ICD10_CODES: dict[str, tuple[str, ...]] = {
    cls: tuple(normalize_icd10_undotted(code) for code in codes)
    for cls, codes in ((ACUTE, _ACUTE_DOTTED), (HISTORY, _HISTORY_DOTTED), (EXCLUDED, _EXCLUDED_DOTTED))
}

# Undotted length of a code with two characters after the dot, e.g. I8240.
_MIN_PARENT_LEN = 5

VTE_PHECODE_PREFIXES = ("415.1", "452.2")


def classify_vte_icd(code) -> Optional[str]:
    """`ACUTE`, `HISTORY`, or None (not VTE, excluded, or ambiguous)."""
    code = normalize_icd10_undotted(code)
    if code is None or not code.startswith(("I2", "I8", "Z8")):
        return None
    # The listed code itself or a more specific child of it.
    for cls, codes in VTE_ICD10_CODES.items():
        if code.startswith(codes):
            return None if cls == EXCLUDED else cls
    # A less specific parent whose listed children all share one class.
    if len(code) < _MIN_PARENT_LEN:
        return None
    classes = {cls for cls, codes in VTE_ICD10_CODES.items() if any(c.startswith(code) for c in codes)}
    if len(classes) != 1:
        return None
    cls = classes.pop()
    return None if cls == EXCLUDED else cls


def is_vte_phecode(phecode) -> bool:
    """True for a normalized phecode string under `VTE_PHECODE_PREFIXES`
    (e.g. '415.1', '415.11', '452.2'; not '452' or '452.0')."""
    if phecode is None:
        return False
    return str(phecode).startswith(VTE_PHECODE_PREFIXES)
