"""Build a comprehensive ICD-10-CM to phecode mapping from the Dana-Farber dataset.

Downloads the official Phecode v1.2 (ICD-10-CM) and PhecodeX v1.0 mappings,
then maps every unique ICD-10 code observed in the cohort to phecodes.
Uses v1.2 as the primary mapping and fills gaps with PhecodeX.

Outputs:
    CODE_PATH/icd10_to_phecode_mapping.csv   – the lookup table used downstream
    CODE_PATH/icd10_unmapped_codes.csv        – ICD-10 codes with no phecode match

Usage:
    python generate_icd10_to_phecode_mapping.py
"""

import io
import os
import re
import zipfile
from typing import Optional

import pandas as pd
import requests

# ── paths ────────────────────────────────────────────────────────────────────
DATA_PATH = "/data/gusev/USERS/jpconnor/data/clinical_text_embedding_project/"
CODE_PATH = os.path.join(DATA_PATH, "code_data/")
SURV_PATH = os.path.join(DATA_PATH, "time-to-event_analysis/")

# ── download URLs ────────────────────────────────────────────────────────────
PHECODE_V12_URL = (
    "https://phewascatalog.org/files/Phecode_map_v1_2_icd10cm_beta.csv.zip"
)
PHECODEX_URL = (
    "https://raw.githubusercontent.com/PheWAS/PhecodeXVocabulary/main/"
    "PhecodeX%20(version%201.0)/phecodeX_ICD_CM_map_flat.csv"
)


# ── helpers ──────────────────────────────────────────────────────────────────
def _normalize_icd10(code: str) -> Optional[str]:
    """Strip to uppercase alphanumeric, remove dots."""
    if pd.isna(code):
        return None
    code = str(code).strip().upper()
    code = re.sub(r"[^A-Z0-9]", "", code)
    return code if code else None


def _normalize_phecode(code) -> Optional[str]:
    if pd.isna(code):
        return None
    code = str(code).strip()
    code = re.sub(r"[^0-9.]", "", code)
    if not code:
        return None
    # fix multiple dots
    if code.count(".") > 1:
        left, right = code.split(".", 1)
        code = f"{left}.{right.replace('.', '')}"
    return code


def _download_v12_mapping() -> pd.DataFrame:
    """Download Phecode v1.2 ICD-10-CM mapping (zipped CSV)."""
    print("Downloading Phecode v1.2 ICD-10-CM mapping …")
    resp = requests.get(PHECODE_V12_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found inside zip: {zf.namelist()}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f)
    print(f"  v1.2 mapping: {len(df)} rows, columns = {list(df.columns)}")
    return df


def _download_phecodex_mapping() -> pd.DataFrame:
    """Download PhecodeX v1.0 ICD-CM flat mapping from GitHub."""
    print("Downloading PhecodeX v1.0 ICD-CM mapping …")
    resp = requests.get(PHECODEX_URL, timeout=120)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    # keep only ICD-10-CM rows
    if "vocabulary_id" in df.columns:
        df = df.loc[df["vocabulary_id"] == "ICD10CM"].copy()
    print(f"  PhecodeX mapping: {len(df)} ICD-10-CM rows, columns = {list(df.columns)}")
    return df


def _resolve_column(df: pd.DataFrame, expected: str) -> str:
    col_map = {col.strip().lower().replace(" ", "_"): col for col in df.columns}
    for candidate in [expected, expected.replace("-", ""), expected.replace("_", "")]:
        if candidate in col_map:
            return col_map[candidate]
    raise ValueError(
        f"Cannot find column '{expected}' in {list(df.columns)}"
    )


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    os.makedirs(CODE_PATH, exist_ok=True)

    # 1. Load all unique ICD-10 codes from the Dana-Farber dataset
    icd_file = os.path.join(SURV_PATH, "timestamped_icd_info.csv")
    print(f"Loading ICD data from {icd_file} …")
    icd_df = pd.read_csv(icd_file, usecols=["DIAGNOSIS_ICD10_CD"])
    raw_codes = icd_df["DIAGNOSIS_ICD10_CD"].dropna().unique()
    print(f"  {len(raw_codes)} unique raw ICD-10 codes in dataset")

    # Build a lookup: normalized code → original code (keep first seen)
    norm_to_raw: dict[str, str] = {}
    for raw in raw_codes:
        norm = _normalize_icd10(raw)
        if norm and norm not in norm_to_raw:
            norm_to_raw[norm] = str(raw).strip()
    unique_norms = set(norm_to_raw.keys())
    print(f"  {len(unique_norms)} unique normalized ICD-10 codes")

    # 2. Download mapping files
    v12_df = _download_v12_mapping()
    phecodeX_df = _download_phecodex_mapping()

    # 3. Normalize the v1.2 mapping
    v12_icd_col = _resolve_column(v12_df, "icd10cm")
    v12_phe_col = _resolve_column(v12_df, "phecode")
    v12_map = v12_df[[v12_icd_col, v12_phe_col]].copy()
    v12_map.columns = ["icd10_raw", "phecode_raw"]
    v12_map["icd10_norm"] = v12_map["icd10_raw"].map(_normalize_icd10)
    v12_map["phecode"] = v12_map["phecode_raw"].map(_normalize_phecode)
    v12_map = v12_map.dropna(subset=["icd10_norm", "phecode"]).drop_duplicates(
        subset=["icd10_norm", "phecode"]
    )
    v12_map["source"] = "phecode_v1.2"
    print(f"  v1.2 normalized: {len(v12_map)} unique (ICD, phecode) pairs")

    # 4. Normalize the PhecodeX mapping
    px_icd_col = _resolve_column(phecodeX_df, "icd")
    px_phe_col = _resolve_column(phecodeX_df, "phecode")
    px_map = phecodeX_df[[px_icd_col, px_phe_col]].copy()
    px_map.columns = ["icd10_raw", "phecode_raw"]
    px_map["icd10_norm"] = px_map["icd10_raw"].map(_normalize_icd10)
    px_map["phecode"] = px_map["phecode_raw"].map(_normalize_phecode)
    px_map = px_map.dropna(subset=["icd10_norm", "phecode"]).drop_duplicates(
        subset=["icd10_norm", "phecode"]
    )
    px_map["source"] = "phecodeX_v1.0"
    print(f"  PhecodeX normalized: {len(px_map)} unique (ICD, phecode) pairs")

    # 5. Merge: v1.2 primary, PhecodeX fills gaps
    v12_covered = set(v12_map["icd10_norm"])
    px_gap_fill = px_map.loc[~px_map["icd10_norm"].isin(v12_covered)].copy()
    combined = pd.concat([v12_map, px_gap_fill], ignore_index=True)
    print(
        f"  Combined: {len(combined)} (ICD, phecode) pairs "
        f"({len(v12_map)} from v1.2 + {len(px_gap_fill)} gap-filled from PhecodeX)"
    )

    # 6. Also try parent-code matching for unmapped codes
    # If a specific code like E1165 isn't mapped, try its parent E116, then E11
    combined_norms = set(combined["icd10_norm"])
    unmapped = unique_norms - combined_norms
    print(f"  {len(unmapped)} codes still unmapped after direct matching — trying parent codes …")

    parent_rows = []
    newly_mapped = 0
    for norm_code in sorted(unmapped):
        # Try progressively shorter prefixes: e.g. E1165 → E116 → E11
        for prefix_len in range(len(norm_code) - 1, 2, -1):
            parent = norm_code[:prefix_len]
            parent_matches = combined.loc[combined["icd10_norm"] == parent, "phecode"].unique()
            if len(parent_matches) > 0:
                for phe in parent_matches:
                    parent_rows.append(
                        {
                            "icd10_raw": norm_to_raw.get(norm_code, norm_code),
                            "phecode_raw": phe,
                            "icd10_norm": norm_code,
                            "phecode": phe,
                            "source": "parent_code_match",
                        }
                    )
                newly_mapped += 1
                break

    if parent_rows:
        parent_df = pd.DataFrame(parent_rows)
        combined = pd.concat([combined, parent_df], ignore_index=True)
    print(f"  Parent-code matching recovered {newly_mapped} additional codes")

    # 7. Filter to only codes in the dataset
    final = combined.loc[combined["icd10_norm"].isin(unique_norms)].copy()
    final = final.drop_duplicates(subset=["icd10_norm", "phecode"])
    mapped_norms = set(final["icd10_norm"])
    still_unmapped = unique_norms - mapped_norms

    print(f"\n=== Summary ===")
    print(f"  Total unique ICD-10 codes in dataset:  {len(unique_norms)}")
    print(f"  Mapped to at least one phecode:        {len(mapped_norms)} ({100*len(mapped_norms)/len(unique_norms):.1f}%)")
    print(f"  Unmapped:                              {len(still_unmapped)} ({100*len(still_unmapped)/len(unique_norms):.1f}%)")
    print(f"  Total (ICD-10, phecode) pairs:         {len(final)}")
    print(f"  Unique phecodes:                       {final['phecode'].nunique()}")

    # 8. Write outputs
    out_cols = ["icd10_norm", "phecode", "source"]
    # rename to match the column names expected downstream
    output = final[out_cols].copy()
    output.columns = ["icd10_code", "phecode", "source"]
    output = output.sort_values(["icd10_code", "phecode"]).reset_index(drop=True)

    out_path = os.path.join(CODE_PATH, "icd10_to_phecode_mapping.csv")
    output.to_csv(out_path, index=False)
    print(f"\nWrote mapping: {out_path}  ({len(output)} rows)")

    if still_unmapped:
        unmapped_df = pd.DataFrame(
            {
                "icd10_code": sorted(still_unmapped),
                "raw_example": [norm_to_raw.get(c, c) for c in sorted(still_unmapped)],
            }
        )
        unmapped_path = os.path.join(CODE_PATH, "icd10_unmapped_codes.csv")
        unmapped_df.to_csv(unmapped_path, index=False)
        print(f"Wrote unmapped codes: {unmapped_path}  ({len(unmapped_df)} rows)")


if __name__ == "__main__":
    main()
